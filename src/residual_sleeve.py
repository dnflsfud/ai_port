"""Explicit OOS residual nonlinear alpha sleeve.

This module is deliberately separate from :mod:`src.rl`.  The historical
``dr_alpha`` path adds a policy score to the base score and was destructive in
its honest re-validation; it does not learn an explicit residual target.

The sleeve implemented here has three causal layers:

1. Convert the *executed-date* specific-return target to a symmetric
   cross-sectional rank and remove the contemporaneous executable base signal,
   style controls, size and sector effects.
2. Fit a small nonlinear LightGBM regressor only on matured dates carrying an
   already-OOS base prediction.  The forward horizon plus execution lag is an
   explicit embargo at every fold boundary.
3. Remove the same controls from the predicted sleeve score without labels,
   then allocate a fixed fraction of ex-ante active-risk variance to the
   covariance-orthogonal residual direction.

All production behaviour remains unchanged unless
``PipelineConfig.residual_sleeve_enabled`` is explicitly enabled.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.config import PipelineConfig
from src.features.nonlinear_confirmation import (
    build_nonlinear_confirmation_features,
)
from src.model_trainer import effective_label_horizon


BASE_SIGNAL_FEATURE = "__base_executable_signal__"
_EPS = 1e-12


@dataclass
class ResidualWalkForwardResult:
    """Compact output of the residual learner before portfolio construction."""

    raw_scores: pd.DataFrame
    scores: pd.DataFrame
    models: Dict[pd.Timestamp, Any]
    feature_names: List[str]
    diagnostics: Dict[str, Any]


def symmetric_cross_sectional_rank(frame: pd.DataFrame) -> pd.DataFrame:
    """Map each valid row to ``[-1, 1]`` with exact zero cross-sectional mean."""

    ranks = frame.rank(axis=1, method="average", na_option="keep")
    counts = frame.notna().sum(axis=1).astype(float)
    denom = (counts - 1.0).replace(0.0, np.nan)
    return ranks.sub(1.0).div(denom, axis=0).mul(2.0).sub(1.0)


def _frame_to_panel_series(frame: pd.DataFrame, name: str) -> pd.Series:
    """Stack a date-by-ticker frame without pandas-version stack semantics."""

    index = pd.MultiIndex.from_product(
        [frame.index, frame.columns], names=["date", "ticker"]
    )
    return pd.Series(frame.to_numpy().reshape(-1), index=index, name=name)


def _finite_zscore(values: pd.Series) -> pd.Series:
    out = values.astype(float).copy()
    finite = np.isfinite(out.to_numpy(dtype=float))
    if int(finite.sum()) < 2:
        return pd.Series(np.nan, index=out.index, dtype=float)
    vals = out.to_numpy(dtype=float, copy=True)
    mean = float(np.mean(vals[finite]))
    std = float(np.std(vals[finite], ddof=0))
    if not np.isfinite(std) or std <= _EPS:
        return pd.Series(np.nan, index=out.index, dtype=float)
    vals[finite] = (vals[finite] - mean) / std
    vals[~finite] = np.nan
    return pd.Series(vals, index=out.index, dtype=float)


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    joined = pd.concat([a.astype(float), b.astype(float)], axis=1).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(joined) < 3:
        return float("nan")
    if joined.iloc[:, 0].std(ddof=0) <= _EPS or joined.iloc[:, 1].std(ddof=0) <= _EPS:
        return float("nan")
    return float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))


def _date_panel_slice(
    panel: pd.DataFrame,
    date: pd.Timestamp,
    tickers: Sequence[str],
) -> pd.DataFrame:
    try:
        sub = panel.xs(date, level="date")
    except (KeyError, ValueError):
        return pd.DataFrame(index=pd.Index(tickers, name="ticker"))
    return sub.reindex(tickers)


def _control_design(
    date: pd.Timestamp,
    tickers: Sequence[str],
    base_signal: pd.Series,
    panel_slice: pd.DataFrame,
    market_cap: Optional[pd.DataFrame],
    sector_map: Optional[Mapping[str, str]],
    control_features: Sequence[str],
    required: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, pd.Index]:
    """Build a numerically stable, label-free daily orthogonalisation basis."""

    idx = pd.Index(tickers, name="ticker")
    base = base_signal.reindex(idx).astype(float)
    valid = pd.Series(np.isfinite(base.to_numpy(dtype=float)), index=idx)
    if required is not None:
        req = required.reindex(idx).astype(float)
        valid &= pd.Series(np.isfinite(req.to_numpy(dtype=float)), index=idx)

    continuous: Dict[str, pd.Series] = {"base_signal": base}
    for name in control_features:
        if name in panel_slice.columns:
            continuous[name] = panel_slice[name].reindex(idx).astype(float)

    if market_cap is not None and date in market_cap.index:
        cap = market_cap.loc[date].reindex(idx).astype(float)
        continuous["log_market_cap"] = np.log(cap.where(cap > 0.0))

    cols: Dict[str, pd.Series] = {}
    for name, series in continuous.items():
        s = series.replace([np.inf, -np.inf], np.nan)
        med = float(s.median()) if s.notna().any() else float("nan")
        if not np.isfinite(med):
            continue
        s = s.fillna(med)
        sd = float(s.std(ddof=0))
        if not np.isfinite(sd) or sd <= _EPS:
            continue
        cols[name] = (s - float(s.mean())) / sd

    X = pd.DataFrame(cols, index=idx, dtype=float)
    X.insert(0, "intercept", 1.0)

    if sector_map:
        sectors = pd.Series(
            [str(sector_map.get(str(t), "Unknown")) for t in idx],
            index=idx,
            dtype="object",
        )
        dummies = pd.get_dummies(sectors, prefix="sector", dtype=float)
        # Intercept + all dummies is rank deficient.  Deterministically drop
        # the lexicographically first sector and retain all relative effects.
        if dummies.shape[1] > 1:
            dummies = dummies.reindex(sorted(dummies.columns), axis=1).iloc[:, 1:]
            X = pd.concat([X, dummies], axis=1)

    valid &= pd.Series(np.isfinite(X.to_numpy(dtype=float)).all(axis=1), index=idx)
    valid_idx = idx[valid.to_numpy(dtype=bool)]
    return X.loc[valid_idx], valid_idx


def _ols_residual(values: pd.Series, design: pd.DataFrame) -> Tuple[pd.Series, float, int]:
    """Return SVD-OLS residual, explained-variance ratio and matrix rank."""

    y = values.reindex(design.index).to_numpy(dtype=float)
    X = design.to_numpy(dtype=float)
    beta, _sum_sq, rank, _singular = np.linalg.lstsq(X, y, rcond=1e-10)
    fitted = X @ beta
    residual = y - fitted
    y_var = float(np.var(y, ddof=0))
    r_var = float(np.var(residual, ddof=0))
    r2 = 1.0 - r_var / y_var if y_var > _EPS else float("nan")
    return pd.Series(residual, index=design.index, dtype=float), float(r2), int(rank)


def build_residual_labels(
    targets: pd.DataFrame,
    base_signal: pd.DataFrame,
    panel: pd.DataFrame,
    market_cap: Optional[pd.DataFrame],
    sector_map: Optional[Mapping[str, str]],
    config: PipelineConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Create executed-date, cross-sectionally residualised training labels.

    At signal date ``s`` and execution lag ``L``, the economic label is
    ``targets[s + L]``.  This function aligns that label but does not decide
    whether it is mature; :func:`walk_forward_residual_scores` enforces the
    ``L + effective_horizon`` fold embargo.
    """

    lag = int(getattr(config, "execution_signal_lag_days", 0))
    executed_targets = targets.shift(-lag) if lag > 0 else targets.copy()
    ranked = symmetric_cross_sectional_rank(executed_targets)
    tickers = list(base_signal.columns)
    out = pd.DataFrame(np.nan, index=base_signal.index, columns=tickers, dtype=float)
    min_names = int(getattr(config, "residual_sleeve_min_names", 50))
    controls = list(getattr(config, "residual_sleeve_orthogonal_features", []))

    r2_values: List[float] = []
    ranks: List[int] = []
    max_abs_corrs: List[float] = []

    for date in base_signal.index.intersection(ranked.index):
        y = ranked.loc[date].reindex(tickers)
        panel_slice = _date_panel_slice(panel, date, tickers)
        X, valid_idx = _control_design(
            date,
            tickers,
            base_signal.loc[date],
            panel_slice,
            market_cap,
            sector_map,
            controls,
            required=y,
        )
        if len(valid_idx) < min_names or len(valid_idx) <= X.shape[1] + 2:
            continue
        residual, r2, matrix_rank = _ols_residual(y, X)
        z = _finite_zscore(residual)
        if z.notna().sum() < min_names:
            continue
        out.loc[date, z.index] = z.to_numpy(dtype=float)
        r2_values.append(r2)
        ranks.append(matrix_rank)
        corr_values = [
            abs(_safe_corr(z, X[col]))
            for col in X.columns
            if col != "intercept"
        ]
        finite_corrs = [v for v in corr_values if np.isfinite(v)]
        if finite_corrs:
            max_abs_corrs.append(float(max(finite_corrs)))

    valid_dates = out.notna().sum(axis=1) >= min_names
    diagnostics = {
        "execution_lag_days": lag,
        "label_dates": int(valid_dates.sum()),
        "first_label_date": (
            str(out.index[valid_dates][0].date()) if valid_dates.any() else None
        ),
        "median_removed_r2": (
            float(np.nanmedian(r2_values)) if r2_values else None
        ),
        "median_design_rank": (
            float(np.nanmedian(ranks)) if ranks else None
        ),
        "max_label_control_abs_corr": (
            float(np.nanmax(max_abs_corrs)) if max_abs_corrs else None
        ),
    }
    return out, diagnostics


def build_residual_feature_panel(
    panel: pd.DataFrame,
    feature_names: Sequence[str],
    base_signal: pd.DataFrame,
    data: Any,
    config: PipelineConfig,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build the secondary learner panel without changing the production panel."""

    names = [name for name in feature_names if name in panel.columns]
    out = panel.loc[:, names].copy()
    base_series = _frame_to_panel_series(base_signal, BASE_SIGNAL_FEATURE)
    out[BASE_SIGNAL_FEATURE] = base_series.reindex(out.index)
    names.append(BASE_SIGNAL_FEATURE)

    if getattr(config, "residual_sleeve_include_confirmation_features", False):
        parent_frames = {
            name: panel[name].unstack("ticker")
            for name in feature_names
            if name in panel.columns
        }
        confirmations = build_nonlinear_confirmation_features(parent_frames, data=data)
        for name, frame in confirmations.items():
            out[name] = _frame_to_panel_series(frame, name).reindex(out.index)
            names.append(name)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out, names


def walk_forward_residual_scores(
    feature_panel: pd.DataFrame,
    residual_labels: pd.DataFrame,
    base_signal: pd.DataFrame,
    feature_names: Sequence[str],
    config: PipelineConfig,
) -> Tuple[pd.DataFrame, Dict[pd.Timestamp, Any], Dict[str, Any]]:
    """Fit and emit causal OOS residual scores on the base signal grid."""

    dates = pd.DatetimeIndex(base_signal.index)
    tickers = list(base_signal.columns)
    raw = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)
    models: Dict[pd.Timestamp, Any] = {}

    prior_any = base_signal.notna().any(axis=1).to_numpy(dtype=bool)
    if not prior_any.any():
        return raw, models, {
            "folds_attempted": 0,
            "folds_trained": 0,
            "reason": "base_signal_has_no_coverage",
        }

    first_prior_pos = int(np.flatnonzero(prior_any)[0])
    train_window = int(getattr(config, "residual_sleeve_train_window", 1260))
    retrain_freq = int(getattr(config, "residual_sleeve_retrain_freq", 63))
    sample_freq = int(getattr(config, "residual_sleeve_sample_freq", 21))
    min_train_dates = int(getattr(config, "residual_sleeve_min_train_dates", 12))
    min_names = int(getattr(config, "residual_sleeve_min_names", 50))
    lag = int(getattr(config, "execution_signal_lag_days", 0))
    horizon = int(effective_label_horizon(config))
    embargo = horizon + lag
    params = copy.deepcopy(getattr(config, "residual_sleeve_lgbm_params", {}))

    folds: List[Dict[str, Any]] = []
    importance_rows: List[np.ndarray] = []
    starts = list(range(first_prior_pos, len(dates), retrain_freq))

    for start_pos in starts:
        fold_date = dates[start_pos]
        cutoff_pos = start_pos - embargo
        fold_info: Dict[str, Any] = {
            "fold_date": str(fold_date.date()),
            "fold_position": int(start_pos),
            "embargo_days": int(embargo),
            "effective_horizon": int(horizon),
            "execution_lag_days": int(lag),
            "trained": False,
        }
        if cutoff_pos < first_prior_pos:
            fold_info["reason"] = "insufficient_mature_oof_history"
            folds.append(fold_info)
            continue

        lo = max(first_prior_pos, cutoff_pos - train_window + 1)
        sampled_positions = [
            p
            for p in range(first_prior_pos, cutoff_pos + 1, sample_freq)
            if p >= lo
        ]
        sampled_dates = [
            dates[p]
            for p in sampled_positions
            if int(residual_labels.loc[dates[p]].notna().sum()) >= min_names
        ]
        fold_info["sampled_train_dates"] = int(len(sampled_dates))
        if len(sampled_dates) < min_train_dates:
            fold_info["reason"] = "insufficient_sampled_train_dates"
            folds.append(fold_info)
            continue

        X_parts: List[pd.DataFrame] = []
        y_parts: List[pd.Series] = []
        accepted_dates: List[pd.Timestamp] = []
        for date in sampled_dates:
            sub = _date_panel_slice(feature_panel, date, tickers).reindex(columns=feature_names)
            y = residual_labels.loc[date].reindex(tickers).astype(float)
            prior = base_signal.loc[date].reindex(tickers).astype(float)
            valid = np.isfinite(y.to_numpy(dtype=float)) & np.isfinite(
                prior.to_numpy(dtype=float)
            )
            if int(valid.sum()) < min_names:
                continue
            X_parts.append(sub.iloc[np.flatnonzero(valid)].astype(float))
            y_parts.append(y.iloc[np.flatnonzero(valid)])
            accepted_dates.append(date)

        if len(accepted_dates) < min_train_dates:
            fold_info["reason"] = "insufficient_accepted_train_dates"
            folds.append(fold_info)
            continue

        X_train = pd.concat(X_parts, axis=0).replace([np.inf, -np.inf], np.nan)
        y_train = pd.concat(y_parts, axis=0)
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        models[fold_date] = model

        latest_train_pos = dates.get_loc(accepted_dates[-1])
        latest_label_realisation_pos = int(latest_train_pos + lag + horizon)
        if latest_label_realisation_pos > start_pos:
            raise RuntimeError(
                "residual sleeve leakage guard failed: latest training label "
                f"realises at pos {latest_label_realisation_pos}, after fold "
                f"boundary {start_pos}"
            )

        block_end = min(start_pos + retrain_freq, len(dates))
        emitted_dates = 0
        for pred_pos in range(start_pos, block_end):
            date = dates[pred_pos]
            sub = _date_panel_slice(feature_panel, date, tickers).reindex(columns=feature_names)
            prior = base_signal.loc[date].reindex(tickers).astype(float)
            valid = np.isfinite(prior.to_numpy(dtype=float))
            if int(valid.sum()) < min_names:
                continue
            values = model.predict(sub.iloc[np.flatnonzero(valid)].astype(float))
            values = np.asarray(values, dtype=float)
            finite_pred = np.isfinite(values)
            if int(finite_pred.sum()) < min_names:
                continue
            names = np.asarray(tickers, dtype=object)[np.flatnonzero(valid)][finite_pred]
            raw.loc[date, list(names)] = values[finite_pred]
            emitted_dates += 1

        booster = getattr(model, "booster_", None)
        tree_count = int(booster.num_trees()) if booster is not None else None
        importance_rows.append(np.asarray(model.feature_importances_, dtype=float))
        fold_info.update({
            "trained": True,
            "train_rows": int(len(X_train)),
            "accepted_train_dates": int(len(accepted_dates)),
            "first_train_date": str(accepted_dates[0].date()),
            "last_train_date": str(accepted_dates[-1].date()),
            "latest_label_realisation_position": latest_label_realisation_pos,
            "emitted_dates": int(emitted_dates),
            "tree_count": tree_count,
        })
        folds.append(fold_info)

    trained = [f for f in folds if f.get("trained")]
    first_active_mask = raw.notna().sum(axis=1) >= min_names
    if importance_rows:
        mean_imp = np.mean(np.vstack(importance_rows), axis=0)
        total_imp = float(mean_imp.sum())
        shares = mean_imp / total_imp if total_imp > 0 else mean_imp
        order = np.argsort(shares)[::-1]
        top_importance = [
            {"feature": str(feature_names[i]), "share": float(shares[i])}
            for i in order[:15]
        ]
    else:
        top_importance = []

    diagnostics = {
        "first_base_signal_date": str(dates[first_prior_pos].date()),
        "first_active_signal_date": (
            str(dates[first_active_mask][0].date()) if first_active_mask.any() else None
        ),
        "folds_attempted": int(len(folds)),
        "folds_trained": int(len(trained)),
        "fold_audit": folds,
        "embargo_days": int(embargo),
        "effective_horizon": int(horizon),
        "execution_lag_days": int(lag),
        "active_signal_dates": int(first_active_mask.sum()),
        "top_feature_importance": top_importance,
        "max_feature_importance_share": (
            float(top_importance[0]["share"]) if top_importance else None
        ),
    }
    return raw, models, diagnostics


def orthogonalize_residual_scores(
    raw_scores: pd.DataFrame,
    base_signal: pd.DataFrame,
    panel: pd.DataFrame,
    market_cap: Optional[pd.DataFrame],
    sector_map: Optional[Mapping[str, str]],
    config: PipelineConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Remove the base/style/size/sector span from predicted scores each day."""

    tickers = list(base_signal.columns)
    out = pd.DataFrame(np.nan, index=base_signal.index, columns=tickers, dtype=float)
    min_names = int(getattr(config, "residual_sleeve_min_names", 50))
    controls = list(getattr(config, "residual_sleeve_orthogonal_features", []))
    clip_q = float(getattr(config, "residual_sleeve_clip_quantile", 0.01))

    r2_values: List[float] = []
    before_base_corr: List[float] = []
    after_base_corr: List[float] = []
    max_control_corr: List[float] = []

    for date in raw_scores.index.intersection(base_signal.index):
        score = raw_scores.loc[date].reindex(tickers).astype(float)
        if score.notna().sum() < min_names:
            continue
        panel_slice = _date_panel_slice(panel, date, tickers)
        X, valid_idx = _control_design(
            date,
            tickers,
            base_signal.loc[date],
            panel_slice,
            market_cap,
            sector_map,
            controls,
            required=score,
        )
        if len(valid_idx) < min_names or len(valid_idx) <= X.shape[1] + 2:
            continue
        clipped = score.reindex(valid_idx)
        if clip_q > 0.0:
            lo, hi = clipped.quantile([clip_q, 1.0 - clip_q]).to_numpy(dtype=float)
            clipped = clipped.clip(lower=float(lo), upper=float(hi))
        residual, r2, _rank = _ols_residual(clipped, X)
        z = _finite_zscore(residual)
        if z.notna().sum() < min_names:
            continue
        out.loc[date, z.index] = z.to_numpy(dtype=float)
        r2_values.append(r2)
        before_base_corr.append(_safe_corr(clipped, base_signal.loc[date]))
        after_base_corr.append(_safe_corr(z, base_signal.loc[date]))
        correlations = [
            abs(_safe_corr(z, X[col]))
            for col in X.columns
            if col != "intercept"
        ]
        finite_corrs = [v for v in correlations if np.isfinite(v)]
        if finite_corrs:
            max_control_corr.append(float(max(finite_corrs)))

    active = out.notna().sum(axis=1) >= min_names
    persistence_values: List[float] = []
    for pos in range(21, len(out)):
        if active.iloc[pos] and active.iloc[pos - 21]:
            persistence_values.append(_safe_corr(out.iloc[pos], out.iloc[pos - 21]))

    values = out.to_numpy(dtype=float)
    total_var = float(np.nanvar(values)) if np.isfinite(values).any() else float("nan")
    ticker_means = out.mean(axis=0, skipna=True).to_numpy(dtype=float)
    between_var = (
        float(np.nanvar(ticker_means)) if np.isfinite(ticker_means).any() else float("nan")
    )
    between_share = (
        between_var / total_var
        if np.isfinite(total_var) and total_var > _EPS and np.isfinite(between_var)
        else float("nan")
    )

    diagnostics = {
        "active_score_dates": int(active.sum()),
        "first_active_score_date": (
            str(out.index[active][0].date()) if active.any() else None
        ),
        "median_r2_removed_at_prediction": (
            float(np.nanmedian(r2_values)) if r2_values else None
        ),
        "median_abs_base_corr_before": (
            float(np.nanmedian(np.abs(before_base_corr))) if before_base_corr else None
        ),
        "median_abs_base_corr_after": (
            float(np.nanmedian(np.abs(after_base_corr))) if after_base_corr else None
        ),
        "max_abs_control_corr_after": (
            float(np.nanmax(max_control_corr)) if max_control_corr else None
        ),
        "score_persistence_21d": (
            float(np.nanmedian(persistence_values)) if persistence_values else None
        ),
        "between_ticker_variance_share": (
            float(between_share) if np.isfinite(between_share) else None
        ),
        "nonzero_dispersion_rate": (
            float(active.mean()) if len(active) else 0.0
        ),
        "post_warmup_nonzero_dispersion_rate": (
            float(active.loc[out.index[active][0]:].mean()) if active.any() else 0.0
        ),
    }
    return out, diagnostics


def smooth_residual_scores(
    raw_scores: pd.DataFrame,
    span: int,
) -> pd.DataFrame:
    """Past-only score stabilisation aligned to the portfolio rebalance span.

    ``adjust=False`` is the recursive live form; no future observation enters
    an earlier score.  The span is not a tuned sleeve parameter: S13.16 pins it
    to the existing production rebalance frequency.
    """

    span = int(span)
    if span < 1:
        raise ValueError("residual score smoothing span must be >= 1")
    if span == 1:
        return raw_scores.copy()
    return raw_scores.ewm(
        span=span,
        adjust=False,
        min_periods=1,
        ignore_na=True,
    ).mean()


def compose_fixed_risk_sleeve(
    base_weights: np.ndarray,
    residual_weights: np.ndarray,
    benchmark_weights: np.ndarray,
    covariance: np.ndarray,
    risk_fraction: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Create reserve-only B and covariance-orthogonal residual C targets.

    ``B`` carries ``1-rho`` of the base active variance.  ``C`` carries the
    same total pre-projection variance as A, with ``rho`` assigned to a
    residual direction that is orthogonal to A under ``covariance``.
    """

    rho = float(risk_fraction)
    if not (0.0 <= rho < 1.0):
        raise ValueError("risk_fraction must be in [0, 1)")
    base_w = np.asarray(base_weights, dtype=float)
    residual_w = np.asarray(residual_weights, dtype=float)
    bm = np.asarray(benchmark_weights, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    a_base = base_w - bm
    a_residual = residual_w - bm
    base_var = float(a_base @ cov @ a_base)
    if rho == 0.0 or not np.isfinite(base_var) or base_var <= _EPS:
        info = {
            "composed": 0.0,
            "base_te_annual": math.sqrt(max(base_var, 0.0) * 252.0),
            "residual_te_annual": 0.0,
            "pre_projection_residual_variance_share": 0.0,
            "pre_projection_risk_correlation": 0.0,
        }
        return base_w.copy(), base_w.copy(), info

    cross = float(a_base @ cov @ a_residual)
    orth = a_residual - a_base * (cross / base_var)
    orth_var = float(orth @ cov @ orth)
    if not np.isfinite(orth_var) or orth_var <= _EPS:
        info = {
            "composed": 0.0,
            "base_te_annual": math.sqrt(base_var * 252.0),
            "residual_te_annual": 0.0,
            "pre_projection_residual_variance_share": 0.0,
            "pre_projection_risk_correlation": 0.0,
        }
        return base_w.copy(), base_w.copy(), info

    residual_unit = orth * math.sqrt(base_var / orth_var)
    post_cross = float(a_base @ cov @ residual_unit)
    residual_unit_var = float(residual_unit @ cov @ residual_unit)
    risk_corr = post_cross / math.sqrt(base_var * residual_unit_var)
    base_scale = math.sqrt(1.0 - rho)
    residual_scale = math.sqrt(rho)
    b_active = base_scale * a_base
    residual_component = residual_scale * residual_unit
    c_active = b_active + residual_component

    b = bm + b_active
    c = bm + c_active
    c_var = float(c_active @ cov @ c_active)
    residual_component_var = float(residual_component @ cov @ residual_component)
    info = {
        "composed": 1.0,
        "base_te_annual": math.sqrt(base_var * 252.0),
        "reserve_te_annual": math.sqrt(float(b_active @ cov @ b_active) * 252.0),
        "residual_te_annual": math.sqrt(residual_component_var * 252.0),
        "candidate_te_annual": math.sqrt(max(c_var, 0.0) * 252.0),
        "pre_projection_residual_variance_share": (
            residual_component_var / c_var if c_var > _EPS else 0.0
        ),
        "pre_projection_risk_correlation": float(risk_corr),
    }
    return b, c, info


def _metric_snapshot(result: Any) -> Dict[str, Any]:
    from src.harness import sub_period_irs

    metrics = result.compute_metrics()
    metrics["sub_periods"] = sub_period_irs(
        result.portfolio_returns.dropna(), result.benchmark_returns.dropna()
    )
    return metrics


def _metric_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]:
    keys = (
        "information_ratio",
        "active_return",
        "tracking_error",
        "avg_annual_turnover",
        "realized_beta",
        "active_share",
        "max_drawdown",
    )
    out: Dict[str, Any] = {}
    for key in keys:
        lv, rv = left.get(key), right.get(key)
        if isinstance(lv, (int, float)) and isinstance(rv, (int, float)):
            out[key] = float(lv - rv)
    lsp = left.get("sub_periods") or {}
    rsp = right.get("sub_periods") or {}
    out["sub_periods"] = {
        key: float(lsp[key] - rsp[key])
        for key in ("P1_ir", "P2_ir", "P3_ir")
        if isinstance(lsp.get(key), (int, float))
        and isinstance(rsp.get(key), (int, float))
    }
    return out


def _copy_backtest_metadata(target: Any, base: Any) -> None:
    for name in (
        "models",
        "targets",
        "panel",
        "feature_names",
        "feature_groups",
        "model_quality",
        "data_quality",
        "benchmark_type",
        "pre_overlay_predictions",
        "pre_execution_predictions",
        "execution_signal_lag_days",
    ):
        if hasattr(base, name):
            setattr(target, name, getattr(base, name))


def _portfolio_divergence(left: Any, right: Any) -> float:
    dates = sorted(set(left.portfolio_weights).intersection(right.portfolio_weights))
    values: List[float] = []
    for date in dates:
        a = left.portfolio_weights[date]
        b = right.portfolio_weights[date]
        idx = a.index.union(b.index)
        values.append(float(0.5 * (a.reindex(idx, fill_value=0.0) - b.reindex(idx, fill_value=0.0)).abs().sum()))
    return float(np.mean(values)) if values else float("nan")


def _incremental_ir(left: pd.Series, right: pd.Series) -> float:
    joined = pd.concat([left, right], axis=1).dropna()
    if len(joined) < 2:
        return float("nan")
    diff = joined.iloc[:, 0] - joined.iloc[:, 1]
    sd = float(diff.std(ddof=1))
    return float(diff.mean() / sd * math.sqrt(252.0)) if sd > _EPS else float("nan")


def _mean_active_share(result: Any, data: Any, config: PipelineConfig) -> float:
    from src.backtest import get_benchmark_fn

    tickers = list(data.tickers)
    bm_fn = get_benchmark_fn(data, tickers, config=config)
    values: List[float] = []
    for date, weights in result.portfolio_weights.items():
        bm = np.asarray(bm_fn(date, tickers, len(tickers)), dtype=float)
        w = weights.reindex(tickers).to_numpy(dtype=float)
        values.append(float(0.5 * np.abs(w - bm).sum()))
    return float(np.mean(values)) if values else float("nan")


def _post_projection_risk_diagnostics(
    reserve_result: Any,
    candidate_result: Any,
    data: Any,
    config: PipelineConfig,
    active_dates: Optional[Iterable[pd.Timestamp]] = None,
) -> Dict[str, Any]:
    from src.backtest import get_benchmark_fn
    from src.portfolio_optimizer import estimate_covariance

    tickers = list(data.tickers)
    bm_fn = get_benchmark_fn(data, tickers, config=config)
    risk_returns = getattr(data, "raw_returns", None)
    if risk_returns is None:
        risk_returns = data.returns
    risk_returns = risk_returns.reindex(index=data.returns.index, columns=tickers)
    pos = {d: i for i, d in enumerate(data.returns.index)}
    dates = sorted(
        set(reserve_result.portfolio_weights).intersection(candidate_result.portfolio_weights)
    )
    if active_dates is not None:
        active_set = {pd.Timestamp(d) for d in active_dates}
        dates = [d for d in dates if pd.Timestamp(d) in active_set]
    shares: List[float] = []
    correlations: List[float] = []
    for date in dates:
        if date not in pos:
            continue
        end = pos[date]
        start = max(0, end - int(getattr(config, "cov_lookback", 126)))
        bm = np.asarray(bm_fn(date, tickers, len(tickers)), dtype=float)
        cov = estimate_covariance(
            risk_returns.iloc[start:end], bm_weights=bm, config=config
        )
        wb = reserve_result.portfolio_weights[date].reindex(tickers).to_numpy(dtype=float)
        wc = candidate_result.portfolio_weights[date].reindex(tickers).to_numpy(dtype=float)
        a_base = wb - bm
        residual = wc - wb
        total = wc - bm
        base_var = float(a_base @ cov @ a_base)
        residual_var = float(residual @ cov @ residual)
        total_var = float(total @ cov @ total)
        cross = float(a_base @ cov @ residual)
        if total_var > _EPS:
            shares.append(residual_var / total_var)
        if base_var > _EPS and residual_var > _EPS:
            correlations.append(cross / math.sqrt(base_var * residual_var))

    def _pct(values: Sequence[float], q: float) -> Optional[float]:
        return float(np.nanquantile(values, q)) if values else None

    return {
        "observations": int(len(shares)),
        "median_residual_variance_share": _pct(shares, 0.50),
        "p10_residual_variance_share": _pct(shares, 0.10),
        "p90_residual_variance_share": _pct(shares, 0.90),
        "median_risk_correlation": _pct(correlations, 0.50),
        "p90_abs_risk_correlation": (
            float(np.nanquantile(np.abs(correlations), 0.90)) if correlations else None
        ),
    }


def run_residual_sleeve_experiment(
    base_result: Any,
    data: Any,
    config: PipelineConfig,
) -> Tuple[Any, Dict[str, Any]]:
    """Run the preregistered A/B/C experiment from one harvested baseline."""

    if not getattr(config, "residual_sleeve_enabled", False):
        raise ValueError("run_residual_sleeve_experiment requires residual_sleeve_enabled=True")
    if getattr(config, "dr_alpha_enabled", False):
        raise ValueError("residual_sleeve_enabled and dr_alpha_enabled are mutually exclusive")

    from src.backtest import (
        get_benchmark_fn,
        get_sector_map,
        simulate_portfolio,
    )
    from src.portfolio_optimizer import estimate_covariance, optimize_portfolio

    lag = int(getattr(config, "execution_signal_lag_days", 0))
    if lag > 0:
        base_signal = getattr(base_result, "pre_execution_predictions", None)
        if base_signal is None:
            raise ValueError(
                "baseline artifact lacks pre_execution_predictions required for "
                f"execution_signal_lag_days={lag}"
            )
    else:
        base_signal = base_result.predictions.copy()

    tickers = list(data.tickers)
    base_signal = base_signal.reindex(index=base_result.targets.index, columns=tickers)
    market_cap = getattr(data, "market_cap", None)
    sector_map = get_sector_map(data)

    labels, label_diag = build_residual_labels(
        base_result.targets,
        base_signal,
        base_result.panel,
        market_cap,
        sector_map,
        config,
    )
    residual_panel, residual_features = build_residual_feature_panel(
        base_result.panel,
        base_result.feature_names,
        base_signal,
        data,
        config,
    )
    raw_scores, residual_models, fold_diag = walk_forward_residual_scores(
        residual_panel,
        labels,
        base_signal,
        residual_features,
        config,
    )
    smoothed_raw_scores = smooth_residual_scores(
        raw_scores, span=int(config.rebalance_freq)
    )
    scores_signal, orth_diag = orthogonalize_residual_scores(
        smoothed_raw_scores,
        base_signal,
        base_result.panel,
        market_cap,
        sector_map,
        config,
    )
    scores_execution = scores_signal.shift(lag) if lag > 0 else scores_signal.copy()

    min_names = int(getattr(config, "residual_sleeve_min_names", 50))
    active_execution = scores_execution.notna().sum(axis=1) >= min_names
    first_active_date = (
        scores_execution.index[active_execution][0] if active_execution.any() else None
    )
    rho = float(getattr(config, "residual_sleeve_risk_fraction", 0.10))
    tickers = list(data.tickers)
    n_tickers = len(tickers)
    returns = data.returns
    all_dates = data.dates
    risk_returns = getattr(data, "raw_returns", None)
    if risk_returns is not None:
        risk_returns = risk_returns.reindex(index=returns.index, columns=tickers)
    bm_fn = get_benchmark_fn(data, tickers, config=config)
    has_spx = bool(
        getattr(config, "sp500_benchmark_enabled", True)
        and data.has_factor_data()
        and str(getattr(config, "sp500_factor_ticker", "SPX")) in data.factor_returns.columns
    )
    spx = (
        data.factor_returns[str(getattr(config, "sp500_factor_ticker", "SPX"))]
        if has_spx
        else None
    )

    residual_config = copy.deepcopy(config)
    residual_config.risk_aversion = float(
        getattr(config, "residual_sleeve_risk_aversion", 1.0)
    )
    residual_config.turnover_penalty = float(
        getattr(config, "residual_sleeve_turnover_penalty", 0.03)
    )
    # This model is a stand-alone residual direction; the final C target still
    # passes through the production base-score OW gate in the shared projection.
    residual_config.enforce_score_gated_ow = True

    telemetry_by_arm: Dict[str, List[Dict[str, Any]]] = {"B": [], "C": []}

    def run_arm(arm: str):
        def optimizer_fn(
            pred_row,
            hist_returns,
            prev_weights,
            s_map,
            bm_weights,
            diagnostics=None,
        ):
            cov = estimate_covariance(
                hist_returns, bm_weights=bm_weights, config=config
            )
            base_diag: Dict[str, Any] = {}
            w_base = optimize_portfolio(
                expected_returns=pred_row,
                cov_matrix=cov,
                prev_weights=prev_weights,
                sector_map=s_map if s_map else None,
                bm_weights=bm_weights,
                config=config,
                diagnostics=base_diag,
            )
            q = (
                scores_execution.loc[pred_row.name].reindex(pred_row.index)
                if pred_row.name in scores_execution.index
                else pd.Series(np.nan, index=pred_row.index)
            )
            active = int(q.notna().sum()) >= min_names
            row_info: Dict[str, Any] = {
                "date": str(pd.Timestamp(pred_row.name).date()),
                "active": bool(active),
                "score_coverage": int(q.notna().sum()),
                "base_solver": base_diag.get("solver"),
                "base_fallback": bool(base_diag.get("used_fallback", False)),
            }
            target = w_base
            residual_diag: Dict[str, Any] = {}
            if active:
                if arm == "B":
                    a = np.asarray(w_base, dtype=float) - np.asarray(bm_weights, dtype=float)
                    target = np.asarray(bm_weights, dtype=float) + math.sqrt(1.0 - rho) * a
                    row_info.update({
                        "composed": True,
                        "pre_projection_residual_variance_share": 0.0,
                        "pre_projection_risk_correlation": 0.0,
                    })
                else:
                    w_residual = optimize_portfolio(
                        expected_returns=q,
                        cov_matrix=cov,
                        prev_weights=prev_weights,
                        sector_map=s_map if s_map else None,
                        bm_weights=bm_weights,
                        config=residual_config,
                        diagnostics=residual_diag,
                    )
                    _reserve, candidate, compose_info = compose_fixed_risk_sleeve(
                        w_base,
                        w_residual,
                        bm_weights,
                        cov,
                        rho,
                    )
                    target = candidate
                    row_info.update(compose_info)
                    row_info["residual_solver"] = residual_diag.get("solver")
                    row_info["residual_fallback"] = bool(
                        residual_diag.get("used_fallback", False)
                    )

            telemetry_by_arm[arm].append(row_info)
            if diagnostics is not None:
                diagnostics.update(base_diag)
                diagnostics["cov_matrix"] = cov
                diagnostics["max_te_annual"] = config.max_te_annual
                diagnostics["sector_deviation"] = config.sector_deviation
                if arm == "C" and residual_diag.get("used_fallback", False):
                    diagnostics["used_fallback"] = True
                    diagnostics["fallback_reason"] = (
                        "residual_direction:" + str(
                            residual_diag.get("fallback_reason") or "unknown"
                        )
                    )
            return np.asarray(target, dtype=float)

        result = simulate_portfolio(
            predictions=base_result.predictions,
            returns=returns,
            tickers=tickers,
            all_dates=all_dates,
            sector_map=sector_map,
            rebalance_freq=config.rebalance_freq,
            one_way_tc=config.one_way_tc,
            optimizer_fn=optimizer_fn,
            targets=base_result.targets,
            bm_weights_fn=bm_fn,
            rebal_check_fn=None,
            weight_drift=True,
            bm_drift=True,
            track_ic=True,
            track_spx=has_spx,
            raw_predictions=base_result.raw_predictions,
            config=config,
            spx_series=spx,
            track_daily_weights=True,
            risk_returns=risk_returns,
        )
        _copy_backtest_metadata(result, base_result)
        result.predictions = base_result.predictions
        result.raw_predictions = base_result.raw_predictions
        return result

    reserve_result = run_arm("B")
    candidate_result = run_arm("C")
    candidate_result.residual_scores_signal = scores_signal
    candidate_result.residual_scores = scores_execution
    candidate_result.residual_raw_scores = raw_scores
    candidate_result.residual_smoothed_raw_scores = smoothed_raw_scores
    candidate_result.residual_models = residual_models

    metrics_a = _metric_snapshot(base_result)
    metrics_b = _metric_snapshot(reserve_result)
    metrics_c = _metric_snapshot(candidate_result)
    post_risk = _post_projection_risk_diagnostics(
        reserve_result,
        candidate_result,
        data,
        config,
        active_dates=scores_execution.index[active_execution],
    )

    active_rebalances = sum(1 for x in telemetry_by_arm["C"] if x.get("active"))
    composed_rebalances = sum(
        1 for x in telemetry_by_arm["C"] if float(x.get("composed", 0.0)) > 0.5
    )
    total_rebalances = len(telemetry_by_arm["C"])
    if first_active_date is not None:
        post_warmup_rows = [
            x
            for x in telemetry_by_arm["C"]
            if pd.Timestamp(x["date"]) >= first_active_date
        ]
        post_warmup_active = sum(1 for x in post_warmup_rows if x.get("active"))
        post_warmup_activation_rate = (
            float(post_warmup_active / len(post_warmup_rows))
            if post_warmup_rows
            else 0.0
        )
    else:
        post_warmup_activation_rate = 0.0

    preactivation_parity = None
    if first_active_date is not None:
        diffs: List[float] = []
        common = sorted(
            set(base_result.portfolio_weights)
            .intersection(reserve_result.portfolio_weights)
            .intersection(candidate_result.portfolio_weights)
        )
        for date in common:
            if date >= first_active_date:
                continue
            wa = base_result.portfolio_weights[date].reindex(tickers).to_numpy(dtype=float)
            wb = reserve_result.portfolio_weights[date].reindex(tickers).to_numpy(dtype=float)
            wc = candidate_result.portfolio_weights[date].reindex(tickers).to_numpy(dtype=float)
            diffs.extend([float(np.max(np.abs(wa - wb))), float(np.max(np.abs(wa - wc)))])
        if diffs:
            preactivation_parity = float(max(diffs))

    # Residual score IC on executable dates, using the same target definition
    # consumed by the portfolio at each close.
    score_ics: List[float] = []
    for date in scores_execution.index[active_execution]:
        if date not in base_result.targets.index:
            continue
        score_ics.append(
            _safe_corr(scores_execution.loc[date], base_result.targets.loc[date])
        )

    joined_active = pd.concat(
        [
            base_result.portfolio_returns - base_result.benchmark_returns,
            candidate_result.portfolio_returns - candidate_result.benchmark_returns,
        ],
        axis=1,
    ).dropna()
    active_return_corr = (
        float(joined_active.iloc[:, 0].corr(joined_active.iloc[:, 1]))
        if len(joined_active) >= 3
        else None
    )

    diagnostics = {
        "configuration": {
            "risk_fraction": rho,
            "base_variance_fraction": 1.0 - rho,
            "train_window": int(config.residual_sleeve_train_window),
            "retrain_freq": int(config.residual_sleeve_retrain_freq),
            "sample_freq": int(config.residual_sleeve_sample_freq),
            "score_smoothing_span": int(config.rebalance_freq),
            "min_train_dates": int(config.residual_sleeve_min_train_dates),
            "min_names": min_names,
            "feature_count": int(len(residual_features)),
            "model_params": copy.deepcopy(config.residual_sleeve_lgbm_params),
        },
        "label": label_diag,
        "walk_forward": fold_diag,
        "orthogonalization": orth_diag,
        "first_active_execution_date": (
            str(first_active_date.date()) if first_active_date is not None else None
        ),
        "active_rebalances": int(active_rebalances),
        "composed_rebalances": int(composed_rebalances),
        "total_rebalances": int(total_rebalances),
        "activation_rate": (
            float(active_rebalances / total_rebalances) if total_rebalances else 0.0
        ),
        "post_warmup_activation_rate": post_warmup_activation_rate,
        "unexplained_passthrough_rebalances": int(active_rebalances - composed_rebalances),
        "preactivation_max_weight_difference": preactivation_parity,
        "mean_residual_score_ic": (
            float(np.nanmean(score_ics)) if score_ics else None
        ),
        "post_projection_risk": post_risk,
        "mean_one_way_book_divergence_c_vs_a": _portfolio_divergence(
            candidate_result, base_result
        ),
        "mean_one_way_book_divergence_c_vs_b": _portfolio_divergence(
            candidate_result, reserve_result
        ),
        "active_return_correlation_c_vs_a": active_return_corr,
        "incremental_ir_c_vs_a": _incremental_ir(
            candidate_result.portfolio_returns, base_result.portfolio_returns
        ),
        "incremental_ir_c_vs_b": _incremental_ir(
            candidate_result.portfolio_returns, reserve_result.portfolio_returns
        ),
        "mean_active_share_a": _mean_active_share(base_result, data, config),
        "mean_active_share_b": _mean_active_share(reserve_result, data, config),
        "mean_active_share_c": _mean_active_share(candidate_result, data, config),
        "optimizer_failure_rate_a": float(
            getattr(base_result, "optimizer_failure_rate", 0.0)
        ),
        "optimizer_failure_rate_b": float(
            getattr(reserve_result, "optimizer_failure_rate", 0.0)
        ),
        "optimizer_failure_rate_c": float(
            getattr(candidate_result, "optimizer_failure_rate", 0.0)
        ),
        "optimizer_telemetry": telemetry_by_arm,
    }

    delta_ca = _metric_delta(metrics_c, metrics_a)
    delta_cb = _metric_delta(metrics_c, metrics_b)
    ca_sub = delta_ca.get("sub_periods") or {}
    cb_sub = delta_cb.get("sub_periods") or {}
    post_risk_share = post_risk.get("median_residual_variance_share")
    post_risk_corr = post_risk.get("median_risk_correlation")
    active_share_a = diagnostics["mean_active_share_a"]
    active_share_c = diagnostics["mean_active_share_c"]
    active_share_ratio = (
        active_share_c / active_share_a
        if np.isfinite(active_share_a) and active_share_a > _EPS
        else float("nan")
    )

    gates = {
        "net_ir_above_0_36": bool(delta_ca.get("information_ratio", -np.inf) > 0.36),
        "net_subperiods_all_positive": bool(
            all(ca_sub.get(k, -np.inf) > 0.0 for k in ("P1_ir", "P2_ir", "P3_ir"))
        ),
        "net_active_return_positive": bool(delta_ca.get("active_return", -np.inf) > 0.0),
        "pure_sleeve_ir_positive": bool(delta_cb.get("information_ratio", -np.inf) > 0.0),
        "pure_sleeve_active_return_positive": bool(
            delta_cb.get("active_return", -np.inf) > 0.0
        ),
        "pure_sleeve_subperiods_nonnegative": bool(
            all(cb_sub.get(k, -np.inf) >= 0.0 for k in ("P1_ir", "P2_ir", "P3_ir"))
        ),
        "tracking_error_within_4_5pct": bool(
            metrics_c.get("tracking_error", np.inf) <= 0.045
        ),
        "turnover_within_baseline_plus_10pct": bool(
            metrics_c.get("avg_annual_turnover", np.inf)
            <= metrics_a.get("avg_annual_turnover", -np.inf) + 0.10
        ),
        "optimizer_failure_within_1pp": bool(
            diagnostics["optimizer_failure_rate_c"]
            <= diagnostics["optimizer_failure_rate_a"] + 0.01
        ),
        "active_beta_within_0_10": bool(
            abs(
                metrics_c.get("realized_active_beta", np.inf)
                - metrics_a.get("realized_active_beta", -np.inf)
            )
            <= 0.10
        ),
        "book_divergence_within_7_5pct": bool(
            diagnostics["mean_one_way_book_divergence_c_vs_a"] <= 0.075
        ),
        "active_share_at_least_75pct": bool(
            np.isfinite(active_share_ratio) and active_share_ratio >= 0.75
        ),
        "median_residual_risk_share_8_to_12pct": bool(
            post_risk_share is not None and 0.08 <= post_risk_share <= 0.12
        ),
        "median_risk_correlation_within_0_05": bool(
            post_risk_corr is not None and abs(post_risk_corr) <= 0.05
        ),
        "post_warmup_activation_at_least_85pct": bool(
            diagnostics["post_warmup_activation_rate"] >= 0.85
        ),
        "no_unexplained_passthrough": bool(
            diagnostics["unexplained_passthrough_rebalances"] == 0
        ),
        "score_persistence_at_least_0_60": bool(
            orth_diag.get("score_persistence_21d") is not None
            and orth_diag["score_persistence_21d"] >= 0.60
        ),
        "between_ticker_variance_at_most_0_30": bool(
            orth_diag.get("between_ticker_variance_share") is not None
            and orth_diag["between_ticker_variance_share"] <= 0.30
        ),
        "post_warmup_dispersion_at_least_95pct": bool(
            orth_diag.get("post_warmup_nonzero_dispersion_rate", 0.0) >= 0.95
        ),
        "control_corr_within_0_05": bool(
            orth_diag.get("max_abs_control_corr_after") is not None
            and orth_diag["max_abs_control_corr_after"] <= 0.05
        ),
        "preactivation_exact_parity": bool(
            preactivation_parity is not None and preactivation_parity <= 1e-10
        ),
    }
    required_gate_names = [
        "net_ir_above_0_36",
        "net_subperiods_all_positive",
        "net_active_return_positive",
        "pure_sleeve_ir_positive",
        "pure_sleeve_active_return_positive",
        "pure_sleeve_subperiods_nonnegative",
        "tracking_error_within_4_5pct",
        "turnover_within_baseline_plus_10pct",
        "optimizer_failure_within_1pp",
        "active_beta_within_0_10",
        "book_divergence_within_7_5pct",
        "active_share_at_least_75pct",
        "median_residual_risk_share_8_to_12pct",
        "median_risk_correlation_within_0_05",
        "post_warmup_activation_at_least_85pct",
        "no_unexplained_passthrough",
        "score_persistence_at_least_0_60",
        "between_ticker_variance_at_most_0_30",
        "post_warmup_dispersion_at_least_95pct",
        "control_corr_within_0_05",
        "preactivation_exact_parity",
    ]
    gates["all_required_pass"] = bool(all(gates[name] for name in required_gate_names))
    diagnostics["active_share_ratio_c_vs_a"] = (
        float(active_share_ratio) if np.isfinite(active_share_ratio) else None
    )
    diagnostics["gates"] = gates

    attribution = {
        "A_production": metrics_a,
        "B_reserve_only": metrics_b,
        "C_residual_sleeve": metrics_c,
        "B_minus_A_reserve_cost": _metric_delta(metrics_b, metrics_a),
        "C_minus_B_pure_sleeve": delta_cb,
        "C_minus_A_net": delta_ca,
        "diagnostics": diagnostics,
    }
    candidate_result.residual_sleeve_attribution = attribution
    return candidate_result, attribution
