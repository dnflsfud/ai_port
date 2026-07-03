"""CS-DR-Alpha true walk-forward driver.

Produces an ``rl_predictions`` DataFrame (date x ticker cross-sectional z) on the
same grid as the LightGBM ``prior_predictions``. The policy is retrained every
``config.retrain_freq`` and only ever predicts dates strictly after its
embargoed training window -> every emitted score is OOS-by-construction.

Leak guard: for a fold boundary at date ``t_k``, training dates are restricted
to panel positions <= pos(t_k) - embargo, with embargo >= the forward-return
horizon (20). At the tight bound (embargo == horizon) the latest training
label's realization window ends exactly at ``t_k``; the OOS score at ``t_k`` is
evaluated on returns t_k+1..t_k+H, disjoint from it — so no training label
reaches INTO the OOS forward window.

Ticker alignment: each fold trains on a fixed universe U (tickers valid on every
sampled train/val date) so the turnover term has a well-defined da. Prediction
scores any ticker with valid features at the predict date (the policy is a pure
feature->score map), so prediction coverage >= baseline.

Pool clipping: each fold's lookback pool is clipped to the prior's coverage
(the LightGBM prior has a ~train_window NaN burn-in head; an uncovered sampled
date would empty U and silently skip the fold). Early folds therefore train on
shorter windows; folds with fewer than ``dr_alpha_min_train_rebal`` sampled
rebalance dates keep the prior (graceful passthrough).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from src.rl.dr_alpha import DRAlphaPolicy, xs_zscore, train_fold


def _np_zscore(v: np.ndarray) -> np.ndarray:
    mu = np.nanmean(v)
    sd = np.nanstd(v)
    if not np.isfinite(sd) or sd < 1e-12:
        return v * 0.0
    return (v - mu) / sd


def run_walkforward(
    panel: pd.DataFrame,                 # MultiIndex (date, ticker) -> features
    targets: pd.DataFrame,               # date x ticker, 20d fwd specific return
    prior_predictions: pd.DataFrame,     # date x ticker, LGBM z (also output grid)
    feature_names: List[str],
    config,
    rebal_dates: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    train_window = int(getattr(config, "train_window", 1260))
    retrain_freq = int(getattr(config, "retrain_freq", 63))
    rebalance_freq = max(int(getattr(config, "rebalance_freq", 21)), 1)
    embargo = int(getattr(config, "dr_alpha_embargo", 20))
    # OOS-by-construction depends on embargo covering the full label realization
    # window. Couple it to the SSOT forward horizon so a horizon change without a
    # matching embargo change fails loudly instead of silently leaking labels.
    forward_horizon = int(getattr(config, "forward_horizon", 20))
    if embargo < forward_horizon:
        raise ValueError(
            f"dr_alpha_embargo ({embargo}) must be >= forward_horizon "
            f"({forward_horizon}): a smaller embargo lets a training label's "
            f"{forward_horizon}d realization window reach INTO the OOS forward "
            f"window (which starts the day after each fold boundary)."
        )
    gamma = float(getattr(config, "dr_alpha_gamma", 1.0))
    residual = bool(getattr(config, "dr_alpha_residual", True))
    use_lgbm = bool(getattr(config, "dr_alpha_use_lgbm_feature", False))
    warm = bool(getattr(config, "dr_alpha_warm_start", True))
    val_months = int(getattr(config, "dr_alpha_val_months", 6))
    min_train_rebal = int(getattr(config, "dr_alpha_min_train_rebal", 12))

    feats = [c for c in feature_names if c in panel.columns]
    # Per-date ticker-indexed feature frames (built once).
    feat_by_date: Dict[pd.Timestamp, pd.DataFrame] = {}
    for d, sub in panel[feats].groupby(level="date"):
        feat_by_date[d] = sub.droplevel("date")
    panel_dates = pd.DatetimeIndex(sorted(feat_by_date.keys()))
    pos_of = {d: i for i, d in enumerate(panel_dates)}

    pred_dates = prior_predictions.index
    tickers = list(prior_predictions.columns)
    n_features = len(feats) + (1 if use_lgbm else 0)

    # Clip every fold's training pool to the prior's coverage. The prior
    # (LightGBM walk-forward z) has a ~train_window NaN burn-in head; a sampled
    # date inside that head empties the fold universe U and silently skips the
    # whole fold, which kept DR inactive for the first train_window of the
    # prediction period (folds only trained once the full lookback window held
    # priors). Early folds now train on a shorter, prior-covered window,
    # gated by min_train_rebal below.
    first_prior_pos: Optional[int] = None
    _prior_any = prior_predictions.notna().any(axis=1)
    for d in prior_predictions.index[_prior_any]:
        p = pos_of.get(d)
        if p is not None:
            first_prior_pos = p
            break

    rl_pred = prior_predictions.copy()

    n_val = max(1, round(val_months * 21.0 / rebalance_freq))

    prev_state: Optional[dict] = None
    starts = list(range(0, len(pred_dates), retrain_freq))

    for si in starts:
        t_k = pred_dates[si]
        block = pred_dates[si: si + retrain_freq]
        if t_k not in pos_of:
            continue
        emb_cut_pos = pos_of[t_k] - embargo
        if emb_cut_pos < 1:
            continue  # no embargoed history yet -> leave prior on this block

        lo = max(0, emb_cut_pos - train_window + 1)
        if first_prior_pos is not None:
            lo = max(lo, first_prior_pos)
        pool = panel_dates[lo: emb_cut_pos + 1]
        sampled = list(pool[::rebalance_freq])
        if len(sampled) < min_train_rebal:
            continue

        # Fold universe U = tickers valid (features + label + prior non-NaN) on
        # every sampled date.
        U: Optional[set] = None
        per_date_valid: Dict[pd.Timestamp, List[str]] = {}
        for d in sampled:
            if d not in feat_by_date or d not in targets.index or d not in prior_predictions.index:
                per_date_valid[d] = []
                continue
            fd = feat_by_date[d].reindex(tickers)
            ok_feat = fd.notna().all(axis=1)
            ok_lab = targets.loc[d].reindex(tickers).notna()
            ok_pri = prior_predictions.loc[d].reindex(tickers).notna()
            valid = [t for t in tickers if bool(ok_feat.get(t, False))
                     and bool(ok_lab.get(t, False)) and bool(ok_pri.get(t, False))]
            per_date_valid[d] = valid
            U = set(valid) if U is None else (U & set(valid))
        if not U or len(U) < 2:
            continue
        U_list = [t for t in tickers if t in U]  # stable order

        def _tensors(date_list):
            Xs, Ls, Ps = [], [], []
            for d in date_list:
                fd = feat_by_date[d].reindex(U_list)[feats].values.astype(np.float32)
                lab = targets.loc[d].reindex(U_list).values.astype(np.float32)
                pri = prior_predictions.loc[d].reindex(U_list).values.astype(np.float32)
                if use_lgbm:
                    fd = np.concatenate([fd, pri.reshape(-1, 1)], axis=1)
                Xs.append(torch.from_numpy(fd))
                Ls.append(torch.from_numpy(lab))
                Ps.append(torch.from_numpy(pri))
            return Xs, Ls, Ps

        n_v = min(n_val, max(1, len(sampled) // 3))
        train_dates = sampled[:-n_v] if len(sampled) > n_v else sampled
        val_dates = sampled[-n_v:] if len(sampled) > n_v else []
        Xtr, Ltr, Ptr = _tensors(train_dates)
        Xva, Lva, Pva = _tensors(val_dates)

        policy, _info = train_fold(
            Xtr, Ltr, Ptr, Xva, Lva, Pva, n_features, config,
            warm_start_state=prev_state if warm else None,
        )
        if warm:
            prev_state = {k: v.clone() for k, v in policy.state_dict().items()}

        # ---- Predict the block (OOS) ----
        policy.eval()
        with torch.no_grad():
            for d in block:
                if d not in feat_by_date:
                    continue
                fd = feat_by_date[d].reindex(tickers)
                ok_feat = fd.notna().all(axis=1)
                need_prior = residual or use_lgbm
                if need_prior:
                    ok_pri = prior_predictions.loc[d].reindex(tickers).notna() \
                        if d in prior_predictions.index else pd.Series(False, index=tickers)
                    valid = [t for t in tickers if bool(ok_feat.get(t, False)) and bool(ok_pri.get(t, False))]
                else:
                    valid = [t for t in tickers if bool(ok_feat.get(t, False))]
                if len(valid) < 2:
                    continue
                fv = fd.reindex(valid)[feats].values.astype(np.float32)
                if use_lgbm:
                    pcol = prior_predictions.loc[d].reindex(valid).values.astype(np.float32).reshape(-1, 1)
                    fv = np.concatenate([fv, pcol], axis=1)
                X = torch.from_numpy(fv)
                score = policy.forward(X)
                score_z = xs_zscore(score).numpy()
                if residual:
                    pri = prior_predictions.loc[d].reindex(valid).values.astype(np.float64)
                    combined = pri + gamma * score_z
                else:
                    combined = score_z
                out_z = _np_zscore(np.asarray(combined, dtype=np.float64))
                rl_pred.loc[d, valid] = out_z

    return rl_pred
