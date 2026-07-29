#!/usr/bin/env python
"""Post-run structural and transmission checks for the S13.15 challenger.

This script deliberately avoids per-feature forward IC so that a single
pre-registered feature block is not turned into an unreported feature sweep.
It measures structure, model usage, prediction stability, and portfolio
divergence against the pinned production baseline.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import NONLINEAR_CONFIRMATION_FEATURES


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_metrics(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("metrics", payload)


def _mean_row_corr(frame_a: pd.DataFrame, frame_b: pd.DataFrame) -> float:
    common_index = frame_a.index.intersection(frame_b.index)
    common_columns = frame_a.columns.intersection(frame_b.columns)
    if len(common_index) == 0 or len(common_columns) == 0:
        return float("nan")
    corr = frame_a.loc[common_index, common_columns].corrwith(
        frame_b.loc[common_index, common_columns], axis=1
    )
    return float(corr.replace([np.inf, -np.inf], np.nan).mean())


def _prediction_persistence(predictions: pd.DataFrame, lag: int = 21) -> float:
    return _mean_row_corr(predictions, predictions.shift(lag))


def _portfolio_divergence(base, arm) -> float:
    base_weights = pd.DataFrame(base.portfolio_weights).T.sort_index()
    arm_weights = pd.DataFrame(arm.portfolio_weights).T.sort_index()
    dates = base_weights.index.intersection(arm_weights.index)
    columns = base_weights.columns.union(arm_weights.columns)
    if len(dates) == 0:
        return float("nan")
    delta = (
        arm_weights.reindex(index=dates, columns=columns, fill_value=0.0)
        - base_weights.reindex(index=dates, columns=columns, fill_value=0.0)
    )
    return float((0.5 * delta.abs().sum(axis=1)).mean())


def _feature_usage(result, candidates: list[str]) -> tuple[pd.DataFrame, dict]:
    all_names = list(result.feature_names)
    gain = pd.Series(0.0, index=all_names)
    split = pd.Series(0.0, index=all_names)
    inclusion = pd.Series(0, index=all_names, dtype=int)
    used_models = 0

    for model in result.models.values():
        names = list(getattr(model, "_active_features", all_names))
        model_gain = model.booster_.feature_importance(importance_type="gain")
        model_split = model.booster_.feature_importance(importance_type="split")
        if len(names) != len(model_gain) or len(names) != len(model_split):
            continue
        gain.loc[names] += model_gain
        split.loc[names] += model_split
        inclusion.loc[names] += 1
        used_models += 1

    gain_total = float(gain.sum())
    split_total = float(split.sum())
    usage = pd.DataFrame(
        {
            "feature": candidates,
            "models_included": [int(inclusion.get(name, 0)) for name in candidates],
            "models_total": used_models,
            "gain_share": [
                float(gain.get(name, 0.0) / gain_total) if gain_total else 0.0
                for name in candidates
            ],
            "split_share": [
                float(split.get(name, 0.0) / split_total) if split_total else 0.0
                for name in candidates
            ],
        }
    )
    summary = {
        "models_total": used_models,
        "block_gain_share": float(
            gain.reindex(candidates).fillna(0.0).sum() / gain_total
        )
        if gain_total
        else 0.0,
        "block_split_share": float(
            split.reindex(candidates).fillna(0.0).sum() / split_total
        )
        if split_total
        else 0.0,
    }
    return usage, summary


def _structural_stats(result, candidates: list[str]) -> pd.DataFrame:
    panel = result.panel
    available = [name for name in candidates if name in panel.columns]
    core = [name for name in result.feature_names if name not in candidates]
    rows = []
    for name in available:
        series = panel[name].replace([np.inf, -np.inf], np.nan)
        wide = series.unstack("ticker")
        persistence = wide.corrwith(wide.shift(21), axis=1).mean()
        total_variance = float(series.var())
        ticker_means = series.groupby(level="ticker").mean()
        between_ticker_share = (
            float(ticker_means.var() / total_variance)
            if np.isfinite(total_variance) and total_variance > 0
            else float("nan")
        )
        correlations = panel[core].corrwith(series).abs()
        max_corr_name = correlations.idxmax() if correlations.notna().any() else None
        rows.append(
            {
                "feature": name,
                "persistence_21d": float(persistence),
                "between_ticker_variance_share": between_ticker_share,
                "max_abs_core_corr": float(correlations.max()),
                "max_corr_core_feature": max_corr_name,
                "nonzero_share": float(series.fillna(0.0).ne(0.0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _metric_delta(base_metrics: dict, arm_metrics: dict, key: str) -> dict:
    base = float(base_metrics.get(key, np.nan))
    arm = float(arm_metrics.get(key, np.nan))
    return {"baseline": base, "challenger": arm, "delta": arm - base}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-dir", default="outputs/codex_causal_rank_65", type=Path
    )
    parser.add_argument(
        "--challenger-dir",
        default="outputs/arm_s13_15_nonlinear_confirmation",
        type=Path,
    )
    args = parser.parse_args()

    base = _load_pickle(args.baseline_dir / "backtest_result.pkl")
    arm = _load_pickle(args.challenger_dir / "backtest_result.pkl")
    base_metrics = _load_metrics(args.baseline_dir / "metrics.json")
    arm_metrics = _load_metrics(args.challenger_dir / "metrics.json")
    candidates = list(NONLINEAR_CONFIRMATION_FEATURES)

    usage, usage_summary = _feature_usage(arm, candidates)
    structure = _structural_stats(arm, candidates)
    feature_stats = structure.merge(usage, on="feature", how="left")
    feature_stats.to_csv(args.challenger_dir / "nonlinear_feature_stats.csv", index=False)

    base_sub = base_metrics.get("sub_periods", {})
    arm_sub = arm_metrics.get("sub_periods", {})
    summary = {
        "as_of": str(arm.panel.index.get_level_values("date").max().date()),
        "comparison": {
            key: _metric_delta(base_metrics, arm_metrics, key)
            for key in (
                "information_ratio",
                "active_return",
                "tracking_error",
                "avg_annual_turnover",
                "avg_ic",
                "annual_return",
                "sharpe_ratio",
            )
        },
        "sub_period_ir": {
            key: {
                "baseline": float(base_sub.get(key, np.nan)),
                "challenger": float(arm_sub.get(key, np.nan)),
                "delta": float(arm_sub.get(key, np.nan) - base_sub.get(key, np.nan)),
            }
            for key in ("P1_ir", "P2_ir", "P3_ir")
        },
        "feature_count": {
            "baseline": len(base.feature_names),
            "challenger": len(arm.feature_names),
            "delta": len(arm.feature_names) - len(base.feature_names),
        },
        "model_quality": {
            "baseline_degenerate_rate": float(
                (base.model_quality or {}).get("degenerate_rate", np.nan)
            ),
            "challenger_degenerate_rate": float(
                (arm.model_quality or {}).get("degenerate_rate", np.nan)
            ),
        },
        "prediction_stability": {
            "baseline_persistence_21d": _prediction_persistence(base.predictions),
            "challenger_persistence_21d": _prediction_persistence(arm.predictions),
            "arm_vs_baseline_mean_daily_corr": _mean_row_corr(
                arm.predictions, base.predictions
            ),
        },
        "portfolio_mean_one_way_divergence": _portfolio_divergence(base, arm),
        "feature_usage": usage_summary,
        "structural_thresholds": {
            "persistence_21d_min": 0.60,
            "between_ticker_variance_share_max": 0.30,
        },
        "structural_pass": bool(
            (feature_stats["persistence_21d"] >= 0.60).all()
            and (feature_stats["between_ticker_variance_share"] <= 0.30).all()
        ),
    }
    (args.challenger_dir / "nonlinear_analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nFeature stats:\n", feature_stats.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
