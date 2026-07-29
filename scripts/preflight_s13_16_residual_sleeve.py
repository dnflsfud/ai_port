"""Structural-only preflight for the S13.16 residual learner.

This intentionally stops before portfolio simulation and never displays a
return, IR or arm comparison.  It is safe to use for causal/coverage checks
before the single preregistered A/B/C return-series evaluation.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_variant import compose_config, load_manifest
from src.backtest import get_sector_map
from src.data_loader import UniverseData
from src.residual_sleeve import (
    build_residual_feature_panel,
    build_residual_labels,
    orthogonalize_residual_scores,
    smooth_residual_scores,
    walk_forward_residual_scores,
)


VARIANT = ROOT / "variants" / "arm_s13_16_residual_nonlinear_sleeve.yaml"
BASE_PKL = ROOT / "outputs" / "codex_causal_rank_65" / "backtest_result.pkl"


def main() -> int:
    manifest = load_manifest(VARIANT)
    config = compose_config(manifest)
    with BASE_PKL.open("rb") as handle:
        base = pickle.load(handle)
    data = UniverseData(config.data_path, config=config)

    lag = int(config.execution_signal_lag_days)
    base_signal = base.pre_execution_predictions if lag > 0 else base.predictions
    base_signal = base_signal.reindex(index=base.targets.index, columns=data.tickers)
    sectors = get_sector_map(data)
    labels, label_diag = build_residual_labels(
        base.targets,
        base_signal,
        base.panel,
        data.market_cap,
        sectors,
        config,
    )
    feature_panel, feature_names = build_residual_feature_panel(
        base.panel, base.feature_names, base_signal, data, config
    )
    raw, _models, fold_diag = walk_forward_residual_scores(
        feature_panel, labels, base_signal, feature_names, config
    )
    smoothed = smooth_residual_scores(raw, span=config.rebalance_freq)
    _scores, orth_diag = orthogonalize_residual_scores(
        smoothed,
        base_signal,
        base.panel,
        data.market_cap,
        sectors,
        config,
    )

    audit_ok = all(
        (not fold.get("trained"))
        or fold["latest_label_realisation_position"] <= fold["fold_position"]
        for fold in fold_diag["fold_audit"]
    )
    summary = {
        "return_series_evaluated": False,
        "label": label_diag,
        "walk_forward": {
            key: value
            for key, value in fold_diag.items()
            if key not in {"fold_audit", "top_feature_importance"}
        },
        "orthogonalization": orth_diag,
        "causal_fold_audit_pass": audit_ok,
        "feature_count": len(feature_names),
        "top_feature_importance": fold_diag.get("top_feature_importance", [])[:10],
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0 if audit_ok and fold_diag.get("folds_trained", 0) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
