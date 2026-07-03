#!/usr/bin/env python
"""Unified variant runner — the single CLI entry point for experiments.

Replaces the `run_iter{N}.py` pattern (see archive/). All experiments now
go through YAML manifests under `variants/`.

Usage
-----
    python run_variant.py --variant variants/iter15_FINAL.yaml
    python run_variant.py --variant variants/exp_p2_mh.yaml --no-cache

Behaviour
---------
1. Load the manifest. Unknown fields under `overrides:` raise at load time.
2. Apply overrides to DEFAULT_CONFIG via harness.build_override_config.
3. Honour `tuning_mode`:
     - "production": enforce_oos_holdout forced OFF (full-period run).
     - "tuning"    : enforce_oos_holdout forced ON. train_cutoff_date
                     must be set in overrides or the run aborts.
     - "oos_verify": enforce_oos_holdout forced OFF; this is the single
                     "peek" allowed per candidate and is logged in the
                     manifest so selection-bias accounting sees it.
4. Reuse Phase 1/2/4 checkpoints if available and --no-cache is not set.
5. Run backtest via src.backtest.run_backtest under the composed config.
6. Dump artifacts to <out_dir>/ (default outputs/<label>/):
     - metrics.json, backtest_result.pkl, experiment_manifest.json
7. Print a concise summary with baseline-comparison deltas.

Compatibility
-------------
This script preserves the iter15 reproduction path used by
`run_finalize_iter15.py`. If you hit a regression, re-run the
`iter15_FINAL.yaml` variant and compare metrics.json to the stored baseline.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

# UTF-8 stdout for Windows consoles.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

try:
    import yaml
except ImportError:
    sys.stderr.write(
        "ERROR: PyYAML is required. Install with `pip install pyyaml`.\n"
    )
    sys.exit(1)

from src.config import DEFAULT_CONFIG, PipelineConfig, dump_experiment_manifest
from src.harness import build_override_config, inject_config
from src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger("run_variant")


VALID_TUNING_MODES = {"production", "tuning", "oos_verify"}


def _valid_config_fields() -> set:
    return set(DEFAULT_CONFIG.__dataclass_fields__.keys())


def load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh) or {}

    required = ("label", "overrides")
    for key in required:
        if key not in manifest:
            raise ValueError(f"{path}: missing required key '{key}'")

    tm = manifest.get("tuning_mode", "production")
    if tm not in VALID_TUNING_MODES:
        raise ValueError(
            f"{path}: tuning_mode='{tm}' must be one of {sorted(VALID_TUNING_MODES)}"
        )

    overrides = manifest.get("overrides") or {}
    if not isinstance(overrides, dict):
        raise ValueError(f"{path}: overrides: must be a mapping")

    unknown = set(overrides.keys()) - _valid_config_fields()
    if unknown:
        raise ValueError(
            f"{path}: overrides contain unknown PipelineConfig fields: "
            f"{sorted(unknown)}. If a field was removed as destructive, "
            f"see docs/rollback_log.md."
        )

    return manifest


def compose_config(manifest: Dict[str, Any]) -> PipelineConfig:
    overrides = dict(manifest.get("overrides") or {})
    tuning_mode = manifest.get("tuning_mode", "production")

    if tuning_mode == "tuning":
        overrides["enforce_oos_holdout"] = True
        if not overrides.get("train_cutoff_date"):
            raise ValueError(
                "tuning_mode='tuning' requires overrides.train_cutoff_date "
                "(e.g. '2024-12-31'). Otherwise the OOS hold-out has no "
                "effect and the run is indistinguishable from 'production'."
            )
    elif tuning_mode in ("production", "oos_verify"):
        overrides["enforce_oos_holdout"] = False

    cfg = build_override_config(overrides)
    return cfg


def resolve_out_dir(manifest: Dict[str, Any]) -> Path:
    out_dir = manifest.get("out_dir")
    if not out_dir:
        out_dir = f"outputs/{manifest['label']}"
    return Path(out_dir)


def _summarize(metrics: Dict[str, Any], baseline_path: Path) -> None:
    """Pretty-print the headline metrics and delta vs baseline."""
    ir = metrics.get("information_ratio")
    sp = metrics.get("sub_periods") or {}
    print("\n" + "=" * 60)
    print("  Variant summary")
    print("=" * 60)
    if ir is not None:
        print(f"  IR          : {ir:.3f}")
    print(f"  Active ret  : {metrics.get('active_return', 0.0) * 100:.2f}%")
    print(f"  TE          : {metrics.get('tracking_error', 0.0) * 100:.2f}%")
    print(f"  Turnover    : {metrics.get('avg_annual_turnover', 0.0) * 100:.1f}%")
    if sp:
        print(f"  P1 IR       : {sp.get('P1_ir', float('nan')):.3f}")
        print(f"  P2 IR       : {sp.get('P2_ir', float('nan')):.3f}")
        print(f"  P3 IR       : {sp.get('P3_ir', float('nan')):.3f}")

    # Baseline delta
    if baseline_path.exists() and ir is not None:
        try:
            with baseline_path.open("r", encoding="utf-8") as fh:
                base = json.load(fh).get("metrics", {})
            base_ir = base.get("information_ratio")
            base_sp = base.get("sub_periods") or {}
            if base_ir is not None:
                print()
                print(f"  vs iter15_FINAL baseline (IR={base_ir:.3f}):")
                print(f"    ΔIR    : {ir - base_ir:+.3f}")
                for k in ("P1_ir", "P2_ir", "P3_ir"):
                    if k in sp and k in base_sp:
                        print(f"    Δ{k}: {sp[k] - base_sp[k]:+.3f}")
        except Exception:
            pass
    print("=" * 60 + "\n")


def run(manifest_path: Path, no_cache: bool = False) -> int:
    manifest = load_manifest(manifest_path)
    label = manifest["label"]
    tuning_mode = manifest.get("tuning_mode", "production")
    cfg = compose_config(manifest)
    inject_config(cfg)

    out_dir = resolve_out_dir(manifest)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run_variant] label={label} tuning_mode={tuning_mode} out_dir={out_dir}")
    print(f"[run_variant] overrides: {manifest.get('overrides') or {}}")

    # Lazy imports so --help doesn't pull pandas/lightgbm.
    from src.backtest import run_backtest, load_checkpoint
    from src.data_loader import UniverseData
    from src.target_engine import build_targets
    from src.harness import sub_period_irs

    t0 = time.time()

    # ------------------------------------------------------------------
    # Checkpoint reuse safety check
    # ------------------------------------------------------------------
    # Phase 1/2/4 checkpoints are only safe to reuse when the variant's
    # overrides do NOT change feature engineering, target generation, or
    # model training. Otherwise the cached panel/predictions/models reflect
    # the WRONG config and the variant result gets mislabeled.
    #
    # Allowlist below covers Phase 5 (backtest loop), Phase 6 (optimizer),
    # and post-Phase-4 score adjustments (VTG / growth_tilt / PEAD /
    # signal_stability) which run_backtest applies AFTER loading the cache.
    # Any override key NOT in this set forces a full pipeline re-run.
    SAFE_FOR_CACHE_REUSE = frozenset({
        # Phase 5: walk-forward loop
        "rebalance_freq", "one_way_tc", "trailing_ic_window",
        # Phase 6: MVO optimizer constraints
        "risk_aversion", "turnover_penalty", "max_te_annual",
        "max_single_turnover", "max_weight", "max_active_per_stock",
        "max_active_share", "max_active_share_ceiling", "sector_deviation",
        "bm_weight_floor", "cov_lookback", "benchmark_type",
        "use_score_based", "allow_scs_on_ecos_exception",
        # Portfolio style (optimizer-level)
        "portfolio_style", "satellite_budget", "satellite_max_per_stock",
        "enforce_score_gated_ow", "score_threshold_for_ow",
        # Trade throttling (optimizer-level)
        "no_trade_band", "partial_rebalance_eta",
        # Position overlays (post-optimizer, applied per rebal in run_backtest)
        "mega_cap_protection_enabled", "mega_cap_bm_threshold",
        "mega_cap_wide_uw_cap", "mega_cap_funding_mode",
        "mega_cap_funding_k", "mega_cap_funding_score_max",
        "bm_proportional_cap_enabled",
        "bm_proportional_cap_bm_scale_at_top",
        "bm_proportional_cap_vol_scale_floor",
        "bm_proportional_cap_vol_lookback",
        # Post-Phase-4 score adjustments (applied AFTER cache load — see
        # src/backtest.py:1325-1339)
        "value_trap_gate_enabled", "vtg_pe_z_threshold",
        "vtg_momentum_threshold", "vtg_accel_threshold", "vtg_scale",
        "growth_tilt_enabled", "growth_tilt_weight",
        "growth_tilt_rev_weight", "growth_tilt_fundamental_weight",
        "growth_tilt_eps_skew", "growth_tilt_rev_eps_share",
        "growth_tilt_rev_sales_share", "growth_tilt_rev_tg_share",
        "pead_boost_enabled", "pead_boost_weight",
        "pead_decay_days", "pead_max_days",
        "signal_stability_lambda",
        # CS-DR-Alpha production overlay — applied AFTER the LightGBM baseline
        # harvest (see DR block in run()), so it does not invalidate the
        # Phase 1/2/4 cache.
        "dr_alpha_enabled", "dr_alpha_arch", "dr_alpha_hidden", "dr_alpha_lr",
        "dr_alpha_epochs", "dr_alpha_l2", "dr_alpha_turnover_lambda",
        "dr_alpha_residual", "dr_alpha_gamma", "dr_alpha_use_lgbm_feature",
        "dr_alpha_embargo", "dr_alpha_warm_start", "dr_alpha_seed",
        "dr_alpha_val_months", "dr_alpha_min_train_rebal", "dr_alpha_apply_ema",
        # Alpha-source attribution — read-only post-hoc SHAP/decomp over the
        # harvested models; changes no features/targets/weights => cache-safe.
        "alpha_attribution_enabled", "alpha_attribution_n_dates",
        # Factor-neutral (P3) — adds an MVO objective term + per-date loadings
        # read from the already-cached panel; no features/targets/models change
        # (same class as risk_aversion/max_te_annual above) => cache-safe.
        "factor_neutral_enabled", "factor_neutral_penalty",
        "factor_neutral_axes", "factor_neutral_loadings",
        "max_name_active_risk_share", "max_sector_active_risk_share",
    })
    overrides = manifest.get("overrides") or {}
    unsafe_keys = []
    for key in sorted(set(overrides.keys()) - SAFE_FOR_CACHE_REUSE):
        if overrides.get(key) != getattr(DEFAULT_CONFIG, key):
            unsafe_keys.append(key)

    cache_disabled_reason = None
    if no_cache:
        cache_disabled_reason = "user passed --no-cache"
    elif unsafe_keys:
        cache_disabled_reason = (
            f"variant overrides Phase 1/2/4 keys: {unsafe_keys}"
        )

    if cache_disabled_reason:
        cp1 = cp2 = cp4 = None
        print(f"[run_variant] cache DISABLED — {cache_disabled_reason}")
    else:
        cp1 = load_checkpoint("phase1")
        cp2 = load_checkpoint("phase2")
        cp4 = load_checkpoint("phase4")

    if cp1 and cp2 and cp4:
        print("[run_variant] reusing Phase 1/2/4 checkpoints "
              "(overrides are Phase 5/6/post-prediction-only)")
        data = cp1["data"]
        panel = cp2["panel"]
        feature_names = cp2["feature_names"]
        feature_groups = cp2["feature_groups"]
        models = cp4["models"]
        predictions = cp4["predictions"]
        raw_predictions = cp4.get("raw_predictions")
        result_targets = build_targets(data)
        targets = result_targets[0] if isinstance(result_targets, tuple) else result_targets

        result = run_backtest(
            data,
            precomputed_panel=panel,
            precomputed_feature_names=feature_names,
            precomputed_feature_groups=feature_groups,
            precomputed_targets=targets,
            precomputed_models=models,
            precomputed_predictions=predictions,
            precomputed_raw_predictions=raw_predictions,
            config=cfg,
        )
    else:
        print("[run_variant] no checkpoints found — running full pipeline")
        data_path = cfg.data_path
        data = UniverseData(data_path, config=cfg)
        result = run_backtest(data, config=cfg)

    # ------------------------------------------------------------------
    # CS-DR-Alpha production overlay (when dr_alpha_enabled)
    # ------------------------------------------------------------------
    # The run_backtest above is the LightGBM baseline harvest. When DR-Alpha is
    # on (production variant), train the cross-sectional Direct-Reinforcement
    # policy true walk-forward on the harvested panel/targets/raw-LGBM-z, then
    # re-run the UNCHANGED production MVO on the DR scores (mirrors
    # scripts/run_dr_alpha.py).
    if getattr(cfg, "dr_alpha_enabled", False):
        from src.rl.dr_walkforward import run_walkforward
        from src.model_trainer import apply_prediction_ema
        base = result
        print("[run_variant] dr_alpha_enabled — training DR walk-forward on "
              "the harvested LightGBM baseline")
        rl_pred = run_walkforward(
            base.panel, base.targets, base.raw_predictions,
            base.feature_names, cfg,
        )
        # Baseline parity: the LightGBM path EMA-blends predictions inside
        # walk_forward_train, which the precomputed path bypasses. Apply the
        # same smoothing to the DR scores (raw stays pre-EMA, mirroring the
        # baseline's raw_predictions semantics for IC/confidence).
        rl_for_mvo = rl_pred
        ema_alpha = float(getattr(cfg, "prediction_ema_alpha", 1.0))
        if getattr(cfg, "dr_alpha_apply_ema", True) and 0.0 < ema_alpha < 1.0:
            rl_for_mvo = apply_prediction_ema(rl_pred, ema_alpha)
            print(f"[run_variant] prediction EMA (alpha={ema_alpha}) applied "
                  "to DR scores (baseline parity)")
        result = run_backtest(
            data,
            precomputed_panel=base.panel,
            precomputed_feature_names=base.feature_names,
            precomputed_feature_groups=base.feature_groups,
            precomputed_targets=base.targets,
            precomputed_models=base.models,
            precomputed_predictions=rl_for_mvo,
            precomputed_raw_predictions=rl_pred,
            config=cfg,
        )
        print("[run_variant] DR-Alpha scores applied through production MVO")

    # Metrics + sub-period IRs
    metrics = result.compute_metrics()
    port = result.portfolio_returns.dropna()
    bm = result.benchmark_returns.dropna()
    metrics["sub_periods"] = sub_period_irs(port, bm)

    if getattr(cfg, "alpha_attribution_enabled", False):
        from src.harness import compute_alpha_attribution
        metrics["alpha_attribution"] = compute_alpha_attribution(
            result, n_dates=getattr(cfg, "alpha_attribution_n_dates", 8)
        )

    # Persist artifacts
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "label": label,
                "tuning_mode": tuning_mode,
                "manifest_path": str(manifest_path),
                "overrides": manifest.get("overrides") or {},
                "metrics": metrics,
                "model_quality": getattr(result, "model_quality", None),
                "data_quality": getattr(result, "data_quality", None),
                "optimizer_solver_counts": getattr(result, "optimizer_solver_counts", {}),
                "optimizer_solver_fallback_rate": getattr(
                    result, "optimizer_solver_fallback_rate", None
                ),
                "elapsed_sec": round(time.time() - t0, 1),
            },
            fh,
            indent=2,
            default=str,
        )

    with (out_dir / "backtest_result.pkl").open("wb") as fh:
        pickle.dump(result, fh)

    dump_experiment_manifest(
        config=cfg,
        output_dir=str(out_dir),
        extra={
            "variant_label": label,
            "tuning_mode": tuning_mode,
            "manifest_path": str(manifest_path),
            "description": manifest.get("description", ""),
        },
    )

    baseline_path = Path("outputs/iter15_FINAL/metrics.json")
    _summarize(metrics, baseline_path)
    print(f"[run_variant] done in {time.time() - t0:.1f}s — artifacts: {out_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", required=True, help="Path to variants/<name>.yaml")
    parser.add_argument("--no-cache", action="store_true",
                        help="Skip Phase 1/2/4 checkpoint reuse (full rebuild).")
    args = parser.parse_args()
    return run(Path(args.variant), no_cache=args.no_cache)


if __name__ == "__main__":
    sys.exit(main())
