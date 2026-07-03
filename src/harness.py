"""Shared helpers for iteration / variant runners.

Extracted from run_iter20.py, run_iter21_full.py, run_finalize_iter15.py
which previously duplicated the same _sub_ir / run_variant / config-injection
boilerplate and drifted (cf. the `D_both` variant-key bug in run_iter20).

Semantics are preserved verbatim — this is a refactor, not a behavior change.
"""

from __future__ import annotations

import dataclasses
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import DEFAULT_CONFIG, PipelineConfig
from src.backtest import run_backtest


# Canonical P1/P2/P3 boundaries used by the iter reports.
SUB_PERIODS = {
    "P1": ("2018-11-23", "2021-05-11"),
    "P2": ("2021-05-12", "2023-10-27"),
    "P3": ("2023-10-30", "2026-04-13"),
}


def sub_ir(port: pd.Series, bm: pd.Series, start: str, end: str) -> float:
    """Information ratio for an arbitrary date window.

    Matches the _sub_ir implementation that run_iter20 / run_iter21_full /
    run_finalize_iter15 all independently re-implemented. Uses ddof=1 and
    sqrt(252) annualization.
    """
    mask = (port.index >= pd.Timestamp(start)) & (port.index <= pd.Timestamp(end))
    pp, bb = port[mask], bm.reindex(port[mask].index)
    if len(pp) < 20:
        return float("nan")
    active = pp.values - bb.values
    if active.std(ddof=1) == 0:
        return 0.0
    return float(active.mean() / active.std(ddof=1) * np.sqrt(252))


def sub_period_irs(port: pd.Series, bm: pd.Series) -> Dict[str, float]:
    """Compute IR for each canonical sub-period (P1/P2/P3)."""
    return {
        f"{label}_ir": sub_ir(port, bm, start, end)
        for label, (start, end) in SUB_PERIODS.items()
    }


def compute_alpha_attribution(result, n_dates: int = 8) -> dict:
    """Wrap src.attribution.run_attribution into a compact, JSON-safe summary.

    Legs A/B only (signal-variance shares). Leg C (construction) is a separate
    re-MVO counterfactual in scripts/run_alpha_attribution.py. Returns {} on
    any failure so attach never breaks a run. interaction_ratio is an UPPER
    BOUND (clamped residual), labeled as such by the caller's reporting.
    """
    try:
        from src.attribution import run_attribution
    except Exception as exc:  # surface a broken shap/lightgbm install (review L3)
        return {"error": f"import failed: {exc}"}
    try:
        detail = run_attribution(
            models=result.models,
            panel=result.panel,
            feature_names=result.feature_names,
            feature_groups=result.feature_groups,
            n_sample_dates=n_dates,
        )
    except Exception as exc:  # never break the run on an attribution failure
        return {"error": str(exc)}
    # Average per-date linear/nonlinear ratios into headline shares, guarding
    # the degenerate (total_var<1e-10) branch that omits 'nonlinear_ratio'.
    lin, nl, n = 0.0, 0.0, 0
    for _date, pair in (detail.get("linear_ratios") or {}).items():
        try:
            l, nlr = float(pair[0]), float(pair[1])
        except (TypeError, ValueError, IndexError):
            continue
        if l == l and nlr == nlr:  # skip nan
            lin += l; nl += nlr; n += 1
    headline = {
        "linear_share": (lin / n) if n else float("nan"),
        "nonlinear_share_upper_bound": (nl / n) if n else float("nan"),
        "n_dates_used": n,
    }
    # group_contributions is keyed by Timestamp ({date: {group: ratio}});
    # stringify the date keys so the summary is JSON-safe (json requires
    # str/int/float/bool/None keys; default=str only coerces values).
    gc = detail.get("group_contributions")
    if isinstance(gc, dict):
        gc = {str(k): v for k, v in gc.items()}
    return {
        "headline": headline,
        "group_contributions": gc,
        "feature_importance": (
            detail["feature_importance"].to_dict()
            if detail.get("feature_importance") is not None else None
        ),
    }


def inject_config(cfg: PipelineConfig) -> None:
    """Inject ``cfg`` into every module that caches DEFAULT_CONFIG at import.

    The original scripts did this inline with three assignments. Centralizing
    it avoids drift if a new module starts caching DEFAULT_CONFIG.
    """
    # Local imports so this module doesn't force circular imports at load time.
    import src.backtest as _bt_mod
    import src.portfolio_optimizer as _po_mod
    import src.config as _config_mod

    _bt_mod.DEFAULT_CONFIG = cfg
    _po_mod.DEFAULT_CONFIG = cfg
    _config_mod.DEFAULT_CONFIG = cfg


def build_override_config(overrides: Dict[str, Any], base: Optional[PipelineConfig] = None) -> PipelineConfig:
    """Build a new PipelineConfig by applying overrides to ``base`` (or DEFAULT_CONFIG).

    Re-runs __post_init__ so portfolio-style derivations reapply. Matches the
    pattern used by run_iter20.run_variant / run_iter21_full.run_variant.
    """
    base = base or DEFAULT_CONFIG
    cfg = dataclasses.replace(base, **overrides)
    if hasattr(cfg, "__post_init__"):
        cfg.__post_init__()
    return cfg


def run_variant(
    label: str,
    overrides: Dict[str, Any],
    *,
    data,
    panel,
    feature_names,
    feature_groups,
    targets,
    models,
    predictions,
    raw_predictions=None,
    out_dir: Optional[Path] = None,
    attach_sub_periods: bool = True,
    sub_period_mode: str = "nested",
    attach_turnover_alias: bool = True,
    verbose: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """Execute a single backtest variant and persist artifacts.

    Preserves the semantics of the original run_iter20 / run_iter21_full:
      1. Build a new config by overriding DEFAULT_CONFIG fields
      2. Inject it into every module global that caches DEFAULT_CONFIG
      3. Call run_backtest with the precomputed phase 1-4 artifacts
      4. Attach P1/P2/P3 sub-period IRs to the metrics dict
      5. Optionally dump metrics.json and result.pkl under out_dir

    Returns (result, metrics).
    """
    cfg = build_override_config(overrides)
    inject_config(cfg)

    if verbose:
        print(f"\n=== [{label}] running ===")

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
    metrics = result.compute_metrics()

    if attach_sub_periods:
        port = result.portfolio_returns.dropna()
        bm = result.benchmark_returns.dropna()
        sub_periods = sub_period_irs(port, bm)
        if sub_period_mode == "nested":
            metrics["sub_periods"] = sub_periods
        elif sub_period_mode == "flat":
            metrics.update(sub_periods)  # top-level P1_ir/P2_ir/P3_ir
        else:
            raise ValueError(f"sub_period_mode must be 'nested' or 'flat', got {sub_period_mode!r}")

    if getattr(cfg, "alpha_attribution_enabled", False):
        metrics["alpha_attribution"] = compute_alpha_attribution(
            result, n_dates=getattr(cfg, "alpha_attribution_n_dates", 8)
        )

    if attach_turnover_alias:
        metrics["annual_turnover_2way"] = metrics.get("avg_annual_turnover", 0.0)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(
            json.dumps(
                {"label": label, "overrides": overrides, "metrics": metrics},
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        with open(out_dir / "result.pkl", "wb") as f:
            pickle.dump(result, f)

    return result, metrics
