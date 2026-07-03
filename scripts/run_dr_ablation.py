#!/usr/bin/env python
"""CS-DR-Alpha 2x2 attribution ablation: {baseline, DR} x {EMA, no-EMA}.

Why: the original grid (docs/RL_DR_ALPHA_RESULTS.md) had two confounds —
(1) the fold pool reached into the prior's NaN burn-in head, silently skipping
    every fold for the first train_window of the prediction period (DR was
    only active in late P3); fixed by pool clipping in dr_walkforward.py.
(2) the DR 2-pass bypassed walk_forward_train's prediction EMA, so passthrough
    regions traded the UN-smoothed raw LGBM signal — the doc's "P2 repaired in
    all 7 configs" was the EMA-removal effect, not RL.

This script decomposes the two effects honestly. One LightGBM harvest, one DR
walk-forward (deterministic; the EMA arm is pure post-processing), four MVO
evaluations:

  A baseline_ema    EMA'd LGBM predictions          (= production baseline)
  B baseline_noema  raw LGBM predictions            (the missing ablation —
                                                     what old DR passthrough
                                                     actually traded)
  C dr_noema        DR scores, no EMA               (old DR treatment, now
                                                     active over the full
                                                     prior-covered period)
  D dr_ema          DR scores + baseline-parity EMA (apples-to-apples vs A)

RL's marginal contribution = C - B (no-EMA pair) and D - A (EMA pair).
Disk-safe: JSON summary only (no pkl).
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import copy
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Production variant equivalent (sans DR) — mirrors scripts/run_dr_alpha.py.
BASE_OVERRIDES = {
    "rebalance_freq": 21,
    "value_trap_gate_enabled": True,
    "vtg_pe_z_threshold": -0.5,
    "vtg_momentum_threshold": -0.5,
    "vtg_accel_threshold": 0.5,
    "vtg_scale": 0.0,
    "enforce_oos_holdout": False,
}

# Production best DR config (variants/iter15_65tkr_reb21_vtg.yaml).
DR_OVERRIDES = {
    "dr_alpha_enabled": True,
    "dr_alpha_residual": False,
    "dr_alpha_use_lgbm_feature": True,
    "dr_alpha_arch": "linear",
    "dr_alpha_turnover_lambda": 0.50,
}


def _ir(active: pd.Series) -> float:
    a = active.dropna()
    if len(a) < 2:
        return float("nan")
    sd = a.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return float("nan")
    return float(a.mean() / sd * np.sqrt(252))


def _headline(result) -> dict:
    from src.harness import sub_period_irs
    m = result.compute_metrics()
    port = result.portfolio_returns.dropna()
    bm = result.benchmark_returns.dropna()
    m["sub_periods"] = sub_period_irs(port, bm)
    return m


def _row(label: str, m: dict) -> str:
    sp = m["sub_periods"]  # keys: P1_ir / P2_ir / P3_ir (src/harness.sub_period_irs)
    return (f"{label:<16} IR {m['information_ratio']:+.3f}  "
            f"P1 {sp['P1_ir']:+.2f} P2 {sp['P2_ir']:+.2f} P3 {sp['P3_ir']:+.2f}  "
            f"turn {m['avg_annual_turnover']:.3f}  "
            f"TE {m['tracking_error']:.4f}  IC {m.get('avg_ic', float('nan')):.4f}")


def _dr_coverage(rl_pred: pd.DataFrame, prior: pd.DataFrame) -> dict:
    """Dates where DR actually modified the score vs the prior."""
    cols = prior.columns.intersection(rl_pred.columns)
    idx = prior.index.intersection(rl_pred.index)
    A, B = prior.loc[idx, cols], rl_pred.loc[idx, cols]
    both = A.notna() & B.notna()
    mod = (A - B).abs().where(both).max(axis=1) > 1e-9
    nonempty = prior.loc[idx].notna().any(axis=1)
    mod_dates = mod[mod].index
    return {
        "n_pred_dates": int(nonempty.sum()),
        "n_dr_modified_dates": int(len(mod_dates)),
        "first_dr_date": str(mod_dates.min().date()) if len(mod_dates) else None,
        "last_dr_date": str(mod_dates.max().date()) if len(mod_dates) else None,
        "coverage_frac": float(len(mod_dates) / max(int(nonempty.sum()), 1)),
    }


def main() -> int:
    from src.harness import build_override_config, inject_config
    from src.backtest import run_backtest
    from src.data_loader import UniverseData
    from src.model_trainer import apply_prediction_ema
    from src.rl.dr_walkforward import run_walkforward

    base_cfg = build_override_config(dict(BASE_OVERRIDES))
    inject_config(base_cfg)

    print("[ablation] harvesting LightGBM baseline (single-thread BLAS)…")
    t0 = time.time()
    data = UniverseData(base_cfg.data_path, config=base_cfg)
    base = run_backtest(data, config=base_cfg)
    raw = base.raw_predictions
    m_A = _headline(base)
    print(f"[ablation] harvest done in {time.time()-t0:.0f}s")
    print(_row("A baseline_ema", m_A))

    def _mvo(pred, raw_pred, cfg):
        return run_backtest(
            data,
            precomputed_panel=base.panel,
            precomputed_feature_names=base.feature_names,
            precomputed_feature_groups=base.feature_groups,
            precomputed_targets=base.targets,
            precomputed_models=base.models,
            precomputed_predictions=pred,
            precomputed_raw_predictions=raw_pred,
            config=cfg,
        )

    # --- B: baseline, no EMA (the missing counterfactual) -------------------
    t1 = time.time()
    m_B = _headline(_mvo(raw, raw, base_cfg))
    print(f"[ablation] B done in {time.time()-t1:.0f}s")
    print(_row("B baseline_noema", m_B))

    # --- DR walk-forward (once; deterministic) ------------------------------
    cfg_dr = copy.deepcopy(base_cfg)
    for k, v in DR_OVERRIDES.items():
        setattr(cfg_dr, k, v)
    t2 = time.time()
    rl_pred = run_walkforward(base.panel, base.targets, raw, base.feature_names, cfg_dr)
    cov = _dr_coverage(rl_pred, raw)
    print(f"[ablation] DR walk-forward done in {time.time()-t2:.0f}s — "
          f"active {cov['n_dr_modified_dates']}/{cov['n_pred_dates']} dates "
          f"({cov['coverage_frac']:.0%}), first {cov['first_dr_date']}")

    # --- C: DR, no EMA -------------------------------------------------------
    t3 = time.time()
    m_C = _headline(_mvo(rl_pred, rl_pred, cfg_dr))
    print(f"[ablation] C done in {time.time()-t3:.0f}s")
    print(_row("C dr_noema", m_C))

    # --- D: DR + baseline-parity EMA -----------------------------------------
    alpha = float(getattr(cfg_dr, "prediction_ema_alpha", 1.0))
    rl_ema = apply_prediction_ema(rl_pred, alpha) if 0.0 < alpha < 1.0 else rl_pred
    t4 = time.time()
    m_D = _headline(_mvo(rl_ema, rl_pred, cfg_dr))
    print(f"[ablation] D done in {time.time()-t4:.0f}s")
    print(_row("D dr_ema", m_D))

    # --- summary --------------------------------------------------------------
    print("\n[ablation] ==== 2x2 summary " + "=" * 50)
    for lab, m in (("A baseline_ema", m_A), ("B baseline_noema", m_B),
                   ("C dr_noema", m_C), ("D dr_ema", m_D)):
        print(_row(lab, m))
    print(f"\nRL marginal (no-EMA pair, C-B): "
          f"dIR {m_C['information_ratio']-m_B['information_ratio']:+.3f}  "
          f"dturn {m_C['avg_annual_turnover']-m_B['avg_annual_turnover']:+.3f}")
    print(f"RL marginal (EMA pair,    D-A): "
          f"dIR {m_D['information_ratio']-m_A['information_ratio']:+.3f}  "
          f"dturn {m_D['avg_annual_turnover']-m_A['avg_annual_turnover']:+.3f}")
    print(f"EMA effect on baseline (B-A): "
          f"dIR {m_B['information_ratio']-m_A['information_ratio']:+.3f}")

    out_dir = ROOT / "outputs" / "dr_ablation_2x2"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump({
            "base_overrides": BASE_OVERRIDES,
            "dr_overrides": DR_OVERRIDES,
            "prediction_ema_alpha": alpha,
            "dr_coverage": cov,
            "arms": {"A_baseline_ema": m_A, "B_baseline_noema": m_B,
                     "C_dr_noema": m_C, "D_dr_ema": m_D},
        }, fh, indent=2, default=str)
    print(f"\n[ablation] done in {time.time()-t0:.0f}s — {out_dir/'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
