#!/usr/bin/env python
"""CS-DR-Alpha driver.

Runs the LightGBM baseline ONCE (in memory), harvests panel/targets/raw
predictions, trains the cross-sectional Direct-Reinforcement alpha walk-forward,
feeds its scores through the UNCHANGED production MVO via
run_backtest(precomputed_predictions=...), and reports an honest baseline-vs-RL
comparison (full-sample = all-OOS, plus a 2024-12-31 sub-split for continuity
with the prior failed-RL analysis).

Disk-safe: everything stays in one process; only small JSON/CSV is written
(the C: drive is near-full). Use --save-pkl to persist ONE result for DSR.

Usage
-----
  # single config (Task 5)
  python scripts/run_dr_alpha.py --label rl_dr_alpha_v1 --save-pkl
  # ablation grid in one process (Task 6)
  python scripts/run_dr_alpha.py --grid
"""
from __future__ import annotations

import os
# Single-thread BLAS BEFORE numpy import (memory + determinism).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
import json
import sys
import time
from pathlib import Path

# UTF-8, line-buffered stdout so unicode (em-dash, Delta) survives a cp949
# console / file redirect and progress flushes promptly.
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

# (heavy imports are done lazily inside main() so --help stays fast)

OOS_CUTOFF = pd.Timestamp("2024-12-31")
BASE_OVERRIDES = {
    "rebalance_freq": 21,
    "value_trap_gate_enabled": True,
    "vtg_pe_z_threshold": -0.5,
    "vtg_momentum_threshold": -0.5,
    "vtg_accel_threshold": 0.5,
    "vtg_scale": 0.0,
    "enforce_oos_holdout": False,
}

GRID = [
    # gentle residual corrections (small gamma => do-little-harm)
    {"dr_alpha_arch": "linear", "dr_alpha_residual": True, "dr_alpha_gamma": 0.15, "dr_alpha_turnover_lambda": 0.10},
    {"dr_alpha_arch": "linear", "dr_alpha_residual": True, "dr_alpha_gamma": 0.30, "dr_alpha_turnover_lambda": 0.10},
    # LGBM-anchored standalone (policy sees LGBM as a feature) + turnover smoothing
    {"dr_alpha_arch": "linear", "dr_alpha_residual": False, "dr_alpha_use_lgbm_feature": True, "dr_alpha_turnover_lambda": 0.10},
    {"dr_alpha_arch": "linear", "dr_alpha_residual": False, "dr_alpha_use_lgbm_feature": True, "dr_alpha_turnover_lambda": 0.50},
    {"dr_alpha_arch": "linear", "dr_alpha_residual": False, "dr_alpha_use_lgbm_feature": True, "dr_alpha_turnover_lambda": 1.00},
    {"dr_alpha_arch": "tiny",   "dr_alpha_residual": False, "dr_alpha_use_lgbm_feature": True, "dr_alpha_turnover_lambda": 0.50},
]


def _ir(active: pd.Series) -> float:
    a = active.dropna()
    if len(a) < 2:
        return float("nan")
    sd = a.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return float("nan")
    return float(a.mean() / sd * np.sqrt(252))


def _oos_split(result) -> dict:
    port = result.portfolio_returns.dropna()
    bm = result.benchmark_returns.reindex(port.index)
    active = (port - bm).dropna()
    ins = active[active.index <= OOS_CUTOFF]
    oos = active[active.index > OOS_CUTOFF]
    return {
        "full_ir": _ir(active), "full_n": int(len(active)),
        "insample_ir": _ir(ins), "insample_n": int(len(ins)),
        "oos_ir": _ir(oos), "oos_n": int(len(oos)),
    }


def _headline(result) -> dict:
    from src.harness import sub_period_irs
    m = result.compute_metrics()
    port = result.portfolio_returns.dropna()
    bm = result.benchmark_returns.dropna()
    m["sub_periods"] = sub_period_irs(port, bm)
    m["oos_split"] = _oos_split(result)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="rl_dr_alpha_v1")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--standalone", action="store_true")
    ap.add_argument("--use-lgbm", action="store_true",
                    help="append LGBM z as a policy input feature")
    ap.add_argument("--no-ema", action="store_true",
                    help="skip the baseline-parity prediction EMA on DR scores")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--save-pkl", action="store_true")
    args = ap.parse_args()

    from src.harness import build_override_config, inject_config
    from src.backtest import run_backtest
    from src.data_loader import UniverseData
    from src.model_trainer import apply_prediction_ema
    from src.rl.dr_walkforward import run_walkforward

    base_cfg = build_override_config(dict(BASE_OVERRIDES))
    inject_config(base_cfg)

    print("[dr_alpha] loading data + harvesting LightGBM baseline (single-thread BLAS)…")
    t0 = time.time()
    data = UniverseData(base_cfg.data_path, config=base_cfg)
    base = run_backtest(data, config=base_cfg)
    panel = base.panel
    targets = base.targets
    feature_names = base.feature_names
    feature_groups = base.feature_groups
    models = base.models
    raw_predictions = base.raw_predictions
    base_metrics = _headline(base)
    print(f"[dr_alpha] baseline harvested in {time.time()-t0:.0f}s — "
          f"IR={base_metrics['information_ratio']:.3f} "
          f"OOS_IR={base_metrics['oos_split']['oos_ir']:.3f} "
          f"turn={base_metrics['avg_annual_turnover']:.3f}")

    configs = []
    if args.grid:
        configs = [(f"{args.label}_g{i}", g) for i, g in enumerate(GRID)]
    else:
        ov = {}
        if args.arch: ov["dr_alpha_arch"] = args.arch
        if args.gamma is not None: ov["dr_alpha_gamma"] = args.gamma
        if args.lam is not None: ov["dr_alpha_turnover_lambda"] = args.lam
        if args.standalone: ov["dr_alpha_residual"] = False
        if args.use_lgbm: ov["dr_alpha_use_lgbm_feature"] = True
        if args.no_ema: ov["dr_alpha_apply_ema"] = False
        configs = [(args.label, ov)]

    results_summary = {"baseline": base_metrics, "n_trials": len(configs), "runs": {}}

    for label, ov in configs:
        cfg = copy.deepcopy(base_cfg)
        cfg.dr_alpha_enabled = True
        for k, v in ov.items():
            setattr(cfg, k, v)
        if args.epochs is not None:
            cfg.dr_alpha_epochs = args.epochs
        print(f"\n[dr_alpha] === {label} === arch={cfg.dr_alpha_arch} "
              f"gamma={cfg.dr_alpha_gamma} lam={cfg.dr_alpha_turnover_lambda} "
              f"residual={cfg.dr_alpha_residual} epochs={cfg.dr_alpha_epochs}")
        tw = time.time()
        rl_pred = run_walkforward(panel, targets, raw_predictions, feature_names, cfg)
        print(f"[dr_alpha]   walk-forward done in {time.time()-tw:.0f}s")
        # Baseline parity: mirror run_variant — EMA-blend the DR scores unless
        # dr_alpha_apply_ema is off (the baseline's predictions are EMA'd
        # inside walk_forward_train, which the precomputed path bypasses).
        rl_for_mvo = rl_pred
        ema_alpha = float(getattr(cfg, "prediction_ema_alpha", 1.0))
        if getattr(cfg, "dr_alpha_apply_ema", True) and 0.0 < ema_alpha < 1.0:
            rl_for_mvo = apply_prediction_ema(rl_pred, ema_alpha)
            print(f"[dr_alpha]   prediction EMA (alpha={ema_alpha}) applied to DR scores")
        rl = run_backtest(
            data,
            precomputed_panel=panel,
            precomputed_feature_names=feature_names,
            precomputed_feature_groups=feature_groups,
            precomputed_targets=targets,
            precomputed_models=models,
            precomputed_predictions=rl_for_mvo,
            precomputed_raw_predictions=rl_pred,
            config=cfg,
        )
        m = _headline(rl)
        results_summary["runs"][label] = m
        b, r = base_metrics, m
        print(f"[dr_alpha]   IR {b['information_ratio']:.3f} -> {r['information_ratio']:.3f} "
              f"(Δ{r['information_ratio']-b['information_ratio']:+.3f}) | "
              f"OOS_IR {b['oos_split']['oos_ir']:.3f} -> {r['oos_split']['oos_ir']:.3f} "
              f"(Δ{r['oos_split']['oos_ir']-b['oos_split']['oos_ir']:+.3f}) | "
              f"turn {b['avg_annual_turnover']:.3f} -> {r['avg_annual_turnover']:.3f}")

        out_dir = ROOT / "outputs" / label
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
            json.dump({"label": label, "overrides": {**BASE_OVERRIDES, **ov,
                       "dr_alpha_enabled": True, "dr_alpha_epochs": cfg.dr_alpha_epochs},
                       "metrics": m, "baseline_metrics": base_metrics},
                      fh, indent=2, default=str)
        try:
            rl_pred.to_csv(out_dir / "rl_predictions_head.csv")  # small: full grid CSV
        except Exception:
            pass
        if args.save_pkl and not args.grid:
            import pickle
            with (out_dir / "backtest_result.pkl").open("wb") as fh:
                pickle.dump(rl, fh)
            print(f"[dr_alpha]   saved backtest_result.pkl for DSR")

    # Grid/summary
    summary_dir = ROOT / "outputs" / (args.label if not args.grid else f"{args.label}_grid")
    summary_dir.mkdir(parents=True, exist_ok=True)
    with (summary_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(results_summary, fh, indent=2, default=str)
    print(f"\n[dr_alpha] done in {time.time()-t0:.0f}s — summary: {summary_dir/'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
