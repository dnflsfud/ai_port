#!/usr/bin/env python
"""Overlay ablation: 2^3 grid (value_trap x growth_tilt x pead_boost), single harvest.

CONFOUND FIXES:
  (1) EMA: harvest ONCE with ALL overlays OFF. base.predictions is then the
      EMA-blended, overlay-FREE base (NOT base.raw_predictions, which is
      PRE-EMA — see model_trainer.py:293). Feed base.predictions as
      precomputed_predictions so each arm re-applies only its OWN overlays
      once (no double-apply, no un-smoothed signal).
  (2) double-overlay: never feed post-overlay predictions back in.

JUDGMENT WINDOW (deviation from the plan's OOS-holdout note, documented in the
decision log §S3): the harvest-once design re-MVOs each arm on precomputed
predictions and never calls walk_forward_train, so an arm-level
enforce_oos_holdout flag is inert. We therefore judge FULL-PERIOD marginal dIR
+ sub-period (P1/P2/P3) sign consistency — the same bar this repo used for the
CS-DR-Alpha decision. Overlays are EXISTING production components; the question
is do-no-harm-to-keep, which sub-period sign consistency tests directly.

on-baseline arm = vtg1_grw1_pead1 (production: all three ON) — must reproduce
S0 IR (~1.485). all-off = vtg0_grw0_pead0 (pure LGBM EMA base).
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import itertools
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    import yaml
    from src.harness import build_override_config, inject_config, sub_period_irs
    from src.backtest import run_backtest
    from src.data_loader import UniverseData

    variant = ROOT / "variants" / "iter15_65tkr_reb21_vtg.yaml"
    overrides = (yaml.safe_load(open(variant, encoding="utf-8")) or {}).get("overrides", {})

    # Harvest ONCE with all overlays OFF -> overlay-free EMA base predictions.
    off = dict(overrides)
    off.update(
        value_trap_gate_enabled=False, growth_tilt_enabled=False,
        pead_boost_enabled=False, signal_stability_lambda=0.0,
    )
    base_cfg = build_override_config(off)
    inject_config(base_cfg)
    print("[overlay] harvesting overlay-free baseline (single-thread BLAS)…")
    t0 = time.time()
    data = UniverseData(base_cfg.data_path, config=base_cfg)
    base = run_backtest(data, config=base_cfg)
    overlay_free = base.predictions  # overlays were OFF in this harvest
    print(f"[overlay] harvest done in {time.time()-t0:.0f}s")

    def _arm(vtg, growth, pead):
        arm_over = dict(overrides)
        arm_over.update(
            value_trap_gate_enabled=vtg, growth_tilt_enabled=growth,
            pead_boost_enabled=pead,
        )
        cfg = build_override_config(arm_over)
        inject_config(cfg)
        res = run_backtest(
            data, config=cfg,
            precomputed_panel=base.panel, precomputed_feature_names=base.feature_names,
            precomputed_feature_groups=base.feature_groups, precomputed_targets=base.targets,
            precomputed_models=base.models,
            precomputed_predictions=overlay_free,             # overlay-free EMA base
            precomputed_raw_predictions=base.raw_predictions,
        )
        m = res.compute_metrics()
        p = res.portfolio_returns.dropna()
        b = res.benchmark_returns.reindex(p.index).ffill().fillna(0.0)
        sp = sub_period_irs(p, b)
        return {
            "information_ratio": m.get("information_ratio"),
            "active_return": m.get("active_return"),
            "tracking_error": m.get("tracking_error"),
            "avg_annual_turnover": m.get("avg_annual_turnover"),
            "realized_beta": m.get("realized_beta"),
            "sub_periods": sp,
        }

    rows = {}
    for vtg, growth, pead in itertools.product((False, True), repeat=3):
        key = f"vtg{int(vtg)}_grw{int(growth)}_pead{int(pead)}"
        print(f"[overlay] arm {key} …")
        rows[key] = _arm(vtg, growth, pead)
    inject_config(base_cfg)

    out_dir = Path("outputs/overlay_ablation")   # CWD-relative => ai_port
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out_dir / "summary.json", "w", encoding="utf-8"),
              indent=2, default=str)

    # Headline table + leave-one-out marginals vs the full (1,1,1) arm.
    full = rows["vtg1_grw1_pead1"]
    alloff = rows["vtg0_grw0_pead0"]

    # GAP2: the on-baseline (1,1,1) arm = production overlays -> must reproduce
    # S0. Enforce as a gate when S0 metrics exist, so a drifted baseline can't
    # silently poison every leave-one-out marginal dIR.
    s0_path = Path("outputs/iter15_65tkr_reb21_vtg/metrics.json")
    if s0_path.exists():
        s0m = json.load(open(s0_path, encoding="utf-8"))
        s0m = s0m.get("metrics", s0m)
        s0_ir = s0m.get("information_ratio")
        if s0_ir is not None and full["information_ratio"] is not None:
            rt = abs(full["information_ratio"] - s0_ir)
            if rt > 1e-3:
                print(f"[overlay] ROUND-TRIP FAIL: on-baseline (1,1,1) IR "
                      f"{full['information_ratio']:.4f} != S0 {s0_ir:.4f} (|d|={rt:.4f} "
                      f"> 1e-3) — harvest confound; marginals not trustworthy.")
                return 1
    else:
        print("[overlay] WARN: S0 metrics.json absent — round-trip identity unverified.")

    print("\n=== overlay ablation (full-period) ===")
    for k, v in rows.items():
        sp = v["sub_periods"]
        print(f"  {k}: IR {v['information_ratio']:+.3f}  act {v['active_return']*100:+.2f}%  "
              f"TE {v['tracking_error']*100:.2f}%  turn {v['avg_annual_turnover']*100:.0f}%  "
              f"P1 {sp['P1_ir']:+.2f} P2 {sp['P2_ir']:+.2f} P3 {sp['P3_ir']:+.2f}")
    print(f"\n  on-baseline (vtg1_grw1_pead1) IR={full['information_ratio']:.3f}  "
          f"all-off IR={alloff['information_ratio']:.3f}")
    # leave-one-out: turn each overlay OFF from the full arm.
    loo = {
        "drop_vtg": rows["vtg0_grw1_pead1"],
        "drop_growth": rows["vtg1_grw0_pead1"],
        "drop_pead": rows["vtg1_grw1_pead0"],
    }
    print("\n  leave-one-out marginal dIR (full - dropped) [+ => overlay helps]:")
    for name, r in loo.items():
        dIR = full["information_ratio"] - r["information_ratio"]
        fsp, rsp = full["sub_periods"], r["sub_periods"]
        dsub = {p: fsp[f"{p}_ir"] - rsp[f"{p}_ir"] for p in ("P1", "P2", "P3")}
        print(f"    {name:12}: dIR {dIR:+.3f}  "
              f"dP1 {dsub['P1']:+.2f} dP2 {dsub['P2']:+.2f} dP3 {dsub['P3']:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
