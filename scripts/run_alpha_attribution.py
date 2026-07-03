#!/usr/bin/env python
"""Alpha-source attribution: legs A/B (run_attribution) + leg C (construction).

Leg C is the annualized active-return DELTA between the full production
pipeline and an overlay-OFF re-MVO of the SAME single harvest. Cloned from
run_dr_ablation.py (harvest-once / re-MVO-many).

CONFOUND FIX (vs the original plan snippet): result.raw_predictions is the
PRE-EMA, pre-overlay raw model output (src/model_trainer.py:293 "블렌딩 전
순수 모델 예측값"), NOT an "overlay-free EMA base". Feeding it as
precomputed_predictions would trade the un-smoothed signal (the EMA confound).
So we HARVEST ONCE WITH ALL OVERLAYS OFF -> base_off.predictions IS the
EMA-blended, overlay-free panel (= pre_overlay_ema_predictions, CLAUDE.md
§4.2), and re-apply the production overlays on that same harvest for the full
arm. Round-trip identity: full_active must reproduce S0's active return.
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

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
    from src.harness import (build_override_config, inject_config,
                             compute_alpha_attribution)
    from src.backtest import run_backtest
    from src.data_loader import UniverseData
    from src.utils import annualise_return

    variant = ROOT / "variants" / "iter15_65tkr_reb21_vtg.yaml"
    overrides = (yaml.safe_load(open(variant, encoding="utf-8")) or {}).get("overrides", {})

    # --- harvest ONCE with all overlays OFF -> overlay-free EMA base ---------
    off = dict(overrides)
    off.update(
        value_trap_gate_enabled=False, growth_tilt_enabled=False,
        pead_boost_enabled=False, signal_stability_lambda=0.0,
    )
    off_cfg = build_override_config(off)
    inject_config(off_cfg)
    print("[legC] harvesting overlay-free baseline (single-thread BLAS)…")
    t0 = time.time()
    data = UniverseData(off_cfg.data_path, config=off_cfg)
    base_off = run_backtest(data, config=off_cfg)            # overlays OFF
    overlay_free = base_off.predictions                       # EMA-blended, overlay-free
    print(f"[legC] harvest done in {time.time()-t0:.0f}s")

    # --- full production arm: re-apply overlays on the SAME harvest ----------
    prod_cfg = build_override_config(dict(overrides))
    inject_config(prod_cfg)
    base = run_backtest(
        data, config=prod_cfg,
        precomputed_panel=base_off.panel,
        precomputed_feature_names=base_off.feature_names,
        precomputed_feature_groups=base_off.feature_groups,
        precomputed_targets=base_off.targets,
        precomputed_models=base_off.models,
        precomputed_predictions=overlay_free,                 # overlay-free EMA base
        precomputed_raw_predictions=base_off.raw_predictions,
    )
    inject_config(prod_cfg)  # leave production config active

    def _ann_active(res):
        p = res.portfolio_returns.dropna()
        b = res.benchmark_returns.reindex(p.index).ffill().fillna(0.0)
        return float(annualise_return((p - b), 252))

    full_active = _ann_active(base)
    overlay_off_active = _ann_active(base_off)
    construction_delta = full_active - overlay_off_active

    # Round-trip identity vs S0 (ai_port/outputs, CWD-relative).
    s0_active = None
    s0_path = Path("outputs/iter15_65tkr_reb21_vtg/metrics.json")
    if s0_path.exists():
        s0m = json.load(open(s0_path, encoding="utf-8"))
        s0m = s0m.get("metrics", s0m)
        s0_active = s0m.get("active_return")

    out = {
        "legA_B": compute_alpha_attribution(
            base, n_dates=getattr(prod_cfg, "alpha_attribution_n_dates", 8)),
        "legC_construction_active_delta": construction_delta,
        "full_active": full_active,
        "overlay_off_active": overlay_off_active,
        "s0_active_return": s0_active,
        "roundtrip_full_vs_s0_abs_diff": (
            abs(full_active - s0_active) if s0_active is not None else None),
        "note": ("interaction is an upper bound; legC is an annualized "
                 "active-return delta, not summed with A/B. Overlay-free base "
                 "is base_off.predictions (EMA-blended, overlay-OFF harvest)."),
    }
    out_dir = Path("outputs/alpha_attribution")   # CWD-relative => ai_port
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_dir / "summary.json", "w", encoding="utf-8"),
              indent=2, default=str)
    print(json.dumps({k: out[k] for k in (
        "full_active", "overlay_off_active", "legC_construction_active_delta",
        "s0_active_return", "roundtrip_full_vs_s0_abs_diff")}, indent=2, default=str))
    print("legA_B headline:", json.dumps(
        out["legA_B"].get("headline"), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
