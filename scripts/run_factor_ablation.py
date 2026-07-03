#!/usr/bin/env python
"""Factor-neutral ablation: OFF vs single pre-committed penalty (harvest-once).

Both arms re-MVO the SAME production harvest (overlays ON) on identical
predictions; only the optimizer objective differs (factor_neutral penalty),
so there is no EMA/overlay confound. The OFF arm must reproduce S0.

Judgment (CLAUDE.md §4.3 / spec): judge by EXPOSURE BINDING, not IR. For each
penalized axis we measure the mean |active style exposure| = mean_t |L_k(t) .
(w_t - bm_t)| across rebalance dates. If the ON arm's exposure drops measurably
vs OFF, the penalty binds; if it does not move, conclude "TE-var already
neutralizes style" and SHELVE. Also reports the §4.3 pre-checks: applied-date
count (>0 required) and the non-finite loading impute fraction.
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    import yaml
    from src.harness import build_override_config, inject_config, sub_period_irs
    from src.backtest import run_backtest, get_benchmark_fn
    from src.data_loader import UniverseData

    variant = ROOT / "variants" / "iter15_65tkr_reb21_vtg.yaml"
    overrides = (yaml.safe_load(open(variant, encoding="utf-8")) or {}).get("overrides", {})

    prod_cfg = build_override_config(dict(overrides))  # metadata (axes/loadings/bm)

    # Harvest ONCE with all overlays OFF -> overlay-free EMA base. Feeding this
    # (not post-overlay predictions) avoids the §4.2 double-overlay confound:
    # each arm re-applies the production overlays exactly once, so the OFF arm
    # reproduces S0.
    off = dict(overrides)
    off.update(
        value_trap_gate_enabled=False, growth_tilt_enabled=False,
        pead_boost_enabled=False, signal_stability_lambda=0.0,
    )
    base_cfg = build_override_config(off)
    inject_config(base_cfg)
    print("[factor] harvesting overlay-free baseline (single-thread BLAS)…")
    t0 = time.time()
    data = UniverseData(base_cfg.data_path, config=base_cfg)
    base = run_backtest(data, config=base_cfg)
    overlay_free = base.predictions
    print(f"[factor] harvest done in {time.time()-t0:.0f}s")

    def _mvo(cfg):
        return run_backtest(
            data, config=cfg,
            precomputed_panel=base.panel, precomputed_feature_names=base.feature_names,
            precomputed_feature_groups=base.feature_groups, precomputed_targets=base.targets,
            precomputed_models=base.models,
            precomputed_predictions=overlay_free,
            precomputed_raw_predictions=base.raw_predictions,
        )

    off_cfg = build_override_config(dict(overrides)); inject_config(off_cfg)
    res_off = _mvo(off_cfg)
    on_over = dict(overrides); on_over.update(factor_neutral_enabled=True)
    on_cfg = build_override_config(on_over); inject_config(on_cfg)
    res_on = _mvo(on_cfg)
    inject_config(prod_cfg)

    # Match production's guarded lookup (src/backtest.py _optimizer_fn): drop any
    # axis lacking a loading column so axes/cols stay aligned and the measurement
    # can't KeyError-crash where the optimizer would silently skip it (review L7).
    axes = [a for a in prod_cfg.factor_neutral_axes
            if a in prod_cfg.factor_neutral_loadings]
    cols = [prod_cfg.factor_neutral_loadings[a] for a in axes]
    tickers = list(data.tickers)
    bm_fn = get_benchmark_fn(data, tickers, config=prod_cfg)
    panel = base.panel

    def _exposures(res):
        per_axis = {a: [] for a in axes}
        applied, cells_total, cells_imputed = 0, 0, 0
        active_shares = []
        for date, wser in res.portfolio_weights.items():
            try:
                sub = panel.xs(date, level="date").reindex(tickers)[cols]
            except (KeyError, ValueError):
                continue
            Lraw = sub.values.astype(float)
            finite = np.isfinite(Lraw)
            cells_total += Lraw.size
            cells_imputed += int((~finite).sum())
            L = np.where(finite, Lraw, 0.0)
            w = wser.reindex(tickers).values.astype(float)
            bm = np.asarray(bm_fn(date, tickers, len(tickers)), dtype=float)
            act = w - bm
            active_shares.append(0.5 * float(np.abs(act).sum()))  # collapse signal (M3)
            for k, a in enumerate(axes):
                per_axis[a].append(abs(float(L[:, k] @ act)))
            applied += 1
        mean_abs = {a: (float(np.mean(v)) if v else None) for a, v in per_axis.items()}
        impute = (cells_imputed / cells_total) if cells_total else None
        ashare = float(np.mean(active_shares)) if active_shares else None
        return mean_abs, applied, impute, ashare

    exp_off, applied_off, impute_off, ashare_off = _exposures(res_off)
    exp_on, applied_on, impute_on, ashare_on = _exposures(res_on)

    def _m(res):
        m = res.compute_metrics()
        p = res.portfolio_returns.dropna()
        b = res.benchmark_returns.reindex(p.index).ffill().fillna(0.0)
        m["sub_periods"] = sub_period_irs(p, b)
        out = {k: m.get(k) for k in (
            "information_ratio", "active_return", "tracking_error",
            "avg_annual_turnover", "realized_beta", "sub_periods")}
        # fallback-to-benchmark rate: a too-strong penalty can push solves to bm
        # (portfolio_optimizer returns bm_weights.copy() on failure) — the §2-5
        # collapse signal the verdict must see (review M3).
        out["optimizer_failure_rate"] = getattr(res, "optimizer_failure_rate", None)
        return out

    summary = {
        "penalty": prod_cfg.factor_neutral_penalty,
        "axes": axes,
        "loading_cols": cols,
        "applied_dates": {"off": applied_off, "on": applied_on},
        "impute_frac": {"off": impute_off, "on": impute_on},
        "mean_abs_active_exposure": {"off": exp_off, "on": exp_on},
        "exposure_drop_pct": {
            a: ((exp_off[a] - exp_on[a]) / exp_off[a] * 100.0)
            if (exp_off.get(a) and exp_on.get(a) is not None and exp_off[a] != 0) else None
            for a in axes},
        "active_share": {"off": ashare_off, "on": ashare_on},  # §2-5 collapse signal
        "metrics": {"off": _m(res_off), "on": _m(res_on)},
    }

    # GAP2: the OFF arm must reproduce S0 (harvest-once round-trip identity).
    # Enforce as a gate when S0 metrics exist — a broken harvest (a re-introduced
    # double-overlay confound) would otherwise yield an authoritative-looking but
    # wrong exposure verdict.
    off_ir = summary["metrics"]["off"]["information_ratio"]
    s0_path = Path("outputs/iter15_65tkr_reb21_vtg/metrics.json")
    rt = None
    if s0_path.exists():
        s0m = json.load(open(s0_path, encoding="utf-8"))
        s0m = s0m.get("metrics", s0m)
        s0_ir = s0m.get("information_ratio")
        if s0_ir is not None and off_ir is not None:
            rt = abs(off_ir - s0_ir)
    summary["roundtrip_off_vs_s0_abs_diff"] = rt

    out_dir = Path("outputs/factor_ablation")   # CWD-relative => ai_port
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out_dir / "summary.json", "w", encoding="utf-8"),
              indent=2, default=str)
    if rt is not None and rt > 1e-3:
        print(f"[factor] ROUND-TRIP FAIL: OFF arm IR {off_ir:.4f} != S0 (|d|={rt:.4f} "
              f"> 1e-3) — harvest confound; exposure verdict not trustworthy.")
        return 1
    if rt is None:
        print("[factor] WARN: S0 metrics.json absent — round-trip identity unverified.")

    print(f"\n=== factor ablation (penalty={summary['penalty']}, axes={axes}) ===")
    print(f"  applied_dates off={applied_off} on={applied_on}  "
          f"impute_frac off={impute_off} on={impute_on}")
    print("  mean |active style exposure| per axis (off -> on, drop%):")
    for a in axes:
        eo, en = exp_off.get(a), exp_on.get(a)
        dp = summary["exposure_drop_pct"][a]
        print(f"    {a:10}: {eo} -> {en}  ({dp:+.1f}% )" if dp is not None
              else f"    {a:10}: {eo} -> {en}")
    mo, mn = summary["metrics"]["off"], summary["metrics"]["on"]
    print(f"  IR  off={mo['information_ratio']:.3f}  on={mn['information_ratio']:.3f}")
    print(f"  TE  off={mo['tracking_error']:.4f}  on={mn['tracking_error']:.4f}")
    print(f"  turn off={mo['avg_annual_turnover']:.3f}  on={mn['avg_annual_turnover']:.3f}")
    print(f"  active_share off={ashare_off} on={ashare_on}  "
          f"fallback off={mo['optimizer_failure_rate']} on={mn['optimizer_failure_rate']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
