#!/usr/bin/env python
"""A3 — trailing-IC adaptive-EMA arm driver (2-pass injection, src/ untouched).

The production prediction EMA blends with a FIXED alpha=0.5
(`src.model_trainer.apply_prediction_ema`). This arm makes alpha time-varying
from the trailing information coefficient, evaluated with the pre-registered
functional form (D0-distribution anchored, no free parameters):

    alpha_t = clip( 0.5 + (tIC_t - m) / (2*iqr), 0.25, 0.75 )

where tIC_t is the trailing-63-trading-day mean of realized ICs whose
REALIZATION date (= rebalance date + forward_horizon) lies strictly before t.
The blended panel is fed to the UNCHANGED production MVO via
run_backtest(precomputed_predictions=...), so nothing in src/ is edited.

Two pure functions (`compute_adaptive_alpha`, `apply_adaptive_ema`) carry the
whole contract and are import-only for the acceptance tests; the heavy backtest
plumbing lives in main() with lazy imports (so `--help` / test import stay fast).

Usage
-----
  python scripts/run_adaptive_ema_arm.py            # identity + arm
  python scripts/run_adaptive_ema_arm.py --identity-only
"""
from __future__ import annotations

import os
# Single-thread BLAS BEFORE numpy import (memory + determinism), matching
# scripts/run_dr_alpha.py.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from pathlib import Path

# UTF-8, line-buffered stdout so unicode (em-dash, Delta) survives a cp949
# console / redirect and progress flushes promptly.
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

# (heavy imports — src.backtest / UniverseData / yaml — are done lazily in
# main() so importing this module for the pure functions stays cheap.)

TRAILING_TD = 63  # pinned trailing trading-day window (spec §사전등록)

VARIANT_PATH = ROOT / "variants" / "iter15_65tkr_reb21_vtg.yaml"
PKL_PATH = ROOT / "outputs" / "iter15_65tkr_reb21_vtg" / "backtest_result.pkl"
S0_METRICS_PATH = ROOT / "outputs" / "iter15_65tkr_reb21_vtg" / "metrics.json"
D0_REPORT_PATH = ROOT / "outputs" / "degenerate_retrain_report.json"
OUT_DIR = ROOT / "outputs" / "exp_adaptive_ema"
OOS_CUTOFF = pd.Timestamp("2024-12-31")


# ===========================================================================
# Pure functions — the pinned contract (acceptance tests import exactly these)
# ===========================================================================
def compute_adaptive_alpha(
    ic_events: pd.Series,
    dates: pd.DatetimeIndex,
    m: float,
    iqr: float,
    window: int = TRAILING_TD,
) -> pd.Series:
    """Causal trailing-IC adaptive alpha series.

    ic_events : Series indexed by each IC event's REALIZATION-completion date
        (Timestamp = prediction_date + forward_horizon), value = realized IC.
        Duplicate index timestamps are allowed (averaged into the window mean);
        may be empty (still DatetimeIndex-typed). Because the caller keys events
        by realization date, this function is causal by construction.
    dates : ordered DatetimeIndex the alphas are wanted for (== the prediction /
        rebalance calendar; in production == raw_predictions.index).
    m, iqr : D0 trailing-IC distribution anchors (median, IQR).

    Returns a float Series indexed EXACTLY by `dates`:
        alpha_t = clip( 0.5 + (tIC_t - m) / (2*iqr), 0.25, 0.75 )
      with tIC_t = mean of ic_events in the trailing window
        [ dates[max(0, i-window)] , dates[i-1] ]  (upper bound STRICTLY < t).
      i == 0 or no in-window event => tIC undefined => alpha_t = 0.5 exactly.
    """
    ev_idx = np.asarray(ic_events.index.values, dtype="datetime64[ns]")
    ev_val = np.asarray(ic_events.values, dtype=float)
    out = np.empty(len(dates), dtype=float)
    for i in range(len(dates)):
        if i == 0:
            out[i] = 0.5
            continue
        lo = np.datetime64(dates[max(0, i - window)], "ns")
        hi = np.datetime64(dates[i - 1], "ns")          # STRICTLY before dates[i]
        mask = (ev_idx >= lo) & (ev_idx <= hi)
        if mask.sum() == 0:
            out[i] = 0.5
            continue
        tic = float(ev_val[mask].mean())
        a = 0.5 + (tic - m) / (2.0 * iqr)
        out[i] = min(max(a, 0.25), 0.75)
    return pd.Series(out, index=dates, dtype=float)


def apply_adaptive_ema(
    raw_predictions: pd.DataFrame, alpha_series: pd.Series
) -> pd.DataFrame:
    """Time-varying-alpha generalization of src.model_trainer.apply_prediction_ema.

    Same recursion / initialization / NaN handling as apply_prediction_ema, but
    the blend weight is read per date from `alpha_series`:
        blended_t[c] = alpha_t * raw_t[c] + (1 - alpha_t) * blended_{t-1}[c]
    over the ticker intersection c = (tickers on t) ∩ (tickers on prev row).
    All-NaN rows are skipped; the first non-empty row is passed through (no prev,
    so its alpha is not applied); tickers absent from prev keep their raw value;
    the output NaN mask equals the input's. With alpha_series ≡ 0.5 this is
    byte-identical to apply_prediction_ema(raw_predictions, 0.5).
    """
    out = raw_predictions.copy()
    prev = None
    for d in out.index:
        cur = out.loc[d].dropna()
        if len(cur) == 0:
            continue
        if prev is not None:
            common = cur.index.intersection(prev.index)
            if len(common) > 0:
                a = float(alpha_series.loc[d])
                blended = a * cur[common] + (1 - a) * prev[common]
                cur.loc[common] = blended
                out.loc[d, common] = blended.values
        prev = cur
    return out


# ===========================================================================
# main() — identity reproduction + arm run (lazy heavy imports; not run at import)
# ===========================================================================
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


def _load_anchors() -> tuple[float, float]:
    """m (median), iqr from the D0 trailing-IC distribution (report is canonical —
    values are never hardcoded)."""
    with D0_REPORT_PATH.open("r", encoding="utf-8") as fh:
        rep = json.load(fh)
    dist = rep["report"]["trailing_ic_dist"]
    return float(dist["median"]), float(dist["iqr"])


def _build_ic_events(ic_series: pd.Series, calendar: pd.DatetimeIndex,
                     horizon: int) -> pd.Series:
    """Shift each IC event from its rebalance/prediction date to its
    REALIZATION-completion date (= prediction date + `horizon` trading days along
    `calendar`). Events whose realization falls past the calendar end are dropped
    (they never complete in-sample, so they inform no in-sample alpha)."""
    pos = calendar.get_indexer(ic_series.index)
    real_dates, real_vals = [], []
    for p, v in zip(pos, ic_series.values):
        if p < 0:
            continue  # rebalance date absent from calendar (should not happen)
        rp = p + horizon
        if rp < len(calendar):
            real_dates.append(calendar[rp])
            real_vals.append(float(v))
    return pd.Series(
        real_vals, index=pd.DatetimeIndex(real_dates), dtype=float
    ).sort_index()


def _compose_variant_config():
    """Rebuild the S0 variant config exactly as run_variant.compose_config does
    (the pkl carries no config), so the injected identity/arm runs use the same
    optimizer/overlay settings that produced S0."""
    import yaml
    from src.harness import build_override_config
    with VARIANT_PATH.open("r", encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh) or {}
    overrides = dict(manifest.get("overrides") or {})
    tuning_mode = manifest.get("tuning_mode", "production")
    if tuning_mode == "tuning":
        overrides["enforce_oos_holdout"] = True
    else:  # production / oos_verify
        overrides["enforce_oos_holdout"] = False
    return build_override_config(overrides)


def _inject_run(data, base, cfg, predictions, raw_predictions):
    """Run the UNCHANGED production MVO on an injected prediction panel, mirroring
    run_variant's checkpoint-reuse injection (overlays applied exactly once)."""
    from src.backtest import run_backtest
    return run_backtest(
        data,
        precomputed_panel=base.panel,
        precomputed_feature_names=base.feature_names,
        precomputed_feature_groups=base.feature_groups,
        precomputed_targets=base.targets,
        precomputed_models=base.models,
        precomputed_predictions=predictions,
        precomputed_raw_predictions=raw_predictions,
        config=cfg,
    )


def _alpha_distribution(alpha: pd.Series) -> dict:
    v = alpha.values.astype(float)
    return {
        "min": float(np.min(v)),
        "median": float(np.median(v)),
        "max": float(np.max(v)),
        "mean": float(np.mean(v)),
        "frac_off_half": float(np.mean(np.abs(v - 0.5) > 1e-12)),
        "frac_at_upper_clip": float(np.mean(v >= 0.75 - 1e-12)),
        "frac_at_lower_clip": float(np.mean(v <= 0.25 + 1e-12)),
        "n": int(len(v)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--identity-only", action="store_true",
                    help="run only the alpha=0.5 identity reproduction of S0")
    args = ap.parse_args()

    import pickle
    from src.harness import inject_config

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m, iqr = _load_anchors()
    print(f"[a3] D0 anchors: m(median)={m:.6f} iqr={iqr:.6f}")

    # --- Load the certified S0 harvest (raw predictions + IC + reusable stages).
    print(f"[a3] loading S0 harvest: {PKL_PATH}")
    with PKL_PATH.open("rb") as fh:
        base = pickle.load(fh)
    raw = base.raw_predictions
    ic_series = base.ic_series
    calendar = raw.index

    cfg = _compose_variant_config()
    inject_config(cfg)
    horizon = int(getattr(cfg, "forward_horizon", 20))
    print(f"[a3] forward_horizon={horizon} | raw_predictions {raw.shape} "
          f"| ic_series n={len(ic_series)}")

    ic_events = _build_ic_events(ic_series, calendar, horizon)
    print(f"[a3] ic_events (realization-dated): n={len(ic_events)} "
          f"span {ic_events.index[0].date()}..{ic_events.index[-1].date()}")

    # UniverseData is the one stage the pkl does not carry (needed by run_backtest
    # for the simulation loop / benchmark). Everything else is injected precomputed.
    from src.data_loader import UniverseData
    data = UniverseData(cfg.data_path, config=cfg)

    with S0_METRICS_PATH.open("r", encoding="utf-8") as fh:
        s0 = json.load(fh)["metrics"]

    # --- Pass 1: identity (alpha ≡ 0.5) must reproduce S0 under the same solver.
    print("[a3] identity pass: apply_adaptive_ema(raw, 0.5) -> production MVO")
    t0 = time.time()
    alpha_half = pd.Series(0.5, index=calendar)
    pred_half = apply_adaptive_ema(raw, alpha_half)
    res_id = _inject_run(data, base, cfg, pred_half, raw)
    m_id = _headline(res_id)
    id_dir = OUT_DIR / "identity"
    id_dir.mkdir(parents=True, exist_ok=True)
    keys = ("information_ratio", "tracking_error", "avg_annual_turnover",
            "active_return")
    deltas = {k: (m_id.get(k), s0.get(k),
                  (None if (m_id.get(k) is None or s0.get(k) is None)
                   else float(m_id[k] - s0[k]))) for k in keys}
    print(f"[a3] identity done in {time.time()-t0:.0f}s — vs S0:")
    for k, (got, ref, d) in deltas.items():
        print(f"      {k}: {got} vs S0 {ref}  (Δ {d})")
    with (id_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump({"metrics": m_id, "s0_metrics": s0, "deltas": deltas},
                  fh, indent=2, default=str)

    # Identity gate: near-exact (same solver / same everything). Report loudly on
    # any material mismatch; per CLAUDE.md §9 the caller decides whether to halt.
    max_abs_delta = max(
        abs(d[2]) for d in deltas.values() if d[2] is not None
    )
    print(f"[a3] identity max |Δ| over {list(keys)} = {max_abs_delta:.3e}")
    if max_abs_delta > 1e-6:
        print("[a3] WARNING: identity did NOT reproduce S0 within 1e-6 — "
              "investigate injection path / panel semantics before trusting arm.")

    if args.identity_only:
        print("[a3] --identity-only set; skipping arm.")
        return 0

    # --- Pass 2: arm (time-varying alpha from trailing IC).
    print("[a3] arm pass: adaptive alpha -> apply_adaptive_ema -> production MVO")
    t1 = time.time()
    alpha = compute_adaptive_alpha(ic_events, calendar, m, iqr)
    a_dist = _alpha_distribution(alpha)
    print(f"[a3] alpha dist: min={a_dist['min']:.3f} med={a_dist['median']:.3f} "
          f"max={a_dist['max']:.3f} frac_off_half={a_dist['frac_off_half']:.3f}")
    pred_arm = apply_adaptive_ema(raw, alpha)
    res_arm = _inject_run(data, base, cfg, pred_arm, raw)
    m_arm = _headline(res_arm)
    arm_dir = OUT_DIR / "arm"
    arm_dir.mkdir(parents=True, exist_ok=True)
    b, r = m_id, m_arm
    print(f"[a3] arm done in {time.time()-t1:.0f}s — "
          f"IR {b['information_ratio']:.3f} -> {r['information_ratio']:.3f} "
          f"(Δ{r['information_ratio']-b['information_ratio']:+.3f}) | "
          f"TE {b['tracking_error']:.4f} -> {r['tracking_error']:.4f} | "
          f"turn {b['avg_annual_turnover']:.3f} -> {r['avg_annual_turnover']:.3f}")
    with (arm_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump({"metrics": m_arm, "identity_metrics": m_id, "s0_metrics": s0,
                   "alpha_distribution": a_dist,
                   "anchors": {"m": m, "iqr": iqr, "window": TRAILING_TD,
                               "forward_horizon": horizon}},
                  fh, indent=2, default=str)
    alpha.to_frame("alpha").to_csv(arm_dir / "alpha_series.csv")
    print(f"[a3] artifacts: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
