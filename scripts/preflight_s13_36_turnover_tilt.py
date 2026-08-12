#!/usr/bin/env python
"""§S13.36 preflight: is share_turnover_63d incremental to the final score?

Read-only diagnostic (no backtest, no retrain) pre-registered in the decision
log §S13.36 BEFORE this ran. §S13.35 arm A proved the model *consumes* the
volume signal (IC 0.020->0.027) yet loses money through the feature-layer
path (ranking rearrangement pays out of carry, §S13.12). The candidate is a
tilt-layer promotion (score' = score + 0.25*sd*z_st on all scored names), so
the live questions are:

  P1  does the information survive orthogonalisation to the final prediction
      score?  residual rank-IC (z_st residualised on winsor-z of the final
      score, fwd 21d) mean > 0 AND t >= 2.0 AND subperiod sign >= 2/3
  P2  is it not just the structural vol tilt again?  mean cross-sectional
      Spearman rho(z_st, z_idio_vol) < 0.6
  P3  is there room in the book?  mean active-weighted z_st exposure
      (sum(active*z), §S13.31 `active_exposure` unit) < +0.25

PROCEED requires all three; otherwise SHELVE (no implementation, no
inventory entry — read-only precedent §S13.7/§S13.30).
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.preflight_s13_30_vol_quality import (  # noqa: E402
    _fwd_return,
    _summarise,
    _zscore_row,
)

HARVEST = Path("outputs/s0_recert_s13_35/backtest_result.pkl")
VARIANT = Path("variants/s0_recert_s13_35.yaml")
OUT_DIR = Path("outputs/s13_36_turnover_precheck")

VOL_COL = "idio_vol_63d"
ST_COL = "share_turnover_63d"
HORIZON = 21
MIN_NAMES = 30   # same floor as the tilt's own scored-names threshold


def residualize(y: pd.Series, x: pd.Series) -> pd.Series:
    """OLS residual of y on [1, x] over the common non-NaN index."""
    common = y.dropna().index.intersection(x.dropna().index)
    if len(common) < 3:
        return pd.Series(dtype=float)
    yv = y.loc[common].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(common)),
                         x.loc[common].to_numpy(dtype=float)])
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    return pd.Series(yv - X @ beta, index=common)


def _spearman(a: pd.Series, b: pd.Series) -> float:
    common = a.dropna().index.intersection(b.dropna().index)
    if len(common) < 3:
        return float("nan")
    r = a.loc[common].rank().corr(b.loc[common].rank())
    return float(r) if np.isfinite(r) else float("nan")


def compute_window_row(z_st, z_pred, z_vol, fwd, min_names: int = MIN_NAMES):
    """Per-rebalance raw/residual IC and vol overlap among scored names."""
    common = z_st.dropna().index.intersection(z_pred.dropna().index)
    common = common.intersection(fwd.dropna().index)
    if len(common) < min_names:
        return None
    raw_ic = _spearman(z_st.loc[common], fwd.loc[common])
    resid = residualize(z_st.loc[common], z_pred.loc[common])
    resid_ic = _spearman(resid, fwd.loc[common])
    overlap = _spearman(z_st.loc[common], z_vol.loc[common])
    return {"n": int(len(common)), "raw_ic": raw_ic,
            "resid_ic": resid_ic, "vol_overlap": overlap}


def evaluate_gates(resid_summary: dict, overlap_mean: float,
                   expo_mean: float) -> dict:
    p1 = bool(np.isfinite(resid_summary["mean"])
              and resid_summary["mean"] > 0
              and resid_summary["t"] >= 2.0
              and resid_summary["subperiod_pos"] >= 2)
    p2 = bool(np.isfinite(overlap_mean) and overlap_mean < 0.6)
    p3 = bool(np.isfinite(expo_mean) and expo_mean < 0.25)
    return {"P1_residual_ic": p1, "P2_vol_overlap": p2,
            "P3_room_in_book": p3, "PROCEED": p1 and p2 and p3}


def main() -> int:
    import yaml
    from src.backtest import get_benchmark_fn
    from src.data_loader import UniverseData
    from src.features.volume_flow import build_volume_flow_features
    from src.harness import build_override_config, inject_config

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[S13.36] loading S0' harvest ...", flush=True)
    with HARVEST.open("rb") as fh:
        base = pickle.load(fh)

    if VOL_COL not in set(base.panel.columns):
        print(f"[S13.36] FATAL missing panel column: {VOL_COL}", flush=True)
        return 2
    if getattr(base, "predictions", None) is None:
        print("[S13.36] FATAL: harvest has no final predictions", flush=True)
        return 2

    overrides = (yaml.safe_load(
        (ROOT / VARIANT).read_text(encoding="utf-8")
    ) or {}).get("overrides", {})
    cfg = build_override_config(dict(overrides))
    inject_config(cfg)
    print("[S13.36] loading universe data ...", flush=True)
    data = UniverseData(cfg.data_path, config=cfg)

    st_panel = build_volume_flow_features(data).get(ST_COL)
    if st_panel is None:
        print(f"[S13.36] FATAL: {ST_COL} could not be built", flush=True)
        return 2

    returns = data.returns_masked
    preds = base.predictions
    vol_panel = base.panel[VOL_COL].unstack("ticker")
    dates = sorted(base.portfolio_weights.keys())
    tickers = list(data.tickers)
    bm_fn = get_benchmark_fn(data, tickers, config=cfg)
    print(f"[S13.36] rebalances={len(dates)} "
          f"({pd.Timestamp(dates[0]).date()} .. {pd.Timestamp(dates[-1]).date()}) "
          f"st_coverage={float(st_panel.notna().mean(axis=1).mean()):.3f}",
          flush=True)

    rows, expo_rows, skipped = [], [], 0
    for dt in dates:
        if (dt not in preds.index or dt not in st_panel.index
                or dt not in vol_panel.index):
            skipped += 1
            continue
        s = preds.loc[dt]
        scored = s.index[s.notna()]
        if len(scored) < MIN_NAMES:
            skipped += 1
            continue

        # P3 exposure on the full-universe z (§S13.31 unit) — every rebalance.
        z_full = _zscore_row(st_panel.loc[dt])
        if not z_full.empty:
            w = base.portfolio_weights[dt].reindex(tickers).fillna(0.0)
            bm_w = pd.Series(
                np.asarray(bm_fn(dt, tickers, len(tickers)), dtype=float),
                index=tickers)
            a = (w - bm_w).reindex(z_full.index).fillna(0.0)
            expo_rows.append({"date": str(pd.Timestamp(dt).date()),
                              "expo": float((a * z_full).sum())})

        fwd = _fwd_return(returns, dt, HORIZON)
        if fwd is None:
            continue
        row = compute_window_row(
            _zscore_row(st_panel.loc[dt].reindex(scored)),
            _zscore_row(s[scored]),
            _zscore_row(vol_panel.loc[dt].reindex(scored)),
            fwd,
        )
        if row is None:
            continue
        row["date"] = str(pd.Timestamp(dt).date())
        rows.append(row)

    raw_sum = _summarise(rows, "raw_ic")
    resid_sum = _summarise(rows, "resid_ic")
    overlap_sum = _summarise(rows, "vol_overlap")
    expo_sum = _summarise(expo_rows, "expo")
    gates = evaluate_gates(resid_sum, overlap_sum["mean"], expo_sum["mean"])

    summary = {
        "section": "S13.36 preflight",
        "harvest": str(HARVEST),
        "variant": str(VARIANT),
        "st_col": ST_COL,
        "vol_col": VOL_COL,
        "horizon_bd": HORIZON,
        "windows": len(rows),
        "expo_rebalances": len(expo_rows),
        "skipped_dates": skipped,
        "raw_ic": raw_sum,
        "residual_ic": resid_sum,
        "vol_overlap": overlap_sum,
        "active_weighted_expo": expo_sum,
        "gates": gates,
        "runtime_s": round(time.time() - t0, 1),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT_DIR / "windows_21d.csv", index=False)
    pd.DataFrame(expo_rows).to_csv(OUT_DIR / "exposure_rebalances.csv",
                                   index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"[S13.36] PROCEED={gates['PROCEED']} "
          f"(P1={gates['P1_residual_ic']} P2={gates['P2_vol_overlap']} "
          f"P3={gates['P3_room_in_book']})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
