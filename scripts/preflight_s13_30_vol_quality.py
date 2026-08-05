#!/usr/bin/env python
"""§S13.30 preflight: does quality discriminate WITHIN the high-vol bucket?

Read-only diagnostic (no backtest, no retrain) pre-registered in the decision
log §S13.30 BEFORE this ran. Purpose: gate the vol×quality confirmation
feature arm (soft_and(z(idio_vol_63d), z(best_roe_level_z))) — vol alpha is
structural (§S13.27/28/29), so the only live question is whether quality
separates junk-vol from alpha-vol *inside* the top-vol tercile.

Gates (PROCEED requires all three):
  P1  conditional rank-IC (top-vol tercile, quality vs fwd 21d) mean >= 0
      AND median-split spread (high-Q minus low-Q, EW) mean >= 0
  P2  spread in BM-down 21d windows > 0 AND t >= 2.0 AND subperiod sign >= 2/3
  P3  final-rebalance OW top-12 quality z median < +0.5 (room to fix)

Secondary diagnostics (explanatory ONLY, never for arm parameter selection):
earnings_quality_252d, and 63d-forward repeats (overlapping windows — naive t).
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

HARVEST = Path("outputs/codex_causal_rank_65/backtest_result.pkl")
OUT_DIR = Path("outputs/vol_quality_precheck")

VOL_COL = "idio_vol_63d"
PRIMARY_Q = "best_roe_level_z"
SECONDARY_Q = "earnings_quality_252d"
HORIZONS = {"21d": 21, "63d": 63}
MIN_TERCILE_NAMES = 15   # sanity floor for a usable tercile cross-section


def _zscore_row(s: pd.Series) -> pd.Series:
    """1/99-winsorised cross-sectional z (row version of §S13.28 _zscore)."""
    s = s.dropna()
    if len(s) < 3:
        return pd.Series(dtype=float)
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    sw = s.clip(lo, hi)
    sd = sw.std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(dtype=float)
    return (sw - sw.mean()) / sd


def _window_after(index: pd.Index, date, horizon: int):
    pos = index.searchsorted(date)
    if pos < len(index) and index[pos] == date:
        pos += 1  # window starts strictly after the rebalance date
    window = index[pos:pos + horizon]
    return window if len(window) == horizon else None


def _fwd_return(returns: pd.DataFrame, date, horizon: int):
    window = _window_after(returns.index, date, horizon)
    if window is None:
        return None
    return (1.0 + returns.loc[window]).prod() - 1.0


def _bm_fwd(bm_ret: pd.Series, date, horizon: int):
    window = _window_after(bm_ret.index, date, horizon)
    if window is None:
        return None
    return float((1.0 + bm_ret.loc[window]).prod() - 1.0)


def _tstat(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float("nan")
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(len(x))) if sd > 0 else float("nan")


def _summarise(rows: list, key: str) -> dict:
    vals = np.array([r[key] for r in rows if np.isfinite(r.get(key, np.nan))])
    thirds = np.array_split(np.arange(len(rows)), 3)
    sub_means = []
    for block in thirds:
        v = np.array([rows[i][key] for i in block
                      if np.isfinite(rows[i].get(key, np.nan))])
        sub_means.append(float(v.mean()) if len(v) else float("nan"))
    return {
        "n": int(len(vals)),
        "mean": float(vals.mean()) if len(vals) else float("nan"),
        "t": _tstat(vals),
        "subperiod_means": sub_means,
        "subperiod_pos": int(sum(1 for m in sub_means if np.isfinite(m) and m > 0)),
    }


def run_metric(dates, vol_panel, q_panel, returns, bm_ret, horizon: int) -> dict:
    """Per-rebalance conditional IC and median-split spread in top-vol tercile."""
    rows = []
    for dt in dates:
        if dt not in vol_panel.index or dt not in q_panel.index:
            continue
        vz = _zscore_row(vol_panel.loc[dt])
        if len(vz) < 3 * MIN_TERCILE_NAMES:
            continue
        top = vz.sort_values(ascending=False).head(len(vz) // 3).index
        q = q_panel.loc[dt].reindex(top).dropna()
        fwd = _fwd_return(returns, dt, horizon)
        if fwd is None:
            continue
        common = q.index.intersection(fwd.dropna().index)
        if len(common) < MIN_TERCILE_NAMES:
            continue
        q, f = q.loc[common], fwd.loc[common]
        ic = q.rank().corr(f.rank())  # Spearman
        med = q.median()
        hi, lo = f[q > med], f[q <= med]
        spread = float(hi.mean() - lo.mean()) if len(hi) and len(lo) else np.nan
        rows.append({"date": str(pd.Timestamp(dt).date()), "n": int(len(common)),
                     "ic": float(ic) if np.isfinite(ic) else np.nan,
                     "spread": spread, "bm_fwd": _bm_fwd(bm_ret, dt, horizon)})

    down = [r for r in rows if r["bm_fwd"] is not None and r["bm_fwd"] < 0]
    up = [r for r in rows if r["bm_fwd"] is not None and r["bm_fwd"] >= 0]
    return {
        "windows": len(rows),
        "ic": _summarise(rows, "ic"),
        "spread": _summarise(rows, "spread"),
        "spread_bm_down": _summarise(down, "spread"),
        "spread_bm_up": _summarise(up, "spread"),
        "rows": rows,
    }


def main() -> int:
    import yaml
    from src.backtest import get_benchmark_fn
    from src.data_loader import UniverseData
    from src.harness import build_override_config, inject_config

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[S13.30] loading production harvest ...", flush=True)
    with HARVEST.open("rb") as fh:
        base = pickle.load(fh)

    # §4.3-style existence check: every configured column must exist.
    cols = set(base.panel.columns)
    missing = [c for c in (VOL_COL, PRIMARY_Q, SECONDARY_Q) if c not in cols]
    if missing:
        print(f"[S13.30] FATAL missing panel columns: {missing}", flush=True)
        return 2

    overrides = (yaml.safe_load(
        (ROOT / "variants" / "codex_causal_rank_65.yaml").read_text(encoding="utf-8")
    ) or {}).get("overrides", {})
    cfg = build_override_config(dict(overrides))
    inject_config(cfg)
    print("[S13.30] loading universe data ...", flush=True)
    data = UniverseData(cfg.data_path, config=cfg)

    returns = data.returns_masked
    bm_ret = base.benchmark_returns
    dates = sorted(base.portfolio_weights.keys())
    print(f"[S13.30] rebalances={len(dates)} "
          f"({pd.Timestamp(dates[0]).date()} .. {pd.Timestamp(dates[-1]).date()})",
          flush=True)

    vol_panel = base.panel[VOL_COL].unstack("ticker")
    panels = {PRIMARY_Q: base.panel[PRIMARY_Q].unstack("ticker"),
              SECONDARY_Q: base.panel[SECONDARY_Q].unstack("ticker")}
    coverage = {name: float(p.reindex(index=vol_panel.index).notna().mean(axis=1).mean())
                for name, p in panels.items()}
    print(f"[S13.30] mean cross-sectional coverage: {coverage}", flush=True)

    metrics = {}
    for qname, qpanel in panels.items():
        for hname, h in HORIZONS.items():
            key = f"{qname}__{hname}"
            print(f"[S13.30] computing {key} ...", flush=True)
            metrics[key] = run_metric(dates, vol_panel, qpanel, returns, bm_ret, h)

    # P3: final-rebalance OW top-12 quality/vol z.
    last = dates[-1]
    tickers = list(data.tickers)
    bm_fn = get_benchmark_fn(data, tickers, config=cfg)
    w = base.portfolio_weights[last].reindex(tickers).fillna(0.0)
    bm_w = pd.Series(np.asarray(bm_fn(last, tickers, len(tickers)), dtype=float),
                     index=tickers)
    active = (w - bm_w).sort_values(ascending=False)
    qz = _zscore_row(panels[PRIMARY_Q].loc[last])
    vz = _zscore_row(vol_panel.loc[last])
    ow12 = [{"ticker": t, "active_w_pct": round(100 * float(active[t]), 2),
             "quality_z": round(float(qz.get(t, np.nan)), 3),
             "vol_z": round(float(vz.get(t, np.nan)), 3)}
            for t in active.head(12).index]
    ow_qz = np.array([r["quality_z"] for r in ow12 if np.isfinite(r["quality_z"])])
    ow_median_qz = float(np.median(ow_qz)) if len(ow_qz) else float("nan")

    # Pre-registered gates on the PRIMARY metric (best_roe_level_z, 21d).
    m = metrics[f"{PRIMARY_Q}__21d"]
    p1 = bool(m["ic"]["mean"] >= 0 and m["spread"]["mean"] >= 0)
    p2 = bool(m["spread_bm_down"]["mean"] > 0
              and m["spread_bm_down"]["t"] >= 2.0
              and m["spread_bm_down"]["subperiod_pos"] >= 2)
    p3 = bool(np.isfinite(ow_median_qz) and ow_median_qz < 0.5)
    proceed = p1 and p2 and p3

    summary = {
        "section": "S13.30 preflight",
        "harvest": str(HARVEST),
        "vol_col": VOL_COL,
        "primary_quality": PRIMARY_Q,
        "secondary_quality": SECONDARY_Q,
        "coverage": coverage,
        "metrics": {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                    for k, v in metrics.items()},
        "ow_top12_last_rebalance": {"date": str(pd.Timestamp(last).date()),
                                    "rows": ow12,
                                    "median_quality_z": ow_median_qz},
        "gates": {"P1_discrimination": p1, "P2_downside_defence": p2,
                  "P3_room_in_book": p3, "PROCEED": proceed},
        "runtime_s": round(time.time() - t0, 1),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(metrics[f"{PRIMARY_Q}__21d"]["rows"]).to_csv(
        OUT_DIR / "primary_21d_windows.csv", index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"[S13.30] PROCEED={proceed} (P1={p1} P2={p2} P3={p3})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
