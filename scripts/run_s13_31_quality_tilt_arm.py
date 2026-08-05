#!/usr/bin/env python
"""§S13.31: 63d-horizon quality tilt inside the top-vol tercile (re-MVO arm).

Harvest-once / re-MVO on the SAVED production harvest (§S13.28 pattern):
inject ``pre_overlay_predictions`` (CLAUDE.md §4.2 canonical object) so the
production overlays are re-applied exactly once; no model retrain.

Pre-registered in the decision log §S13.31 BEFORE this ran:
  Q0  harvest injected unchanged     -> round-trip gate vs production
  Q1  score' = score + LAM*sd(scored)*z_q inside the top idio_vol tercile
      (z_q = winsorised z of best_roe_level_z WITHIN the tercile; cells
      outside the tercile or with missing quality are byte-unchanged)

LAM = 0.25 is the single pre-committed value — no sweep. The tilt is the
prediction-layer carrier of the §S13.30 finding that quality's defensive
payoff inside high-vol lives at the 63d horizon (quality z persists across
rebalances, so a standing tilt accrues the slow effect without touching the
21d label structure).
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
PROD_METRICS = Path("outputs/codex_causal_rank_65/metrics.json")
OUT_DIR = Path("outputs/s13_31_quality_tilt")

VOL_COL = "idio_vol_63d"
Q_COL = "best_roe_level_z"
LAM = 0.25
MIN_NAMES = 30


def _zscore_row(s: pd.Series) -> pd.Series:
    """1/99-winsorised cross-sectional z (§S13.30 helper, duplicated per
    script-self-containment convention)."""
    s = s.dropna()
    if len(s) < 3:
        return pd.Series(dtype=float)
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    sw = s.clip(lo, hi)
    sd = sw.std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(dtype=float)
    return (sw - sw.mean()) / sd


def apply_quality_tilt(preds: pd.DataFrame, vol_panel: pd.DataFrame,
                       q_panel: pd.DataFrame, lam: float,
                       min_names: int = MIN_NAMES):
    """Tilt scores toward quality inside the top-vol tercile, per date."""
    out = preds.copy()
    tilted = passthrough = 0
    deltas = []
    for dt in preds.index:
        s = preds.loc[dt]
        scored = s.index[s.notna()]
        if (len(scored) < min_names or dt not in vol_panel.index
                or dt not in q_panel.index):
            passthrough += 1
            continue
        vz = _zscore_row(vol_panel.loc[dt].reindex(scored))
        if len(vz) < min_names:
            passthrough += 1
            continue
        top = vz.sort_values(ascending=False).head(len(vz) // 3).index
        zq = _zscore_row(q_panel.loc[dt].reindex(top))
        sd = float(s[scored].std(ddof=1))
        if zq.empty or not np.isfinite(sd) or sd <= 0:
            passthrough += 1
            continue
        out.loc[dt, zq.index] = s[zq.index] + lam * sd * zq
        deltas.append(float((lam * sd * zq).abs().mean()))
        tilted += 1
    diag = {
        "tilted_dates": int(tilted),
        "passthrough_dates": int(passthrough),
        "mean_abs_delta": float(np.mean(deltas)) if deltas else 0.0,
    }
    return out, diag


def _max_drawdown(returns: pd.Series) -> float:
    cum = (1.0 + returns.dropna()).cumprod()
    return float((cum / cum.cummax() - 1.0).min())


def _subperiod_irs(act: pd.Series) -> list:
    out = []
    for block in np.array_split(act.dropna(), 3):
        sd = block.std(ddof=1)
        out.append(float(block.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan"))
    return out


def _downcap(port: pd.Series, bm: pd.Series) -> float:
    down = bm < 0
    denom = bm[down].mean()
    return float(port[down].mean() / denom) if down.any() and denom != 0 else float("nan")


def main() -> int:
    import yaml
    from src.backtest import get_benchmark_fn, run_backtest
    from src.data_loader import UniverseData
    from src.harness import build_override_config, inject_config

    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[S13.31] loading saved production harvest ...", flush=True)
    with HARVEST.open("rb") as fh:
        base = pickle.load(fh)
    prod_m = json.loads(PROD_METRICS.read_text(encoding="utf-8"))
    prod_m = prod_m.get("metrics", prod_m)

    cols = set(base.panel.columns)
    missing = [c for c in (VOL_COL, Q_COL) if c not in cols]
    if missing:
        print(f"[S13.31] FATAL missing panel columns: {missing}", flush=True)
        return 2

    overrides = (yaml.safe_load(
        (ROOT / "variants" / "codex_causal_rank_65.yaml").read_text(encoding="utf-8")
    ) or {}).get("overrides", {})
    cfg = build_override_config(dict(overrides))
    inject_config(cfg)
    print("[S13.31] loading universe data ...", flush=True)
    data = UniverseData(cfg.data_path, config=cfg)

    vol_panel = base.panel[VOL_COL].unstack("ticker")
    q_panel = base.panel[Q_COL].unstack("ticker")
    preds = base.pre_overlay_predictions
    print("[S13.31] applying quality tilt ...", flush=True)
    tilted, diag = apply_quality_tilt(preds, vol_panel, q_panel, LAM)
    print("[S13.31] tilt diagnostics: " + json.dumps(diag), flush=True)

    tickers = list(data.tickers)
    bm_fn = get_benchmark_fn(data, tickers, config=cfg)

    def active_share(res) -> float:
        vals = []
        for date, wser in res.portfolio_weights.items():
            w = np.nan_to_num(wser.reindex(tickers).to_numpy(dtype=float))
            bm = np.asarray(bm_fn(date, tickers, len(tickers)), dtype=float)
            vals.append(0.5 * float(np.abs(w - bm).sum()))
        return float(np.mean(vals)) if vals else float("nan")

    def active_exposure(res, z_panel: pd.DataFrame) -> float:
        vals = []
        for date, wser in res.portfolio_weights.items():
            if date not in z_panel.index:
                continue
            z = _zscore_row(z_panel.loc[date])
            if z.empty:
                continue
            w = wser.reindex(tickers).fillna(0.0)
            bm = pd.Series(np.asarray(bm_fn(date, tickers, len(tickers)),
                                      dtype=float), index=tickers)
            a = (w - bm).reindex(z.index).fillna(0.0)
            vals.append(float((a * z).sum()))
        return float(np.mean(vals)) if vals else float("nan")

    results = {}
    for name, panel_preds in (("Q0_identity", preds), ("Q1_quality_tilt", tilted)):
        print(f"\n[S13.31] === arm {name}: re-MVO on the same harvest ===", flush=True)
        t0 = time.time()
        inject_config(cfg)
        res = run_backtest(
            data, config=cfg,
            precomputed_panel=base.panel,
            precomputed_feature_names=base.feature_names,
            precomputed_feature_groups=base.feature_groups,
            precomputed_targets=base.targets,
            precomputed_models=base.models,
            precomputed_predictions=panel_preds,
            precomputed_raw_predictions=base.raw_predictions,
        )
        m = res.compute_metrics()
        act = (res.portfolio_returns - res.benchmark_returns).dropna()
        sub_irs = _subperiod_irs(act)
        results[name] = {
            "information_ratio": float(m["information_ratio"]),
            "active_return": float(m["active_return"]),
            "tracking_error": float(m["tracking_error"]),
            "realized_beta": float(m.get("realized_beta", float("nan"))),
            "avg_annual_turnover": float(m.get("avg_annual_turnover", float("nan"))),
            "optimizer_failure_rate": float(getattr(res, "optimizer_failure_rate", float("nan"))),
            "active_share": active_share(res),
            "max_drawdown": _max_drawdown(res.portfolio_returns),
            "active_max_drawdown": _max_drawdown(act),
            "subperiod_irs": sub_irs,
            "worst_subperiod_ir": float(np.nanmin(sub_irs)),
            "downside_capture": _downcap(res.portfolio_returns, res.benchmark_returns),
            "quality_z_exposure": active_exposure(res, q_panel),
            "vol_z_exposure": active_exposure(res, vol_panel),
            "runtime_s": round(time.time() - t0, 1),
        }
        print(f"[S13.31] {name}: " + json.dumps(results[name]), flush=True)
        act.to_csv(OUT_DIR / f"{name}_active_returns.csv")

    q0, q1 = results["Q0_identity"], results["Q1_quality_tilt"]
    prod_ir, prod_ar = float(prod_m["information_ratio"]), float(prod_m["active_return"])
    g0 = (abs(q0["information_ratio"] - prod_ir) <= 0.005
          and abs(q0["active_return"] - prod_ar) <= 1e-4)

    d1a = q1["quality_z_exposure"] > q0["quality_z_exposure"]
    tail_improvements = {
        "max_drawdown": q1["max_drawdown"] > q0["max_drawdown"],
        "active_max_drawdown": q1["active_max_drawdown"] > q0["active_max_drawdown"],
        "worst_subperiod_ir": q1["worst_subperiod_ir"] > q0["worst_subperiod_ir"],
        "downside_capture": q1["downside_capture"] < q0["downside_capture"],
    }
    d1b = sum(tail_improvements.values()) >= 1
    d2_dir = q1["information_ratio"] - q0["information_ratio"]
    e2 = (q1["tracking_error"] <= 0.045
          and q1["active_share"] >= 0.5 * q0["active_share"]
          and q1["optimizer_failure_rate"] <= q0["optimizer_failure_rate"] + 0.10
          and abs(q1["vol_z_exposure"] - q0["vol_z_exposure"]) <= 0.10)

    summary = {
        "section": "S13.31",
        "purpose": ("63d-horizon quality tilt inside top-vol tercile "
                    "(defence arm, no auto-flip)"),
        "harvest": str(HARVEST),
        "lam": LAM,
        "tilt_diagnostics": diag,
        "production_reference": {"information_ratio": prod_ir, "active_return": prod_ar},
        "arms": results,
        "gates": {
            "G0_roundtrip_pass": bool(g0),
            "G0_ir_abs_diff": abs(q0["information_ratio"] - prod_ir),
            "D1a_quality_exposure_up": bool(d1a),
            "D1b_tail_improved": bool(d1b),
            "D1b_detail": {k: bool(v) for k, v in tail_improvements.items()},
            "D2_delta_ir": d2_dir,
            "E2_character_pass": bool(e2),
        },
        "total_runtime_s": round(time.time() - t_start, 1),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if g0 else 1


if __name__ == "__main__":
    sys.exit(main())
