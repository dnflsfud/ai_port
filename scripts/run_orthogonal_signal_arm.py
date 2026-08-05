#!/usr/bin/env python
"""§S13.28: characteristic-orthogonal residual-signal IR (diagnostic, no flip).

Harvest-once / re-MVO-many on the SAVED production harvest
(``outputs/codex_causal_rank_65/backtest_result.pkl``). That artefact carries
``pre_overlay_predictions`` — the canonical pre_overlay_ema_predictions object
(CLAUDE.md §4.2) — so injecting it re-applies the production overlays exactly
once. No model retrain, no double overlay.

Arms (pre-registered in the decision log §S13.28 before this ran):
  R0  harvest injected unchanged                  -> round-trip gate vs production
  R1  characteristic-orthogonal residual score    -> "pure alpha" IR
  R2  characteristic-fitted (style) score         -> style-only IR

The 7 characteristics are the ones used by the §S13.28 return-space spanning
regression: SMB / VALUE / MOM / QUALITY / LOWVOL / BAB / GROWTH. Both R1 and R2
are affinely rescaled to the per-date mean/std of the original score, so the
optimizer sees an identical signal SCALE and only the cross-sectional SHAPE
changes.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

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
OUT_DIR = Path("outputs/orthogonal_signal_arm")

# characteristic -> (panel column | "__logcap__", sign making "high = the side
# the classic factor is long")
CHARS = {
    "SMB": ("__logcap__", -1.0),
    "VALUE": ("best_px_bps_ratio_level_z", -1.0),
    "MOM": ("momentum_252d", +1.0),
    "QUALITY": ("best_roe_level_z", +1.0),
    "LOWVOL": ("realized_vol_126d", -1.0),
    "BAB": ("beta_63d", -1.0),
    "GROWTH": ("best_sales_chg_252d", +1.0),
}
MIN_NAMES = 30


def _zscore(x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional winsorised (1/99) z-score, per date."""
    lo, hi = x.quantile(0.01, axis=1), x.quantile(0.99, axis=1)
    xw = x.clip(lo, hi, axis=0)
    sd = xw.std(axis=1, ddof=1)
    return xw.sub(xw.mean(axis=1), axis=0).div(sd.where(sd > 0), axis=0)


def build_char_panels(panel: pd.DataFrame, market_cap: pd.DataFrame,
                      index, columns) -> dict:
    """date x ticker z-score panel per characteristic, signed long-side-high."""
    logcap = np.log(market_cap.where(market_cap > 0))
    out = {}
    for name, (col, sign) in CHARS.items():
        raw = logcap if col == "__logcap__" else panel[col].unstack("ticker")
        out[name] = _zscore((raw * sign).reindex(index=index).reindex(columns=columns))
    return out


def split_signal(preds: pd.DataFrame, chars: dict):
    """Per date, split the score into characteristic-fitted and residual parts.

    Missing characteristic cells count as neutral (z = 0). A date with fewer
    than MIN_NAMES scored names is passed through unchanged in both arms; so is
    an arm whose own cross-sectional dispersion is degenerate (a fully spanned
    score leaves no residual to trade).
    """
    names = list(CHARS)
    resid, fitted = preds.copy(), preds.copy()
    r2s = []
    passthrough = degenerate = 0

    for dt in preds.index:
        s = preds.loc[dt]
        scored = s.index[s.notna()]
        if len(scored) < MIN_NAMES:
            passthrough += 1
            continue
        y = s[scored].to_numpy(dtype=float)
        cols = [chars[n].loc[dt, scored].to_numpy(dtype=float)
                if dt in chars[n].index else np.zeros(len(scored)) for n in names]
        X = np.nan_to_num(np.column_stack([np.ones(len(scored))] + cols),
                          nan=0.0, posinf=0.0, neginf=0.0)
        b = np.linalg.pinv(X.T @ X) @ (X.T @ y)
        fit = X @ b
        e = y - fit
        var_y = float(((y - y.mean()) ** 2).sum())
        r2s.append(1 - float(e @ e) / var_y if var_y > 0 else np.nan)

        mu, sd = float(y.mean()), float(y.std(ddof=1))
        for target, vec in ((resid, e), (fitted, fit)):
            vec_sd = float(np.std(vec, ddof=1))
            if not np.isfinite(vec_sd) or vec_sd <= 1e-12 or sd <= 1e-12:
                degenerate += 1
                continue
            target.loc[dt, scored] = (vec - vec.mean()) * (sd / vec_sd) + mu

    diag = {
        "dates_fitted": int(len(r2s)),
        "dates_passthrough": int(passthrough),
        "degenerate_rescale": int(degenerate),
        "mean_cross_sectional_r2": float(np.nanmean(r2s)) if r2s else None,
        "median_cross_sectional_r2": float(np.nanmedian(r2s)) if r2s else None,
    }
    return resid, fitted, diag


def main() -> int:
    import yaml
    from src.backtest import get_benchmark_fn, run_backtest
    from src.data_loader import UniverseData
    from src.harness import build_override_config, inject_config

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("[S13.28] loading saved production harvest ...", flush=True)
    with HARVEST.open("rb") as fh:
        base = pickle.load(fh)
    prod_m = json.loads(PROD_METRICS.read_text(encoding="utf-8"))
    prod_m = prod_m.get("metrics", prod_m)

    overrides = (yaml.safe_load(
        (ROOT / "variants" / "codex_causal_rank_65.yaml").read_text(encoding="utf-8")
    ) or {}).get("overrides", {})
    cfg = build_override_config(dict(overrides))
    inject_config(cfg)

    print("[S13.28] loading universe data ...", flush=True)
    data = UniverseData(cfg.data_path, config=cfg)

    preds = base.pre_overlay_predictions
    chars = build_char_panels(base.panel, data.market_cap, preds.index, preds.columns)
    print("[S13.28] orthogonalising signal ...", flush=True)
    resid, fitted, diag = split_signal(preds, chars)
    print("[S13.28] split diagnostics: " + json.dumps(diag), flush=True)

    tickers = list(data.tickers)
    bm_fn = get_benchmark_fn(data, tickers, config=cfg)

    def active_share(res) -> float:
        vals = []
        for date, wser in res.portfolio_weights.items():
            w = np.nan_to_num(wser.reindex(tickers).to_numpy(dtype=float))
            bm = np.asarray(bm_fn(date, tickers, len(tickers)), dtype=float)
            vals.append(0.5 * float(np.abs(w - bm).sum()))
        return float(np.mean(vals)) if vals else float("nan")

    results = {}
    for name, panel_preds in (("R0_identity", preds), ("R1_residual", resid),
                              ("R2_style", fitted)):
        print(f"\n[S13.28] === arm {name}: re-MVO on the same harvest ===", flush=True)
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
        results[name] = {
            "information_ratio": float(m["information_ratio"]),
            "active_return": float(m["active_return"]),
            "tracking_error": float(m["tracking_error"]),
            "annual_return": float(m["annual_return"]),
            "realized_beta": float(m.get("realized_beta", float("nan"))),
            "avg_annual_turnover": float(m.get("avg_annual_turnover", float("nan"))),
            "optimizer_failure_rate": float(getattr(res, "optimizer_failure_rate", float("nan"))),
            "active_share": active_share(res),
            "runtime_s": round(time.time() - t0, 1),
        }
        print(f"[S13.28] {name}: " + json.dumps(results[name]), flush=True)
        res.active_returns.to_csv(OUT_DIR / f"{name}_active_returns.csv")

    r0, r1, r2 = results["R0_identity"], results["R1_residual"], results["R2_style"]
    prod_ir, prod_ar = float(prod_m["information_ratio"]), float(prod_m["active_return"])
    g0 = (abs(r0["information_ratio"] - prod_ir) <= 0.005
          and abs(r0["active_return"] - prod_ar) <= 1e-4)

    def g2(a) -> bool:
        return (a["tracking_error"] <= 0.045
                and a["optimizer_failure_rate"] <= r0["optimizer_failure_rate"] + 0.10
                and a["active_share"] >= 0.5 * r0["active_share"])

    summary = {
        "section": "S13.28",
        "purpose": ("diagnostic only (no production flip): IR of the "
                    "characteristic-orthogonal residual signal"),
        "harvest": str(HARVEST),
        "characteristics": {k: v[0] for k, v in CHARS.items()},
        "split_diagnostics": diag,
        "production_reference": {"information_ratio": prod_ir, "active_return": prod_ar},
        "arms": results,
        "gates": {
            "G0_roundtrip_pass": bool(g0),
            "G0_ir_abs_diff": abs(r0["information_ratio"] - prod_ir),
            "G0_active_abs_diff": abs(r0["active_return"] - prod_ar),
            "G2_R1_character_pass": bool(g2(r1)),
            "G2_R2_character_pass": bool(g2(r2)),
        },
        "deltas_vs_R0": {
            "R1_dIR": r1["information_ratio"] - r0["information_ratio"],
            "R2_dIR": r2["information_ratio"] - r0["information_ratio"],
        },
        "total_runtime_s": round(time.time() - t_start, 1),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2), flush=True)
    return 0 if g0 else 1


if __name__ == "__main__":
    sys.exit(main())
