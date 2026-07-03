#!/usr/bin/env python
"""Export operating data for the dashboard — production backtest -> JSON/CSV.

Runs the production variant ONCE and extracts everything an operator needs that
the scalar metrics.json does not carry:

  returns.csv      daily portfolio vs benchmark returns + cumulative
  performance.json annual/active return, vol, sharpe, IR, TE, beta, max DD,
                   per-year returns, sub-period IR
  holdings.json    latest-rebalance weights vs benchmark -> per-name active,
                   sector, top OW / top UW, concentration
  contribution.json arithmetic stock/sector contribution to portfolio and
                   active return, plus transaction-cost residual
  risk.json        latest absolute/active risk contribution by stock and sector
  monitoring.json  turnover, active-share, rolling IR/TE/beta, monthly active,
                   drawdown events and guardrail diagnostics
  features.json    gain-based feature importance averaged across the walk-forward
                   models (which features actually drove the book) + group rollup
  operations.json  next-rebalance target weights, trade list (target - prev),
                   sector exposure (port vs bm), turnover, fallback rate

Run FROM the project root (ai_port), engine vendored under ./src:
    PYTHONPATH=. <PY> scripts/export_operating_data.py
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
from typing import Optional

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

OUT = ROOT / "outputs" / "operating"


def _safe_float(x, ndigits: Optional[int] = 6):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return round(v, ndigits) if ndigits is not None else v


def _drift_weights(weights: np.ndarray, daily_ret: np.ndarray) -> np.ndarray:
    vals = np.asarray(weights, dtype=float) * (1.0 + np.asarray(daily_ret, dtype=float))
    total = float(np.nansum(vals))
    if not np.isfinite(total) or total <= 0:
        return np.asarray(weights, dtype=float)
    return vals / total


def _drawdown_events(cum: pd.Series, dd: pd.Series, max_events: int = 5) -> list[dict]:
    """Contiguous underwater periods, sorted by worst trough."""
    if cum.empty or dd.empty:
        return []
    events = []
    idx = list(dd.index)
    i = 0
    while i < len(idx):
        if dd.iloc[i] >= 0:
            i += 1
            continue
        start_i = i
        while i < len(idx) and dd.iloc[i] < 0:
            i += 1
        end_i = i
        seg = dd.iloc[start_i:end_i]
        trough = seg.idxmin()
        peak_hist = cum.loc[:idx[start_i]]
        peak = peak_hist.idxmax() if not peak_hist.empty else idx[start_i]
        recovery = idx[end_i] if end_i < len(idx) else None
        events.append({
            "peak": str(peak)[:10],
            "start": str(idx[start_i])[:10],
            "trough": str(trough)[:10],
            "recovery": str(recovery)[:10] if recovery is not None else None,
            "max_drawdown": _safe_float(seg.min(), 6),
            "days_underwater": int(end_i - start_i),
        })
    events.sort(key=lambda r: r["max_drawdown"] if r["max_drawdown"] is not None else 0)
    return events[:max_events]


def _serialize_records(df: pd.DataFrame, date_col: str = "date") -> list[dict]:
    if df is None or df.empty:
        return []
    out = df.copy()
    if out.index.name or isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index().rename(columns={out.index.name or "index": date_col})
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d")
    rows = []
    for rec in out.to_dict(orient="records"):
        rows.append({
            k: (_safe_float(v, 6) if isinstance(v, (int, float, np.integer, np.floating)) else v)
            for k, v in rec.items()
        })
    return rows


def main() -> int:
    import yaml
    from src.harness import build_override_config, inject_config, sub_period_irs
    from src.backtest import run_backtest, get_benchmark_fn, get_sector_map
    from src.data_loader import UniverseData
    from src.attribution import compute_feature_importance
    from src.portfolio_optimizer import estimate_covariance

    variant = ROOT / "variants" / "iter15_65tkr_reb21_vtg.yaml"
    overrides = (yaml.safe_load(open(variant, encoding="utf-8")) or {}).get("overrides", {})
    cfg = build_override_config(dict(overrides))
    inject_config(cfg)

    # Prefer the cached full result pkl (run_variant saves it) — loading it is
    # ~1GB vs a fresh walk-forward backtest's ~3-4GB peak, so the export still
    # runs under a tight commit limit. Falls back to a fresh backtest if absent.
    t0 = time.time()
    pkl = ROOT / "outputs" / "iter15_65tkr_reb21_vtg" / "backtest_result.pkl"
    data = UniverseData(cfg.data_path, config=cfg)   # needed for sector map + benchmark weights
    if pkl.exists():
        import pickle
        print(f"[export] loading cached backtest_result.pkl ({pkl.stat().st_size//1_000_000}MB)…")
        with open(pkl, "rb") as fh:
            res = pickle.load(fh)
        print(f"[export] loaded in {time.time()-t0:.0f}s")
    else:
        print("[export] no cached result — running production backtest (single-thread BLAS)…")
        res = run_backtest(data, config=cfg)
        print(f"[export] backtest done in {time.time()-t0:.0f}s")

    OUT.mkdir(parents=True, exist_ok=True)
    tickers = list(data.tickers)
    sector_map = get_sector_map(data)
    bm_fn = get_benchmark_fn(data, tickers, config=cfg)

    # ---- returns.csv + performance.json -----------------------------------
    port = res.portfolio_returns.dropna()
    bm = res.benchmark_returns.reindex(port.index).ffill().fillna(0.0)
    cum_p = (1 + port).cumprod()
    cum_b = (1 + bm).cumprod()
    dd = cum_p / cum_p.cummax() - 1.0
    ret_df = pd.DataFrame({
        "portfolio_ret": port, "benchmark_ret": bm,
        "portfolio_cum": cum_p, "benchmark_cum": cum_b, "drawdown": dd,
    })
    ret_df.index.name = "date"
    ret_df.to_csv(OUT / "returns.csv")

    m = res.compute_metrics()
    by_year = {}
    for yr, g in port.groupby(port.index.year):
        b_g = bm.reindex(g.index)
        p_r = float((1 + g).prod() - 1); b_r = float((1 + b_g).prod() - 1)
        by_year[int(yr)] = {"portfolio": round(p_r, 4), "benchmark": round(b_r, 4),
                            "active": round(p_r - b_r, 4)}
    perf = {
        "as_of": str(port.index.max())[:10],
        "annual_return": m.get("annual_return"), "annual_vol": m.get("annual_vol"),
        "sharpe_ratio": m.get("sharpe_ratio"), "active_return": m.get("active_return"),
        "tracking_error": m.get("tracking_error"), "information_ratio": m.get("information_ratio"),
        "realized_beta": m.get("realized_beta"), "avg_annual_turnover": m.get("avg_annual_turnover"),
        "avg_ic": m.get("avg_ic"),
        "total_return": float(cum_p.iloc[-1] - 1.0), "bm_total_return": float(cum_b.iloc[-1] - 1.0),
        "max_drawdown": float(dd.min()),
        "sub_period_ir": sub_period_irs(port, bm),
        "by_year_returns": by_year,
        "optimizer_failure_rate": getattr(res, "optimizer_failure_rate", None),
        "optimizer_solver_counts": getattr(res, "optimizer_solver_counts", {}),
        "optimizer_solver_fallback_rate": getattr(res, "optimizer_solver_fallback_rate", None),
        "model_quality": getattr(res, "model_quality", None),
        "data_quality": getattr(res, "data_quality", None) or getattr(data, "data_quality", None),
    }
    json.dump(perf, open(OUT / "performance.json", "w", encoding="utf-8"), indent=2, default=str)

    # ---- holdings.json (latest rebalance OW/UW) ---------------------------
    reb_dates = sorted(res.portfolio_weights.keys())
    last = reb_dates[-1]
    w = res.portfolio_weights[last].reindex(tickers).fillna(0.0)
    bm_w = pd.Series(np.asarray(bm_fn(last, tickers, len(tickers)), dtype=float), index=tickers)
    active = w - bm_w
    rows = [{"ticker": t, "weight": round(float(w[t]), 5), "bm_weight": round(float(bm_w[t]), 5),
             "active": round(float(active[t]), 5), "sector": sector_map.get(t, "Unknown")}
            for t in tickers]
    rows.sort(key=lambda r: r["active"], reverse=True)
    active_share_l1 = float(active.abs().sum())
    abs_active = active.abs().sort_values(ascending=False)
    active_budget = float(abs_active.sum())
    concentration = {
        "top5_weight": _safe_float(w.sort_values(ascending=False).head(5).sum(), 6),
        "top10_weight": _safe_float(w.sort_values(ascending=False).head(10).sum(), 6),
        "effective_names": _safe_float(1.0 / float((w.pow(2)).sum()), 2)
        if float((w.pow(2)).sum()) > 0 else None,
        "max_name_weight": _safe_float(w.max(), 6),
        "max_name_active_abs": _safe_float(abs_active.iloc[0], 6) if len(abs_active) else None,
        "top5_active_budget_share": _safe_float(abs_active.head(5).sum() / active_budget, 6)
        if active_budget > 0 else None,
        "top10_active_budget_share": _safe_float(abs_active.head(10).sum() / active_budget, 6)
        if active_budget > 0 else None,
    }
    holdings = {
        "as_of": str(last)[:10], "n_holdings": int((w > 1e-6).sum()),
        "active_share_l1": round(active_share_l1, 4),
        "active_share_one_way": round(active_share_l1 / 2, 4),
        "concentration": concentration,
        "top_ow": rows[:12], "top_uw": rows[-12:][::-1], "all": rows,
    }
    json.dump(holdings, open(OUT / "holdings.json", "w", encoding="utf-8"), indent=2, default=str)

    # ---- contribution.json (name/sector contribution) ---------------------
    # Entering-day weights are previous day's recorded end-of-day weights.
    # This reconciles to gross portfolio return before transaction costs; the
    # cost/residual line keeps the displayed active contribution auditable.
    dates = port.index
    stock_rets = (
        data.returns.reindex(dates)[tickers]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(float)
    )
    w_daily = pd.DataFrame(res.daily_weights).T.sort_index().reindex(dates).ffill()
    init_bm = pd.Series(np.asarray(bm_fn(dates[0], tickers, len(tickers)), dtype=float), index=tickers)
    w_enter = w_daily.shift(1)
    if not w_enter.empty:
        w_enter.iloc[0] = init_bm
    w_enter = w_enter.fillna(0.0)[tickers]

    rebal_set = set(pd.Timestamp(d) for d in reb_dates)
    bm_enter_rows = []
    bm_curr = init_bm.values.copy()
    for d in dates:
        bm_enter_rows.append(pd.Series(bm_curr.copy(), index=tickers, name=d))
        bm_curr = _drift_weights(bm_curr, stock_rets.loc[d].values)
        if pd.Timestamp(d) in rebal_set:
            bm_curr = np.asarray(bm_fn(d, tickers, len(tickers)), dtype=float)
    bm_enter = pd.DataFrame(bm_enter_rows).reindex(dates)[tickers]

    port_name_contrib = (w_enter * stock_rets).sum(axis=0)
    bm_name_contrib = (bm_enter * stock_rets).sum(axis=0)
    active_name_contrib = port_name_contrib - bm_name_contrib
    gross_port = (w_enter * stock_rets).sum(axis=1)
    gross_bm = (bm_enter * stock_rets).sum(axis=1)
    contribution_residual = {
        "portfolio_gross_arithmetic": _safe_float(gross_port.sum(), 6),
        "portfolio_reported_arithmetic": _safe_float(port.sum(), 6),
        "transaction_cost_and_timing_residual": _safe_float(gross_port.sum() - port.sum(), 6),
        "benchmark_reconstructed_arithmetic": _safe_float(gross_bm.sum(), 6),
        "benchmark_reported_arithmetic": _safe_float(bm.sum(), 6),
        "active_reconstructed_arithmetic": _safe_float((gross_port - gross_bm).sum(), 6),
        "active_reported_arithmetic": _safe_float((port - bm).sum(), 6),
    }
    contrib_rows = []
    latest_active = w - bm_w
    for t in tickers:
        contrib_rows.append({
            "ticker": t,
            "sector": sector_map.get(t, "Unknown"),
            "portfolio_contribution": _safe_float(port_name_contrib.get(t), 6),
            "benchmark_contribution": _safe_float(bm_name_contrib.get(t), 6),
            "active_contribution": _safe_float(active_name_contrib.get(t), 6),
            "avg_weight": _safe_float(w_enter[t].mean(), 6),
            "avg_bm_weight": _safe_float(bm_enter[t].mean(), 6),
            "avg_active_weight": _safe_float((w_enter[t] - bm_enter[t]).mean(), 6),
            "latest_weight": _safe_float(w.get(t), 6),
            "latest_bm_weight": _safe_float(bm_w.get(t), 6),
            "latest_active": _safe_float(latest_active.get(t), 6),
        })
    contrib_rows.sort(key=lambda r: r["active_contribution"] or 0, reverse=True)
    contrib_df = pd.DataFrame(contrib_rows)
    sector_contrib = (
        contrib_df.groupby("sector")[["portfolio_contribution", "benchmark_contribution", "active_contribution"]]
        .sum()
        .sort_values("active_contribution", ascending=False)
    )
    contribution = {
        "period": {"start": str(dates.min())[:10], "end": str(dates.max())[:10]},
        "method": "Arithmetic contribution using entering-day weights; excludes compounding. Residual captures transaction costs/timing.",
        "residual": contribution_residual,
        "top_active_contributors": contrib_rows[:15],
        "bottom_active_contributors": sorted(contrib_rows, key=lambda r: r["active_contribution"] or 0)[:15],
        "by_ticker": contrib_rows,
        "by_sector": _serialize_records(sector_contrib, date_col="sector"),
    }
    json.dump(contribution, open(OUT / "contribution.json", "w", encoding="utf-8"), indent=2, default=str)

    # ---- risk.json (latest total/active risk contribution) ----------------
    risk = {"as_of": str(last)[:10], "method": "Latest rebalance, Ledoit-Wolf covariance, annualized."}
    risk_guardrails = {}
    try:
        cov_lookback = int(getattr(cfg, "cov_lookback", 126))
        hist_returns = data.returns[tickers].loc[data.returns.index < last].tail(cov_lookback)
        cov = np.asarray(estimate_covariance(hist_returns, bm_weights=bm_w.values, config=cfg), dtype=float)
        wv = w.values.astype(float)
        av = (w - bm_w).values.astype(float)
        port_var = float(wv @ cov @ wv)
        active_var = float(av @ cov @ av)
        port_vol = float(np.sqrt(max(port_var, 0.0)) * np.sqrt(252.0))
        active_te = float(np.sqrt(max(active_var, 0.0)) * np.sqrt(252.0))
        total_marginal = cov @ wv
        active_marginal = cov @ av
        total_component = wv * total_marginal
        active_component = av * active_marginal
        risk_rows = []
        for i, t in enumerate(tickers):
            risk_rows.append({
                "ticker": t,
                "sector": sector_map.get(t, "Unknown"),
                "weight": _safe_float(wv[i], 6),
                "bm_weight": _safe_float(bm_w.iloc[i], 6),
                "active_weight": _safe_float(av[i], 6),
                "total_risk_pct": _safe_float(total_component[i] / port_var, 6)
                if port_var > 0 else None,
                "total_vol_contribution": _safe_float(total_component[i] / np.sqrt(port_var) * np.sqrt(252.0), 6)
                if port_var > 0 else None,
                "active_risk_pct": _safe_float(active_component[i] / active_var, 6)
                if active_var > 0 else None,
                "active_te_contribution": _safe_float(active_component[i] / np.sqrt(active_var) * np.sqrt(252.0), 6)
                if active_var > 0 else None,
            })
        risk_rows.sort(key=lambda r: abs(r["active_te_contribution"] or 0), reverse=True)
        risk_df = pd.DataFrame(risk_rows)
        sector_risk = (
            risk_df.groupby("sector")[["total_vol_contribution", "active_te_contribution"]]
            .sum()
            .sort_values("active_te_contribution", key=lambda s: s.abs(), ascending=False)
        )
        top_name = risk_rows[0] if risk_rows else {}
        top_name_active_risk_share = None
        top_sector_active_risk_share = None
        top_sector_name = None
        if active_te > 0 and top_name.get("active_te_contribution") is not None:
            top_name_active_risk_share = abs(float(top_name["active_te_contribution"])) / active_te
        if active_te > 0 and not sector_risk.empty:
            top_sector_name = str(sector_risk.index[0])
            top_sector_active_risk_share = (
                abs(float(sector_risk.iloc[0]["active_te_contribution"])) / active_te
            )
        max_name_risk = float(getattr(cfg, "max_name_active_risk_share", 0.35))
        max_sector_risk = float(getattr(cfg, "max_sector_active_risk_share", 0.75))
        risk_guardrails = {
            "max_name_active_risk_share": _safe_float(max_name_risk, 6),
            "top_name": top_name.get("ticker"),
            "top_name_active_risk_share": _safe_float(top_name_active_risk_share, 6),
            "top_name_active_risk_breached": (
                top_name_active_risk_share is not None
                and top_name_active_risk_share > max_name_risk
            ),
            "max_sector_active_risk_share": _safe_float(max_sector_risk, 6),
            "top_sector": top_sector_name,
            "top_sector_active_risk_share": _safe_float(top_sector_active_risk_share, 6),
            "top_sector_active_risk_breached": (
                top_sector_active_risk_share is not None
                and top_sector_active_risk_share > max_sector_risk
            ),
        }
        risk.update({
            "cov_lookback_days": cov_lookback,
            "estimated_portfolio_vol": _safe_float(port_vol, 6),
            "estimated_tracking_error": _safe_float(active_te, 6),
            "guardrails": risk_guardrails,
            "by_ticker": risk_rows,
            "by_sector": _serialize_records(sector_risk, date_col="sector"),
        })
    except Exception as exc:
        risk["error"] = str(exc)
    json.dump(risk, open(OUT / "risk.json", "w", encoding="utf-8"), indent=2, default=str)

    # ---- monitoring.json (trend/guardrail diagnostics) --------------------
    turnover = res.turnover.sort_index()
    turnover_df = pd.DataFrame({
        "turnover_two_way": turnover,
        "turnover_one_way": turnover * 0.5,
        "rolling6_two_way": turnover.rolling(6, min_periods=1).mean(),
        "rolling6_one_way": turnover.rolling(6, min_periods=1).mean() * 0.5,
    })
    active_share_rows = []
    for d in reb_dates:
        wd = res.portfolio_weights[d].reindex(tickers).fillna(0.0)
        bmd = pd.Series(np.asarray(bm_fn(d, tickers, len(tickers)), dtype=float), index=tickers)
        ad = wd - bmd
        budget = float(ad.abs().sum())
        active_share_rows.append({
            "date": str(d)[:10],
            "active_share_one_way": _safe_float(0.5 * budget, 6),
            "max_name_active_abs": _safe_float(ad.abs().max(), 6),
            "top5_active_budget_share": _safe_float(ad.abs().sort_values(ascending=False).head(5).sum() / budget, 6)
            if budget > 0 else None,
            "effective_names": _safe_float(1.0 / float(wd.pow(2).sum()), 2)
            if float(wd.pow(2).sum()) > 0 else None,
        })

    active_ret = port - bm
    rolling_df = pd.DataFrame(index=port.index)
    for window in (63, 126, 252):
        ar = active_ret.rolling(window).mean() * 252.0
        te = active_ret.rolling(window).std() * np.sqrt(252.0)
        beta = port.rolling(window).cov(bm) / bm.rolling(window).var()
        rolling_df[f"active_return_{window}d_ann"] = ar
        rolling_df[f"tracking_error_{window}d_ann"] = te
        rolling_df[f"information_ratio_{window}d"] = ar / te.replace(0.0, np.nan)
        rolling_df[f"beta_{window}d"] = beta
    rolling_df = rolling_df.dropna(how="all")

    monthly = (1.0 + ret_df[["portfolio_ret", "benchmark_ret"]]).resample("ME").prod() - 1.0
    monthly["active"] = monthly["portfolio_ret"] - monthly["benchmark_ret"]
    monthly.index = monthly.index.to_period("M").astype(str)
    monthly.index.name = "month"

    model_quality = getattr(res, "model_quality", None) or {}
    data_quality = getattr(res, "data_quality", None) or getattr(data, "data_quality", None) or {}
    tail_ffill_days = data_quality.get("tail_ffill_days")
    max_tail_ffill_days = data_quality.get(
        "max_tail_ffill_days", getattr(cfg, "max_tail_ffill_days", None)
    )
    monitoring = {
        "period": {"start": str(port.index.min())[:10], "end": str(port.index.max())[:10]},
        "turnover": _serialize_records(turnover_df, date_col="date"),
        "active_share": active_share_rows,
        "rolling": _serialize_records(rolling_df, date_col="date"),
        "monthly_returns": _serialize_records(monthly, date_col="month"),
        "drawdown_events": _drawdown_events(cum_p, dd),
        "guardrails": {
            "optimizer_failure_rate": _safe_float(getattr(res, "optimizer_failure_rate", None), 6),
            "optimizer_solver_fallback_rate": _safe_float(
                getattr(res, "optimizer_solver_fallback_rate", None), 6
            ),
            "latest_active_share_one_way": holdings["active_share_one_way"],
            "latest_turnover_two_way": _safe_float(turnover.iloc[-1], 6) if len(turnover) else None,
            "latest_estimated_te": risk.get("estimated_tracking_error"),
            "top5_active_budget_share": concentration["top5_active_budget_share"],
            "top_name_active_risk_share": risk_guardrails.get("top_name_active_risk_share"),
            "top_name_active_risk_breached": risk_guardrails.get("top_name_active_risk_breached"),
            "top_sector_active_risk_share": risk_guardrails.get("top_sector_active_risk_share"),
            "top_sector_active_risk_breached": risk_guardrails.get("top_sector_active_risk_breached"),
            "model_degenerate_rate": _safe_float(model_quality.get("degenerate_rate"), 6),
            "model_degenerate_rate_breached": (
                model_quality.get("degenerate_rate") is not None
                and model_quality.get("max_degenerate_model_rate") is not None
                and model_quality.get("degenerate_rate") > model_quality.get("max_degenerate_model_rate")
            ),
            "tail_ffill_days": tail_ffill_days,
            "max_tail_ffill_days": max_tail_ffill_days,
            "tail_ffill_breached": (
                tail_ffill_days is not None
                and max_tail_ffill_days is not None
                and tail_ffill_days > max_tail_ffill_days
            ),
        },
    }
    json.dump(monitoring, open(OUT / "monitoring.json", "w", encoding="utf-8"), indent=2, default=str)

    # ---- features.json (gain importance across walk-forward models) --------
    feat_to_group = {}
    for grp, feats in (res.feature_groups or {}).items():
        for fn in feats:
            feat_to_group[fn] = grp
    imps = []
    for d, model in res.models.items():
        feats = getattr(model, "_active_features", None) or res.feature_names
        try:
            imps.append(compute_feature_importance(model, feats))
        except Exception:
            continue
    features = {"n_models": len(imps), "top_features": [], "group_importance": {}}
    if imps:
        avg = pd.concat(imps, axis=1).fillna(0.0).mean(axis=1).sort_values(ascending=False)
        total = float(avg.sum()) or 1.0
        features["top_features"] = [
            {"feature": f, "importance": round(float(v), 1),
             "share_pct": round(float(v) / total * 100, 2),
             "group": feat_to_group.get(f, "?")}
            for f, v in avg.head(30).items()]
        grp = avg.groupby(avg.index.map(lambda f: feat_to_group.get(f, "?"))).sum().sort_values(ascending=False)
        features["group_importance"] = {g: round(float(v) / total * 100, 2) for g, v in grp.items()}
    json.dump(features, open(OUT / "features.json", "w", encoding="utf-8"), indent=2, default=str)

    # ---- operations.json (next target + trade list + sector exposure) -----
    prev = reb_dates[-2] if len(reb_dates) >= 2 else last
    w_prev = res.portfolio_weights[prev].reindex(tickers).fillna(0.0)
    trades = [{"ticker": t, "prev": round(float(w_prev[t]), 5), "target": round(float(w[t]), 5),
               "delta": round(float(w[t] - w_prev[t]), 5)} for t in tickers]
    trades = [tr for tr in trades if abs(tr["delta"]) > 1e-5]
    trades.sort(key=lambda r: abs(r["delta"]), reverse=True)
    sec_port, sec_bm = {}, {}
    for t in tickers:
        s = sector_map.get(t, "Unknown")
        sec_port[s] = sec_port.get(s, 0.0) + float(w[t])
        sec_bm[s] = sec_bm.get(s, 0.0) + float(bm_w[t])
    sector_exposure = {s: {"portfolio": round(sec_port[s], 4), "benchmark": round(sec_bm.get(s, 0.0), 4),
                           "active": round(sec_port[s] - sec_bm.get(s, 0.0), 4)}
                       for s in sorted(sec_port, key=lambda x: -sec_port[x])}
    ops = {
        "as_of": str(last)[:10], "prev_rebalance": str(prev)[:10],
        "rebalance_freq_days": getattr(cfg, "rebalance_freq", None),
        "n_trades": len(trades), "trade_list": trades,
        "sector_exposure": sector_exposure,
        "turnover_two_way_latest": round(float(w.sub(w_prev).abs().sum()), 4),
        "optimizer_failure_rate": getattr(res, "optimizer_failure_rate", None),
        "optimizer_failures": getattr(res, "optimizer_failures", None),
        "solver_protocol": (
            "ECOS-only"
            if not getattr(cfg, "allow_scs_on_ecos_exception", False)
            else "ECOS with SCS on ECOS exception"
        ),
        "optimizer_solver_counts": getattr(res, "optimizer_solver_counts", {}),
        "optimizer_solver_fallbacks": getattr(res, "optimizer_solver_fallbacks", None),
        "optimizer_solver_fallback_rate": getattr(res, "optimizer_solver_fallback_rate", None),
    }
    json.dump(ops, open(OUT / "operations.json", "w", encoding="utf-8"), indent=2, default=str)

    print(f"\n=== export summary ===")
    print(f"  perf: annual {perf['annual_return']*100:.1f}% | active {perf['active_return']*100:.1f}% | "
          f"IR {perf['information_ratio']:.3f} | maxDD {perf['max_drawdown']*100:.1f}%")
    print(f"  holdings: {holdings['n_holdings']} names, active_share L1 {holdings['active_share_l1']:.3f} "
          f"(one-way {holdings['active_share_one_way']:.3f}), as_of {holdings['as_of']}")
    print(f"  top OW: " + ", ".join(f"{r['ticker']}{r['active']*100:+.1f}%" for r in holdings['top_ow'][:5]))
    print(f"  top UW: " + ", ".join(f"{r['ticker']}{r['active']*100:+.1f}%" for r in holdings['top_uw'][:5]))
    print(f"  contribution: active arithmetic {contribution_residual['active_reconstructed_arithmetic']:+.3f} "
          f"| residual {contribution_residual['transaction_cost_and_timing_residual']:+.3f}")
    if "error" in risk:
        print(f"  risk: skipped ({risk['error']})")
    else:
        print(f"  risk: est vol {risk['estimated_portfolio_vol']*100:.1f}% | "
              f"est TE {risk['estimated_tracking_error']*100:.2f}%")
    print(f"  monitoring: turnover points {len(turnover)} | rolling rows {len(rolling_df)}")
    print(f"  features({features['n_models']} models) top: " +
          ", ".join(f"{f['feature']}" for f in features['top_features'][:5]))
    print(f"  ops: {ops['n_trades']} trades at next rebalance, wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
