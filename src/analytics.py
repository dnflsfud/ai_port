"""
Analytics / Business Logic extracted from export_csv.py.

Contains:
  - Market regime classification (direction, volatility regime, sector rotation)
  - OW/UW stock explanation generation (natural-language signal interpretation)
  - Sector/style tilt computation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple



def _resolve_bm_weights(
    rebal_date,
    tickers: List[str],
    bm_weights_history: Optional[Dict] = None,
) -> pd.Series:
    """Return cap-weighted BM weights for a rebal date, or fall back to 1/n.

    Parameters
    ----------
    rebal_date : Timestamp
        The rebalancing date.
    tickers : List[str]
        Ticker order to use in the returned Series.
    bm_weights_history : Optional[Dict[Timestamp, pd.Series]]
        Cap-weighted BM weights per rebalancing date. If None or the date
        is missing, falls back to equal-weight (1/n). Renormalizes if some
        tickers are missing from the BM series (e.g. universe expansion).

    Notes
    -----
    Production runs `benchmark_type='cap_weighted'` (REDESIGN A). Analytics
    must agree with the optimizer's BM definition or active-weight tables
    will be misleading. Caller is responsible for building
    `bm_weights_history` via `get_benchmark_fn(data, tickers, config)`
    once and passing it down.
    """
    n_tickers = len(tickers)
    if not n_tickers:
        return pd.Series(dtype=float)
    if bm_weights_history is not None and rebal_date in bm_weights_history:
        bm = bm_weights_history[rebal_date].reindex(tickers).fillna(0.0)
        s = float(bm.sum())
        if s > 0:
            return bm / s
    return pd.Series(1.0 / n_tickers, index=tickers)


# ---------------------------------------------------------------------------
# Market Regime Classification
# ---------------------------------------------------------------------------
def classify_market_direction(ew_ret_21d: float) -> str:
    """21-day equal-weight return -> market direction label."""
    if ew_ret_21d > 0.02:
        return "Bullish"
    elif ew_ret_21d < -0.02:
        return "Bearish"
    else:
        return "Sideways"


def classify_volatility_regime(vol_21d_ann: float) -> str:
    """Annualised 21-day EW volatility -> regime label."""
    if vol_21d_ann > 0.25:
        return "High Volatility"
    elif vol_21d_ann > 0.15:
        return "Normal Volatility"
    else:
        return "Low Volatility"


def compute_regime_stats(
    returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    rebal_date: pd.Timestamp,
) -> Dict:
    """Compute 21-day EW return and annualised volatility for regime classification.

    Returns dict with keys:
        ew_ret_21d, vol_21d_ann, market_direction, volatility_regime
    """
    idx = dates.get_loc(rebal_date)
    lookback_start = max(0, idx - 21)
    recent_ret = returns.iloc[lookback_start:idx + 1]
    ew_ret = recent_ret.mean(axis=1).sum()
    vol_21d = recent_ret.mean(axis=1).std() * np.sqrt(252)

    return {
        "ew_ret_21d": ew_ret,
        "vol_21d_ann": vol_21d,
        "market_direction": classify_market_direction(ew_ret),
        "volatility_regime": classify_volatility_regime(vol_21d),
    }


def classify_sector_rotation(
    recent_ret: pd.DataFrame,
    tickers: List[str],
) -> str:
    """Classify sector rotation as Asset-Heavy, Asset-Light, or Neutral."""
    # M5: bucket lists live in src.metadata — avoid hardcoding ticker lists
    # in two places. Imported lazily so metadata.py changes take effect
    # without reloading analytics.py in long-running sessions.
    from src.metadata import get_asset_rotation_buckets
    all_light, all_heavy = get_asset_rotation_buckets()
    asset_light = [c for c in all_light if c in tickers]
    asset_heavy = [c for c in all_heavy if c in tickers]

    al_ret = recent_ret[asset_light].mean(axis=1).sum() if asset_light else 0
    ah_ret = recent_ret[asset_heavy].mean(axis=1).sum() if asset_heavy else 0

    if ah_ret > al_ret + 0.01:
        return "Asset-Heavy"
    elif al_ret > ah_ret + 0.01:
        return "Asset-Light"
    else:
        return "Neutral"


# ---------------------------------------------------------------------------
# Signal Interpretation (Natural Language)
# ---------------------------------------------------------------------------
def interpret_signal_ow(pred_val: float) -> str:
    """Natural-language interpretation for an overweight stock's prediction z-score."""
    if np.isnan(pred_val):
        return "Signal not available"
    if pred_val > 1.0:
        return f"Very strong positive signal (z={pred_val:.2f})"
    elif pred_val > 0.5:
        return f"Strong positive signal (z={pred_val:.2f})"
    elif pred_val > 0:
        return f"Mild positive signal (z={pred_val:.2f})"
    else:
        return f"Optimizer risk-adjusted OW despite negative signal (z={pred_val:.2f})"


def interpret_signal_uw(pred_val: float) -> str:
    """Natural-language interpretation for an underweight stock's prediction z-score."""
    if np.isnan(pred_val):
        return "Signal not available"
    if pred_val < -0.5:
        return f"Strong negative signal (z={pred_val:.2f})"
    elif pred_val < 0:
        return f"Mild negative signal (z={pred_val:.2f})"
    else:
        return f"Positive signal but BM cap-weight dominates (z={pred_val:.2f})"


def interpret_signal_brief_ow(pred_val: float) -> str:
    """Shorter signal interpretation for regime CSV OW."""
    if np.isnan(pred_val):
        return "Optimizer allocation"
    if pred_val > 0.5:
        return f"Strong positive signal (z={pred_val:.2f})"
    elif pred_val > 0:
        return f"Mild positive signal (z={pred_val:.2f})"
    else:
        return f"Negative signal but optimizer kept OW (z={pred_val:.2f})"


def interpret_signal_brief_uw(pred_val: float) -> str:
    """Shorter signal interpretation for regime CSV UW."""
    if np.isnan(pred_val):
        return "Optimizer allocation"
    if pred_val < -0.5:
        return f"Strong negative signal (z={pred_val:.2f})"
    elif pred_val < 0:
        return f"Mild negative signal (z={pred_val:.2f})"
    else:
        return f"Positive signal but optimizer UW (z={pred_val:.2f})"


# ---------------------------------------------------------------------------
# Monthly Regime Analysis (pure data, no CSV writing)
# ---------------------------------------------------------------------------
def compute_monthly_regime_rows(
    portfolio_weights: Dict,
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: List[str],
    n_months: int = 6,
    bm_weights_history: Optional[Dict] = None,
) -> List[Dict]:
    """Compute monthly regime rows (market direction, volatility, OW/UW stocks).

    Returns a list of dicts ready to be turned into a DataFrame.

    `bm_weights_history`: optional cap-weighted BM weights per rebal date
    (matches `benchmark_type` in config). When None, falls back to 1/n —
    keep this fallback ONLY for legacy callers; production should always
    supply cap-weighted history to keep OW/UW labels consistent with the
    optimizer's active-weight definition.
    """
    rebal_dates = sorted(portfolio_weights.keys(), reverse=True)
    if not rebal_dates:
        return []

    last_date = rebal_dates[0]
    cutoff = last_date - pd.DateOffset(months=n_months)
    recent_rebal = [d for d in rebal_dates if d >= cutoff]

    # Monthly grouping (first rebalancing per month)
    monthly = {}
    for d in sorted(recent_rebal):
        month_key = d.strftime("%Y-%m")
        if month_key not in monthly:
            monthly[month_key] = d

    rows = []
    for month_key, rebal_date in sorted(monthly.items()):
        w = portfolio_weights[rebal_date]
        bm_w = _resolve_bm_weights(rebal_date, tickers, bm_weights_history)
        active_w = (w - bm_w).sort_values(ascending=False)

        # Regime stats
        regime = compute_regime_stats(returns, dates, rebal_date)
        ew_ret = regime["ew_ret_21d"]
        vol_21d = regime["vol_21d_ann"]
        market_dir = regime["market_direction"]
        vol_regime = regime["volatility_regime"]

        # Recent returns for sector rotation
        idx = dates.get_loc(rebal_date)
        lookback_start = max(0, idx - 21)
        recent_ret = returns.iloc[lookback_start:idx + 1]
        rotation = classify_sector_rotation(recent_ret, tickers)

        # Predictions
        pred_row = predictions.loc[rebal_date, tickers] if rebal_date in predictions.index else pd.Series()

        # OW explanations
        ow_stocks = active_w.head(5)
        uw_stocks = active_w.tail(5)

        ow_explanations = []
        for ticker, aw in ow_stocks.items():
            pred_val = pred_row.get(ticker, np.nan) if not pred_row.empty else np.nan
            reason = interpret_signal_brief_ow(pred_val)
            ow_explanations.append(f"{ticker}(+{aw:.1%}, {reason})")

        uw_explanations = []
        for ticker, aw in uw_stocks.items():
            pred_val = pred_row.get(ticker, np.nan) if not pred_row.empty else np.nan
            reason = interpret_signal_brief_uw(pred_val)
            uw_explanations.append(f"{ticker}({aw:.1%}, {reason})")

        rows.append({
            "year_month": month_key,
            "rebal_date": rebal_date.strftime("%Y-%m-%d"),
            "market_direction": market_dir,
            "volatility_regime": vol_regime,
            "sector_rotation": rotation,
            "ew_return_21d": round(ew_ret, 4),
            "vol_21d_ann": round(vol_21d, 4),
            "top_ow_stocks": " | ".join(ow_explanations),
            "top_uw_stocks": " | ".join(uw_explanations),
            "n_ow_stocks": int((active_w > 0.002).sum()),
            "n_uw_stocks": int((active_w < -0.002).sum()),
            "total_active_share": round(active_w.abs().sum() / 2, 4),
        })

    return rows


# ---------------------------------------------------------------------------
# Monthly OW Explanations (detailed, pure data)
# ---------------------------------------------------------------------------
def compute_monthly_ow_explanation_rows(
    portfolio_weights: Dict,
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: List[str],
    group_contributions: Dict,
    n_months: int = 6,
    bm_weights_history: Optional[Dict] = None,
    ticker_meta: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[Dict]:
    """Compute detailed monthly OW explanation rows.

    Returns a list of dicts ready to be turned into a DataFrame.

    `ticker_meta` (§S11.4, 2026-07-21 필수화): Universe_Meta 기반
    build_ticker_meta(data.meta) 결과를 주입한다. None이면 60/150 스테일
    정적 TICKER_META를 무음 사용하게 되므로 ValueError. 레거시 정적 동작이
    필요하면 ticker_meta=TICKER_META를 명시적으로 넘긴다.

    See `compute_monthly_regime_rows` re: `bm_weights_history`.
    """
    if ticker_meta is None:
        raise ValueError(
            "ticker_meta is required — pass build_ticker_meta(data.meta) "
            "(or TICKER_META explicitly for the legacy static behaviour)"
        )
    meta_map = ticker_meta
    ticker_sectors = {t: meta_map.get(t, {}).get("sector", "Other") for t in tickers}
    ticker_styles = {t: meta_map.get(t, {}).get("style", "Other") for t in tickers}
    ticker_subs = {t: meta_map.get(t, {}).get("sub", "N/A") for t in tickers}

    rebal_dates = sorted(portfolio_weights.keys(), reverse=True)
    if not rebal_dates:
        return []

    last_date = rebal_dates[0]
    cutoff = last_date - pd.DateOffset(months=n_months)
    recent_rebal = [d for d in rebal_dates if d >= cutoff]

    # Monthly first rebalancing
    monthly = {}
    for d in sorted(recent_rebal):
        mk = d.strftime("%Y-%m")
        if mk not in monthly:
            monthly[mk] = d

    rows = []
    for month_key, rebal_date in sorted(monthly.items()):
        w = portfolio_weights[rebal_date]
        bm_w = _resolve_bm_weights(rebal_date, tickers, bm_weights_history)
        active_w = (w - bm_w).sort_values(ascending=False)

        # Regime analysis
        regime = compute_regime_stats(returns, dates, rebal_date)
        ew_ret = regime["ew_ret_21d"]
        vol_21d = regime["vol_21d_ann"]
        market_dir = regime["market_direction"]
        vol_regime = regime["volatility_regime"]
        regime_label = f"{market_dir} / {vol_regime}"

        # Regime reasons
        idx = dates.get_loc(rebal_date)
        lookback_start = max(0, idx - 21)
        recent_ret = returns.iloc[lookback_start:idx + 1]

        regime_reasons = []
        regime_reasons.append(f"21d EW Ret={ew_ret:+.2%} -> {market_dir}")
        regime_reasons.append(f"21d Vol(ann)={vol_21d:.1%} -> {vol_regime}")

        # Sector-level 21d returns
        sec_rets = {}
        for sec in set(ticker_sectors.values()):
            sec_t = [t for t in tickers if ticker_sectors[t] == sec]
            if sec_t:
                sec_rets[sec] = recent_ret[sec_t].mean(axis=1).sum()

        if sec_rets:
            top_sec = max(sec_rets, key=sec_rets.get)
            bot_sec = min(sec_rets, key=sec_rets.get)
            regime_reasons.append(
                f"Leading: {top_sec}({sec_rets[top_sec]:+.2%}), "
                f"Lagging: {bot_sec}({sec_rets[bot_sec]:+.2%})"
            )
        regime_reason = " | ".join(regime_reasons)

        # Dominant category effects
        closest_attr_date = None
        if group_contributions:
            attr_dates = sorted(group_contributions.keys())
            diffs = [abs((pd.Timestamp(ad) - rebal_date).days) for ad in attr_dates]
            if diffs:
                min_idx = diffs.index(min(diffs))
                closest_attr_date = attr_dates[min_idx]

        if closest_attr_date and closest_attr_date in group_contributions:
            gc = group_contributions[closest_attr_date]
            gc_sorted = sorted(gc.items(), key=lambda x: x[1], reverse=True)
            dom_effects = " > ".join([f"{g}({v:.1%})" for g, v in gc_sorted])
        else:
            dom_effects = "N/A"

        # Sector/Style tilt summary
        sec_active = {}
        for sec in set(ticker_sectors.values()):
            sec_t = [t for t in tickers if ticker_sectors[t] == sec]
            if sec_t:
                sec_active[sec] = active_w[sec_t].sum()

        sty_active = {}
        for sty in set(ticker_styles.values()):
            sty_t = [t for t in tickers if ticker_styles[t] == sty]
            if sty_t:
                sty_active[sty] = active_w[sty_t].sum()

        sec_sorted = sorted(sec_active.items(), key=lambda x: x[1], reverse=True)
        sty_sorted = sorted(sty_active.items(), key=lambda x: x[1], reverse=True)

        sector_tilt_str = " | ".join([f"{s}({v:+.1%})" for s, v in sec_sorted if abs(v) > 0.005])
        style_tilt_str = " | ".join([f"{s}({v:+.1%})" for s, v in sty_sorted if abs(v) > 0.005])

        # Per-stock OW detail
        pred_row = predictions.loc[rebal_date, tickers] if rebal_date in predictions.index else pd.Series()

        ow_details = []
        for ticker in active_w.index:
            aw = active_w[ticker]
            if aw < 0.002:
                continue
            pred_val = pred_row.get(ticker, np.nan) if not pred_row.empty else np.nan
            sec = ticker_sectors.get(ticker, "N/A")
            sty = ticker_styles.get(ticker, "N/A")
            sub = ticker_subs.get(ticker, "N/A")
            signal_desc = interpret_signal_ow(pred_val)
            ow_details.append(
                f"{ticker}[{sec}/{sty}/{sub}](AW={aw:+.1%}, {signal_desc})"
            )

        uw_details = []
        for ticker in reversed(active_w.index):
            aw = active_w[ticker]
            if aw > -0.002:
                continue
            pred_val = pred_row.get(ticker, np.nan) if not pred_row.empty else np.nan
            sec = ticker_sectors.get(ticker, "N/A")
            sty = ticker_styles.get(ticker, "N/A")
            signal_desc = interpret_signal_uw(pred_val)
            uw_details.append(
                f"{ticker}[{sec}/{sty}](AW={aw:+.1%}, {signal_desc})"
            )

        rows.append({
            "year_month": month_key,
            "rebal_date": rebal_date.strftime("%Y-%m-%d"),
            "regime_label": regime_label,
            "regime_reason": regime_reason,
            "dominant_category_effects": dom_effects,
            "sector_tilt": sector_tilt_str,
            "style_tilt": style_tilt_str,
            "n_ow_stocks": int((active_w > 0.002).sum()),
            "n_uw_stocks": int((active_w < -0.002).sum()),
            "top_ow_details": " | ".join(ow_details[:10]),
            "top_uw_details": " | ".join(uw_details[:10]),
            "all_ow_details": " | ".join(ow_details),
            "all_uw_details": " | ".join(uw_details),
        })

    return rows


# ---------------------------------------------------------------------------
# Sector / Style Tilt computation (pure data)
# ---------------------------------------------------------------------------
def compute_style_sector_tilt_rows(
    portfolio_weights: Dict,
    tickers: List[str],
    bm_weights_history: Optional[Dict] = None,
    ticker_meta: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[Dict]:
    """Compute per-rebalancing sector & style active-weight rows.

    Returns a list of dicts ready to be turned into a DataFrame.

    `ticker_meta` (§S11.4, 2026-07-21 필수화): Universe_Meta 기반
    build_ticker_meta(data.meta) 결과를 주입한다. None이면 60/150 스테일
    정적 TICKER_META를 무음 사용하게 되므로 ValueError. 레거시 정적 동작이
    필요하면 ticker_meta=TICKER_META를 명시적으로 넘긴다.

    See `compute_monthly_regime_rows` re: `bm_weights_history`.
    """
    if ticker_meta is None:
        raise ValueError(
            "ticker_meta is required — pass build_ticker_meta(data.meta) "
            "(or TICKER_META explicitly for the legacy static behaviour)"
        )
    meta_map = ticker_meta
    ticker_sectors = {t: meta_map.get(t, {}).get("sector", "Other") for t in tickers}
    ticker_styles = {t: meta_map.get(t, {}).get("style", "Other") for t in tickers}
    all_sectors = sorted(set(ticker_sectors.values()))
    all_styles = sorted(set(ticker_styles.values()))

    rebal_dates = sorted(portfolio_weights.keys())
    rows = []

    for d in rebal_dates:
        w = portfolio_weights[d]
        bm_w = _resolve_bm_weights(d, tickers, bm_weights_history)
        active_w = w - bm_w

        row = {"date": d.strftime("%Y-%m-%d")}

        for sec in all_sectors:
            sec_tickers = [t for t in tickers if ticker_sectors[t] == sec]
            row[f"sector_{sec}"] = round(active_w[sec_tickers].sum(), 6) if sec_tickers else 0.0
            row[f"port_sector_{sec}"] = round(w[sec_tickers].sum(), 6) if sec_tickers else 0.0
            row[f"bm_sector_{sec}"] = round(bm_w[sec_tickers].sum(), 6) if sec_tickers else 0.0

        for sty in all_styles:
            sty_tickers = [t for t in tickers if ticker_styles[t] == sty]
            row[f"style_{sty}"] = round(active_w[sty_tickers].sum(), 6) if sty_tickers else 0.0
            row[f"port_style_{sty}"] = round(w[sty_tickers].sum(), 6) if sty_tickers else 0.0
            row[f"bm_style_{sty}"] = round(bm_w[sty_tickers].sum(), 6) if sty_tickers else 0.0

        sec_active = {sec: row[f"sector_{sec}"] for sec in all_sectors}
        sty_active = {sty: row[f"style_{sty}"] for sty in all_styles}
        row["dominant_ow_sector"] = max(sec_active, key=sec_active.get)
        row["dominant_uw_sector"] = min(sec_active, key=sec_active.get)
        row["dominant_ow_style"] = max(sty_active, key=sty_active.get)
        row["dominant_uw_style"] = min(sty_active, key=sty_active.get)

        rows.append(row)

    return rows
