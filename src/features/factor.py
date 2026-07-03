"""Category 5: Factor / Macro Conditioning (~90 features)."""

import pandas as pd
import numpy as np
from typing import Dict

from src.data_loader import UniverseData, TICKERS, FACTOR_CATEGORIES


def build_factor_features(data: UniverseData) -> Dict[str, pd.DataFrame]:
    features: Dict[str, pd.DataFrame] = {}

    if not data.has_factor_data():
        return features

    factor_ret = data.factor_returns
    factor_px = data.factor_prices
    # Broadcast to the actually-loaded universe (data.tickers), not the
    # canonical TICKERS constant. This keeps the shape of factor features
    # consistent with accounting / price / sellside blocks.
    tickers = list(data.tickers)
    n = len(tickers)
    common_dates = data.dates.intersection(factor_ret.index)

    def bcast(series: pd.Series) -> pd.DataFrame:
        vals = series.reindex(common_dates).values.reshape(-1, 1)
        return pd.DataFrame(np.tile(vals, (1, n)), index=common_dates, columns=tickers)

    # ===== 1. Factor ETF Momentum/Vol/Accel (7 x 5 = 35) =====
    etf_factors = FACTOR_CATEGORIES.get("Factor_ETF", [])
    for f in etf_factors:
        if f not in factor_ret.columns:
            continue
        for w in [21, 63]:
            features[f"fac_{f}_mom_{w}d"] = bcast(factor_ret[f].rolling(w, min_periods=w).sum())
            features[f"fac_{f}_vol_{w}d"] = bcast(factor_ret[f].rolling(w, min_periods=w).std() * np.sqrt(252))
        m21 = factor_ret[f].rolling(21, min_periods=21).sum()
        m63 = factor_ret[f].rolling(63, min_periods=63).sum()
        features[f"fac_{f}_accel"] = bcast(m21 - m63)

    # ===== 2. Factor spreads (7) =====
    def _spread(a, b, w):
        if a in factor_ret.columns and b in factor_ret.columns:
            return factor_ret[a].rolling(w).sum() - factor_ret[b].rolling(w).sum()
        return None

    for w in [21, 63]:
        s = _spread("F_Value", "F_Growth", w)
        if s is not None:
            features[f"fac_value_growth_{w}d"] = bcast(s)

    for name, a, b in [
        ("risk_on_off", "F_HiBeta", "F_MinVol"),
        ("qual_growth", "F_Quality", "F_Growth"),
        ("hidiv_growth", "F_HiDiv", "F_Growth"),
    ]:
        s = _spread(a, b, 21)
        if s is not None:
            features[f"fac_{name}_21d"] = bcast(s)

    # ===== 3. Market index features (~8) =====
    for idx in ["SPX", "NDX"]:
        if idx not in factor_ret.columns:
            continue
        for w in [21, 63]:
            features[f"fac_{idx}_mom_{w}d"] = bcast(factor_ret[idx].rolling(w, min_periods=w).sum())

    if "MXEF" in factor_ret.columns and "MXWD" in factor_ret.columns:
        s = factor_ret["MXEF"].rolling(21).sum() - factor_ret["MXWD"].rolling(21).sum()
        features["fac_em_dev_spread"] = bcast(s)
    if "SPX" in factor_ret.columns and "SX5E" in factor_ret.columns:
        s = factor_ret["SPX"].rolling(21).sum() - factor_ret["SX5E"].rolling(21).sum()
        features["fac_us_eu_spread"] = bcast(s)

    # ===== 4. VIX / SKEW (~8) =====
    if factor_px is not None and "VIX" in factor_px.columns:
        vix = factor_px["VIX"]
        features["fac_VIX_level"] = bcast(vix)
        features["fac_VIX_chg_5d"] = bcast(vix - vix.shift(5))
        rm = vix.rolling(63).mean()
        rs = vix.rolling(63).std().replace(0, np.nan)
        features["fac_VIX_zscore"] = bcast((vix - rm) / rs)
        # Binary VIX regime
        features["fac_VIX_elevated"] = bcast((vix > 20).astype(float))
        features["fac_VIX_panic"] = bcast((vix > 30).astype(float))
        # VRP proxy: VIX vs realized vol of SPX
        if "SPX" in factor_ret.columns:
            rv = factor_ret["SPX"].rolling(21).std() * np.sqrt(252) * 100
            features["fac_vrp_proxy"] = bcast(vix - rv)

    if factor_px is not None and "SKEW" in factor_px.columns:
        features["fac_SKEW_level"] = bcast(factor_px["SKEW"])
        features["fac_SKEW_chg"] = bcast(factor_px["SKEW"] - factor_px["SKEW"].shift(21))

    # ===== 5. Rates (~8) =====
    if factor_px is not None:
        if "UST_10Y" in factor_px.columns and "UST_2Y" in factor_px.columns:
            slope = factor_px["UST_10Y"] - factor_px["UST_2Y"]
            features["fac_yield_slope"] = bcast(slope)
            features["fac_yield_slope_chg"] = bcast(slope - slope.shift(21))
            # Curve regime
            features["fac_curve_inverted"] = bcast((slope < 0).astype(float))
            features["fac_curve_steep"] = bcast((slope > 1.0).astype(float))

        if "UST_10Y" in factor_px.columns and "US_BEI10" in factor_px.columns:
            real = factor_px["UST_10Y"] - factor_px["US_BEI10"]
            features["fac_real_rate"] = bcast(real)
            features["fac_real_rate_chg"] = bcast(real - real.shift(21))

        if "UST_10Y" in factor_px.columns:
            r10 = factor_px["UST_10Y"]
            features["fac_rate_vol"] = bcast(r10.diff().rolling(21).std())

        if "UST_10Y" in factor_px.columns and "GER_10Y" in factor_px.columns:
            features["fac_us_ger_spread"] = bcast(factor_px["UST_10Y"] - factor_px["GER_10Y"])

    # ===== 6. FX (~6) =====
    if factor_px is not None and "DXY" in factor_px.columns:
        dxy = factor_px["DXY"]
        rm = dxy.rolling(63).mean()
        rs = dxy.rolling(63).std().replace(0, np.nan)
        features["fac_dxy_zscore"] = bcast((dxy - rm) / rs)
        features["fac_dxy_mom_21d"] = bcast(dxy.pct_change(21))
        features["fac_usd_strong"] = bcast(((dxy - rm) / rs > 1).astype(float))

    if factor_px is not None and "USDKRW" in factor_px.columns:
        features["fac_usdkrw_chg"] = bcast(factor_px["USDKRW"].pct_change(21))

    if factor_ret is not None:
        em_cols = [c for c in ["USDKRW", "USDCNH"] if c in factor_ret.columns]
        if em_cols:
            em_avg = factor_ret[em_cols].mean(axis=1)
            features["fac_em_fx_mom"] = bcast(em_avg.rolling(21).sum())

    # ===== 7. Commodities (~6) =====
    for c in ["WTI", "GOLD"]:
        if c in factor_ret.columns:
            for w in [21, 63]:
                features[f"fac_{c}_mom_{w}d"] = bcast(factor_ret[c].rolling(w).sum())
    if "COPPER" in factor_ret.columns and "GOLD" in factor_ret.columns:
        # Copper/Gold ratio = risk appetite proxy
        if factor_px is not None and "COPPER" in factor_px.columns and "GOLD" in factor_px.columns:
            cg = factor_px["COPPER"] / factor_px["GOLD"].replace(0, np.nan)
            features["fac_copper_gold"] = bcast(cg.pct_change(21))
    if "BCOM" in factor_ret.columns:
        features["fac_cmd_mom_21d"] = bcast(factor_ret["BCOM"].rolling(21).sum())

    # ===== 8. GS Thematic (~3) =====
    for t in ["GS_AI", "GS_Nuclear", "GS_SemiHW"]:
        if t in factor_ret.columns:
            features[f"fac_{t}_mom"] = bcast(factor_ret[t].rolling(21).sum())

    # ===== 9. Macro Sentiment (~5) =====
    if factor_px is not None:
        if "CESI_US" in factor_px.columns:
            cesi = factor_px["CESI_US"]
            features["fac_cesi_level"] = bcast(cesi)
            features["fac_cesi_chg"] = bcast(cesi - cesi.shift(21))
        if "AAII_Bull" in factor_px.columns and "AAII_Bear" in factor_px.columns:
            spread = factor_px["AAII_Bull"] - factor_px["AAII_Bear"]
            features["fac_aaii_spread"] = bcast(spread)
            features["fac_aaii_chg"] = bcast(spread - spread.shift(21))

    return features
