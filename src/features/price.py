"""Category 2: Price / Market (~50 features)."""

import pandas as pd
import numpy as np
from typing import Dict

from src.data_loader import UniverseData
from src.features.utils import cs_rank


def build_price_features(data: UniverseData) -> Dict[str, pd.DataFrame]:
    features: Dict[str, pd.DataFrame] = {}
    # §S11.7: PIT 뷰(상장 전 NaN) — 유령의 합성 수익률이 횡단면 순위·시장평균·
    # 베타 계산에 참여하지 않도록 dense returns 대신 masked 뷰를 소비한다.
    returns = data.returns_masked
    prices = data.prices
    mktcap = data.market_cap

    # --- Reversal (3) ---
    for w in [5, 10, 21]:
        features[f"reversal_{w}d"] = -1 * returns.rolling(w, min_periods=w).sum()

    # --- Momentum (3) ---
    for w in [63, 126, 252]:
        features[f"momentum_{w}d"] = returns.rolling(w, min_periods=w).sum()

    # --- Risk-adjusted momentum (3) ---
    for w in [63, 126, 252]:
        mom = returns.rolling(w, min_periods=w).sum()
        vol = returns.rolling(w, min_periods=w).std().replace(0, np.nan)
        features[f"risk_adj_mom_{w}d"] = mom / vol

    # --- Realized volatility (3) ---
    for w in [21, 63, 126]:
        features[f"realized_vol_{w}d"] = returns.rolling(w, min_periods=w).std() * np.sqrt(252)

    # --- Volatility ratio & change (2) ---
    v21 = returns.rolling(21).std()
    v126 = returns.rolling(126).std().replace(0, np.nan)
    features["vol_ratio_21_126"] = v21 / v126
    v21_lag = returns.shift(21).rolling(21).std().replace(0, np.nan)
    features["vol_change_21d"] = v21 / v21_lag - 1

    # --- MA distance (3) ---
    for w in [21, 50, 200]:
        ma = prices.rolling(w, min_periods=w).mean().replace(0, np.nan)
        features[f"price_dist_ma_{w}d"] = (prices / ma) - 1

    # --- MA crossover (2) ---
    ma21 = prices.rolling(21).mean()
    ma50 = prices.rolling(50).mean()
    ma200 = prices.rolling(200).mean()
    features["ma_cross_21_50"] = (ma21 / ma50.replace(0, np.nan)) - 1
    features["ma_cross_50_200"] = (ma50 / ma200.replace(0, np.nan)) - 1

    # --- Drawdown & 52-week range (4) ---
    rmax63 = prices.rolling(63, min_periods=1).max().replace(0, np.nan)
    features["drawdown_63d"] = (prices / rmax63) - 1
    rmax252 = prices.rolling(252, min_periods=126).max().replace(0, np.nan)
    rmin252 = prices.rolling(252, min_periods=126).min().replace(0, np.nan)
    features["dist_52w_high"] = (prices / rmax252) - 1
    features["dist_52w_low"] = (prices / rmin252) - 1
    range252 = (rmax252 - rmin252).replace(0, np.nan)
    features["range_position_52w"] = (prices - rmin252) / range252

    # --- Mktcap rank (1) ---
    features["mktcap_rank"] = mktcap.rank(axis=1, pct=True)

    # --- Relative momentum (stock vs market) (3) ---
    ew_ret = returns.mean(axis=1)
    for w in [21, 63, 126]:
        stock_mom = returns.rolling(w, min_periods=w).sum()
        mkt_mom = ew_ret.rolling(w, min_periods=w).sum()
        features[f"rel_mom_{w}d"] = stock_mom.sub(mkt_mom, axis=0)

    # --- Momentum rank (2) ---
    for w in [21, 63]:
        features[f"mom_rank_{w}d"] = cs_rank(returns.rolling(w, min_periods=w).sum())

    # --- Return distribution (4) ---
    for w in [21, 63]:
        features[f"ret_skew_{w}d"] = returns.rolling(w, min_periods=w).skew()
        features[f"ret_kurt_{w}d"] = returns.rolling(w, min_periods=w).kurt()

    # --- Max/Min return (4) ---
    for w in [21, 63]:
        features[f"max_ret_{w}d"] = returns.rolling(w, min_periods=w).max()
        features[f"min_ret_{w}d"] = returns.rolling(w, min_periods=w).min()

    # --- Positive return ratio (2) ---
    # NaN > 0 == False라서 상장 전 셀이 0.0으로 새지 않도록 notna로 가드.
    for w in [21, 63]:
        pos = (returns > 0).astype(float).where(returns.notna())
        features[f"pos_ret_ratio_{w}d"] = pos.rolling(w, min_periods=w).mean()

    # --- Downside deviation & Up/Down ratio (3) ---
    neg_ret = returns.clip(upper=0)
    for w in [21, 63]:
        features[f"downside_vol_{w}d"] = neg_ret.rolling(w, min_periods=w).std() * np.sqrt(252)
    pos_vol21 = returns.clip(lower=0).rolling(21).std()
    neg_vol21 = neg_ret.rolling(21).std().replace(0, np.nan)
    features["up_down_vol_ratio"] = pos_vol21 / neg_vol21

    # --- Rolling beta to EW market (2) ---
    mkt = returns.mean(axis=1)
    for w in [63]:
        xy = returns.mul(mkt, axis=0)
        e_xy = xy.rolling(w, min_periods=w).mean()
        e_x = returns.rolling(w, min_periods=w).mean()
        e_y = mkt.rolling(w, min_periods=w).mean()
        cov_xy = e_xy - e_x.mul(e_y, axis=0)
        var_y = mkt.rolling(w, min_periods=w).var().replace(0, np.nan)
        beta = cov_xy.div(var_y, axis=0)
        features[f"beta_{w}d"] = beta
        # Idiosyncratic vol
        resid = returns - beta.mul(mkt, axis=0)
        features[f"idio_vol_{w}d"] = resid.rolling(w, min_periods=w).std() * np.sqrt(252)

    # --- Trend consistency (1) ---
    # 위 pos_ret_ratio와 동일한 NaN>0 가드.
    ret5d = returns.rolling(5, min_periods=5).sum()
    trend = (ret5d > 0).astype(float).where(ret5d.notna())
    features["trend_consist_63d"] = trend.rolling(63, min_periods=21).mean()

    return features
