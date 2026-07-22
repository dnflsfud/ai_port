"""Category 4: Conditioning (~50 features).

단독 IR=0이지만 GBT interaction에서 강력한 변수들.
Includes earnings timeline features.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict

from src.data_loader import UniverseData, TICKERS
from src.features.sellside import clean_revision_spikes, get_cleaned_revision

logger = logging.getLogger(__name__)


def build_conditioning_features(data: UniverseData, config=None) -> Dict[str, pd.DataFrame]:
    """Build conditioning regime/calendar features.

    `config` is forwarded to `get_cleaned_revision` so the conditioning-side
    revision breadth features use the SAME cleaned revision stream as the
    sellside feature block. Prior to 2026-04-21 this code path hardcoded
    mode="down_only" with a separate cleaner call, which meant Phase 2.4's
    `reversion_gated` mode was silently ignored here.
    """
    features: Dict[str, pd.DataFrame] = {}
    dates = data.dates
    # Use data.tickers (intersection of loaded sheets) as the authoritative
    # universe. Reading TICKERS + returns.columns would silently ignore
    # tickers that were dropped by other sheets but present in returns.
    tickers = list(data.tickers)
    n = len(tickers)
    # §S11.7: PIT 뷰(상장 전 NaN) — regime 시장 통계(ew_ret·평균 vol·분산도)에
    # 유령의 합성 수익률이 참여하지 않도록 masked 뷰를 소비한다.
    returns = data.returns_masked

    def bcast(vals_1d):
        """1D array/series -> broadcast to (dates x tickers)."""
        v = np.asarray(vals_1d).reshape(-1, 1)
        return pd.DataFrame(np.tile(v, (1, n)), index=dates, columns=tickers)

    # ===== Calendar (~15) =====
    month = dates.month
    features["cal_is_Q1"] = bcast((month <= 3).astype(float))
    features["cal_is_Q2"] = bcast(((month >= 4) & (month <= 6)).astype(float))
    features["cal_is_Q3"] = bcast(((month >= 7) & (month <= 9)).astype(float))
    features["cal_is_Q4"] = bcast((month >= 10).astype(float))
    features["cal_is_jan"] = bcast((month == 1).astype(float))
    features["cal_is_qtr_end"] = bcast(month.isin([3, 6, 9, 12]).astype(float))
    features["cal_is_yr_end"] = bcast((month == 12).astype(float))
    features["cal_is_earnings"] = bcast(month.isin([1, 2, 4, 5, 7, 8, 10, 11]).astype(float))
    # month_in_quarter: 1=first month, 2=mid, 3=last (proxy for days to earnings)
    miq = ((month - 1) % 3 + 1).astype(float)
    features["cal_month_in_qtr"] = bcast(miq)
    features["cal_is_mid_qtr"] = bcast((miq == 2).astype(float))
    features["cal_is_first_half"] = bcast((month <= 6).astype(float))
    dow = dates.dayofweek
    features["cal_is_monday"] = bcast((dow == 0).astype(float))
    features["cal_is_friday"] = bcast((dow == 4).astype(float))

    # ===== Sector one-hot (~10) =====
    meta = data.meta
    if isinstance(meta, pd.DataFrame) and "sector" in meta.columns:
        sector_map = meta["sector"]
    elif isinstance(meta, pd.DataFrame) and len(meta.columns) > 0:
        sector_map = meta.iloc[:, 0]
    else:
        sector_map = pd.Series("Unknown", index=tickers)

    for sec in sector_map.unique():
        if str(sec) in ("nan", "Unknown"):
            continue
        vals = np.zeros((len(dates), n))
        for i, t in enumerate(tickers):
            if sector_map.get(t, "") == sec:
                vals[:, i] = 1.0
        features[f"sector_{sec}"] = pd.DataFrame(vals, index=dates, columns=tickers)

    # ===== Size buckets (~5) =====
    mcr = data.market_cap.rank(axis=1, pct=True)
    features["is_mega_cap"] = (mcr > 0.8).astype(float)
    features["is_large_cap"] = ((mcr > 0.6) & (mcr <= 0.8)).astype(float)
    features["is_mid_cap"] = ((mcr > 0.3) & (mcr <= 0.6)).astype(float)
    features["is_small_cap"] = (mcr <= 0.3).astype(float)
    features["size_rank"] = mcr

    # ===== Market regime (continuous + binary) (~15) =====
    ew_ret = returns.mean(axis=1)
    for w in [21, 63]:
        r = ew_ret.rolling(w, min_periods=w).sum()
        features[f"regime_mkt_ret_{w}d"] = bcast(r)
    vol_21 = returns.rolling(21, min_periods=21).std().mean(axis=1) * np.sqrt(252)
    vol_63 = returns.rolling(63, min_periods=63).std().mean(axis=1) * np.sqrt(252)
    features["regime_avg_vol_21d"] = bcast(vol_21)
    features["regime_avg_vol_63d"] = bcast(vol_63)

    # Cross-sectional dispersion
    cs_disp_21 = returns.rolling(21, min_periods=21).mean().std(axis=1)
    features["regime_dispersion_21d"] = bcast(cs_disp_21)

    # Market breadth — denominator is the per-date valid (listed) count, not
    # the fixed universe width (§S11.4 point-in-time universe).
    ma50 = data.prices.rolling(50, min_periods=50).mean()
    valid_n_ma50 = ma50.notna().sum(axis=1).replace(0, np.nan)
    breadth = (data.prices > ma50).sum(axis=1) / valid_n_ma50
    features["regime_breadth_50d"] = bcast(breadth)

    # --- BINARY regime flags (핵심: standalone IR~=0, interaction에서 강력) ---
    vol_med = vol_21.rolling(252, min_periods=126).median()
    features["is_high_vol"] = bcast((vol_21 > vol_med * 1.2).astype(float))
    features["is_low_vol"] = bcast((vol_21 < vol_med * 0.8).astype(float))
    ret_63 = ew_ret.rolling(63, min_periods=63).sum()
    features["is_bull_market"] = bcast((ret_63 > 0.05).astype(float))
    features["is_bear_market"] = bcast((ret_63 < -0.05).astype(float))
    features["is_trending"] = bcast((ret_63.abs() > 0.08).astype(float))
    disp_med = cs_disp_21.rolling(252, min_periods=126).median()
    features["is_high_dispersion"] = bcast((cs_disp_21 > disp_med * 1.2).astype(float))

    # ===== Earnings Timeline Features (~8, per-stock) =====
    earn_tl = getattr(data, "earnings_timeline", None)
    if earn_tl is not None:
        earn = earn_tl.reindex(index=dates, columns=tickers, fill_value=0)

        # is_earnings_day: 당일 실적발표 (per-stock)
        features["earn_is_day"] = earn.astype(float)

        # days_since_earnings: 마지막 실적발표 이후 경과일 (fully vectorized)
        days_since = pd.DataFrame(np.nan, index=dates, columns=tickers)
        dates_ts = dates.values.astype("datetime64[ns]")
        for col in tickers:
            if col not in earn.columns:
                continue
            earn_dates_col = earn.index[earn[col] == 1]
            if len(earn_dates_col) == 0:
                continue
            earn_ts = earn_dates_col.values.astype("datetime64[ns]")
            idx = np.searchsorted(earn_ts, dates_ts, side='right') - 1
            valid = idx >= 0
            deltas = np.where(
                valid,
                (dates_ts - earn_ts[np.clip(idx, 0, len(earn_ts) - 1)]).astype("timedelta64[D]").astype(float),
                np.nan,
            )
            days_since[col] = deltas
        days_since = days_since.ffill().fillna(999)
        features["earn_days_since"] = days_since

        # days_to_next_earnings: 다음 실적발표까지 남은 일수 (fully vectorized)
        days_to = pd.DataFrame(np.nan, index=dates, columns=tickers)
        for col in tickers:
            if col not in earn.columns:
                continue
            earn_dates_col = earn.index[earn[col] == 1]
            if len(earn_dates_col) == 0:
                continue
            earn_ts = earn_dates_col.values.astype("datetime64[ns]")
            idx = np.searchsorted(earn_ts, dates_ts, side='left')
            valid = idx < len(earn_ts)
            deltas = np.where(
                valid,
                (earn_ts[np.clip(idx, 0, len(earn_ts) - 1)] - dates_ts).astype("timedelta64[D]").astype(float),
                np.nan,
            )
            days_to[col] = deltas
        # NaNs occur only on the trailing dates after the last known earnings
        # event (searchsorted side='left' is valid for every prior date), so a
        # bfill has nothing to pull from here — it is a no-op. Drop it: keeping
        # it would import a future earnings delta if the timeline shape ever
        # changed. Fill the unknown tail with the sentinel directly.
        days_to = days_to.fillna(999)
        features["earn_days_to_next"] = days_to

        # Binary flags
        features["earn_pre_5d"] = (days_to <= 5).astype(float)
        features["earn_pre_10d"] = (days_to <= 10).astype(float)
        features["earn_post_5d"] = (days_since <= 5).astype(float)
        features["earn_post_10d"] = (days_since <= 10).astype(float)

        # Earnings cycle position: 0~1 (0=발표 직후, 1=다음 발표 직전)
        cycle = days_since / (days_since + days_to).replace(0, np.nan)
        features["earn_cycle_pos"] = cycle.fillna(0.5)

        print(f"[EarningsFeatures] 8개 실적발표일 피처 생성")
    else:
        print("[EarningsFeatures] Earnings_Timeline 없음 - 스킵")

    # ===== Fundamental regime (~7, spike-cleaned via shared helper) =====
    # Routed through get_cleaned_revision (2026-04-21) so the cleaning mode
    # matches sellside + PEAD + growth_tilt. Note this branch previously
    # passed `earnings_timeline=earn_tl` (Pattern-2 in timeline mode); the
    # helper defaults to None to stay consistent with the sellside call.
    eps_rev = get_cleaned_revision(data, "Factset_EPS_Revision", config=config)
    if eps_rev is not None:
        rev_pos_pct = (
            (eps_rev > 0).sum(axis=1)
            / eps_rev.notna().sum(axis=1).replace(0, np.nan)
        )
        features["regime_rev_breadth_eps"] = bcast(rev_pos_pct)
        features["is_rev_expansion"] = bcast((rev_pos_pct > 0.6).astype(float))
    else:
        logger.debug("conditioning: Factset_EPS_Revision missing — skipping regime_rev_breadth_eps / is_rev_expansion")

    sales_rev = get_cleaned_revision(data, "Factset_Sales_Revision", config=config)
    if sales_rev is not None:
        rev_pos_s = (
            (sales_rev > 0).sum(axis=1)
            / sales_rev.notna().sum(axis=1).replace(0, np.nan)
        )
        features["regime_rev_breadth_sales"] = bcast(rev_pos_s)
    else:
        logger.debug("conditioning: Factset_Sales_Revision missing — skipping regime_rev_breadth_sales")
    try:
        news = data.get_sheet("NEWS_SENTIMENT_DAILY_AVG")
        sent_pos = (
            (news > 0).sum(axis=1)
            / news.notna().sum(axis=1).replace(0, np.nan)
        )
        features["regime_sent_breadth"] = bcast(sent_pos)
    except KeyError:
        logger.debug("conditioning: NEWS_SENTIMENT_DAILY_AVG missing — skipping regime_sent_breadth")
    try:
        rec = data.get_sheet("EQY_REC_CONS")
        rec_mean = rec.mean(axis=1)
        features["regime_avg_rec"] = bcast(rec_mean)
    except KeyError:
        logger.debug("conditioning: EQY_REC_CONS missing — skipping regime_avg_rec")

    return features
