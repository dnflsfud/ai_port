"""Category 6: Rate-Regime features for LightGBM model input.

REDESIGN iter16 (2026-04-16): Adds macro rate-environment features
directly to the model panel. The post-model regime gate lever
(compute_regime_stress / regime_gate_enabled) was deleted on 2026-04-20
after iter21 confirmed it subtracted IR (see docs/rollback_log.md).

Rationale: P2 (2021-05 ~ 2023-10) IR=0.107 is the weakest sub-period.
This period was dominated by the Fed rate-hike cycle. The existing model
has only ONE rate-adjacent feature (regime_mkt_ret_21d) which captures
equity direction, NOT the rate level/speed that drove cross-sectional
alpha rotation. These features give the model direct visibility into the
rate environment so it can learn rate-conditional factor loadings.

All features are BROADCAST (same value across all tickers per date) so
they function as conditioning variables — IR=~0 standalone but powerful
in GBT interaction splits (same principle as cal_is_Q1).

Feature catalogue (5 features):
  1. regime_yield_level_z   — 10Y UST 3-year trailing z-score
  2. regime_yield_chg_63d   — 63d change in 10Y yield (rate momentum)
  3. regime_curve_slope     — 10Y - 2Y slope (inversion signal)
  4. regime_real_rate       — 10Y - BEI10 (real rate proxy)
  5. regime_rate_vol_63d    — 63d realized vol of daily 10Y yield changes
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

from src.data_loader import UniverseData


def build_regime_features(data: UniverseData) -> Dict[str, pd.DataFrame]:
    """Build rate-regime conditioning features from factor price data."""
    features: Dict[str, pd.DataFrame] = {}

    fp = data.factor_prices
    if fp is None:
        print("[RegimeFeatures] No factor_prices available — skipping")
        return features

    dates = data.dates
    tickers = list(data.tickers)
    n = len(tickers)

    def bcast(series_1d: pd.Series) -> pd.DataFrame:
        """Broadcast 1D series -> (dates x tickers) DataFrame."""
        aligned = series_1d.reindex(dates)
        v = aligned.values.reshape(-1, 1)
        return pd.DataFrame(np.tile(v, (1, n)), index=dates, columns=tickers)

    # --- 10Y yield level z-score (3-year trailing) ---
    if "UST_10Y" in fp.columns:
        ust10 = fp["UST_10Y"].reindex(dates).ffill()

        # Trailing 3Y z-score: how extreme is today's yield vs recent history
        ust10_mean = ust10.rolling(756, min_periods=252).mean()
        ust10_std = ust10.rolling(756, min_periods=252).std().replace(0, np.nan)
        yield_z = (ust10 - ust10_mean) / ust10_std
        features["regime_yield_level_z"] = bcast(yield_z)

        # 63d yield change (rate momentum / direction)
        features["regime_yield_chg_63d"] = bcast(ust10.diff(63))

        # 63d realized vol of daily yield changes
        daily_chg = ust10.diff()
        rate_vol = daily_chg.rolling(63, min_periods=21).std() * np.sqrt(252)
        features["regime_rate_vol_63d"] = bcast(rate_vol)

    # --- Yield curve slope (10Y - 2Y) ---
    if "UST_10Y" in fp.columns and "UST_2Y" in fp.columns:
        ust10 = fp["UST_10Y"].reindex(dates).ffill()
        ust2 = fp["UST_2Y"].reindex(dates).ffill()
        slope = ust10 - ust2
        features["regime_curve_slope"] = bcast(slope)

    # --- Real rate proxy (10Y - breakeven inflation) ---
    if "UST_10Y" in fp.columns and "US_BEI10" in fp.columns:
        ust10 = fp["UST_10Y"].reindex(dates).ffill()
        bei10 = fp["US_BEI10"].reindex(dates).ffill()
        real_rate = ust10 - bei10
        features["regime_real_rate"] = bcast(real_rate)

    if features:
        print(f"[RegimeFeatures] Built {len(features)} rate-regime features")
    else:
        print("[RegimeFeatures] Required columns not found in factor_prices — skipping")

    return features
