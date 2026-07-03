"""Category 6: Macro × Ticker cross features (Phase 2 P2 fix, 2026-04-22).

Motivation
----------
factor.py already generates macro features (VIX level, yield slope, DXY
z-score, ...), but they are broadcast **identically across tickers** — zero
cross-sectional variation. GBT can only use them as regime flags on the
split/conditioning side.

This module builds **macro × ticker-specific** products that DO have
cross-sectional variation: every ticker on the same date gets a different
value because the macro scalar is multiplied by the ticker's own signal
(revision, momentum, or realized vol).

Economic hypothesis for P2 (rate-shock 2021-05 ~ 2023-10):
- `rate_up × positive_revision` — firms with rising forward earnings
  *despite* rising rates are the best longs (pricing power, quality).
- `yield_slope_negative × positive_revision` — a recession-signalling
  inversion combined with strong forward revisions flags defensive
  quality.
- `VIX_high × positive_momentum` — crowded trades (positive momentum
  during fear spikes) often reverse; cross term lets the model learn
  that the LONGER mom is less reliable when VIX is high.
- `vol_high × momentum_63d` — pure ticker-level risk-on / momentum
  interaction (not macro); high-vol momentum often overshoots.
- `DXY_up × positive_revision` — USD strength is a revision headwind
  for multinationals; this term captures the relative winners.

Five features total — kept lean to limit selection bias in a 60-ticker
universe.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.data_loader import UniverseData
from src.features.utils import cross_sectional_zscore
from src.features.sellside import get_cleaned_revision

logger = logging.getLogger(__name__)


def _rolling_zscore(s: pd.Series, window: int = 63) -> pd.Series:
    rm = s.rolling(window, min_periods=max(21, window // 3)).mean()
    rs = s.rolling(window, min_periods=max(21, window // 3)).std().replace(0, np.nan)
    return (s - rm) / rs


def _bcast_scalar_to_panel(
    series: pd.Series, dates: pd.DatetimeIndex, tickers: list
) -> pd.DataFrame:
    """Broadcast a date-only series to (dates × tickers) panel."""
    vals = series.reindex(dates).values.reshape(-1, 1)
    return pd.DataFrame(
        np.tile(vals, (1, len(tickers))), index=dates, columns=tickers
    )


def build_macro_cross_features(
    data: UniverseData, config=None
) -> Dict[str, pd.DataFrame]:
    """Build macro × ticker-specific cross features.

    All output panels share the intersection of `data.dates` and factor
    data index, so downstream alignment in assembly.build_all_features
    is identical to factor.build_factor_features.
    """
    features: Dict[str, pd.DataFrame] = {}

    # Ablation gate: config.macro_cross_enabled (default True).
    # When False, return an empty dict so apply_core_filter silently drops
    # the mc_* entries from the whitelist (leaving the feature count at
    # baseline_v3's 56 instead of 61) — lets us run regime-PCA ablation
    # cleanly.
    if config is not None and not getattr(config, "macro_cross_enabled", True):
        logger.info("macro_cross: disabled via config.macro_cross_enabled=False")
        return features

    if not data.has_factor_data():
        logger.debug("macro_cross: factor data missing — skipping all features")
        return features

    factor_px = data.factor_prices
    tickers = list(data.tickers)
    common_dates = data.dates.intersection(factor_px.index)

    # ── Ticker-specific base signals (aligned to common_dates) ──
    returns = data.returns.loc[:, tickers]
    ret_aligned = returns.reindex(index=common_dates)

    mom63 = ret_aligned.rolling(63, min_periods=63).sum()
    mom252 = ret_aligned.rolling(252, min_periods=252).sum()
    vol21 = ret_aligned.rolling(21, min_periods=21).std() * np.sqrt(252)

    mom63_cs = cross_sectional_zscore(mom63)
    mom252_cs = cross_sectional_zscore(mom252)
    vol21_cs = cross_sectional_zscore(vol21)

    # ── Cleaned EPS revision (reuses Phase 2.4 helper) ──
    eps_rev_cleaned = get_cleaned_revision(
        data, "Factset_EPS_Revision", config=config
    )
    if eps_rev_cleaned is not None:
        eps_rev_aligned = eps_rev_cleaned.reindex(
            index=common_dates, columns=tickers
        )
        eps_rev_cs = cross_sectional_zscore(eps_rev_aligned)
    else:
        eps_rev_cs = None

    # ── Macro scalars (z-scored) ──
    def _get_macro_z(col: str, window: int = 63) -> Optional[pd.DataFrame]:
        if col not in factor_px.columns:
            return None
        z = _rolling_zscore(factor_px[col], window=window)
        return _bcast_scalar_to_panel(z, common_dates, tickers)

    ust_10y_z = _get_macro_z("UST_10Y")
    vix_z = _get_macro_z("VIX")
    dxy_z = _get_macro_z("DXY")

    slope_panel = None
    if "UST_10Y" in factor_px.columns and "UST_2Y" in factor_px.columns:
        slope = factor_px["UST_10Y"] - factor_px["UST_2Y"]
        slope_panel = _bcast_scalar_to_panel(slope, common_dates, tickers)

    # ── Cross terms ──

    # 1. Rate level × revision — "pricing power during rate hikes"
    if ust_10y_z is not None and eps_rev_cs is not None:
        features["mc_rate_x_eps_rev"] = ust_10y_z * eps_rev_cs

    # 2. Yield-curve slope × revision — "recession hedge via forward earnings"
    if slope_panel is not None and eps_rev_cs is not None:
        features["mc_slope_x_eps_rev"] = slope_panel * eps_rev_cs

    # 3. VIX × momentum — "crowded-trade reversal indicator"
    if vix_z is not None:
        features["mc_vix_x_mom252"] = vix_z * mom252_cs

    # 4. Realized vol × 63d momentum — "high-vol momentum overshoot"
    features["mc_vol_x_mom63"] = vol21_cs * mom63_cs

    # 5. DXY × revision — "USD headwind for multinationals"
    if dxy_z is not None and eps_rev_cs is not None:
        features["mc_dxy_x_eps_rev"] = dxy_z * eps_rev_cs

    logger.info(
        "macro_cross: built %d cross features: %s",
        len(features),
        sorted(features.keys()),
    )
    return features
