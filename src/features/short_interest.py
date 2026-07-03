"""Category 7: Short Interest features (TSZ-normalized).

REDESIGN iter16d (2026-04-16): SHORT_INT_RATIO (days-to-cover) from
S&P500.xlsx. Micro-structure signal orthogonal to existing features.

Normalization strategy (same as fin_pb_level_z / fin_pe_level_z):
  1. rolling_tsz(756, 252): 3Y trailing time-series z-score per ticker
     → answers "is this stock's SI high vs its own recent history?"
  2. cross_sectional_zscore: across tickers on the same date
     → answers "relative to peers, whose SI is elevated today?"

This two-stage normalization prevents structurally high-SI names (e.g.
TSLA, which always has high SI) from dominating the signal — just like
rolling_tsz prevents NVDA's structurally high PE from biasing valuation.

Feature catalogue (3 features):
  1. si_tsz_level   — 3Y TSZ → CS z-score (core: "unusually shorted vs history & peers")
  2. si_tsz_chg_63d — 63d change in TSZ (direction: covering vs building trend)
  3. si_cs_rank     — raw CS percentile rank (robust complement, no history needed)
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict

from src.data_loader import UniverseData
from src.features.utils import cross_sectional_zscore, cs_rank, rolling_tsz

logger = logging.getLogger(__name__)


def build_short_interest_features(data: UniverseData) -> Dict[str, pd.DataFrame]:
    """Build short interest features from SHORT_INT_RATIO sheet."""
    features: Dict[str, pd.DataFrame] = {}

    try:
        si_raw = data.get_sheet("SHORT_INT_RATIO")
    except KeyError:
        logger.warning("ShortInterestFeatures: SHORT_INT_RATIO sheet not found — skipping")
        return features

    tickers = list(data.tickers)
    dates = data.dates

    # Align to universe tickers and dates
    si = si_raw.reindex(columns=tickers, index=dates).ffill()

    # Drop if too sparse (< 30% coverage)
    coverage = si.notna().sum().sum() / si.size
    if coverage < 0.3:
        logger.warning("ShortInterestFeatures: coverage too low (%.1f%%) — skipping", coverage * 100)
        return features

    # Fill remaining NaN with per-date cross-sectional median
    row_median = si.median(axis=1)
    for col in si.columns:
        mask = si[col].isna()
        if mask.any():
            si.loc[mask, col] = row_median[mask]
    si = si.fillna(0.0)

    # 1. TSZ level: 3Y trailing z-score → cross-sectional z-score
    #    "unusually shorted vs own history AND vs peers today"
    si_tsz = rolling_tsz(si, window=756, min_periods=252)
    features["si_tsz_level"] = cross_sectional_zscore(si_tsz)

    # 2. TSZ 63d change: direction of the history-normalized SI
    #    Positive = SI rising vs own norm (shorts building), negative = covering
    features["si_tsz_chg_63d"] = si_tsz.diff(63)

    # 3. Raw cross-sectional rank (robust: no history dependency, works from day 1)
    features["si_cs_rank"] = cs_rank(si)

    print(f"[ShortInterestFeatures] Built {len(features)} TSZ-normalized SI features "
          f"(coverage={coverage:.1%})")

    return features
