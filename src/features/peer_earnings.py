"""S13.10 — peer earnings cascade (relational features).

Every one of the 61 core features is a single-stock attribute: this name's
valuation, this name's momentum, this name's volatility. Nothing tells the
model what the *rest of the sector* already reported before this name does.

That gap matters because earnings arrive in a staggered sequence. A name that
reports late in its sector's season does so into a tape that has already
repriced the read-across; a name that reports first does not.

Sources are deliberately narrow — ``Earnings_Timeline`` and ``Daily_Returns``
plus the sector map. Both were verified clean in the 2026-07-27 workbook audit
(price/return identity corr 1.0, zero >100bp mismatches). We do NOT use
``Factset_EPS_Surprise``: its structural bank coverage gap is what sank S13.4,
and re-importing it here would confound this arm with that defect.

Point-in-time: every value at date t is built from announcements dated <= t and
returns realised on those days. Same-day inclusion matches the existing
``earn_is_day`` / ``earn_post_5d`` convention in conditioning.py, and the
pipeline's ``execution_signal_lag_days`` applies on top.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Trailing window that defines "this earnings season": one quarter of business
# days, long enough for a full sector reporting cycle and short enough that the
# previous quarter's cascade has rolled off. Matches retrain_freq=63.
SEASON_DAYS = 63

# No-event sentinel, matching conditioning.py's earn_days_since/days_to_next.
NO_EVENT = 999.0


def _leave_one_out_mean(
    values: pd.DataFrame,
    valid: pd.DataFrame,
    sector_members: Dict[str, list],
) -> pd.DataFrame:
    """Per-date sector mean of ``values`` excluding each column itself.

    ``valid`` is the 0/1 mask of which cells count, so a peer with no recent
    announcement contributes to neither numerator nor denominator. A column
    whose sector holds no other qualifying peer stays NaN — never self.
    """
    out = pd.DataFrame(np.nan, index=values.index, columns=values.columns)
    contrib = (values * valid).fillna(0.0)
    for members in sector_members.values():
        total = contrib[members].sum(axis=1)
        count = valid[members].sum(axis=1)
        for name in members:
            peer_n = (count - valid[name]).replace(0, np.nan)
            out[name] = (total - contrib[name]) / peer_n
    return out


def build_peer_earnings_features(data) -> Dict[str, pd.DataFrame]:
    """Build the 3 peer-cascade features. Empty dict if inputs are missing."""
    features: Dict[str, pd.DataFrame] = {}

    earn_tl = getattr(data, "earnings_timeline", None)
    if earn_tl is None:
        logger.info("[PeerEarnings] Earnings_Timeline absent - skipping")
        return features

    meta = getattr(data, "meta", None)
    if not isinstance(meta, pd.DataFrame) or "sector" not in meta.columns:
        logger.info("[PeerEarnings] sector map absent - skipping")
        return features

    dates = data.dates
    tickers = list(data.tickers)
    earn = earn_tl.reindex(index=dates, columns=tickers, fill_value=0).astype(float)

    sector_map = meta["sector"].astype(str)
    sector_members: Dict[str, list] = {}
    for name in tickers:
        sec = sector_map.get(name, "Unknown")
        if sec in ("nan", "Unknown"):
            continue
        sector_members.setdefault(sec, []).append(name)
    # A sector of one has no peers; leaving it out keeps _leave_one_out_mean
    # from ever degenerating into the name's own value.
    sector_members = {k: v for k, v in sector_members.items() if len(v) >= 2}
    if not sector_members:
        logger.info("[PeerEarnings] no sector has 2+ members - skipping")
        return features

    # Has this name reported inside the trailing season?
    reported = (earn.rolling(SEASON_DAYS, min_periods=1).sum() > 0).astype(float)

    # (1) how much of my sector has already reported (excluding me)
    ones = pd.DataFrame(1.0, index=dates, columns=tickers)
    features["peer_earn_reported_frac"] = _leave_one_out_mean(
        reported, ones, sector_members
    )

    # (2) what the tape did to peers when they reported.
    # Announcement-day EXCESS return (day return minus the universe mean that
    # day), so a market-wide move is not misread as a reaction. Each name
    # carries its latest reaction for the season, then we average over peers.
    returns = data.returns.reindex(index=dates, columns=tickers)
    excess = returns.sub(returns.mean(axis=1), axis=0)
    carried = excess.where(earn == 1).ffill(limit=SEASON_DAYS)
    carried = carried.where(reported == 1)
    features["peer_earn_reaction_63d"] = _leave_one_out_mean(
        carried, carried.notna().astype(float), sector_members
    )

    # (3) am I an early or a late reporter this season?
    # my days-since-last-announcement minus the sector median. Positive = my
    # peers reported more recently than I did, i.e. I report into a tape that
    # has already seen the sector read-across.
    pos = pd.Series(np.arange(len(dates), dtype=float), index=dates)
    days_since = pd.DataFrame(NO_EVENT, index=dates, columns=tickers)
    for name in tickers:
        last = pos.where(earn[name] == 1).ffill()
        days_since[name] = (pos - last).fillna(NO_EVENT)

    lead_lag = pd.DataFrame(np.nan, index=dates, columns=tickers)
    for members in sector_members.values():
        median = days_since[members].median(axis=1)
        for name in members:
            lead_lag[name] = days_since[name] - median
    features["peer_earn_lead_lag"] = lead_lag

    logger.info(
        "[PeerEarnings] 3 relational features over %d sectors "
        "(%d tickers, season=%dbd)",
        len(sector_members),
        sum(len(v) for v in sector_members.values()),
        SEASON_DAYS,
    )
    return features
