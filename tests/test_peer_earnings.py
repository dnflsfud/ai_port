# -*- coding: utf-8 -*-
"""S13.10 peer earnings cascade — value contract.

The three features are relational: each is a leave-one-out aggregate over the
name's own sector. The properties that must hold:

  * leave-one-out is exact — in a 2-name sector, my peer aggregate IS the
    other name's value, never my own.
  * point-in-time — an announcement dated t cannot change any value before t.
  * the reaction is an EXCESS return, so a day where the whole universe moves
    together produces no reaction signal.
  * lead_lag sign: positive = my peers reported more recently than I did
    (I am the late reporter).
"""

import types

import numpy as np
import pandas as pd

from src.features.peer_earnings import SEASON_DAYS, build_peer_earnings_features

TICKERS = ["AA", "AB", "BA", "BB"]
SECTORS = {"AA": "Alpha", "AB": "Alpha", "BA": "Beta", "BB": "Beta"}


def _data(earn_marks, returns=None, n=20):
    """earn_marks: {ticker: [row indices where an announcement lands]}."""
    dates = pd.bdate_range("2024-01-01", periods=n)
    earn = pd.DataFrame(0.0, index=dates, columns=TICKERS)
    for t, rows in earn_marks.items():
        for r in rows:
            earn.iloc[r, earn.columns.get_loc(t)] = 1.0
    if returns is None:
        returns = pd.DataFrame(0.0, index=dates, columns=TICKERS)
    meta = pd.DataFrame({"sector": [SECTORS[t] for t in TICKERS]},
                        index=pd.Index(TICKERS, name="ticker"))
    return types.SimpleNamespace(
        earnings_timeline=earn, meta=meta, dates=dates,
        tickers=TICKERS, returns=returns,
    )


def test_missing_timeline_yields_no_features():
    d = _data({})
    d.earnings_timeline = None
    assert build_peer_earnings_features(d) == {}


def test_builds_exactly_three_named_features():
    from src.config import PEER_EARNINGS_FEATURES

    f = build_peer_earnings_features(_data({"AA": [2], "BA": [3]}))
    assert set(f) == set(PEER_EARNINGS_FEATURES)


def test_leave_one_out_excludes_self():
    """AA reported, AB did not. AB's peer frac is 1.0 (AA), AA's is 0.0 (AB)."""
    f = build_peer_earnings_features(_data({"AA": [2]}))
    frac = f["peer_earn_reported_frac"]
    assert frac.loc[frac.index[5], "AB"] == 1.0   # AB sees AA reported
    assert frac.loc[frac.index[5], "AA"] == 0.0   # AA sees only AB, silent


def test_point_in_time_no_leak_backwards():
    """An announcement at row 10 must not move any value at rows < 10."""
    late = build_peer_earnings_features(_data({"AA": [10]}))["peer_earn_reported_frac"]
    never = build_peer_earnings_features(_data({}))["peer_earn_reported_frac"]
    pd.testing.assert_series_equal(
        late["AB"].iloc[:10], never["AB"].iloc[:10], check_names=False
    )
    assert late.loc[late.index[10], "AB"] == 1.0  # visible from the event day


def test_season_window_rolls_off():
    f = build_peer_earnings_features(_data({"AA": [0]}, n=SEASON_DAYS + 5))
    frac = f["peer_earn_reported_frac"]
    assert frac.loc[frac.index[SEASON_DAYS - 1], "AB"] == 1.0
    assert frac.loc[frac.index[SEASON_DAYS + 1], "AB"] == 0.0


def test_reaction_is_excess_not_raw_return():
    """A day where every name moves identically carries no reaction."""
    dates = pd.bdate_range("2024-01-01", periods=20)
    flat = pd.DataFrame(0.0, index=dates, columns=TICKERS)
    flat.iloc[3] = 0.05  # whole universe up 5% on the announcement day
    f = build_peer_earnings_features(_data({"AA": [3]}, returns=flat))
    react = f["peer_earn_reaction_63d"]
    assert abs(react.loc[react.index[6], "AB"]) < 1e-12

    idio = pd.DataFrame(0.0, index=dates, columns=TICKERS)
    idio.iloc[3, idio.columns.get_loc("AA")] = 0.08  # only AA jumps
    g = build_peer_earnings_features(_data({"AA": [3]}, returns=idio))
    # AA's excess = 0.08 - mean(0.08,0,0,0) = 0.06; AB sees exactly that.
    assert np.isclose(g["peer_earn_reaction_63d"].loc[dates[6], "AB"], 0.06)


def test_lead_lag_positive_means_i_am_the_late_reporter():
    """AB reported at row 2, AA has not reported at all -> AA is late."""
    f = build_peer_earnings_features(_data({"AB": [2]}))
    ll = f["peer_earn_lead_lag"]
    assert ll.loc[ll.index[8], "AA"] > 0
    assert ll.loc[ll.index[8], "AB"] < 0


def test_sector_of_one_is_not_aggregated():
    """A lone sector member has no peers; its cells stay NaN, never self."""
    d = _data({"AA": [2]})
    d.meta = pd.DataFrame({"sector": ["Alpha", "Solo", "Beta", "Beta"]},
                          index=pd.Index(TICKERS, name="ticker"))
    f = build_peer_earnings_features(d)
    assert f["peer_earn_reported_frac"]["AA"].isna().all()
    assert f["peer_earn_reported_frac"]["AB"].isna().all()
