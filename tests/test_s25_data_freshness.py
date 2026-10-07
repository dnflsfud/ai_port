"""§S25 (2026-10-07) — production gate must see the DATA date's freshness.

The registry's only staleness signal compared the registry's own generation
time with a 96h limit; the scheduler regenerates the registry daily, so a
workbook frozen at data_as_of 2026-09-29 was published on 10-02, 10-05 and
10-06 without any warning, and rebalance_overdue (row-count based) can never
fire while the workbook is frozen.
"""
from datetime import datetime, timezone

import pandas as pd

from scripts.validate_portfolio_bundles import (
    MAX_DATA_AGE_SESSIONS,
    data_age_sessions,
    evaluate_production,
)
from streamlit_app import data_freshness_warning


def test_data_age_counts_completed_exchange_sessions_after_data_as_of():
    # Registry built 2026-10-06 03:29 UTC = 10-05 23:29 ET: last completed
    # XNYS session is 10-05 -> 09-30, 10-01, 10-02, 10-05 = 4 sessions old.
    assert data_age_sessions("2026-09-29", "2026-10-06T03:29:56+00:00", "XNYS") == 4
    # Same data, built 10-06 21:00 UTC = 17:00 ET (10-06 session closed): 5.
    assert data_age_sessions("2026-09-29", "2026-10-06T21:00:00+00:00", "XNYS") == 5
    # 10-06 20:00 UTC = 16:00 ET: the 10-06 close is not counted yet: 4.
    assert data_age_sessions("2026-09-29", "2026-10-06T20:00:00+00:00", "XNYS") == 4
    # Fresh workbook (pulled the KST morning after the 10-06 close), run 10-07 02:30 UTC: 0.
    assert data_age_sessions("2026-10-06", "2026-10-07T02:30:00+00:00", "XNYS") == 0
    # Weekday calendar fallback counts weekdays.
    assert data_age_sessions("2026-10-02", "2026-10-07T02:30:00+00:00", "weekday_index") == 2
    # Unparseable / missing inputs -> None (fail-closed upstream).
    assert data_age_sessions(None, "2026-10-07T02:30:00+00:00", "XNYS") is None
    assert data_age_sessions("not-a-date", "2026-10-07T02:30:00+00:00", "XNYS") is None
    assert data_age_sessions("2026-10-02", "2026-10-07T02:30:00+00:00", "XLON") is None


def test_production_gate_holds_on_stale_data_as_of():
    stale = evaluate_production(
        {"data_as_of": "2026-09-29", "rebalance_calendar": "XNYS"},
        as_of_utc=datetime(2026, 10, 6, 3, 29, 56, tzinfo=timezone.utc),
    )
    assert stale["checks"]["data_as_of_fresh_ok"] is False
    assert stale["status"] == "HOLD"
    assert stale["values"]["data_as_of"] == "2026-09-29"
    assert stale["values"]["data_age_sessions"] == 4
    assert stale["values"]["max_data_age_sessions"] == MAX_DATA_AGE_SESSIONS == 3

    fresh = evaluate_production(
        {"data_as_of": "2026-10-06", "rebalance_calendar": "XNYS"},
        as_of_utc=datetime(2026, 10, 7, 2, 30, tzinfo=timezone.utc),
    )
    assert fresh["checks"]["data_as_of_fresh_ok"] is True
    assert fresh["values"]["data_age_sessions"] == 0

    boundary = evaluate_production(
        {"data_as_of": "2026-10-01", "rebalance_calendar": "XNYS"},
        as_of_utc=datetime(2026, 10, 7, 2, 30, tzinfo=timezone.utc),   # 10-02, 10-05, 10-06 = 3
    )
    assert boundary["checks"]["data_as_of_fresh_ok"] is True

    missing = evaluate_production({})
    assert missing["checks"]["data_as_of_fresh_ok"] is None   # fail-closed
    assert missing["status"] == "HOLD"


def test_dashboard_warning_names_the_stale_data_date():
    gate = {"checks": {"data_as_of_fresh_ok": False},
            "values": {"data_as_of": "2026-09-29", "data_age_sessions": 4, "max_data_age_sessions": 3}}
    message = data_freshness_warning(gate)
    assert "2026-09-29" in message and "4" in message and "3" in message
    assert data_freshness_warning({"checks": {"data_as_of_fresh_ok": True},
                                   "values": {"data_age_sessions": 0}}) is None
    unknown = data_freshness_warning({"checks": {"data_as_of_fresh_ok": None}, "values": {}})
    assert unknown is not None and "unknown" in unknown.lower()
    assert data_freshness_warning(None) is None
