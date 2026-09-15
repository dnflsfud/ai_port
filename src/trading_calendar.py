"""Shared workbook calendar and exchange-session forecasting.

Historical BusinessDays are authoritative; XNYS extends only beyond the
workbook's last date. Legacy weekday variants explicitly retain their calendar.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd


def business_days_from_raw(raw):
    if "BusinessDays" not in raw:
        raise ValueError("business_day_calendar_enabled=True requires a BusinessDays sheet in the workbook.")
    sheet = raw["BusinessDays"]
    values = sheet["BusinessDay"] if "BusinessDay" in sheet else sheet.index
    days = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce")).dropna()
    days = days.normalize().unique().sort_values()
    if days.empty or (days.dayofweek >= 5).any():
        raise ValueError("BusinessDays sheet has no valid dates or contains weekend dates.")
    return days


@lru_cache(maxsize=16)
def exchange_sessions(start, end):
    try:
        import exchange_calendars as xcals
    except ImportError as exc:
        raise RuntimeError("Future trading dates require exchange-calendars; install requirements-full.txt.") from exc
    # Include boundary weekends/holidays inside the instantiated calendar.
    calendar = xcals.get_calendar("XNYS", start=pd.Timestamp(start) - pd.Timedelta(days=7),
                                  end=pd.Timestamp(end) + pd.Timedelta(days=7))
    return pd.DatetimeIndex(calendar.sessions_in_range(start, end)).tz_localize(None).normalize()


def forecast_session(as_of, rows_until, *, calendar_dates=None, calendar_name="weekday_index"):
    """Return the Nth following session and whether exchange extrapolation was used."""
    as_of = pd.Timestamp(as_of).normalize()
    if rows_until <= 0:
        raise ValueError("rows_until must be positive")
    if calendar_name == "weekday_index":
        return as_of + pd.offsets.BDay(rows_until), True
    if calendar_name != "XNYS":
        raise ValueError(f"Unsupported trading calendar: {calendar_name}")
    known = pd.DatetimeIndex([] if calendar_dates is None else calendar_dates).normalize().unique().sort_values()
    future = known[known > as_of]
    if len(future) >= rows_until:
        return future[rows_until - 1], False
    end_known = max(as_of, known[-1]) if len(known) else as_of
    start = end_known + pd.Timedelta(days=1)
    end = start + pd.Timedelta(days=max(366, rows_until * 3))
    extension = exchange_sessions(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    future = future.union(extension).sort_values()
    if len(future) < rows_until:
        raise ValueError("Insufficient future exchange sessions")
    return future[rows_until - 1], True


def align_earnings_events(events, sessions):
    """Map dated release flags to the first session on/after the release.

    Never backdate future events. Collisions OR the flag, while diagnostics
    retain the number of source events represented by the binary matrix.
    """
    sessions = pd.DatetimeIndex(sessions).normalize()
    if sessions.empty or not sessions.is_monotonic_increasing or sessions.has_duplicates:
        raise ValueError("Earnings sessions must be sorted and unique")
    source_dates = pd.DatetimeIndex(events.index).normalize()
    row, col = np.nonzero(events.eq(1).to_numpy())
    positions = sessions.searchsorted(source_dates[row], side="left")
    keep = (positions < len(sessions)) & (source_dates[row] >= sessions.min())
    values = np.zeros((len(sessions), len(events.columns)), dtype=int)
    values[positions[keep], col[keep]] = 1
    result = pd.DataFrame(values, index=sessions, columns=events.columns)
    moved = int((sessions[positions[keep]] != source_dates[row[keep]]).sum())
    diag = {"source_events": int(len(row)), "represented_events": int(keep.sum()),
            "mapped_flags": int(result.to_numpy().sum()), "shifted_events": moved,
            "outside_calendar_events": int((~keep).sum())}
    return result, diag
