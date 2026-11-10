# -*- coding: utf-8 -*-
"""§S28 src fixes (decision log §S28) — output-invariant on the normal paths.

  tmp leak  tg_basis_guard._save left a 0-byte ``tg_basis_state.json<random>.tmp`` behind when the
            write itself failed (2026-11-09 disk full: OSError 28 at flush) — only os.replace was
            covered by the cleanup.
  H5        the §S22 A-05 tail-truncation guard counted the weekend / US-holiday forward-fill rows
            that sit after the last BusinessDay whenever the workbook cutoff is not a trading day
            (pull on a Monday, pull on the day after a US market holiday) and failed the production
            loader. Only XNYS sessions are rows a stale BusinessDays sheet would really lose.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest


# ------------------------------------------------------------------ tmp leak
def test_tg_basis_save_success_path_round_trips_and_leaves_no_tmp(tmp_path):
    from src import tg_basis_guard as guard

    path = tmp_path / "tg_basis_state.json"
    state = {"schema_version": 1, "pending": {}, "baseline": {"BBB": 2.5, "AAA": 1.1}}
    guard._save(path, state)
    assert json.loads(path.read_text(encoding="utf-8")) == state
    expected = json.dumps(state, indent=2, sort_keys=True, allow_nan=False).replace("\n", os.linesep)
    assert path.read_bytes() == expected.encode("utf-8")           # byte pin of the success path
    assert [p.name for p in tmp_path.iterdir()] == ["tg_basis_state.json"]


def test_tg_basis_save_removes_the_tmp_file_when_the_write_fails(tmp_path, monkeypatch):
    from src import tg_basis_guard as guard

    path = tmp_path / "tg_basis_state.json"
    guard._save(path, {"schema_version": 1, "baseline": {"AAA": 1.1}, "pending": {}})
    before = path.read_bytes()

    def _disk_full(_fd):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("src.tg_basis_guard.os.fsync", _disk_full)
    with pytest.raises(OSError):
        guard._save(path, {"schema_version": 1, "baseline": {"AAA": 9.9}, "pending": {}})
    assert list(tmp_path.glob("*.tmp")) == []                      # no orphan temp file
    assert path.read_bytes() == before                             # previous state untouched


# ------------------------------------------------------------------ H5
def _calendar_day_raw(business_days_end: str, px_end: str):
    """The workbook layout: data sheets on a CALENDAR-day index, BusinessDays = trading days.
    Every weekday from 2026-10-01 to 2026-11-25 is an XNYS session."""
    px_days = pd.date_range("2026-10-01", px_end)
    bdays = pd.bdate_range("2026-10-01", business_days_end)
    px = pd.DataFrame({"AAA": np.linspace(100, 110, len(px_days))}, index=px_days)
    return {
        "PX_LAST": px,
        "Daily_Returns": px.pct_change(),
        "BusinessDays": pd.DataFrame(index=pd.Index(bdays, name="BusinessDay")),
    }


def test_weekend_rows_after_the_last_business_day_are_not_a_tail_truncation():
    from src.data_loader import restrict_to_business_days

    # pull on Monday 2026-11-09 -> cutoff Sunday 11-08; BusinessDays ends Friday 11-06
    out, diag = restrict_to_business_days(
        _calendar_day_raw("2026-11-06", "2026-11-08"), fail_on_tail_truncation=True
    )
    assert diag["tail_rows_dropped"] == 0
    assert diag["tail_dropped_from"] is None
    assert diag["price_last_date"] == "2026-11-08"
    assert out["PX_LAST"].index.max() == pd.Timestamp("2026-11-06")


def test_us_holiday_row_after_the_last_business_day_is_not_a_tail_truncation():
    from src.data_loader import restrict_to_business_days

    # pull on Friday 2026-11-27 -> cutoff Thursday 11-26 (Thanksgiving, XNYS closed)
    out, diag = restrict_to_business_days(
        _calendar_day_raw("2026-11-25", "2026-11-26"), fail_on_tail_truncation=True
    )
    assert diag["tail_rows_dropped"] == 0
    assert diag["tail_dropped_from"] is None
    assert out["PX_LAST"].index.max() == pd.Timestamp("2026-11-25")


def test_a_real_session_after_the_last_business_day_still_fails_closed():
    from src.data_loader import restrict_to_business_days

    # stale BusinessDays sheet: ends Thursday 11-05 while prices run through Friday 11-06 (a session)
    _, diag = restrict_to_business_days(_calendar_day_raw("2026-11-05", "2026-11-06"))
    assert diag["tail_rows_dropped"] == 1
    assert diag["tail_dropped_from"] == "2026-11-06"
    with pytest.raises(ValueError, match="BusinessDays"):
        restrict_to_business_days(
            _calendar_day_raw("2026-11-05", "2026-11-06"), fail_on_tail_truncation=True
        )
