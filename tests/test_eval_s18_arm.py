# -*- coding: utf-8 -*-
"""§S18.1 arm 판정 스크립트 헬퍼 단위테스트 (경량 — pkl 로드 없음)."""
import types

import numpy as np
import pandas as pd

from scripts.eval_s18_arm import (
    BASE, FRAMES, NEG_EQUITY_NAMES, TG_BASIS_GATES, TURNOVER_NEUTRAL_BAND, Z_SD_MIN,
    mechanism_business_day_calendar, mechanism_static, mechanism_static_neutral, mechanism_tg_basis,
    mechanism_tilt,
)


def test_frames_and_base():
    assert set(FRAMES) == {"s18_1_tilt_negative_equity", "s18_2_tg_basis_events", "s18_3_static_execution",
                           "s18_5_static_execution_eta042", "s18_6_business_day_calendar"}
    # §S18.7 calendar arm: correctness track against the 2026-09-11 workbook re-certification.
    assert FRAMES["s18_6_business_day_calendar"]["base"] == "s18_7_s0recert"
    assert FRAMES["s18_6_business_day_calendar"]["mechanism"] == "business_day_calendar"
    assert FRAMES["s18_6_business_day_calendar"]["alpha_identical"] is False
    assert BASE == "s18_s0recert"
    # §S18.3 re-attempt is judged against the two-flip re-certification, not the S18.1 base.
    assert FRAMES["s18_5_static_execution_eta042"]["base"] == "s18_4_flip2_recert"
    assert FRAMES["s18_5_static_execution_eta042"]["mechanism"] == "static_neutral"
    assert TURNOVER_NEUTRAL_BAND == (0.85, 1.15)
    assert FRAMES["s18_3_static_execution"]["alpha_identical"] is True
    assert all(v["turnover_max"] == 1.25 for v in FRAMES.values())
    assert set(TG_BASIS_GATES) == {"RTX", "T", "DELL", "DHR"} and Z_SD_MIN == 0.90
    assert len(NEG_EQUITY_NAMES) == 21


def _res(pre_exec=None, panel=None):
    return types.SimpleNamespace(pre_execution_predictions=pre_exec, panel=panel)


def test_mechanism_tilt_requires_penalty_release_on_the_named_cells():
    idx = pd.bdate_range("2024-01-01", periods=120)
    cols = NEG_EQUITY_NAMES[:5] + ["AAPL"]
    base = pd.DataFrame(0.0, index=idx, columns=cols)
    arm = base.copy()
    arm.loc[:, NEG_EQUITY_NAMES[:5]] = 0.3           # 600 cells released by +0.3
    out = mechanism_tilt(_res(base), _res(arm))
    assert out["pass"] is True and out["changed_cells"] == 600
    weak = base.copy(); weak.loc[idx[:10], NEG_EQUITY_NAMES[0]] = 0.5   # 10 cells only
    assert mechanism_tilt(_res(base), _res(weak))["pass"] is False
    assert mechanism_tilt(_res(None), _res(arm))["pass"] is False


def _panel(values_by_ticker, years=(2014, 2022), n_other=30, spread=1.0):
    dates = pd.bdate_range(f"{years[0]}-01-01", f"{years[1]}-12-31")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(0, spread, (len(dates), n_other)), index=dates,
                         columns=[f"X{i}" for i in range(n_other)])
    for t, v in values_by_ticker.items():
        frame[t] = v
    stacked = frame.stack()
    stacked.index.names = ["date", "ticker"]
    return pd.DataFrame({"tg_upside": stacked})


def test_mechanism_tg_basis_gates():
    base = _panel({"RTX": -3.5, "T": -2.2, "DELL": 5.0, "DHR": 3.0}, spread=0.8)
    arm = _panel({"RTX": -0.3, "T": 0.1, "DELL": 0.5, "DHR": 0.2}, spread=1.0)
    out = mechanism_tg_basis(_res(panel=base), _res(panel=arm))
    assert out["pass"] is True
    assert out["names"]["RTX"]["base"] < -3 and out["names"]["RTX"]["arm"] > -1.5
    assert out["tg_upside_cs_sd_2014_2021"]["arm"] >= Z_SD_MIN
    still_pinned = _panel({"RTX": -3.5, "T": 0.1, "DELL": 0.5, "DHR": 0.2}, spread=1.0)
    assert mechanism_tg_basis(_res(panel=base), _res(panel=still_pinned))["pass"] is False


def test_mechanism_static_execution_bounds():
    g0_ok = {"avg_ic_bit_identical": True, "degenerate_equal": True}
    assert mechanism_static(1.10, g0_ok)["pass"] is True
    assert mechanism_static(0.98, g0_ok)["pass"] is False      # execution-only change must not trade less
    assert mechanism_static(1.30, g0_ok)["pass"] is False
    assert mechanism_static(1.10, {"avg_ic_bit_identical": False, "degenerate_equal": True})["pass"] is False


def test_mechanism_static_neutral_bounds():
    g0_ok = {"avg_ic_bit_identical": True, "degenerate_equal": True}
    assert mechanism_static_neutral(1.00, g0_ok)["pass"] is True
    assert mechanism_static_neutral(0.90, g0_ok)["pass"] is True      # neutral: trading less is allowed
    assert mechanism_static_neutral(0.80, g0_ok)["pass"] is False
    assert mechanism_static_neutral(1.28, g0_ok)["pass"] is False     # the S18.1 arm-3 outcome fails here
    assert mechanism_static_neutral(1.00, {"avg_ic_bit_identical": False, "degenerate_equal": True})["pass"] is False


def _cal_res(dates, panel_dates=None):
    act = pd.Series(0.0, index=pd.DatetimeIndex(dates))
    panel = None
    if panel_dates is not None:
        idx = pd.MultiIndex.from_product([pd.DatetimeIndex(panel_dates), ["AAA", "BBB"]], names=["date", "ticker"])
        panel = pd.DataFrame({"x": 0.0}, index=idx)
    return types.SimpleNamespace(active_returns=act, panel=panel)


def test_mechanism_business_day_calendar_gate():
    bdays = pd.bdate_range("2024-01-01", "2024-12-31")
    holidays = pd.DatetimeIndex(["2024-01-15", "2024-02-19", "2024-05-27", "2024-07-04"])  # US-holiday weekdays
    business = bdays.difference(holidays)
    base = _cal_res(bdays, bdays)                       # daily calendar incl. holiday weekdays
    arm = _cal_res(business, business)                  # restricted to BusinessDays
    arm_doc = {"data_quality": {"business_day_calendar": {"enabled": True, "daily_returns_recomputed": True,
                                                          "business_days": len(business),
                                                          "rows_dropped_by_sheet": {"PX_LAST": 4, "Daily_Returns": 4}},
                                "intersection_dates": len(business), "tail_extended_dates": 0}}
    base_doc = {"data_quality": {"intersection_dates": len(bdays) - 1, "tail_extended_dates": 1}}
    out = mechanism_business_day_calendar(base, arm, arm_doc, base_doc, business)
    assert out["pass"] is True
    assert out["weekday_rows_outside_business_days"] == {"base": 4, "arm": 0}
    assert out["rows_dropped_total"] == 8 and out["calendar_len"]["shrank"] is True
    assert out["calendar_len"] == {"base": len(bdays), "arm": len(business), "shrank": True}
    # arm still carrying a holiday row -> FAIL
    leaky = _cal_res(business.union(holidays[:1]), business)
    assert mechanism_business_day_calendar(base, leaky, arm_doc, base_doc, business)["pass"] is False
    # flag not actually on (no diagnostic) -> FAIL even if dates look right
    assert mechanism_business_day_calendar(base, arm, {"data_quality": {}}, base_doc, business)["pass"] is False
    # base already clean (nothing to fix) -> FAIL (mechanism must be observable)
    assert mechanism_business_day_calendar(arm, arm, arm_doc, arm_doc, business)["pass"] is False
