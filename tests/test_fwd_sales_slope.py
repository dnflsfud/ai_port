# -*- coding: utf-8 -*-
"""§S13.25 fwd sales 기간구조 slope 피처 + 월말 리밸런싱 테스트 (default-OFF·게이트)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.backtest import make_month_end_rebal_check
from src.features.fwd_sales_slope import (
    FWD_SALES_SLOPE_FEATURES,
    FWD_SALES_SLOPE_SHEET,
    NL_CONFIRM_PARENTS,
    admitted_fwd_sales_slope_features,
    build_fwd_sales_slope_features,
)


class _FakeData:
    def __init__(self, sheets, tickers, dates):
        self._sheets = sheets
        self.tickers = tickers
        self.dates = dates

    def get_sheet(self, name):
        if name not in self._sheets:
            raise KeyError(name)
        return self._sheets[name]


def _synthetic():
    dates = pd.bdate_range("2024-01-01", periods=70)
    tickers = ["AAA", "BBB", "CCC"]
    rng = np.random.default_rng(7)
    slope = pd.DataFrame(
        rng.normal(0.05, 0.02, (len(dates), 3)), index=dates, columns=tickers
    )
    parents = {
        parent: pd.DataFrame(
            rng.normal(0.0, 1.0, (len(dates), 3)), index=dates, columns=tickers
        )
        for parent in NL_CONFIRM_PARENTS.values()
    }
    data = _FakeData({FWD_SALES_SLOPE_SHEET: slope}, tickers, dates)
    return data, parents, slope


def test_admitted_gate_off_is_empty_on_is_full():
    off = SimpleNamespace(fwd_sales_slope_features_enabled=False)
    on = SimpleNamespace(fwd_sales_slope_features_enabled=True)
    assert admitted_fwd_sales_slope_features(off) == set()
    assert admitted_fwd_sales_slope_features(SimpleNamespace()) == set()
    assert admitted_fwd_sales_slope_features(on) == set(FWD_SALES_SLOPE_FEATURES)


def test_build_produces_declared_features():
    data, parents, slope = _synthetic()
    out = build_fwd_sales_slope_features(parents, data)
    assert set(out) == set(FWD_SALES_SLOPE_FEATURES)
    pd.testing.assert_frame_equal(out["fwd_sales_slope_level"], slope)
    pd.testing.assert_frame_equal(out["fwd_sales_slope_chg_63d"], slope.diff(63))


def test_nl_confirm_is_elementwise_min_of_zscores():
    """soft-AND: 두 z-score의 최소값 — 한 다리가 약하면 확인 점수도 약해야 한다."""
    data, parents, slope = _synthetic()
    out = build_fwd_sales_slope_features(parents, data)
    zs = (slope.sub(slope.mean(axis=1), axis=0)).div(slope.std(axis=1, ddof=1), axis=0)
    for name, parent in NL_CONFIRM_PARENTS.items():
        p = parents[parent]
        zp = (p.sub(p.mean(axis=1), axis=0)).div(p.std(axis=1, ddof=1), axis=0)
        expected = zs.where(zs.le(zp), zp)
        got = out[name]
        assert np.allclose(got.values, expected.values, atol=1e-9, equal_nan=True)


def test_build_skips_missing_sheet_and_missing_parents():
    dates = pd.bdate_range("2024-01-01", periods=5)
    empty = _FakeData({}, ["AAA"], dates)
    assert build_fwd_sales_slope_features({}, empty) == {}

    data, _, _ = _synthetic()
    out = build_fwd_sales_slope_features({}, data)  # 파트너 피처 부재
    assert set(out) == {"fwd_sales_slope_level", "fwd_sales_slope_chg_63d"}


def test_month_end_rebal_check_hits_last_trading_day_of_month():
    # 6월 말 금요일(06-28), 7월 말 수요일(07-31)을 포함하는 거래일 달력.
    dates = pd.bdate_range("2024-06-24", "2024-08-06")
    check = make_month_end_rebal_check(dates)

    state = {"first_rebal": True}
    assert check(0, dates[0], 0, 21, state) is True  # 첫날 초기 편입
    state = {"first_rebal": False}

    expected_eom = {pd.Timestamp("2024-06-28"), pd.Timestamp("2024-07-31")}
    hits = {
        dates[i]
        for i in range(len(dates))
        if check(i, dates[i], 0, 21, state)
    }
    assert hits == expected_eom  # 마지막 날짜(08-06)는 월말 판정 불가 -> 미포함
