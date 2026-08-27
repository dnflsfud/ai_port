# -*- coding: utf-8 -*-
"""factor.py — §S15 fix-pack 캘린더 정렬 테스트."""
import numpy as np
import pandas as pd
import pytest

from src.config import PipelineConfig
from src.features.factor import build_factor_features


class _FakeFactorData:
    def __init__(self, dates, factor_returns=None, factor_prices=None):
        self.dates = dates
        self.tickers = ["A", "B"]
        self.factor_returns = factor_returns
        self.factor_prices = factor_prices

    def has_factor_data(self):
        return self.factor_returns is not None or self.factor_prices is not None


def _setup(fixpack):
    dates = pd.bdate_range("2020-01-01", periods=120)
    factor_cal = dates.delete(80)     # 80번째 날 = 팩터 캘린더 결손(휴장)
    fr = pd.DataFrame({"SPX": 0.01}, index=factor_cal)
    data = _FakeFactorData(dates, factor_returns=fr)
    cfg = PipelineConfig(s15_fixpack_enabled=fixpack)
    return build_factor_features(data, config=cfg), dates


def test_off_parity_restricted_calendar():
    feats, dates = _setup(fixpack=False)
    f = feats["fac_SPX_mom_21d"]
    assert dates[80] not in f.index
    assert len(f.index) == 119


def test_no_config_matches_off_parity():
    dates = pd.bdate_range("2020-01-01", periods=120)
    factor_cal = dates.delete(80)
    fr = pd.DataFrame({"SPX": 0.01}, index=factor_cal)
    data = _FakeFactorData(dates, factor_returns=fr)
    feats = build_factor_features(data)          # 기존 호출 형태 (config 없음)
    off, _ = _setup(fixpack=False)
    pd.testing.assert_frame_equal(feats["fac_SPX_mom_21d"],
                                  off["fac_SPX_mom_21d"])


def test_on_full_calendar_with_ffill():
    feats, dates = _setup(fixpack=True)
    f = feats["fac_SPX_mom_21d"]
    assert f.index.equals(dates)
    # 휴장일 값 = 직전 팩터 값 (ffill), 0.0이 아님
    assert f.loc[dates[80], "A"] == pytest.approx(0.21, rel=1e-9)
    # 워밍업 NaN은 그대로 보존
    assert np.isnan(f.loc[dates[0], "A"])
