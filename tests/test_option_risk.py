# -*- coding: utf-8 -*-
"""§S13.38 옵션 리스크 표준화 피처 테스트 (default-OFF 게이트·패스스루).

파생(252D TS z·Δ5)은 데이터 계층(§S13.37, create_ai_signal_data.py)이 정본 —
이 모듈은 워크북 시트를 (dates×tickers)로 정렬해 그대로 싣는 패스스루다.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.features.option_risk import (
    OPTION_RISK_FEATURES,
    admitted_option_risk_features,
    build_option_risk_features,
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


def _synthetic(n=60):
    dates = pd.bdate_range("2025-01-01", periods=n)
    tickers = ["AAA", "BBB", "CCC"]
    rng = np.random.default_rng(38)

    def sheet(cols):
        return pd.DataFrame(
            rng.normal(0.0, 1.0, (len(dates), len(cols))),
            index=dates,
            columns=cols,
        )

    sheets = {name: sheet(tickers) for name in OPTION_RISK_FEATURES}
    return _FakeData(sheets, tickers, dates), sheets


def test_admitted_gate_off_empty_on_full():
    assert admitted_option_risk_features(SimpleNamespace()) == set()
    on = SimpleNamespace(option_risk_features_enabled=True)
    assert admitted_option_risk_features(on) == set(OPTION_RISK_FEATURES)
    assert "days_to_earnings" in OPTION_RISK_FEATURES
    # vrp_z는 사전등록 제외(사용자 지시 — 4피처 + earnings 분리 장치만)
    assert "vol_risk_premium_z" not in OPTION_RISK_FEATURES


def test_build_is_aligned_passthrough():
    data, sheets = _synthetic()
    out = build_option_risk_features(data)
    assert set(out) == set(OPTION_RISK_FEATURES)
    for name, df in out.items():
        assert list(df.index) == list(data.dates)
        assert list(df.columns) == data.tickers
        assert np.allclose(
            df.fillna(0).values, sheets[name].fillna(0).values, atol=1e-12
        ), name


def test_build_drops_extra_columns_and_nans_missing():
    """SPX Index 등 비유니버스 열은 버리고, 시트에 없는 종목은 NaN 열."""
    data, sheets = _synthetic()
    wide = sheets["iv30_z"].copy()
    wide["SPX Index"] = 99.0
    data._sheets["iv30_z"] = wide[["AAA", "BBB", "SPX Index"]]  # CCC 열 부재
    out = build_option_risk_features(data)
    assert list(out["iv30_z"].columns) == ["AAA", "BBB", "CCC"]
    assert out["iv30_z"]["CCC"].isna().all()
    assert np.allclose(out["iv30_z"]["AAA"].values, wide["AAA"].values)


def test_build_skips_missing_sheets_independently():
    data, _ = _synthetic()
    del data._sheets["days_to_earnings"]
    out = build_option_risk_features(data)
    assert set(out) == set(OPTION_RISK_FEATURES) - {"days_to_earnings"}

    empty = _FakeData({}, ["AAA"], pd.bdate_range("2025-01-01", periods=3))
    assert build_option_risk_features(empty) == {}
