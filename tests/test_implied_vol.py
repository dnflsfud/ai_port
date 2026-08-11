# -*- coding: utf-8 -*-
"""§S13.34 implied-vol 서피스 피처 테스트 (default-OFF 게이트·inline reference)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.features.implied_vol import (
    IMPLIED_VOL_FEATURES,
    IV_30D_100_SHEET,
    IV_30D_90_SHEET,
    IV_30D_110_SHEET,
    IV_3M_100_SHEET,
    REALIZED_VOL_SHEET,
    admitted_implied_vol_features,
    build_implied_vol_features,
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
    dates = pd.bdate_range("2024-01-01", periods=40)
    tickers = ["AAA", "BBB", "CCC"]
    rng = np.random.default_rng(11)

    def sheet(base):
        return pd.DataFrame(
            rng.uniform(base, base + 15.0, (len(dates), 3)),
            index=dates,
            columns=tickers,
        )

    sheets = {
        IV_30D_100_SHEET: sheet(28.0),
        IV_30D_90_SHEET: sheet(31.0),
        IV_30D_110_SHEET: sheet(27.0),
        IV_3M_100_SHEET: sheet(29.0),
        REALIZED_VOL_SHEET: sheet(30.0),
    }
    return _FakeData(sheets, tickers, dates), sheets


def test_admitted_gate_off_is_empty_on_is_full():
    off = SimpleNamespace(implied_vol_features_enabled=False)
    on = SimpleNamespace(implied_vol_features_enabled=True)
    assert admitted_implied_vol_features(off) == set()
    assert admitted_implied_vol_features(SimpleNamespace()) == set()
    assert admitted_implied_vol_features(on) == set(IMPLIED_VOL_FEATURES)


def test_build_matches_inline_reference():
    data, sheets = _synthetic()
    out = build_implied_vol_features(data)
    assert set(out) == set(IMPLIED_VOL_FEATURES)
    iv100 = sheets[IV_30D_100_SHEET]
    reference = {
        "iv_skew_level": (sheets[IV_30D_90_SHEET] - sheets[IV_30D_110_SHEET]) / iv100,
        "iv_vrp_level": iv100 - sheets[REALIZED_VOL_SHEET],
        "iv_term_level": (sheets[IV_3M_100_SHEET] - iv100) / iv100,
    }
    for name, ref in reference.items():
        assert np.allclose(
            out[name].fillna(0).values, ref.fillna(0).values, atol=1e-9
        ), name


def test_non_positive_iv100_guard_yields_nan():
    data, sheets = _synthetic()
    iv100 = sheets[IV_30D_100_SHEET].copy()
    iv100.iloc[0, 0] = 0.0
    iv100.iloc[1, 1] = -3.0
    data._sheets[IV_30D_100_SHEET] = iv100
    out = build_implied_vol_features(data)
    assert np.isnan(out["iv_skew_level"].iloc[0, 0])
    assert np.isnan(out["iv_term_level"].iloc[0, 0])
    assert np.isnan(out["iv_skew_level"].iloc[1, 1])
    assert np.isnan(out["iv_term_level"].iloc[1, 1])
    # 가드 셀 밖은 전부 유한해야 한다 (inf 유출 금지).
    assert np.isfinite(out["iv_skew_level"].values[2:]).all()
    assert np.isfinite(out["iv_vrp_level"].values).all()


def test_build_skips_missing_inputs():
    dates = pd.bdate_range("2024-01-01", periods=5)
    empty = _FakeData({}, ["AAA"], dates)
    assert build_implied_vol_features(empty) == {}

    # 실현 vol 시트만 없으면 vrp만 빠지고 skew/term은 생성된다 (§9: 추정 금지).
    data, _ = _synthetic()
    del data._sheets[REALIZED_VOL_SHEET]
    out = build_implied_vol_features(data)
    assert set(out) == {"iv_skew_level", "iv_term_level"}
