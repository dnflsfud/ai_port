# -*- coding: utf-8 -*-
"""§S13.35 volume/put-call flow 피처 테스트 (default-OFF 게이트·inline reference).

두 arm이 한 모듈을 공유한다:
  arm A (volume 3):  vol_abnormal_21_126 / share_turnover_63d / amihud_illiq_63d
  arm B (putcall 2): pcr_oi_level / pcr_oi_chg_21d
공식·윈도우 관례는 모듈 docstring에 사전등록 — 이 파일의 inline reference가
그 정본을 잠근다.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.features.volume_flow import (
    PUTCALL_FEATURES,
    PUTCALL_SHEET,
    VOLUME_FEATURES,
    admitted_putcall_features,
    admitted_volume_features,
    build_volume_flow_features,
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


def _synthetic(n=160):
    dates = pd.bdate_range("2024-01-01", periods=n)
    tickers = ["AAA", "BBB", "CCC"]
    rng = np.random.default_rng(35)

    def sheet(low, high):
        return pd.DataFrame(
            rng.uniform(low, high, (len(dates), 3)), index=dates, columns=tickers
        )

    sheets = {
        "PX_VOLUME": sheet(1e6, 5e6),
        "PX_LAST": sheet(50.0, 150.0),
        "CUR_MKT_CAP": sheet(1e10, 5e10),
        "Daily_Returns": sheet(-0.03, 0.03),
        PUTCALL_SHEET: sheet(0.3, 1.5),
    }
    return _FakeData(sheets, tickers, dates), sheets


def _reference(sheets):
    vol = sheets["PX_VOLUME"]
    px = sheets["PX_LAST"]
    mc = sheets["CUR_MKT_CAP"].where(sheets["CUR_MKT_CAP"] > 0)
    ret = sheets["Daily_Returns"]
    pcr = sheets[PUTCALL_SHEET]

    adv21 = vol.rolling(21, min_periods=10).mean()
    adv126 = vol.rolling(126, min_periods=63).mean()
    dollar = (vol * px).where(vol * px > 0)

    return {
        "vol_abnormal_21_126": np.log(adv21 / adv126.where(adv126 > 0)),
        "share_turnover_63d": dollar.rolling(63, min_periods=31).mean() / mc,
        "amihud_illiq_63d": np.log(
            (ret.abs() * mc / dollar)
            .rolling(63, min_periods=31)
            .mean()
            .where(lambda x: x > 0)
        ),
        "pcr_oi_level": pcr,
        "pcr_oi_chg_21d": pcr - pcr.shift(21),
    }


def test_admitted_gates_off_empty_on_full_and_independent():
    off = SimpleNamespace()
    on_vol = SimpleNamespace(volume_features_enabled=True)
    on_pcr = SimpleNamespace(putcall_features_enabled=True)
    assert admitted_volume_features(off) == set()
    assert admitted_putcall_features(off) == set()
    assert admitted_volume_features(on_vol) == set(VOLUME_FEATURES)
    assert admitted_putcall_features(on_vol) == set()
    assert admitted_putcall_features(on_pcr) == set(PUTCALL_FEATURES)
    assert admitted_volume_features(on_pcr) == set()


def test_build_matches_inline_reference():
    data, sheets = _synthetic()
    out = build_volume_flow_features(data)
    assert set(out) == set(VOLUME_FEATURES) | set(PUTCALL_FEATURES)
    reference = _reference(sheets)
    for name, ref in reference.items():
        assert np.allclose(
            out[name].fillna(0).values, ref.fillna(0).values, atol=1e-9
        ), name


def test_guards_never_leak_inf():
    data, sheets = _synthetic()
    bad_mc = sheets["CUR_MKT_CAP"].copy()
    bad_mc.iloc[130, 0] = 0.0
    bad_vol = sheets["PX_VOLUME"].copy()
    bad_vol.iloc[131, 1] = 0.0
    data._sheets["CUR_MKT_CAP"] = bad_mc
    data._sheets["PX_VOLUME"] = bad_vol
    out = build_volume_flow_features(data)
    for name in VOLUME_FEATURES:
        vals = out[name].values
        assert not np.isinf(vals[~np.isnan(vals)]).any(), name
    # MC=0 셀은 해당일 turnover가 NaN이어야 한다 (0 나눗셈 가드).
    assert np.isnan(out["share_turnover_63d"].iloc[130, 0])


def test_build_skips_missing_sheet_groups():
    dates = pd.bdate_range("2024-01-01", periods=5)
    empty = _FakeData({}, ["AAA"], dates)
    assert build_volume_flow_features(empty) == {}

    # putcall 시트만 없으면 volume 3종은 생성된다 (§9: 추정 금지).
    data, _ = _synthetic()
    del data._sheets[PUTCALL_SHEET]
    out = build_volume_flow_features(data)
    assert set(out) == set(VOLUME_FEATURES)

    # PX_VOLUME이 없으면 volume 축 전체가 빠지고 putcall만 남는다.
    data2, _ = _synthetic()
    del data2._sheets["PX_VOLUME"]
    out2 = build_volume_flow_features(data2)
    assert set(out2) == set(PUTCALL_FEATURES)


def test_putcall_missing_columns_stay_nan():
    """비US 48종처럼 시트에 열이 없는 종목은 정직한 NaN 열로 남아야 한다."""
    data, sheets = _synthetic()
    data._sheets[PUTCALL_SHEET] = sheets[PUTCALL_SHEET][["AAA", "BBB"]]
    out = build_volume_flow_features(data)
    assert out["pcr_oi_level"]["CCC"].isna().all()
    assert out["pcr_oi_chg_21d"]["CCC"].isna().all()
    assert out["pcr_oi_level"]["AAA"].notna().all()
