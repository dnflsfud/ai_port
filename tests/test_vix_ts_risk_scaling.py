# -*- coding: utf-8 -*-
"""§S13.34 VIX 텀스트럭처 디리스킹 오버레이 테스트 (OFF-parity·재진입 로직)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.backtest import apply_vix_ts_risk_scaling


def _cfg(**kw):
    base = dict(
        vix_ts_risk_scaling_enabled=True,
        vix_ts_risk_scale=0.5,
        vix_ts_improve_window=5,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _panel(dates):
    return pd.DataFrame(1.0, index=dates, columns=["AAA", "BBB"])


def test_off_returns_input_object():
    dates = pd.bdate_range("2024-01-01", periods=10)
    preds = _panel(dates)
    fpx = pd.DataFrame({"VIX": 20.0, "VIX3M": 22.0}, index=dates)
    out = apply_vix_ts_risk_scaling(
        preds, fpx, _cfg(vix_ts_risk_scaling_enabled=False)
    )
    assert out is preds  # 구조적 parity: OFF는 입력 객체 그대로


def test_missing_vix_columns_is_inert():
    dates = pd.bdate_range("2024-01-01", periods=10)
    preds = _panel(dates)
    fpx = pd.DataFrame({"VIX": 20.0}, index=dates)
    pd.testing.assert_frame_equal(
        apply_vix_ts_risk_scaling(preds, fpx, _cfg()), preds
    )
    pd.testing.assert_frame_equal(
        apply_vix_ts_risk_scaling(preds, None, _cfg()), preds
    )


def test_backwardation_scaling_and_reentry():
    """콘탱고 유지 → 백워데이션 심화 시 μ×0.5 → 축소(개선) 전환 시 원복."""
    dates = pd.bdate_range("2024-01-01", periods=20)
    slope = np.array(
        [0.10] * 10
        + [-0.02, -0.04, -0.06, -0.08, -0.10]   # 심화: risk-off
        + [-0.08, -0.06, -0.04, -0.02, -0.01]   # 회복 경로
    )
    vix = pd.Series(20.0, index=dates)
    fpx = pd.DataFrame({"VIX": vix, "VIX3M": vix * (1.0 + slope)}, index=dates)
    preds = _panel(dates)
    out = apply_vix_ts_risk_scaling(preds, fpx, _cfg())

    # 5일 창: t−5 대비 slope 하락(악화)인 −0.02…−0.06(idx 10..16)은 ×0.5,
    # idx 17부터 slope_t > slope_{t−5} (백워데이션 축소) → 리스크 재사용(원복).
    expected = np.ones(20)
    expected[10:17] = 0.5
    assert np.allclose(out["AAA"].values, expected)
    assert np.allclose(out["BBB"].values, expected)
    # 입력은 불변 (copy-on-write)
    assert (preds.values == 1.0).all()


def test_nan_slope_dates_are_inert_and_nan_preds_stay_nan():
    dates = pd.bdate_range("2024-01-01", periods=8)
    vix = pd.Series([np.nan] * 8, index=dates)
    fpx = pd.DataFrame({"VIX": vix, "VIX3M": vix}, index=dates)
    preds = _panel(dates)
    preds.iloc[3, 0] = np.nan
    out = apply_vix_ts_risk_scaling(preds, fpx, _cfg())
    # slope 전부 NaN → risk-off 없음 → 수치 동일, NaN 위치 보존
    pd.testing.assert_frame_equal(out, preds)
