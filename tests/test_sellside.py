# -*- coding: utf-8 -*-
"""sellside 빌더 단위 테스트 — §S13.4 신규 피처 3종 값 계약.

수용 게이트(플래그·whitelist·yaml)는 tests/acceptance/
test_s13_surprise_opcf_features.py 가 담당하고, 이 파일은 빌더 값만 본다.
"""

import numpy as np
import pandas as pd

from src.config import PipelineConfig

NEW_KEYS = {"eps_surprise", "sales_surprise", "fwd_opcf_yield"}


class _FakeData:
    def __init__(self, sheets, local_prices):
        self._sheets = dict(sheets)
        self.local_prices = local_prices
        self.prices = local_prices

    def get_sheet(self, name):
        if name not in self._sheets:
            raise KeyError(name)
        return self._sheets[name]


def _panel(values):
    idx = pd.date_range("2026-01-01", periods=len(values), freq="B")
    return pd.DataFrame({"AAA": values, "BBB": [v * 2 for v in values]}, index=idx)


def test_builder_surprise_levels_and_opcf_yield():
    from src.features.sellside import build_sellside_features

    eps = _panel([1.0, 1.0, 3.0])
    sales = _panel([-2.0, -2.0, 4.0])
    opcf = _panel([10.0, 10.0, 12.0])
    px = _panel([100.0, 0.0, 120.0])  # 0 -> NaN 가드 확인

    data = _FakeData(
        {
            "Factset_EPS_Surprise": eps,
            "Factset_Sales_Surprise": sales,
            "Factset_Fwd_OpCashflow": opcf,
        },
        local_prices=px,
    )
    feats = build_sellside_features(data, config=PipelineConfig())

    # 원 레벨 pass-through (스무딩/클리닝 없음 — 서프라이즈는 이벤트 계단열)
    pd.testing.assert_frame_equal(feats["eps_surprise"], eps)
    pd.testing.assert_frame_equal(feats["sales_surprise"], sales)

    # fwd CF yield = opcf / px, 가격 0은 NaN
    y = feats["fwd_opcf_yield"]
    assert np.isclose(y.iloc[0]["AAA"], 10.0 / 100.0)
    assert np.isnan(y.iloc[1]["AAA"])
    assert np.isclose(y.iloc[2]["BBB"], 24.0 / 240.0)


def test_builder_missing_sheets_are_skipped():
    from src.features.sellside import build_sellside_features

    data = _FakeData({}, local_prices=_panel([100.0, 101.0, 102.0]))
    feats = build_sellside_features(data, config=PipelineConfig())
    for key in NEW_KEYS | REV_KEYS:
        assert key not in feats


# --- §S13.5: fwd_opcf 리비전 3윈도우 (safe_pct_change 관용구) ---------------
REV_KEYS = {"fwd_opcf_rev_63d", "fwd_opcf_rev_126d", "fwd_opcf_rev_252d"}


def test_builder_fwd_opcf_rev_windows():
    from src.features.sellside import build_sellside_features
    from src.features.utils import safe_pct_change

    n = 260
    opcf = pd.DataFrame(
        {
            # 음수 -> 양수 전환 포함: |기저값| 분모의 부호 보존 계약 확인
            "AAA": np.linspace(-5.0, 15.0, n),
            "BBB": np.linspace(10.0, 30.0, n),
        },
        index=pd.date_range("2025-01-01", periods=n, freq="B"),
    )
    px = opcf * 0.0 + 100.0

    data = _FakeData({"Factset_Fwd_OpCashflow": opcf}, local_prices=px)
    feats = build_sellside_features(data, config=PipelineConfig())

    for window, key in [(63, "fwd_opcf_rev_63d"), (126, "fwd_opcf_rev_126d"),
                        (252, "fwd_opcf_rev_252d")]:
        assert key in feats, key
        pd.testing.assert_frame_equal(feats[key], safe_pct_change(opcf, window))

    # 부호 보존 스팟체크: 기저 -5 -> +상승이면 리비전은 양수여야 한다.
    v = feats["fwd_opcf_rev_63d"]["AAA"].dropna().iloc[0]
    assert v > 0
