# -*- coding: utf-8 -*-
"""macro_cross 피처의 PIT returns 소비 계약 (§S11.7).

mom/vol 횡단면 z-score는 dense `data.returns`가 아니라
`data.returns_masked`에서 계산돼야 한다 (유령 median 시계열 제외).
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.features.macro_cross import build_macro_cross_features


def _raise_key_error(name):
    raise KeyError(name)


def _stub(n_dates=80, listing_pos=70, seed=2):
    dates = pd.bdate_range("2022-01-03", periods=n_dates)
    rng = np.random.default_rng(seed)
    dense = pd.DataFrame(
        rng.normal(0.0, 0.02, size=(n_dates, 3)),
        index=dates,
        columns=["AAA", "BBB", "NEW"],
    )
    masked = dense.copy()
    masked.iloc[:listing_pos, 2] = np.nan
    factor_px = pd.DataFrame(index=dates)  # macro 컬럼 없음 -> mc_vol_x_mom63만 생성
    data = SimpleNamespace(
        returns=dense,
        returns_masked=masked,
        dates=dates,
        tickers=["AAA", "BBB", "NEW"],
        factor_prices=factor_px,
        has_factor_data=lambda: True,
        get_sheet=_raise_key_error,  # revision 시트 없음 -> eps_rev_cs=None
    )
    return data, dates


def test_vol_x_mom_uses_masked_returns():
    data, dates = _stub()
    features = build_macro_cross_features(data)
    panel = features["mc_vol_x_mom63"]
    # NEW는 실데이터 10일뿐 -> mom63/vol21 미형성 -> 전 구간 NaN이어야 함
    assert panel["NEW"].isna().all()
    # 상시 상장 종목은 63일 이후 정의됨
    assert np.isfinite(panel["AAA"].iloc[-1])
