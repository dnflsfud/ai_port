# -*- coding: utf-8 -*-
"""§S13.22 캐리 TE-캡 조건화 순수 함수 단위테스트 (I/O 없음)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.carry_te_conditioning import (build_te_cap_multipliers,
                                       expanding_percentile, index_eps_series,
                                       te_multiplier_from_composite)


def _factor_px(n: int = 700, seed: int = 9) -> pd.DataFrame:
    bd = pd.bdate_range("2020-01-02", periods=n)
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.008, (n, 3)), axis=0)),
        index=bd, columns=["SPX_FWD_EPS", "NDX_FWD_EPS", "MXWD_FWD_EPS"])


def test_index_eps_series_matches_index_eps_formulas():
    px = _factor_px()
    g63 = index_eps_series(px, "fac_eps_g63")
    np.testing.assert_allclose(
        g63, np.log(px["MXWD_FWD_EPS"]).diff(63), rtol=0.0, atol=1e-12)
    tech = index_eps_series(px, "fac_eps_tech_lead63")
    np.testing.assert_allclose(
        tech,
        np.log(px["NDX_FWD_EPS"]).diff(63) - np.log(px["SPX_FWD_EPS"]).diff(63),
        rtol=0.0, atol=1e-12)


def test_expanding_percentile_is_pit():
    s = pd.Series([3.0, 1.0, 2.0, 5.0, 4.0])
    pct = expanding_percentile(s)
    # hand-match: t시점 값의 과거(자기 포함) 내 순위 비율
    np.testing.assert_allclose(pct.values, [1.0, 0.5, 2.0 / 3.0, 1.0, 0.8],
                               atol=1e-12)
    # 트렁케이션 불변 (미래 비의존)
    part = expanding_percentile(s.iloc[:3])
    np.testing.assert_allclose(part.values, pct.values[:3], atol=1e-12)


def test_te_multiplier_mapping():
    comp = pd.Series([0.10, 0.50, 0.90, 0.333, 0.334, np.nan])
    mult = te_multiplier_from_composite(comp, kappa=0.25)
    np.testing.assert_allclose(
        mult.values, [0.75, 1.0, 1.25, 0.75, 1.0, 1.0], atol=1e-12)


def test_build_multipliers_warmup_and_composite():
    px = _factor_px()
    cfg = SimpleNamespace(
        carry_te_conditioning_enabled=True,
        carry_te_conditioning_features=["fac_eps_g63", "fac_eps_tech_lead63"],
        carry_te_conditioning_kappa=0.25,
        carry_te_conditioning_min_history=504,
    )
    mult = build_te_cap_multipliers(px, cfg)
    assert mult is not None
    # 워밍업(피처 63d + 히스토리 504) 이전은 1.0
    assert (mult.iloc[:400] == 1.0).all()
    # 이후에는 세 값만 등장
    tail = mult.iloc[600:]
    assert set(np.round(tail.unique(), 6)).issubset({0.75, 1.0, 1.25})
    # 결합 상태 = 두 피처 확장 백분위 평균의 터실 (hand-match 1개 시점)
    g = expanding_percentile(index_eps_series(px, "fac_eps_g63").dropna())
    t2 = expanding_percentile(
        index_eps_series(px, "fac_eps_tech_lead63").dropna())
    t = mult.index[650]
    comp = 0.5 * (g.loc[t] + t2.loc[t])
    expected = 1.25 if comp > 2.0 / 3.0 else (0.75 if comp <= 1.0 / 3.0 else 1.0)
    assert mult.loc[t] == expected


def test_build_multipliers_disabled_returns_none():
    px = _factor_px()
    off = SimpleNamespace(carry_te_conditioning_enabled=False)
    assert build_te_cap_multipliers(px, off) is None
    # factor_prices 부재 → None (무음 실패 대신 명시 로그 후 비활성)
    on = SimpleNamespace(
        carry_te_conditioning_enabled=True,
        carry_te_conditioning_features=["fac_eps_g63"],
        carry_te_conditioning_kappa=0.25,
        carry_te_conditioning_min_history=504,
    )
    assert build_te_cap_multipliers(None, on) is None
