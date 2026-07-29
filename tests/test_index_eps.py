# -*- coding: utf-8 -*-
"""S13.18 index forward-EPS feature block 단위테스트: 수식 hand-match,
per-date 상수 브로드캐스트, 결측 컬럼 graceful skip, whitelist 게이팅."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

import pytest

from src.features.assembly import apply_core_filter
from src.features.index_eps import (INDEX_EPS_FEATURES,
                                    admitted_index_eps_features,
                                    build_index_eps_features)


def _data(n: int = 300, seed: int = 5):
    dates = pd.bdate_range("2022-01-03", periods=n)
    rng = np.random.default_rng(seed)
    fp = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, (n, 3)), axis=0)),
        index=dates, columns=["SPX_FWD_EPS", "NDX_FWD_EPS", "MXWD_FWD_EPS"])
    return SimpleNamespace(dates=dates, tickers=["AAA", "BBB", "CCC"],
                           factor_prices=fp)


def test_build_hand_match_and_broadcast():
    data = _data()
    feats = build_index_eps_features(data)
    assert set(feats.keys()) == set(INDEX_EPS_FEATURES)
    fp = data.factor_prices
    t = data.dates[280]
    spx = np.log(fp["SPX_FWD_EPS"])
    ndx = np.log(fp["NDX_FWD_EPS"])
    mxwd = np.log(fp["MXWD_FWD_EPS"])
    expected = {
        "fac_eps_g63": mxwd.diff(63).loc[t],
        "fac_eps_us_lead63": (spx.diff(63) - mxwd.diff(63)).loc[t],
        "fac_eps_us_lead252": (spx.diff(252) - mxwd.diff(252)).loc[t],
        "fac_eps_tech_lead63": (ndx.diff(63) - spx.diff(63)).loc[t],
    }
    for name, val in expected.items():
        row = feats[name].loc[t]
        # per-date 상수: 모든 종목에 동일 값 브로드캐스트
        assert row.nunique() == 1
        np.testing.assert_allclose(row.iloc[0], val, rtol=0.0, atol=1e-12)
    # 워밍업 구간은 NaN 보존 (252d 스프레드가 바인딩)
    assert feats["fac_eps_us_lead252"].iloc[100].isna().all()
    assert feats["fac_eps_g63"].iloc[100].notna().all()


def test_missing_columns_graceful_skip():
    data = _data()
    data.factor_prices = data.factor_prices.drop(columns=["NDX_FWD_EPS"])
    assert build_index_eps_features(data) == {}
    data.factor_prices = None
    assert build_index_eps_features(data) == {}


def test_whitelist_gating_off_and_on():
    data = _data()
    feats = build_index_eps_features(data)
    groups = {"Factor": list(feats.keys())}
    # OFF (extra=None): core whitelist가 전부 제거 → 패널 불변 경로
    off = dict(feats)
    off_groups = {k: list(v) for k, v in groups.items()}
    apply_core_filter(off, off_groups, extra_whitelist=None)
    assert not any(n in off for n in INDEX_EPS_FEATURES)
    assert "Factor" not in off_groups or off_groups["Factor"] == []
    # ON: extra_whitelist로 4종 전부 생존
    on = dict(feats)
    on_groups = {k: list(v) for k, v in groups.items()}
    apply_core_filter(on, on_groups, extra_whitelist=set(INDEX_EPS_FEATURES))
    assert all(n in on for n in INDEX_EPS_FEATURES)


def test_admitted_subset_gating():
    # S13.21: flag OFF → 빈 셋 (기존 parity 경로 불변)
    off = SimpleNamespace(index_eps_features_enabled=False)
    assert admitted_index_eps_features(off) == set()
    # ON + subset 미지정(None) → 전체 4종 (S13.18 동작 동일)
    on_all = SimpleNamespace(index_eps_features_enabled=True,
                             index_eps_feature_names=None)
    assert admitted_index_eps_features(on_all) == set(INDEX_EPS_FEATURES)
    # ON + 필드 자체 부재 → 전체 4종 (하위호환)
    legacy = SimpleNamespace(index_eps_features_enabled=True)
    assert admitted_index_eps_features(legacy) == set(INDEX_EPS_FEATURES)
    # ON + 2종 subset → 그 2종만
    on_two = SimpleNamespace(
        index_eps_features_enabled=True,
        index_eps_feature_names=["fac_eps_g63", "fac_eps_tech_lead63"])
    assert admitted_index_eps_features(on_two) == {"fac_eps_g63",
                                                   "fac_eps_tech_lead63"}
    # 미지의 이름 → 무음 inert 금지, ValueError
    bad = SimpleNamespace(index_eps_features_enabled=True,
                          index_eps_feature_names=["fac_eps_typo"])
    with pytest.raises(ValueError):
        admitted_index_eps_features(bad)
