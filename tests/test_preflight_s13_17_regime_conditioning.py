# -*- coding: utf-8 -*-
"""§S13.17 사전점검 스크립트 순수 함수 단위테스트 (워크북/pkl I/O 없음)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.preflight_s13_17_regime_conditioning import (
    STATE_NAMES, bucket_stats, build_market_features, filter_probs, fit_hmm)


def _synthetic_px(n: int = 400, seed: int = 3) -> pd.DataFrame:
    bd = pd.bdate_range("2020-01-02", periods=n)
    rng = np.random.default_rng(seed)
    cols = ["SPX", "NDX", "MXWD", "VIX", "UST_10Y", "UST_3M",
            "F_HiBeta", "F_MinVol", "SPX_FWD_EPS", "NDX_FWD_EPS",
            "MXWD_FWD_EPS"]
    px = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, (n, len(cols))),
                                 axis=0)),
        index=bd, columns=cols)
    px["UST_10Y"] = 3.0 + rng.normal(0.0, 0.1, n).cumsum() * 0.01
    px["UST_3M"] = 2.5 + rng.normal(0.0, 0.1, n).cumsum() * 0.01
    return px


def test_build_market_features_contract():
    feats = build_market_features(_synthetic_px())
    assert list(feats.columns) == ["mxwd_ret21", "spx_rv21", "vix_log",
                                   "vix_chg21", "ust_slope", "risk_appetite",
                                   "eps_g63", "eps_us_lead63",
                                   "eps_us_lead252", "eps_tech_lead63"]
    clean = feats.dropna()
    # 252BD 슬로우-스프레드 창이 워밍업 바인딩 (regime_v2와 동일)
    assert clean.index[0] == feats.index[252]
    assert np.isfinite(clean.to_numpy(dtype=np.float64)).all()


def test_fit_and_filter_probs_are_valid_simplex():
    rng = np.random.default_rng(11)
    # 3개 분리 상태(느린 전이)의 2-D 합성 데이터
    means = np.array([[-2.0, -1.0], [0.0, 0.0], [2.0, 1.5]])
    states = np.repeat([0, 1, 2, 1, 0, 2], 120)
    X = means[states] + rng.normal(0.0, 0.4, (len(states), 2))
    params = fit_hmm(X, warm=None, seed=42)
    probs = filter_probs(params, X)
    assert probs.shape == (len(X), 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=0.0, atol=1e-9)
    assert (probs >= 0.0).all()
    # 트렁케이션 불변(필터드 = 미래 비의존)
    part = filter_probs(params, X[:200])
    np.testing.assert_allclose(part, probs[:200], rtol=0.0, atol=1e-12)


def test_bucket_stats_hand_match():
    bd = pd.bdate_range("2024-01-02", periods=10)
    active = pd.Series([0.01, -0.02, 0.01, 0.01, -0.03,
                        0.02, 0.00, -0.01, 0.01, -0.02], index=bd)
    modal = pd.Series(["calm"] * 4 + ["stress"] * 3 + ["mid"] * 3, index=bd)
    stats = bucket_stats(active, modal)
    assert list(stats.index) == list(STATE_NAMES)
    assert stats["n_days"].sum() == 10
    np.testing.assert_allclose(stats["share"].sum(), 1.0)
    np.testing.assert_allclose(
        stats.loc["stress", "ann_active"],
        active[modal == "stress"].mean() * 252.0)
    np.testing.assert_allclose(
        stats.loc["calm", "ann_active"],
        active[modal == "calm"].mean() * 252.0)
