# -*- coding: utf-8 -*-
"""§S13.20 사전점검 스크립트 순수 함수 단위테스트 (워크북/pkl I/O 없음)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.preflight_s13_20_eps_loading import (
    daily_innovation, judge_gates, rank_autocorr, rank_ic, window_loadings)


def test_daily_innovation_hand_match():
    bd = pd.bdate_range("2024-01-02", periods=5)
    px = pd.DataFrame({"NDX_FWD_EPS": [100.0, 102.0, 101.0, 103.0, 103.0],
                       "SPX_FWD_EPS": [50.0, 50.5, 50.0, 51.0, 51.5]},
                      index=bd)
    innov = daily_innovation(px)
    expected = (np.log(px["NDX_FWD_EPS"]).diff()
                - np.log(px["SPX_FWD_EPS"]).diff())
    np.testing.assert_allclose(innov, expected, rtol=0.0, atol=1e-12)


def test_window_loadings_recovers_known_beta():
    rng = np.random.default_rng(5)
    T = 252
    x = pd.Series(rng.normal(0.0, 1e-3, T))
    y = pd.DataFrame({
        "A": 2.0 * x + rng.normal(0.0, 1e-5, T),   # 강한 양(+) 감응
        "B": -1.0 * x + rng.normal(0.0, 1e-5, T),  # 음(−) 감응
        "C": rng.normal(0.0, 1e-3, T),             # 무감응
    })
    beta, tstat = window_loadings(y, x, min_obs=200)
    np.testing.assert_allclose(beta["A"], 2.0, atol=0.01)
    np.testing.assert_allclose(beta["B"], -1.0, atol=0.01)
    assert abs(tstat["A"]) > 2.0 and abs(tstat["B"]) > 2.0
    assert abs(tstat["C"]) < 3.0  # 무감응은 극단 t 아님
    # min_obs 미달 → NaN
    y_short = y.copy()
    y_short.loc[y_short.index[:100], "A"] = np.nan
    beta2, tstat2 = window_loadings(y_short, x, min_obs=200)
    assert np.isnan(beta2["A"]) and np.isnan(tstat2["A"])
    assert np.isfinite(beta2["B"])


def test_rank_autocorr_identity_and_lag():
    idx = pd.RangeIndex(6)
    row = np.arange(10, dtype=float)
    loadings = pd.DataFrame([row] * 6, index=idx)
    # 모든 행 동일 → lag-3 순위 자기상관 1.0
    np.testing.assert_allclose(rank_autocorr(loadings, lag=3), 1.0, atol=1e-12)
    # 행마다 순위 반전 → -1.0
    alt = pd.DataFrame([row if i % 2 == 0 else row[::-1] for i in range(6)],
                       index=idx)
    np.testing.assert_allclose(rank_autocorr(alt, lag=1), -1.0, atol=1e-12)


def test_rank_ic_sign_follows_state():
    sig = pd.Series(np.arange(20, dtype=float),
                    index=[f"T{i}" for i in range(20)])
    tgt = sig.copy() / 100.0  # 완전 정렬
    np.testing.assert_allclose(rank_ic(sig, tgt), 1.0, atol=1e-12)
    np.testing.assert_allclose(rank_ic(-sig, tgt), -1.0, atol=1e-12)
    # 유효 표본 부족 → NaN
    assert np.isnan(rank_ic(sig.iloc[:5], tgt.iloc[:5]))


def test_judge_gates_logic():
    g = judge_gates(share_sig=0.25, autocorr=0.7, mean_ic=0.01,
                    sub_ics={"P1": 0.02, "P2": -0.01, "P3": 0.01})
    assert g["G1"] and g["G2"] and g["G3"] and g["verdict"] == "PROCEED"
    # G1 실패
    g = judge_gates(0.15, 0.7, 0.01, {"P1": 0.02, "P2": 0.01, "P3": 0.01})
    assert not g["G1"] and g["verdict"] == "SHELVE"
    # G2 실패
    g = judge_gates(0.25, 0.5, 0.01, {"P1": 0.02, "P2": 0.01, "P3": 0.01})
    assert not g["G2"] and g["verdict"] == "SHELVE"
    # G3 실패: mean IC 음수
    g = judge_gates(0.25, 0.7, -0.01, {"P1": 0.02, "P2": 0.01, "P3": 0.01})
    assert not g["G3"] and g["verdict"] == "SHELVE"
    # G3 실패: 서브기간 양(+)이 1개뿐
    g = judge_gates(0.25, 0.7, 0.01, {"P1": 0.02, "P2": -0.01, "P3": -0.01})
    assert not g["G3"] and g["verdict"] == "SHELVE"
