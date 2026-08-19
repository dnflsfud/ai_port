# -*- coding: utf-8 -*-
"""§S13.43 옵션 기반 risk on-off 레짐 사전점검 — plain-function 단위 테스트."""
import numpy as np
import pandas as pd

from scripts.precheck_s13_43_regime import (build_composite, expanding_z,
                                            forward_targets, newey_west_t,
                                            separation_test, state_stats, z252)


def test_z252_standardizes_and_expanding_z_burn_in():
    rng = np.random.default_rng(7)
    s = pd.Series(rng.normal(3.0, 2.0, 1500))
    z = z252(s)
    tail = z.iloc[500:]
    assert abs(float(tail.mean())) < 0.15
    assert abs(float(tail.std()) - 1.0) < 0.25
    ez = expanding_z(s, burn_in=252)
    assert ez.iloc[:251].isna().all()
    assert ez.iloc[251:].notna().all()


def test_build_composite_requires_min_components():
    idx = pd.RangeIndex(3)
    comp = pd.DataFrame({
        "a": [1.0, 1.0, np.nan], "b": [1.0, 1.0, np.nan],
        "c": [1.0, 1.0, 1.0], "d": [1.0, np.nan, 1.0],
        "e": [np.nan, np.nan, 1.0], "f": [np.nan, np.nan, np.nan],
    }, index=idx)
    out = build_composite(comp, min_components=4)
    assert float(out.iloc[0]) == 1.0          # 4성분 가용 → 평균
    assert np.isnan(out.iloc[1])              # 3성분 → NaN
    assert np.isnan(out.iloc[2])              # 3성분 → NaN


def test_forward_targets_use_next_day_onward_only():
    idx = pd.bdate_range("2024-01-01", periods=10)
    bm = pd.Series([0.01, -0.02, 0.03, 0.0, 0.01, -0.01, 0.02, 0.0, 0.01, 0.005],
                   index=idx)
    act = pd.Series(0.001, index=idx)
    t1, t2, t3, dd = forward_targets(bm, act, fwd=3)
    # t=0 신호 → 타깃은 t+1..t+3 (−0.02, 0.03, 0.0)
    expected = (1 - 0.02) * (1 + 0.03) * (1 + 0.0) - 1
    assert abs(float(t1.iloc[0]) - expected) < 1e-12
    assert abs(float(t2.iloc[0]) - float(np.std([-0.02, 0.03, 0.0], ddof=1)
                                         * np.sqrt(252))) < 1e-9
    assert abs(float(t3.iloc[0]) - (1.001 ** 3 - 1)) < 1e-12
    # 마지막 fwd일은 미래 부족 → NaN
    assert t1.iloc[-3:].isna().all()
    # 창내 최대낙폭 진단: t=0 창 (−0.02, +0.03, 0.0) → 낙폭 = −0.02
    assert abs(float(dd.iloc[0]) + 0.02) < 1e-12


def test_separation_test_detects_direction():
    rng = np.random.default_rng(11)
    n = 90
    sig = pd.Series(rng.normal(0, 1, n))
    tgt = pd.Series(2.0 * sig.values + rng.normal(0, 0.5, n))
    up = separation_test(sig, tgt, direction=+1)
    assert up["t"] > 2 and up["pass"]
    assert all(s > 0 for s in up["thirds"])
    down = separation_test(sig, tgt, direction=-1)
    assert not down["pass"]


def test_newey_west_t_signal_vs_placebo():
    rng = np.random.default_rng(3)
    n = 1500
    x = rng.normal(0, 1, n)
    placebo = rng.normal(0, 1, n)
    # AR(1) 잔차로 중첩창 자기상관 흉내
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.9 * e[i - 1] + rng.normal(0, 0.3)
    y = 0.8 * x + e
    coef, tstats = newey_west_t(y, np.column_stack([x, placebo]), lag=21)
    assert abs(coef[1] - 0.8) < 0.1
    assert tstats[1] > 5
    assert abs(tstats[2]) < 2


def test_state_stats_share_and_mean_spell():
    state = pd.Series([False, True, True, False, True, True, True, False])
    out = state_stats(state)
    assert abs(out["share"] - 5 / 8) < 1e-12
    assert abs(out["mean_spell"] - 2.5) < 1e-12
    assert out["n_spells"] == 2
