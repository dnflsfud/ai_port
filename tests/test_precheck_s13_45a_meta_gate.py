# -*- coding: utf-8 -*-
"""§S13.45-A 메타 라벨링 게이트 사전점검 — plain-function 단위 테스트."""
import numpy as np
import pandas as pd

from scripts.precheck_s13_43_regime import newey_west_t
from scripts.preflight_s13_30_vol_quality import _fwd_return
from scripts.precheck_s13_45a_meta_gate import (mad_dispersion,
                                                rank_stability,
                                                subperiod_tercile_signs,
                                                tercile_diff,
                                                trailing_ic)


def test_rank_stability_spearman_common_nonnan_only():
    prev = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": np.nan})
    # 공통 비-NaN = A,B,C,D. 동일 순위 → +1, E(NaN)·F(prev 부재)는 무시
    curr = pd.Series({"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4, "E": 9.9, "F": 5.0})
    assert abs(rank_stability(prev, curr) - 1.0) < 1e-12
    # 완전 역순 → -1
    rev = pd.Series({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0})
    assert abs(rank_stability(prev, rev) + 1.0) < 1e-12
    # 공통 종목 < 3 → NaN
    assert np.isnan(rank_stability(pd.Series({"A": 1.0, "B": 2.0}), curr))


def test_trailing_ic_excludes_current_takes_last6():
    idx = pd.bdate_range("2024-01-01", periods=10)
    ic = pd.Series(np.arange(10, dtype=float) / 100.0, index=idx)
    # 당일(idx[8]) 미포함: 직전 6개 = idx[2..7] 값 0.02..0.07 → 평균 0.045
    assert abs(trailing_ic(ic, idx[8]) - 0.045) < 1e-12
    # production 미러: 직전 IC < 2개면 0.0
    assert trailing_ic(ic, idx[1]) == 0.0
    assert trailing_ic(ic, idx[0]) == 0.0
    # 직전 2개(idx[0..1] = 0.00, 0.01) → 평균 0.005
    assert abs(trailing_ic(ic, idx[2]) - 0.005) < 1e-12


def test_fwd_return_window_strictly_after_date():
    idx = pd.bdate_range("2024-01-01", periods=30)
    rets = pd.DataFrame(0.0, index=idx, columns=["A", "B"])
    rets.loc[idx[5], "A"] = -0.50            # 당일 수익은 창 밖
    rets.loc[idx[6]:idx[26], "A"] = 0.01     # t+1..t+21
    fwd = _fwd_return(rets, idx[5], 21)
    assert abs(fwd["A"] - (1.01 ** 21 - 1.0)) < 1e-12
    assert fwd["B"] == 0.0
    # 잔여 일수 < 21 → None
    assert _fwd_return(rets, idx[15], 21) is None


def test_newey_west_lag1_slope_sign_and_null():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 200)
    y = 2.0 * x + rng.normal(0, 0.1, 200)
    coef, t = newey_west_t(y, x[:, None], lag=1)
    assert abs(coef[1] - 2.0) < 0.05
    assert t[1] > 10.0
    y_null = rng.normal(0, 1, 200)
    _, t_null = newey_west_t(y_null, x[:, None], lag=1)
    assert abs(t_null[1]) < 2.0


def test_tercile_diff_and_subperiod_signs():
    stab = pd.Series(np.arange(9, dtype=float))
    y = pd.Series([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    # 상위 tercile(stab 6,7,8)→y 평균 2, 하위(0,1,2)→0 → diff = 2
    assert abs(tercile_diff(stab, y) - 2.0) < 1e-12
    # 역방향이면 음수
    assert abs(tercile_diff(stab, y.iloc[::-1].reset_index(drop=True)) + 2.0) < 1e-12
    # 표본 < 6 → tercile당 2개 미만 → NaN
    assert np.isnan(tercile_diff(stab.head(5), y.head(5)))
    # 3분할: 각 분할 6개, 분할 내 자체 tercile(2개씩)로 diff 계산
    stab18 = pd.Series(np.tile(np.arange(6, dtype=float), 3))
    y18 = pd.Series(np.concatenate([np.arange(6.0), np.arange(6.0),
                                    -np.arange(6.0)]))
    signs = subperiod_tercile_signs(stab18, y18)
    assert len(signs) == 3
    assert signs[0] > 0 and signs[1] > 0 and signs[2] < 0
