# -*- coding: utf-8 -*-
"""§S13.50 알파-가중 투영 사전점검 — plain-function 단위 테스트."""
import numpy as np
import pandas as pd

from scripts.precheck_s13_50_alpha_weighted_projection import (
    alpha_weight_vector, trailing_ic_mean)


def test_alpha_weight_vector_is_all_ones_at_kappa_zero():
    mu = np.array([-2.0, 0.0, 0.5, 3.0, np.nan])
    v = alpha_weight_vector(mu, 0.0)
    assert np.allclose(v, np.ones(5), atol=1e-12)


def test_alpha_weight_vector_spans_one_to_one_plus_kappa_and_is_monotone():
    mu = np.array([0.1, -0.4, 2.0, 0.7, -1.5])
    v = alpha_weight_vector(mu, 3.0)
    assert np.isclose(v[np.argmax(mu)], 4.0, atol=1e-12)     # 최대 mu -> 1 + kappa
    assert np.isclose(v[np.argmin(mu)], 1.0, atol=1e-12)     # 최소 mu -> 1
    order = np.argsort(mu)
    assert np.all(np.diff(v[order]) > 0)                     # mu 순으로 단조 증가


def test_alpha_weight_vector_pins_nonfinite_mu_to_one_and_excludes_from_rank():
    mu = np.array([np.nan, -1.0, 0.0, 1.0, np.inf])
    v = alpha_weight_vector(mu, 3.0)
    # 유한 3개만으로 pct_rank = [0, 0.5, 1] -> v = [1, 2.5, 4]
    assert np.allclose(v, [1.0, 1.0, 2.5, 4.0, 1.0], atol=1e-12)


def test_trailing_ic_mean_returns_zero_with_fewer_than_two_prior_values():
    idx = pd.to_datetime(["2020-01-02", "2020-02-03"])
    s = pd.Series([0.05, 0.09], index=idx)
    # t = 두 번째 날짜 -> 이전 값은 1개뿐 -> 0.0 (backtest.py len(ic_values) >= 2)
    assert trailing_ic_mean(s, pd.Timestamp("2020-02-03"), 6) == 0.0
    assert trailing_ic_mean(s, pd.Timestamp("2019-01-01"), 6) == 0.0


def test_trailing_ic_mean_ignores_values_dated_on_or_after_t():
    idx = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    base = pd.Series([0.02, 0.04, 0.06], index=idx)
    t = pd.Timestamp("2020-01-06")
    expected = trailing_ic_mean(base, t, 6)
    assert np.isclose(expected, 0.03, atol=1e-12)
    future = pd.concat([base, pd.Series(
        [9.0, -9.0], index=pd.to_datetime(["2020-01-06", "2020-01-07"]))])
    assert np.isclose(trailing_ic_mean(future, t, 6), expected, atol=1e-12)


def test_trailing_ic_mean_uses_only_the_last_window_values():
    idx = pd.date_range("2020-01-01", periods=9, freq="D")
    s = pd.Series(np.arange(1.0, 10.0), index=idx)      # 1..9
    t = pd.Timestamp("2020-01-10")                      # 전 9개 모두 사용 가능
    # window=6 -> 마지막 6개(4..9) 평균 = 6.5 (앞의 3개는 무시)
    assert np.isclose(trailing_ic_mean(s, t, 6), 6.5, atol=1e-12)
