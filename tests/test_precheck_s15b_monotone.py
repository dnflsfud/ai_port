# -*- coding: utf-8 -*-
"""§S15 후보 B 사전점검(monotone) — plain-function 단위 테스트."""
import numpy as np
import pandas as pd

from scripts.precheck_s15b_monotone import qualify_features, window_mean_ics


def test_window_mean_ics_blocks_and_min_dates():
    dates = pd.bdate_range("2020-01-01", periods=130)
    ic = pd.DataFrame({"f0": [0.1] * 130}, index=dates)
    win = window_mean_ics(ic, window=63, min_dates=30)
    # 63+63+4 -> 마지막 4행 블록은 탈락
    assert len(win) == 2
    assert np.allclose(win["f0"], 0.1)


def test_window_mean_ics_nan_column_below_min_dates_is_nan():
    dates = pd.bdate_range("2020-01-01", periods=63)
    vals = np.full(63, np.nan)
    vals[:10] = 0.2                      # 유효 10 < min 30
    ic = pd.DataFrame({"f0": vals, "f1": [0.05] * 63}, index=dates)
    win = window_mean_ics(ic, window=63, min_dates=30)
    assert len(win) == 1
    assert np.isnan(win["f0"].iloc[0])
    assert np.isclose(win["f1"].iloc[0], 0.05)


def test_qualify_features_stable_vs_noisy():
    rng = np.random.default_rng(1)
    n = 30
    win = pd.DataFrame({
        "stable_pos": 0.05 + 0.01 * rng.standard_normal(n),
        "stable_neg": -0.05 + 0.01 * rng.standard_normal(n),
        "noisy": 0.002 * rng.standard_normal(n),
    })
    q = qualify_features(win, consistency_min=0.75, t_min=2.5)
    assert bool(q.loc["stable_pos", "qualified"]) is True
    assert q.loc["stable_pos", "sign"] == 1
    assert bool(q.loc["stable_neg", "qualified"]) is True
    assert q.loc["stable_neg", "sign"] == -1
    assert bool(q.loc["noisy", "qualified"]) is False


def test_qualify_features_too_few_windows_disqualifies():
    win = pd.DataFrame({"f0": [0.1, 0.1, 0.1]})
    q = qualify_features(win)
    assert bool(q.loc["f0", "qualified"]) is False
