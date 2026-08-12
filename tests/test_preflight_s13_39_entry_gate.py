# -*- coding: utf-8 -*-
"""§S13.39 진입 게이트 preflight 헬퍼 — plain-function 단위 테스트."""
import numpy as np
import pandas as pd

from scripts.preflight_s13_39_entry_gate import (
    detect_increases,
    event_spread,
)


def test_detect_increases_band_and_missing_prev():
    prev = pd.Series({"AAA": 0.05, "BBB": 0.02})
    curr = pd.Series({"AAA": 0.052, "BBB": 0.028, "CCC": 0.01})
    # band 0.003: AAA +0.002 미달, BBB +0.008 포함, CCC 신규 +0.010 포함
    out = detect_increases(prev, curr, band=0.003)
    assert out == {"BBB": 0.008, "CCC": 0.01}


def test_detect_increases_ignores_decreases():
    prev = pd.Series({"AAA": 0.05})
    curr = pd.Series({"AAA": 0.01})
    assert detect_increases(prev, curr, band=0.003) == {}


def test_event_spread_separates_groups():
    rng = np.random.default_rng(39)
    n = 200
    extreme = np.r_[np.ones(50, dtype=bool), np.zeros(150, dtype=bool)]
    fwd = np.where(extreme, -0.02, 0.01) + rng.normal(0, 0.005, n)
    df = pd.DataFrame({"fwd": fwd, "extreme": extreme})
    out = event_spread(df)
    assert out["n_extreme"] == 50 and out["n_rest"] == 150
    assert out["mean_diff"] < -0.025  # extreme − rest ≈ −0.03
    assert out["t"] < -10

    # 극단 그룹이 비면 NaN
    empty = event_spread(pd.DataFrame({"fwd": fwd, "extreme": np.zeros(n, bool)}))
    assert np.isnan(empty["t"])
