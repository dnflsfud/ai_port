# -*- coding: utf-8 -*-
"""build_dashboard_data 벤치마크 가중치 가드 테스트 (legacy 스크립트 최소 방어)."""

import numpy as np
import pandas as pd

import scripts.build_dashboard_data as dash_mod
from scripts.build_dashboard_data import compute_benchmark_weights


def test_benchmark_weights_guard_missing_and_nonpositive_caps():
    dates = pd.bdate_range("2026-01-02", periods=3)
    cap = pd.DataFrame(
        {"AAA": [100.0, np.nan, 100.0], "BBB": [-5.0, 300.0, 0.0]}, index=dates
    )
    # MISSING column must not raise KeyError; non-positive caps must be zeroed.
    out = compute_benchmark_weights(cap, dates, ["AAA", "BBB", "MISSING"])
    assert list(out.columns) == ["AAA", "BBB", "MISSING"]
    assert (out["MISSING"] == 0.0).all()
    # Day 1: BBB cap is negative -> zeroed -> AAA takes full weight.
    assert out.loc[dates[0], "AAA"] == 1.0
    assert out.loc[dates[0], "BBB"] == 0.0
    # Day 2: AAA NaN is ffilled to 100, BBB 300 -> 0.25 / 0.75.
    assert out.loc[dates[1], "AAA"] == 0.25
    assert out.loc[dates[1], "BBB"] == 0.75
    # Day 3: BBB cap 0 -> zeroed.
    assert out.loc[dates[2], "AAA"] == 1.0
    np.testing.assert_allclose(out.sum(axis=1).to_numpy(), 1.0)


def test_module_docstring_points_to_production_export_path():
    assert "export_operating_data" in dash_mod.__doc__
