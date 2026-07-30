# -*- coding: utf-8 -*-
"""§S13.23 국가-매핑 지수 리비전 사전점검 순수 함수 테스트."""

import numpy as np
import pandas as pd
import pytest

from scripts.preflight_s13_23_index_rev_mapping import (
    EXCHANGE_TO_REV,
    build_mapped_panel,
    daily_spearman_ic,
    forward_returns,
    judge_gates,
    map_universe,
)


def test_map_universe_covers_known_and_raises_on_unknown():
    codes = pd.Series(["US", "FP", "GR", "JP", "NA", "SM", "LN", "SW", "DC", "KS"],
                      index=[f"T{i}" for i in range(10)])
    mapped = map_universe(codes)
    assert mapped.loc["T0"] == "SPX_REV"
    assert mapped.loc["T1"] == "CAC_REV"
    assert mapped.loc["T2"] == "DAX_REV"
    assert mapped.loc["T3"] == "JPN_REV"
    assert set(mapped.loc[["T4", "T5", "T6", "T7", "T8"]]) == {"SX5E_REV"}
    assert mapped.loc["T9"] == "SPX_REV"
    with pytest.raises(ValueError):
        map_universe(pd.Series(["XX"], index=["BAD"]))


def test_build_mapped_panel_assigns_country_series():
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    factor_px = pd.DataFrame(
        {"SPX_REV": [1.0, 2.0, 3.0], "CAC_REV": [10.0, 20.0, 30.0]}, index=dates
    )
    mapping = pd.Series({"AAA": "SPX_REV", "BBB": "CAC_REV"})
    panel = build_mapped_panel(factor_px, mapping, dates, ["AAA", "BBB"])
    assert list(panel.columns) == ["AAA", "BBB"]
    assert panel["AAA"].tolist() == [1.0, 2.0, 3.0]
    assert panel["BBB"].tolist() == [10.0, 20.0, 30.0]


def test_forward_returns_sums_next_horizon_days():
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    r = pd.DataFrame({"A": [0.01, 0.02, 0.03, 0.04, 0.05]}, index=dates)
    fwd = forward_returns(r, horizon=2)
    assert fwd.loc[dates[0], "A"] == pytest.approx(0.02 + 0.03)
    assert fwd.loc[dates[2], "A"] == pytest.approx(0.04 + 0.05)
    assert np.isnan(fwd.loc[dates[3], "A"]) and np.isnan(fwd.loc[dates[4], "A"])


def test_daily_spearman_ic_monotone_and_min_pairs():
    dates = pd.date_range("2024-01-01", periods=2, freq="B")
    n = 6
    feat = pd.DataFrame([np.arange(n), np.arange(n)], index=dates,
                        columns=[f"T{i}" for i in range(n)], dtype=float)
    fwd = feat.copy()  # 완전 단조 -> IC 1.0
    ic = daily_spearman_ic(feat, fwd, min_pairs=4)
    assert ic.loc[dates[0]] == pytest.approx(1.0)
    # 유효쌍 부족 날짜는 제외
    fwd2 = fwd.copy()
    fwd2.iloc[1, :4] = np.nan
    ic2 = daily_spearman_ic(feat, fwd2, min_pairs=4)
    assert dates[1] not in ic2.index


def test_judge_gates_thresholds():
    g = judge_gates(nonanchor_share=0.23, mean_ic=0.02, h1=0.01, h2=0.03)
    assert g["G1"] and g["G2"] and g["verdict"] == "PROCEED"
    g_lowvar = judge_gates(nonanchor_share=0.05, mean_ic=0.02, h1=0.01, h2=0.03)
    assert not g_lowvar["G1"] and g_lowvar["verdict"] == "SHELVE"
    g_weak = judge_gates(nonanchor_share=0.23, mean_ic=0.005, h1=0.01, h2=0.001)
    assert not g_weak["G2"] and g_weak["verdict"] == "SHELVE"
    g_flip = judge_gates(nonanchor_share=0.23, mean_ic=-0.02, h1=-0.01, h2=-0.03)
    assert g_flip["G2"] and g_flip["verdict"] == "PROCEED"  # 방향 무관
