# -*- coding: utf-8 -*-
"""§S13.38 precheck helpers — plain-function unit tests (no fixtures)."""
import numpy as np
import pandas as pd

from scripts.run_s13_38_option_risk_precheck import (
    cs_spearman,
    simple,
    summarize_ic,
)


def test_simple_strips_bloomberg_suffix():
    assert simple("AAPL US Equity") == "AAPL"
    assert simple("005930 KS Equity") == "005930"
    assert simple("BRK/B US Equity") == "BRK/B"
    assert simple("AAPL") == "AAPL"


def test_cs_spearman_perfect_rank_and_min_pairs():
    idx = [f"t{i}" for i in range(40)]
    a = pd.Series(np.arange(40, dtype=float), index=idx)
    b = pd.Series(np.arange(40, dtype=float) ** 3, index=idx)  # monotone
    assert np.isclose(cs_spearman(a, b), 1.0)
    assert np.isclose(cs_spearman(a, -b), -1.0)
    # MIN_PAIRS(30) 미만이면 NaN
    small = a.iloc[:10]
    assert np.isnan(cs_spearman(small, b.iloc[:10]))


def test_cs_spearman_ignores_nan_pairs():
    idx = [f"t{i}" for i in range(60)]
    a = pd.Series(np.arange(60, dtype=float), index=idx)
    b = a.copy()
    b.iloc[:5] = np.nan
    assert np.isclose(cs_spearman(a, b), 1.0)


def test_summarize_ic_sign_consistency_and_conservative_t():
    ic = pd.Series(0.05, index=pd.RangeIndex(90)) + pd.Series(
        np.random.default_rng(38).normal(0, 0.01, 90)
    )
    out = summarize_ic(ic, horizon=21)
    assert out["sign_consistent"] is True
    assert out["n_dates"] == 90
    # 보수적 t = naive/√(21/5)
    assert np.isclose(out["t_conservative"], out["t_naive"] / np.sqrt(21 / 5), atol=0.01)

    flip = pd.Series(
        np.r_[np.full(45, 0.05), np.full(45, -0.05)], index=pd.RangeIndex(90)
    )
    assert summarize_ic(flip, horizon=21)["sign_consistent"] is False
