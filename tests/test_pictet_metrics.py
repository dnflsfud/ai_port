# -*- coding: utf-8 -*-
"""Pictet-convention headline metrics (S13.35 Task 0).

active_share = mean over rebalance dates of sum(|w - w_bm|) / 2  (p.41 footnote
of the Pictet PDF: "sum of all absolute active weights divided by two").
Matches the operating-dashboard definition in analytics.py (total_active_share).

avg_annual_turnover_one_way = 0.5 * two-way L1 (already present — locked here).
"""

import pandas as pd
import pytest

from src.backtest import BacktestResult


def _minimal_result() -> BacktestResult:
    r = BacktestResult()
    idx = pd.bdate_range("2026-01-05", periods=6)
    r.portfolio_returns = pd.Series([0.01, -0.005, 0.002, 0.0, 0.004, -0.001], index=idx)
    r.benchmark_returns = pd.Series([0.008, -0.004, 0.001, 0.001, 0.003, -0.002], index=idx)
    return r


def test_active_share_is_mean_of_rebalance_half_l1():
    r = _minimal_result()
    r.active_share_series = pd.Series(
        {
            pd.Timestamp("2026-01-05"): 0.04,
            pd.Timestamp("2026-01-12"): 0.06,
        }
    )
    m = r.compute_metrics()
    assert m["active_share"] == pytest.approx(0.05)


def test_active_share_defaults_to_zero_when_untracked():
    r = _minimal_result()
    m = r.compute_metrics()
    assert m["active_share"] == 0.0


def test_one_way_turnover_is_half_of_two_way():
    r = _minimal_result()
    r.turnover = pd.Series(
        {
            pd.Timestamp("2026-01-06"): 0.30,
            pd.Timestamp("2026-01-08"): 0.10,
        }
    )
    m = r.compute_metrics()
    assert m["avg_annual_turnover_one_way"] == pytest.approx(
        0.5 * m["avg_annual_turnover"]
    )
