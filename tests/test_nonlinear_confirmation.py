# -*- coding: utf-8 -*-
"""Value semantics for the nonlinear winner-confirmation block."""

import numpy as np
import pandas as pd

from src.features.nonlinear_confirmation import (
    CONFIRMATION_PARENTS,
    build_nonlinear_confirmation_features,
)


def _frame(rows, cols=("A", "B", "C", "D")):
    return pd.DataFrame(
        rows, index=pd.date_range("2024-01-01", periods=len(rows)), columns=list(cols)
    )


def _zs(row):
    return (row - row.mean()) / row.std()


def test_confirmation_is_minimum_of_parent_zscores():
    a = _frame([[1.0, 2.0, 3.0, 4.0]])
    b = _frame([[4.0, 1.0, 3.0, 2.0]])
    feats = build_nonlinear_confirmation_features(
        {"momentum_252d": a, "dist_52w_high": b}
    )
    want = np.minimum(_zs(a.iloc[0]), _zs(b.iloc[0]))
    assert np.allclose(feats["nl_mom_breakout_confirm"].iloc[0], want)


def test_confirmation_propagates_nan_from_either_parent():
    a = _frame([[1.0, np.nan, 3.0, 4.0]])
    b = _frame([[4.0, 1.0, np.nan, 2.0]])
    feats = build_nonlinear_confirmation_features(
        {"momentum_252d": a, "dist_52w_high": b}
    )
    row = feats["nl_mom_breakout_confirm"].iloc[0]
    assert np.isnan(row["B"]) and np.isnan(row["C"])
    assert not np.isnan(row["A"]) and not np.isnan(row["D"])


def test_missing_parent_skips_only_that_confirmation():
    a = _frame([[1.0, 2.0, 3.0, 4.0]])
    b = _frame([[4.0, 1.0, 3.0, 2.0]])
    feats = build_nonlinear_confirmation_features(
        {"momentum_252d": a, "dist_52w_high": b}
    )
    assert set(feats) == {"nl_mom_breakout_confirm"}


def test_all_confirmations_build_when_all_parents_exist():
    parents = {parent for pair in CONFIRMATION_PARENTS.values() for parent in pair}
    rng = np.random.default_rng(42)
    inputs = {parent: _frame(rng.normal(size=(3, 4))) for parent in parents}
    feats = build_nonlinear_confirmation_features(inputs)
    assert set(feats) == set(CONFIRMATION_PARENTS)


class _FakeData:
    def __init__(self, returns):
        self.returns_masked = returns


def test_trend_efficiency_distinguishes_smooth_paths_and_is_signed():
    n = 252
    returns = _frame(
        np.column_stack(
            [
                np.full(n, 0.01),
                np.full(n, -0.01),
                np.where(np.arange(n) % 2 == 0, 0.01, -0.01),
                np.full(n, np.nan),
            ]
        )
    )
    result = build_nonlinear_confirmation_features({}, _FakeData(returns))[
        "nl_trend_efficiency_252"
    ]
    row = result.iloc[-1]
    assert np.isclose(row["A"], 1.0)
    assert np.isclose(row["B"], -1.0)
    assert abs(row["C"]) < 1e-12
    assert np.isnan(row["D"])
    assert result.iloc[100].isna().all()


def test_zero_distance_path_yields_nan_not_infinity():
    returns = _frame(np.zeros((130, 4)))
    result = build_nonlinear_confirmation_features({}, _FakeData(returns))[
        "nl_trend_efficiency_252"
    ]
    assert result.iloc[-1].isna().all()
