# -*- coding: utf-8 -*-
"""Value semantics of the S13.13 interaction block builder.

The block is products of CS z-scores of already-whitelisted parents, so the
builder must (a) z-score each parent per date before multiplying, (b) let NaN
propagate from either parent, and (c) skip a feature whose parent is absent
instead of raising — assembly calls it unconditionally (S8 idiom).
"""

import numpy as np
import pandas as pd

from src.features.interactions import INTERACTION_PARENTS, build_interaction_features


def _frame(rows, cols=("A", "B", "C", "D")):
    return pd.DataFrame(
        rows, index=pd.date_range("2024-01-01", periods=len(rows)), columns=list(cols)
    )


def _zs(row):
    return (row - row.mean()) / row.std()


def test_product_is_zscore_times_zscore():
    a = _frame([[1.0, 2.0, 3.0, 4.0]])
    b = _frame([[4.0, 1.0, 3.0, 2.0]])
    feats = build_interaction_features({"idio_vol_63d": a, "momentum_252d": b})
    got = feats["ix_vol_mom"].iloc[0]
    want = _zs(a.iloc[0]) * _zs(b.iloc[0])
    assert np.allclose(got, want)


def test_nan_in_either_parent_propagates():
    a = _frame([[1.0, np.nan, 3.0, 4.0]])
    b = _frame([[4.0, 1.0, np.nan, 2.0]])
    feats = build_interaction_features({"idio_vol_63d": a, "momentum_252d": b})
    row = feats["ix_vol_mom"].iloc[0]
    assert np.isnan(row["B"]) and np.isnan(row["C"])
    assert not np.isnan(row["A"]) and not np.isnan(row["D"])


def test_missing_parent_skips_feature_instead_of_raising():
    a = _frame([[1.0, 2.0, 3.0, 4.0]])
    b = _frame([[4.0, 1.0, 3.0, 2.0]])
    feats = build_interaction_features({"idio_vol_63d": a, "momentum_252d": b})
    assert set(feats) == {"ix_vol_mom"}


def test_all_four_products_built_when_all_parents_present():
    parents = {p for pair in INTERACTION_PARENTS.values() for p in pair}
    rng = np.random.default_rng(0)
    inputs = {p: _frame(rng.normal(size=(3, 4))) for p in parents}
    feats = build_interaction_features(inputs)
    assert set(feats) == set(INTERACTION_PARENTS)
    for df in feats.values():
        assert df.shape == (3, 4)


class _FakeData:
    def __init__(self, returns):
        self.returns = returns


def test_mom_consistency_is_share_of_up_days():
    n = 260
    rets = _frame(
        np.column_stack([
            np.full(n, 0.01),                     # A: always up -> 1.0
            np.full(n, -0.01),                    # B: always down -> 0.0
            np.where(np.arange(n) % 2 == 0, 0.01, -0.01),  # C: ~0.5
            np.full(n, np.nan),                   # D: no data -> NaN
        ])
    )
    feats = build_interaction_features({}, data=_FakeData(rets))
    row = feats["mom_consistency_252"].iloc[-1]
    assert np.isclose(row["A"], 1.0)
    assert np.isclose(row["B"], 0.0)
    assert abs(row["C"] - 0.5) < 0.01
    assert np.isnan(row["D"])
    # min_periods 126: too-early rows stay NaN
    assert feats["mom_consistency_252"].iloc[60].isna().all()


def test_mom_consistency_skipped_without_data():
    feats = build_interaction_features({})
    assert "mom_consistency_252" not in feats


def test_constant_cross_section_yields_nan_not_inf():
    a = _frame([[2.0, 2.0, 2.0, 2.0]])
    b = _frame([[4.0, 1.0, 3.0, 2.0]])
    feats = build_interaction_features({"idio_vol_63d": a, "momentum_252d": b})
    assert feats["ix_vol_mom"].iloc[0].isna().all()
