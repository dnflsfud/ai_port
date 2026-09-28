# -*- coding: utf-8 -*-
"""§S20 Sales_Revision(FY2) 사전점검 — plain-function 단위 테스트."""
import numpy as np
import pandas as pd

from scripts.precheck_s20_fy2_sales_revision import (
    nw_t,
    residualize,
    rowwise_rank_corr,
    thirds,
)


def _frame(values, n_dates=3):
    idx = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    cols = [f"T{i}" for i in range(len(values[0]))]
    return pd.DataFrame(values, index=idx, columns=cols)


def test_rowwise_rank_corr_monotone_rows_and_min_pairs():
    base = list(range(12))
    a = _frame([base, base, base])
    b = _frame([base, base[::-1], base])
    rho = rowwise_rank_corr(a, b)
    assert np.isclose(rho.iloc[0], 1.0)
    assert np.isclose(rho.iloc[1], -1.0)
    # 유효쌍 10개 미만 행은 제외
    b2 = b.copy()
    b2.iloc[2, :5] = np.nan
    rho2 = rowwise_rank_corr(a, b2)
    assert len(rho2) == 2


def test_residualize_is_orthogonal_per_date_and_keeps_nan():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    cols = [f"T{i}" for i in range(30)]
    x = pd.DataFrame(rng.normal(size=(4, 30)), index=idx, columns=cols)
    y = 0.5 * x + pd.DataFrame(rng.normal(size=(4, 30)), index=idx, columns=cols)
    y.iloc[1, 3] = np.nan
    res = residualize(y, [x])
    assert np.isnan(res.iloc[1, 3])
    for i in range(4):
        m = res.iloc[i].notna()
        assert abs(float((res.iloc[i][m] * x.iloc[i][m]).sum())) < 1e-8
        assert abs(float(res.iloc[i][m].mean())) < 1e-8


def test_nw_t_scale_and_short_series():
    rng = np.random.default_rng(3)
    s = pd.Series(0.02 + rng.normal(0.0, 0.1, 2000))
    t = nw_t(s)
    # iid 잡음이면 NW t ≈ mean / (std/√n)
    naive = s.mean() / (s.std(ddof=0) / np.sqrt(len(s)))
    assert abs(t - naive) / abs(naive) < 0.25
    assert np.isnan(nw_t(pd.Series([0.1, 0.2, 0.3])))


def test_thirds_splits_remainder_into_last_block():
    s = pd.Series([1.0] * 3 + [2.0] * 3 + [3.0] * 4)
    assert thirds(s) == [1.0, 2.0, 3.0]
