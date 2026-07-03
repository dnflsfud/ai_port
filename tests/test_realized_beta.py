"""compute_beta: realized regression beta of a return series vs benchmark.

Ground truth: if port = k*bm + noise, beta -> k. Tests the pure helper that
src/backtest.py compute_metrics() uses for the 'realized_beta' diagnostic.
"""
import numpy as np
import pandas as pd

from src.utils import compute_beta


def _series(arr, start="2015-01-01"):
    idx = pd.bdate_range(start, periods=len(arr))
    return pd.Series(arr, index=idx)


def test_beta_recovers_known_slope():
    rng = np.random.default_rng(0)
    bm = _series(rng.normal(0, 0.01, 500))
    port = 1.20 * bm  # exact, no noise
    assert abs(compute_beta(port, bm) - 1.20) < 1e-9


def test_beta_with_noise_is_close():
    rng = np.random.default_rng(1)
    bm = _series(rng.normal(0, 0.01, 2000))
    port = _series(0.90 * bm.values + rng.normal(0, 0.001, 2000))
    assert abs(compute_beta(port, bm) - 0.90) < 0.05


def test_beta_zero_variance_benchmark_is_nan():
    bm = _series(np.zeros(100))
    port = _series(np.ones(100) * 0.01)
    assert np.isnan(compute_beta(port, bm))


def test_beta_aligns_and_dropna():
    bm = _series([0.01, -0.02, np.nan, 0.03, 0.0])
    port = _series([0.02, -0.04, 0.01, 0.06, np.nan])
    # only rows where both finite are used; port=2*bm on those -> 2.0
    assert abs(compute_beta(port, bm) - 2.0) < 1e-9


def test_active_beta_is_beta_minus_one():
    # realized_active_beta = compute_beta(port-bm, bm) is the algebraic identity
    # beta(port,bm) - 1 (same bm in both legs), NOT independent neutrality
    # evidence. Pins the derivative nature flagged in review GAP5.
    rng = np.random.default_rng(2)
    bm = _series(rng.normal(0, 0.01, 800))
    port = _series(1.15 * bm.values + rng.normal(0, 0.002, 800))
    beta = compute_beta(port, bm)
    active_beta = compute_beta(port - bm, bm)
    assert abs(active_beta - (beta - 1.0)) < 1e-9
