"""factor_neutral: OFF-default, axes exclude growth+momentum, single pre-committed penalty."""
from src.config import PipelineConfig


def test_factor_neutral_off_by_default():
    c = PipelineConfig()
    assert c.factor_neutral_enabled is False
    assert c.factor_neutral_penalty >= 0.0
    # growth/momentum excluded a priori (conflict with growth_tilt/PEAD)
    axes = set(c.factor_neutral_axes)
    assert "growth" not in axes and "momentum" not in axes
    assert len(c.factor_neutral_axes) >= 1


import numpy as np


def test_factor_penalty_disabled_identical():
    from src.portfolio_optimizer import optimize_portfolio
    from src.config import PipelineConfig
    import pandas as pd
    n = 8; tk = [f"T{i}" for i in range(n)]; rng = np.random.default_rng(0)
    mu = pd.Series(rng.normal(0, 0.02, n), index=tk)
    A = rng.normal(0, 0.01, (n, n)); cov = A @ A.T / 252 + np.eye(n) * 1e-4
    bm = np.ones(n) / n; L = rng.normal(0, 1, (n, 2))
    c0 = PipelineConfig(); c0.factor_neutral_enabled = False
    w_none = optimize_portfolio(mu, cov, bm_weights=bm, config=c0)
    w_load = optimize_portfolio(mu, cov, bm_weights=bm, config=c0, factor_loadings=L)
    assert np.allclose(w_none, w_load, atol=1e-9)  # loadings ignored when disabled


def test_factor_penalty_reduces_active_exposure():
    from src.portfolio_optimizer import optimize_portfolio
    from src.config import PipelineConfig
    import pandas as pd
    n = 8; tk = [f"T{i}" for i in range(n)]; rng = np.random.default_rng(2)
    mu = pd.Series(rng.normal(0, 0.02, n), index=tk)
    A = rng.normal(0, 0.01, (n, n)); cov = A @ A.T / 252 + np.eye(n) * 1e-4
    bm = np.ones(n) / n; L = rng.normal(0, 1, (n, 2))
    c_off = PipelineConfig(); c_off.factor_neutral_enabled = False
    c_on = PipelineConfig(); c_on.factor_neutral_enabled = True; c_on.factor_neutral_penalty = 50.0
    w_off = optimize_portfolio(mu, cov, bm_weights=bm, config=c_off, factor_loadings=L)
    w_on = optimize_portfolio(mu, cov, bm_weights=bm, config=c_on, factor_loadings=L)
    e_off = np.abs(L.T @ (w_off - bm)).sum()
    e_on = np.abs(L.T @ (w_on - bm)).sum()
    assert e_on <= e_off + 1e-9


def test_factor_loadings_nonfinite_imputed_no_crash():
    from src.portfolio_optimizer import optimize_portfolio
    from src.config import PipelineConfig
    import pandas as pd
    n = 6; tk = [f"T{i}" for i in range(n)]
    mu = pd.Series(np.zeros(n), index=tk)
    cov = np.eye(n) * 1e-4; bm = np.ones(n) / n
    L = np.full((n, 2), np.nan)  # all non-finite -> impute 0 -> inert
    c = PipelineConfig(); c.factor_neutral_enabled = True; c.factor_neutral_penalty = 10.0
    w = optimize_portfolio(mu, cov, bm_weights=bm, config=c, factor_loadings=L)
    assert np.all(np.isfinite(w))
