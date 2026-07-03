import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src.config import PipelineConfig
from src.portfolio_optimizer import estimate_covariance, optimize_portfolio


def _loose_config() -> PipelineConfig:
    return PipelineConfig(
        portfolio_style="unconstrained",
        max_weight=0.80,
        max_active_per_stock=0.80,
        max_active_share=1.50,
        max_single_turnover=2.0,
        max_te_annual=1.0,
        bm_weight_floor=0.0,
        sector_deviation=1.0,
        enforce_score_gated_ow=False,
    )


def test_nonfinite_alpha_neutralized_without_full_bm_fallback():
    tickers = [f"T{i}" for i in range(8)]
    mu = pd.Series([np.nan, 0.10, 0.02, 0.01, -0.01, -0.02, -0.03, -0.04], index=tickers)
    cov = np.eye(len(tickers)) * 1e-6
    bm = np.ones(len(tickers)) / len(tickers)
    diag = {}

    weights = optimize_portfolio(mu, cov, bm_weights=bm, config=_loose_config(), diagnostics=diag)

    assert diag["missing_alpha_count"] == 1
    assert diag["missing_alpha_tickers"] == ["T0"]
    assert abs(weights[0] - bm[0]) < 1e-7
    assert not np.allclose(weights, bm, atol=1e-6)
    assert diag["used_fallback"] is False


def test_mega_cap_vol_shrinkage_uses_vol_scale_not_sqrt_scale():
    rng = np.random.default_rng(7)
    returns = pd.DataFrame(
        rng.normal(size=(90, 4)) * np.array([0.040, 0.015, 0.012, 0.010]),
        columns=list("ABCD"),
    )
    bm = np.array([0.70, 0.10, 0.10, 0.10])

    lw = LedoitWolf()
    lw.fit(returns.values)
    base_cov = lw.covariance_.copy()
    vols = np.sqrt(np.diag(base_cov))
    shrink_factor = (0.5 * vols.mean() + 0.5 * vols[0]) / vols[0]

    cov = estimate_covariance(returns, lookback=90, bm_weights=bm, config=PipelineConfig())

    expected_var = base_cov[0, 0] * shrink_factor ** 2
    assert abs(cov[0, 0] - expected_var) < 1e-12


def test_pairwise_covariance_handles_staggered_raw_return_gaps():
    rng = np.random.default_rng(11)
    n = 100
    frame = pd.DataFrame(
        {
            "A": rng.normal(0.0, 0.010, n),
            "B": np.r_[rng.normal(0.0, 0.030, 50), np.full(50, np.nan)],
            "C": np.r_[np.full(50, np.nan), rng.normal(0.0, 0.006, 50)],
        }
    )

    cov = estimate_covariance(frame, lookback=100, bm_weights=None, config=PipelineConfig())

    assert cov.shape == (3, 3)
    assert np.all(np.isfinite(cov))
    assert np.linalg.eigvalsh(cov).min() > -1e-12
    assert np.std(np.diag(cov)) > 1e-8
