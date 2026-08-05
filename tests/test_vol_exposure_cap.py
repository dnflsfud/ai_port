"""S13.27: portfolio-level vol exposure cap (default-OFF).

vols @ w <= (1 + excess) * (vols @ bm), vols = sqrt(diag(cov)).
OFF must leave the optimiser untouched; ON must bind exactly when the
unconstrained solution tilts into high-vol names.
"""
import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.portfolio_optimizer import optimize_portfolio


def _cfg(**kw) -> PipelineConfig:
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
        mega_cap_protection_enabled=False,
        **kw,
    )


def _problem():
    n = 8
    tickers = [f"T{i}" for i in range(n)]
    ann = np.array([0.60] * 4 + [0.20] * 4)          # 4 high-vol, 4 low-vol
    cov = np.diag((ann / np.sqrt(252.0)) ** 2)        # daily variance
    mu = pd.Series(
        [0.10, 0.09, 0.08, 0.07, -0.01, -0.02, -0.03, -0.04], index=tickers
    )                                                 # alpha loves high vol
    bm = np.ones(n) / n
    return mu, cov, bm


def test_off_tilts_into_high_vol():
    mu, cov, bm = _problem()
    w = optimize_portfolio(mu, cov, bm_weights=bm, config=_cfg(), diagnostics={})
    vols = np.sqrt(np.diag(cov))
    assert float(vols @ w) > 1.05 * float(vols @ bm)


def test_on_enforces_cap():
    mu, cov, bm = _problem()
    cfg = _cfg(vol_exposure_cap_enabled=True, vol_exposure_cap_excess=0.05)
    w = optimize_portfolio(mu, cov, bm_weights=bm, config=cfg, diagnostics={})
    vols = np.sqrt(np.diag(cov))
    assert float(vols @ w) <= 1.05 * float(vols @ bm) + 1e-8


def test_slack_cap_is_inert():
    mu, cov, bm = _problem()
    w_off = optimize_portfolio(mu, cov, bm_weights=bm, config=_cfg(), diagnostics={})
    cfg = _cfg(vol_exposure_cap_enabled=True, vol_exposure_cap_excess=10.0)
    w_slack = optimize_portfolio(mu, cov, bm_weights=bm, config=cfg, diagnostics={})
    assert np.allclose(w_off, w_slack, atol=1e-6)
