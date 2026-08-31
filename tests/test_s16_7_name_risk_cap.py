"""S16.7: per-name active-risk share cap enforcement (default-OFF).

When ON, optimize_portfolio / project_portfolio_weights iteratively tighten
symmetric per-name active bounds (sequential convex re-solve) until every
name's Euler active-risk share a_i*(cov@a)_i / (a@cov@a) is
<= max_name_active_risk_share (+ tol). OFF executes no new code.
"""
import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.portfolio_optimizer import (
    NAME_RISK_CAP_TOL,
    optimize_portfolio,
    project_portfolio_weights,
)

CAP = 0.35


def _cfg(**kw) -> PipelineConfig:
    return PipelineConfig(
        portfolio_style="unconstrained",
        max_weight=0.80,
        max_active_per_stock=0.30,
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
    n = 6
    tickers = [f"T{i}" for i in range(n)]
    ann = np.array([0.75, 0.50, 0.50, 0.20, 0.20, 0.20])
    vols = ann / np.sqrt(252.0)
    corr = np.eye(n)
    for i in (0, 1, 2):
        for j in (0, 1, 2):
            if i != j:
                corr[i, j] = 0.5          # correlated high-vol cluster
    cov = np.outer(vols, vols) * corr
    mu = pd.Series([0.12, 0.06, 0.05, -0.01, -0.02, -0.03], index=tickers)
    bm = np.ones(n) / n
    return mu, cov, bm


def _max_share(w, bm, cov):
    a = np.asarray(w, float) - bm
    var = float(a @ cov @ a)
    assert var > 1e-18
    return float(np.max(a * (cov @ a) / var))


def test_default_off():
    assert PipelineConfig().name_risk_share_cap_enabled is False


def test_off_breaches_and_records_no_diag():
    mu, cov, bm = _problem()
    diag = {}
    w = optimize_portfolio(mu, cov, bm_weights=bm, config=_cfg(), diagnostics=diag)
    assert _max_share(w, bm, cov) > 0.40
    assert not any(k.startswith("name_risk_cap") for k in diag)


def test_on_enforces_cap():
    mu, cov, bm = _problem()
    cfg = _cfg(name_risk_share_cap_enabled=True, max_name_active_risk_share=CAP)
    diag = {}
    w = optimize_portfolio(mu, cov, bm_weights=bm, config=cfg, diagnostics=diag)
    assert _max_share(w, bm, cov) <= CAP + NAME_RISK_CAP_TOL + 1e-8
    assert diag["name_risk_cap_iterations"] >= 1
    assert diag["name_risk_cap_converged"] is True
    assert abs(float(np.sum(w)) - 1.0) < 1e-8


def test_slack_cap_is_inert_bytes():
    mu, cov, bm = _problem()
    w_off = optimize_portfolio(mu, cov, bm_weights=bm, config=_cfg(), diagnostics={})
    cfg = _cfg(name_risk_share_cap_enabled=True, max_name_active_risk_share=0.99)
    diag = {}
    w_on = optimize_portfolio(mu, cov, bm_weights=bm, config=cfg, diagnostics=diag)
    assert np.array_equal(w_off, w_on)          # zero iterations -> byte identical
    assert diag["name_risk_cap_iterations"] == 0


def test_projection_enforces_cap():
    mu, cov, bm = _problem()
    candidate = optimize_portfolio(mu, cov, bm_weights=bm, config=_cfg())
    p_off = project_portfolio_weights(
        candidate_weights=candidate, expected_returns=mu, cov_matrix=cov,
        bm_weights=bm, config=_cfg(),
    )
    assert _max_share(p_off, bm, cov) > 0.40    # feasible candidate kept as-is
    cfg = _cfg(name_risk_share_cap_enabled=True, max_name_active_risk_share=CAP)
    p_on = project_portfolio_weights(
        candidate_weights=candidate, expected_returns=mu, cov_matrix=cov,
        bm_weights=bm, config=cfg,
    )
    assert _max_share(p_on, bm, cov) <= CAP + NAME_RISK_CAP_TOL + 1e-8
