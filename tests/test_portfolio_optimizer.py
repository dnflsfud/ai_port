"""Minimal stem tests for src/portfolio_optimizer.

Required by the TDD guard hook so edits to portfolio_optimizer.py are allowed.
The authoritative acceptance coverage for the structural fixes lives in
tests/acceptance/test_optimizer_structural_fixes.py — this file only smoke-checks
importability and the OFF-default parity of the new cov-shrink flag.
"""

import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.portfolio_optimizer import estimate_covariance, optimize_portfolio


def test_module_imports():
    assert callable(estimate_covariance)
    assert callable(optimize_portfolio)


def test_cov_shrink_flag_default_on():
    # New flag defaults to True (current behavior preserved).
    assert PipelineConfig().cov_megacap_vol_shrink_enabled is True


# ---------------------------------------------------------------------------
# Sector active-risk soft penalty (§S11.5 candidate) — default-OFF + parity
# ---------------------------------------------------------------------------
def _toy_inputs():
    # 8 names so bm_i = 0.125 < max_weight 0.15 (no mega-cap w>=bm pin —
    # with the pin active on every name the solution collapses to w == bm
    # and the test is vacuous).
    tickers = [f"T{i}" for i in range(8)]
    # Alpha concentrated in sector A (first 4 names)
    mu = pd.Series([0.004] * 4 + [0.0] * 4, index=tickers)
    cov = np.eye(8) * (0.02 ** 2)
    bm = np.full(8, 1.0 / 8.0)
    sector_map = {t: ("A" if i < 4 else "B") for i, t in enumerate(tickers)}
    return tickers, mu, cov, bm, sector_map


def _sector_block_var(w, bm, cov, sector_map, tickers, sector):
    a = np.asarray(w, dtype=float) - bm
    m = np.array([1.0 if sector_map[t] == sector else 0.0 for t in tickers])
    a_s = a * m
    return float(a_s @ cov @ a_s)


def test_sector_active_risk_penalty_defaults_off():
    c = PipelineConfig()
    assert c.sector_active_risk_penalty_enabled is False
    assert c.sector_active_risk_penalty == 5.0  # single pre-registered weight


def test_sector_penalty_expr_inert_when_disabled():
    from src.portfolio_optimizer import _sector_active_risk_penalty_expr
    import cvxpy as cp

    tickers, mu, cov, bm, sector_map = _toy_inputs()
    w = cp.Variable(len(tickers))
    off = PipelineConfig()
    assert _sector_active_risk_penalty_expr(w, bm, cov, sector_map, tickers, off) == 0
    on = PipelineConfig(sector_active_risk_penalty_enabled=True)
    # No sector map -> inert even when enabled
    assert _sector_active_risk_penalty_expr(w, bm, cov, None, tickers, on) == 0


def test_sector_penalty_reduces_sector_concentration():
    tickers, mu, cov, bm, sector_map = _toy_inputs()

    # turnover_penalty=0 so the daily-scale toy mu is not frozen at w == bm by
    # the trading cost term, and mega-cap protection off because EVERY toy name
    # has bm 12.5% >= 4% threshold (all-UW-forbidden would pin w == bm).
    # Both isolate the sector-risk penalty effect.
    w_off = optimize_portfolio(
        mu, cov, bm_weights=bm, sector_map=sector_map, turnover_penalty=0.0,
        config=PipelineConfig(mega_cap_protection_enabled=False),
    )
    w_on = optimize_portfolio(
        mu, cov, bm_weights=bm, sector_map=sector_map, turnover_penalty=0.0,
        config=PipelineConfig(
            mega_cap_protection_enabled=False,
            sector_active_risk_penalty_enabled=True,
            sector_active_risk_penalty=200.0,  # decisive on the toy scale
        ),
    )

    var_a_off = _sector_block_var(w_off, bm, cov, sector_map, tickers, "A")
    var_a_on = _sector_block_var(w_on, bm, cov, sector_map, tickers, "A")
    assert np.isfinite(w_on).all()
    assert abs(np.sum(w_on) - 1.0) < 1e-6
    # Fixture sanity: the OFF book must carry real sector-A active risk,
    # otherwise the comparison is vacuous solver noise.
    assert var_a_off > 1e-10
    # The penalty must shrink sector A's standalone active variance.
    assert var_a_on < var_a_off * 0.9
