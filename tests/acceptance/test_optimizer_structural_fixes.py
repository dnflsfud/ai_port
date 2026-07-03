"""Acceptance tests for the 4 structural fixes in src/portfolio_optimizer.py.

Written BEFORE implementation (test-designer). New-behavior tests are expected
to FAIL against the current code; regression-lock tests are expected to PASS.

Idioms: plain pytest functions (no fixtures), synthetic data only, fast.
`caplog` used only to verify warnings. Numeric expectations are computed
independently in-test (LedoitWolf fit reproduced here), never reusing the
implementation's own arithmetic.

------------------------------------------------------------------------------
ACCEPTANCE-CRITERION  <->  TEST MAPPING
------------------------------------------------------------------------------
(B-1) per-name max_weight relaxed up to bm so mega-cap `w>=bm` stops being
      infeasible; warn once when any bm_i > config.max_weight; inert when all
      bm_i <= max_weight.
        - test_max_weight_relaxed_to_bm_restores_feasibility   (feasibility)   [NEW -> fails now]
        - test_max_weight_relaxation_warns                     (warn once)     [NEW -> fails now]
        - test_max_weight_inert_when_bm_below_cap              (inert path)    [LOCK -> passes now]

(B-2) mega_cap_protection_enabled=True AND funding_mode=False (or k<=0) is a
      silent no-op today; add a WARNING. Constraint/weight behavior unchanged.
        - test_megacap_noop_warns                              (warn / no-warn) [NEW -> fails now]

(B-3) new config flag cov_megacap_vol_shrink_enabled (default True). False
      skips the bm-based D@S@D vol shrinkage; True is bit-identical to today.
        - test_config_cov_shrink_default_on                    (default True)  [NEW -> fails now]
        - test_cov_shrink_off_returns_unshrunk                 (OFF path)      [NEW -> fails now]
        - test_cov_shrink_on_matches_current_formula           (formula lock)  [LOCK -> passes now]

(B-4) dead `bottom_indices` variable removal -- behavior unchanged. No dedicated
      test; the mega-cap funding code path (line ~350-352 comprehension that
      referenced bottom_indices) is exercised, with unchanged results, by
      test_max_weight_relaxed_to_bm_restores_feasibility,
      test_max_weight_inert_when_bm_below_cap and test_megacap_noop_warns.
------------------------------------------------------------------------------
"""

import logging

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src.config import PipelineConfig
from src.portfolio_optimizer import estimate_covariance, optimize_portfolio


# ---------------------------------------------------------------------------
# Shared synthetic builders
# ---------------------------------------------------------------------------
def _shrink_returns_and_bm():
    """60 days x 6 names, seed-fixed, no NaN, name 0 high-vol.

    bm[0] = 0.5 > 2/n (= 0.333) triggers the mega-cap vol-shrink branch for
    name 0 only; all other names have bm = 0.1 < 0.333 (untouched).
    """
    rng = np.random.default_rng(20260702)
    vols = np.array([0.040, 0.012, 0.011, 0.013, 0.010, 0.014])
    data = rng.normal(size=(60, 6)) * vols
    returns = pd.DataFrame(data, columns=[f"S{i}" for i in range(6)])
    bm = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
    return returns, bm


def _ledoitwolf_base(returns: pd.DataFrame) -> np.ndarray:
    """Independent LedoitWolf covariance (the UNSHRUNK reference)."""
    lw = LedoitWolf()
    lw.fit(returns.values)
    return lw.covariance_.copy()


def _shrunk_reference(base: np.ndarray, bm: np.ndarray) -> np.ndarray:
    """Independent reproduction of the CURRENT bm-based D@S@D shrinkage."""
    n = len(bm)
    mean_bm = 1.0 / n
    vols = np.sqrt(np.diag(base))
    avg_vol = vols.mean()
    scale = np.ones(n)
    for i in range(n):
        if bm[i] > mean_bm * 2:
            scale[i] = (0.5 * avg_vol + 0.5 * vols[i]) / vols[i] if vols[i] > 0 else 1.0
    return np.diag(scale) @ base @ np.diag(scale)


def _megacap_funding_config(max_weight: float) -> PipelineConfig:
    """Loose MVO config with concentrated mega-cap funding active.

    portfolio_style='unconstrained' so __post_init__ does NOT overwrite
    max_active_share / max_active_per_stock (verified below in the tests).
    """
    return PipelineConfig(
        portfolio_style="unconstrained",
        max_weight=max_weight,
        max_active_per_stock=0.80,
        max_active_share=2.0,
        max_single_turnover=2.0,
        max_te_annual=1.0,
        bm_weight_floor=0.0,
        sector_deviation=1.0,
        enforce_score_gated_ow=True,
        mega_cap_protection_enabled=True,
        mega_cap_funding_mode=True,
        mega_cap_funding_k=1,
        mega_cap_bm_threshold=0.04,
    )


def _noop_config(funding_mode: bool) -> PipelineConfig:
    """Mega-cap protection ON; funding_mode toggled to hit the no-op branch."""
    return PipelineConfig(
        portfolio_style="unconstrained",
        max_weight=0.30,
        max_active_per_stock=0.80,
        max_active_share=2.0,
        max_single_turnover=2.0,
        max_te_annual=1.0,
        bm_weight_floor=0.0,
        sector_deviation=1.0,
        enforce_score_gated_ow=False,
        mega_cap_protection_enabled=True,
        mega_cap_funding_mode=funding_mode,
        mega_cap_funding_k=1,
        mega_cap_bm_threshold=0.04,
    )


# ---------------------------------------------------------------------------
# (B-3) covariance vol-shrink flag
# ---------------------------------------------------------------------------
def test_config_cov_shrink_default_on():
    """[NEW] New flag exists and defaults to True (current behavior preserved)."""
    assert PipelineConfig().cov_megacap_vol_shrink_enabled is True


def test_cov_shrink_off_returns_unshrunk():
    """[NEW] Flag OFF -> plain LedoitWolf cov, no D@S@D bm shrinkage."""
    returns, bm = _shrink_returns_and_bm()
    base = _ledoitwolf_base(returns)
    shrunk = _shrunk_reference(base, bm)

    # Sanity: the shrink is non-trivial here, so OFF vs ON is distinguishable.
    assert not np.allclose(base, shrunk, atol=1e-9)

    cfg_off = PipelineConfig(cov_megacap_vol_shrink_enabled=False)
    cov = estimate_covariance(returns, lookback=60, bm_weights=bm, config=cfg_off)

    assert np.allclose(cov, base, atol=1e-9)
    assert not np.allclose(cov, shrunk, atol=1e-9)


def test_cov_shrink_on_matches_current_formula():
    """[LOCK] Flag ON (default) reproduces the exact current shrink formula."""
    returns, bm = _shrink_returns_and_bm()
    base = _ledoitwolf_base(returns)
    shrunk = _shrunk_reference(base, bm)

    cov = estimate_covariance(returns, lookback=60, bm_weights=bm, config=PipelineConfig())

    assert np.allclose(cov, shrunk, atol=1e-9)


# ---------------------------------------------------------------------------
# (B-1) per-name max_weight relaxed up to bm
# ---------------------------------------------------------------------------
def test_max_weight_relaxed_to_bm_restores_feasibility():
    """[NEW] bm[0]=0.40 > max_weight=0.15 with a forced `w>=bm` mega pin.

    Current code caps w[0] <= 0.15 while the non-funding mega pin forces
    w[0] >= 0.40 -> INFEASIBLE -> bm fallback (used_fallback True). After the
    fix the per-name cap relaxes to bm[0], so the solve is optimal and
    w[0] == 0.40 with NO fallback.
    """
    cfg = _megacap_funding_config(0.15)
    # __post_init__ must not have clamped these under 'unconstrained'.
    assert cfg.max_active_share == 2.0
    assert cfg.max_active_per_stock == 0.80

    tk = [f"T{i}" for i in range(5)]
    bm = np.array([0.40, 0.15, 0.15, 0.15, 0.15])
    mu = pd.Series([1.0, -0.01, -0.02, -0.03, -0.04], index=tk)
    cov = np.eye(5) * 1e-6

    d = {}
    w = optimize_portfolio(mu, cov, prev_weights=bm, bm_weights=bm, config=cfg, diagnostics=d)

    assert d["used_fallback"] is False
    assert w[0] >= 0.40 - 1e-6


def test_max_weight_relaxation_warns(caplog):
    """[NEW] A WARNING is emitted once when any bm_i exceeds config.max_weight."""
    cfg = _megacap_funding_config(0.15)
    tk = [f"T{i}" for i in range(5)]
    bm = np.array([0.40, 0.15, 0.15, 0.15, 0.15])
    mu = pd.Series([1.0, -0.01, -0.02, -0.03, -0.04], index=tk)
    cov = np.eye(5) * 1e-6

    with caplog.at_level(logging.WARNING, logger="src.portfolio_optimizer"):
        optimize_portfolio(mu, cov, prev_weights=bm, bm_weights=bm, config=cfg)

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "expected a WARNING when bm_i > max_weight"
    assert any("weight" in r.getMessage().lower() for r in warnings)


def test_max_weight_inert_when_bm_below_cap(caplog):
    """[LOCK] When all bm_i <= max_weight the relaxation is inert.

    Solution exists (no fallback), every weight stays <= max_weight, and no
    relaxation WARNING is emitted. Holds both before and after the fix.
    """
    cfg = _megacap_funding_config(0.30)
    tk = [f"T{i}" for i in range(5)]
    bm = np.full(5, 0.20)
    mu = pd.Series([1.0, -0.01, -0.02, -0.03, -0.04], index=tk)
    cov = np.eye(5) * 1e-6

    d = {}
    with caplog.at_level(logging.WARNING, logger="src.portfolio_optimizer"):
        w = optimize_portfolio(mu, cov, prev_weights=bm, bm_weights=bm, config=cfg, diagnostics=d)

    assert d["used_fallback"] is False
    assert np.all(w <= cfg.max_weight + 1e-6)
    # No bm-exceeds-cap relaxation warning should fire on the inert path.
    assert not any("max_weight" in r.getMessage().lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# (B-2) mega-cap protection silent no-op warning
# ---------------------------------------------------------------------------
def test_megacap_noop_warns(caplog):
    """[NEW] protection ON + funding_mode OFF -> WARNING; funding ON -> silent."""
    tk = [f"T{i}" for i in range(5)]
    bm = np.full(5, 0.20)
    mu = pd.Series([0.5, -0.1, 0.2, -0.3, 0.1], index=tk)
    cov = np.eye(5) * 1e-4

    # funding_mode OFF: mega protection generates no constraints -> must warn.
    with caplog.at_level(logging.WARNING, logger="src.portfolio_optimizer"):
        optimize_portfolio(mu, cov, prev_weights=bm, bm_weights=bm,
                           config=_noop_config(funding_mode=False))
    off_warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert off_warnings, "expected a WARNING for the mega-cap protection no-op"
    assert any(
        ("mega" in r.getMessage().lower()) or ("protection" in r.getMessage().lower())
        for r in off_warnings
    )

    # funding_mode ON: constraints are generated -> no no-op warning.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="src.portfolio_optimizer"):
        optimize_portfolio(mu, cov, prev_weights=bm, bm_weights=bm,
                           config=_noop_config(funding_mode=True))
    assert not any(
        "mega" in r.getMessage().lower() for r in caplog.records
    ), "no no-op warning expected when funding_mode is active"
