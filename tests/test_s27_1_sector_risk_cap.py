"""§S27.1: per-sector active-risk share cap enforcement (default-OFF).

The production gate (scripts/validate_portfolio_bundles.py sector_active_risk_ok) has
failed every run since 2026-10-02 (Technology 0.916 > max_sector_active_risk_share 0.85)
because the optimiser had no matching constraint. When ON, optimize_portfolio /
project_portfolio_weights extend the §S16.7 sequential convex re-solve: same-sign
contributors of a sector whose Euler share sum_{i in s} a_i*(cov@a)_i / (a@cov@a)
exceeds the cap get their symmetric active bounds shrunk. OFF executes no new code.
"""
import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.portfolio_optimizer import (
    NAME_RISK_CAP_TOL,
    SECTOR_RISK_CAP_TOL,
    optimize_portfolio,
    project_portfolio_weights,
)

SECTOR_CAP = 0.85
SECTORS = {"T0": "Tech", "T1": "Tech", "T2": "Tech", "T3": "Fin", "T4": "Util", "T5": "Health"}


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


def _problem(scale_invariant=False):
    """Tech cluster (T0-T2, correlated, high vol) carries the alpha. Default: a
    non-Tech name has positive alpha too, so capital can be re-allocated out of
    Tech. ``scale_invariant``: every non-Tech name is a pure funding short at the
    long-only floor, so shrinking Tech shrinks the funding pro rata and the
    share ratio cannot move (the disclosed §S16.7 failure mode)."""
    n = 6
    tickers = [f"T{i}" for i in range(n)]
    other_vol = 0.20 if scale_invariant else 0.25
    ann = np.array([0.45, 0.40, 0.40, other_vol, other_vol, other_vol])
    vols = ann / np.sqrt(252.0)
    corr = np.eye(n)
    for i in (0, 1, 2):
        for j in (0, 1, 2):
            if i != j:
                corr[i, j] = 0.6
    cov = np.outer(vols, vols) * corr
    alpha = [0.10, 0.08, 0.07, -0.01, -0.01, -0.01] if scale_invariant else [0.10, 0.08, 0.07, 0.04, -0.01, -0.02]
    mu = pd.Series(alpha, index=tickers)
    bm = np.ones(n) / n
    return mu, cov, bm


def _sector_shares(w, bm, cov):
    a = np.asarray(w, float) - bm
    var = float(a @ cov @ a)
    assert var > 1e-18
    contrib = a * (cov @ a) / var
    out = {}
    for i, t in enumerate(sorted(SECTORS)):
        out[SECTORS[t]] = out.get(SECTORS[t], 0.0) + float(contrib[i])
    return out


def _max_name_share(w, bm, cov):
    a = np.asarray(w, float) - bm
    return float(np.max(a * (cov @ a) / float(a @ cov @ a)))


def test_production_variant_pins_s27_1_flip_state():
    """§8/§S27.1: 사용자 승인 flip(2026-11-03) 이후 production 상태 핀 — sector_risk_share_cap_enabled=True
    (새 S0′ 1.7502, 'OM-PIT' 1.7087 은퇴), 기본값은 False 유지."""
    import yaml

    with open("variants/codex_causal_rank_65.yaml", encoding="utf-8") as fh:
        overrides = yaml.safe_load(fh).get("overrides") or {}
    assert overrides.get("sector_risk_share_cap_enabled") is True
    assert PipelineConfig().sector_risk_share_cap_enabled is False


def test_default_off_and_tolerance_matches_the_gate():
    assert PipelineConfig().sector_risk_share_cap_enabled is False
    assert PipelineConfig().max_sector_active_risk_share == SECTOR_CAP
    assert SECTOR_RISK_CAP_TOL == 0.0                 # the gate tests the sector share with zero tolerance


def test_off_breaches_and_records_no_diag():
    for scale_invariant, floor in ((False, SECTOR_CAP), (True, 0.90)):
        mu, cov, bm = _problem(scale_invariant)
        diag = {}
        w = optimize_portfolio(mu, cov, sector_map=SECTORS, bm_weights=bm, config=_cfg(), diagnostics=diag)
        assert _sector_shares(w, bm, cov)["Tech"] > floor
        assert not any(k.startswith("sector_risk_cap") for k in diag)


def test_on_enforces_sector_cap():
    mu, cov, bm = _problem()
    diag = {}
    w = optimize_portfolio(mu, cov, sector_map=SECTORS, bm_weights=bm,
                           config=_cfg(sector_risk_share_cap_enabled=True), diagnostics=diag)
    assert max(abs(v) for v in _sector_shares(w, bm, cov).values()) <= SECTOR_CAP + 1e-8
    assert diag["sector_risk_cap_converged"] is True and diag["name_risk_cap_iterations"] >= 1
    assert abs(float(np.sum(w)) - 1.0) < 1e-8


def test_slack_sector_cap_is_inert_bytes():
    mu, cov, bm = _problem()
    w_off = optimize_portfolio(mu, cov, sector_map=SECTORS, bm_weights=bm, config=_cfg(), diagnostics={})
    diag = {}
    w_on = optimize_portfolio(mu, cov, sector_map=SECTORS, bm_weights=bm,
                              config=_cfg(sector_risk_share_cap_enabled=True, max_sector_active_risk_share=5.0),
                              diagnostics=diag)
    assert np.array_equal(w_off, w_on) and diag["name_risk_cap_iterations"] == 0


def test_name_cap_alone_is_unchanged_by_a_slack_sector_cap():
    mu, cov, bm = _problem()
    base = _cfg(name_risk_share_cap_enabled=True, max_name_active_risk_share=0.35)
    w_name = optimize_portfolio(mu, cov, sector_map=SECTORS, bm_weights=bm, config=base, diagnostics={})
    both = _cfg(name_risk_share_cap_enabled=True, max_name_active_risk_share=0.35,
                sector_risk_share_cap_enabled=True, max_sector_active_risk_share=5.0)
    w_both = optimize_portfolio(mu, cov, sector_map=SECTORS, bm_weights=bm, config=both, diagnostics={})
    assert np.array_equal(w_name, w_both)


def test_name_and_sector_caps_hold_together():
    mu, cov, bm = _problem()
    cfg = _cfg(name_risk_share_cap_enabled=True, max_name_active_risk_share=0.35, sector_risk_share_cap_enabled=True)
    diag = {}
    w = optimize_portfolio(mu, cov, sector_map=SECTORS, bm_weights=bm, config=cfg, diagnostics=diag)
    assert _max_name_share(w, bm, cov) <= 0.35 + NAME_RISK_CAP_TOL + 1e-8
    assert max(abs(v) for v in _sector_shares(w, bm, cov).values()) <= SECTOR_CAP + 1e-8
    assert diag["name_risk_cap_converged"] is True and diag["sector_risk_cap_converged"] is True


def test_projection_enforces_sector_cap():
    mu, cov, bm = _problem()
    candidate = optimize_portfolio(mu, cov, sector_map=SECTORS, bm_weights=bm, config=_cfg())
    p_off = project_portfolio_weights(candidate_weights=candidate, expected_returns=mu, cov_matrix=cov,
                                      sector_map=SECTORS, bm_weights=bm, config=_cfg())
    assert _sector_shares(p_off, bm, cov)["Tech"] > SECTOR_CAP  # feasible candidate kept as-is
    p_on = project_portfolio_weights(candidate_weights=candidate, expected_returns=mu, cov_matrix=cov,
                                     sector_map=SECTORS, bm_weights=bm,
                                     config=_cfg(sector_risk_share_cap_enabled=True))
    assert max(abs(v) for v in _sector_shares(p_on, bm, cov).values()) <= SECTOR_CAP + 1e-8


def test_scale_invariant_book_stalls_reports_and_never_falls_back_to_bm():
    mu, cov, bm = _problem(scale_invariant=True)
    w_off = optimize_portfolio(mu, cov, sector_map=SECTORS, bm_weights=bm, config=_cfg())
    diag = {}
    w = optimize_portfolio(mu, cov, sector_map=SECTORS, bm_weights=bm,
                           config=_cfg(sector_risk_share_cap_enabled=True), diagnostics=diag)
    assert diag["sector_risk_cap_converged"] is False                      # reported, not hidden
    assert not diag.get("used_fallback") and not np.allclose(w, bm)        # invariant 5: no silent bm
    assert abs(float(np.sum(w)) - 1.0) < 1e-8 and float(np.min(w)) >= -1e-9
    assert _sector_shares(w, bm, cov)["Tech"] <= _sector_shares(w_off, bm, cov)["Tech"] + 1e-6
