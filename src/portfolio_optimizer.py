"""
Phase 6: portfolio optimisation.

The core optimizer is long-only benchmark-aware MVO:
    maximize(mu @ w - lambda * risk - tc * turnover)

Hard constraints are reused by both:
- the target optimizer
- the post-smoothing execution projection

That keeps the realised book inside the same feasible region as the target
book without changing any configured parameter values.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src.config import DEFAULT_CONFIG, PipelineConfig

logger = logging.getLogger(__name__)

# M3: condition-number threshold. Covariance matrices above this are ill-
# conditioned enough that cvxpy quad_form solves start producing garbage or
# NaN weights. ~1e8 ≈ 8 digits of precision loss on a 64-bit solve.
_COV_COND_WARN_THRESHOLD = 1e8

# ---------------------------------------------------------------------------
# Backwards-compatible module-level aliases (read from DEFAULT_CONFIG)
# ---------------------------------------------------------------------------
RISK_AVERSION = DEFAULT_CONFIG.risk_aversion
TURNOVER_PENALTY = DEFAULT_CONFIG.turnover_penalty
MAX_TE_ANNUAL = DEFAULT_CONFIG.max_te_annual
MAX_SINGLE_TURNOVER = DEFAULT_CONFIG.max_single_turnover
SECTOR_DEVIATION = DEFAULT_CONFIG.sector_deviation
COV_LOOKBACK = DEFAULT_CONFIG.cov_lookback
BM_WEIGHT_FLOOR = DEFAULT_CONFIG.bm_weight_floor
MAX_ACTIVE_SHARE = DEFAULT_CONFIG.max_active_share
MAX_WEIGHT = DEFAULT_CONFIG.max_weight
MAX_ACTIVE_PER_STOCK = DEFAULT_CONFIG.max_active_per_stock
USE_SCORE_BASED = DEFAULT_CONFIG.use_score_based


def print_optimizer_config(n_tickers: int = 15, config: PipelineConfig = None):
    """Print the effective optimiser constraints."""
    config = config or DEFAULT_CONFIG
    ew = 1.0 / n_tickers
    lines = [
        "+" + "-" * 48 + "+",
        "|       Portfolio Optimizer Constraints        |",
        "+" + "-" * 48 + "+",
        f"| MAX_WEIGHT           : {config.max_weight:>6.1%}            |",
        f"| MAX_ACTIVE_PER_STOCK : +/-{config.max_active_per_stock:>4.1%}         |",
        f"| BM_WEIGHT_FLOOR      : {config.bm_weight_floor:.0%} of BM ({ew * config.bm_weight_floor:.1%}) |",
        f"| MAX_ACTIVE_SHARE     : {config.max_active_share:.0%}               |",
        f"| MAX_TE_ANNUAL        : {config.max_te_annual:.1%}            |",
        f"| SECTOR_DEVIATION     : +/-{config.sector_deviation:.0%}            |",
        f"| RISK_AVERSION        : {config.risk_aversion:<6}           |",
        f"| TURNOVER_PENALTY     : {config.turnover_penalty:<6}           |",
        f"| MAX_SINGLE_TURNOVER  : {config.max_single_turnover:.0%}               |",
        f"| MODE                 : {'SCORE-BASED' if config.use_score_based else 'MVO':<12} |",
        "+" + "-" * 48 + "+",
    ]
    print("\n".join(lines))


def estimate_covariance(
    returns: pd.DataFrame,
    lookback: int = COV_LOOKBACK,
    bm_weights: Optional[np.ndarray] = None,
    config: PipelineConfig = None,
) -> np.ndarray:
    """Estimate covariance via Ledoit-Wolf shrinkage."""
    config = config or DEFAULT_CONFIG
    if lookback == COV_LOOKBACK:
        lookback = config.cov_lookback

    recent = returns.iloc[-lookback:].replace([np.inf, -np.inf], np.nan)
    if len(recent) < 30 or int(recent.count().max()) < 30:
        return np.eye(returns.shape[1]) * 0.04 / 252.0

    if recent.isna().any().any():
        cov = _pairwise_covariance(recent)
    else:
        lw = LedoitWolf()
        lw.fit(recent.values)
        cov = lw.covariance_.copy()

    # Mild mega-cap volatility shrinkage while preserving PSD via D @ S @ D.
    if bm_weights is not None and getattr(config, "cov_megacap_vol_shrink_enabled", True):
        n = len(bm_weights)
        mean_bm = 1.0 / n
        vols = np.sqrt(np.diag(cov))
        avg_vol = vols.mean()
        scale = np.ones(n)
        for i in range(n):
            if bm_weights[i] > mean_bm * 2:
                if vols[i] > 0:
                    shrink_factor = (0.5 * avg_vol + 0.5 * vols[i]) / vols[i]
                else:
                    shrink_factor = 1.0
                scale[i] = shrink_factor
        cov = np.diag(scale) @ cov @ np.diag(scale)

    # M3: warn on ill-conditioned covariance before it reaches cvxpy.
    # Using np.linalg.cond is O(n^3) but covs are 50x50 at most — negligible.
    try:
        cond = float(np.linalg.cond(cov))
    except np.linalg.LinAlgError:
        cond = float("inf")
    if not np.isfinite(cond) or cond > _COV_COND_WARN_THRESHOLD:
        logger.warning(
            "estimate_covariance: ill-conditioned covariance (cond=%.2e, threshold=%.0e). "
            "MVO may produce unstable or fallback weights.",
            cond, _COV_COND_WARN_THRESHOLD,
        )

    return cov


def _pairwise_covariance(recent: pd.DataFrame) -> np.ndarray:
    """Estimate a PSD covariance matrix without imputing missing returns."""
    n = recent.shape[1]
    default_var = 0.04 / 252.0
    cov_df = recent.cov(min_periods=30)
    cov = cov_df.reindex(index=recent.columns, columns=recent.columns).to_numpy(copy=True)

    var = recent.var(axis=0, skipna=True).reindex(recent.columns).values
    finite_var = var[np.isfinite(var) & (var > 0)]
    fallback_var = float(np.median(finite_var)) if len(finite_var) else default_var
    var = np.where(np.isfinite(var) & (var > 0), var, fallback_var)

    cov = np.asarray(cov, dtype=float)
    missing = ~np.isfinite(cov)
    cov[missing] = 0.0
    np.fill_diagonal(cov, var)
    cov = 0.5 * (cov + cov.T)

    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
        floor = max(fallback_var * 1e-4, 1e-10)
        cov = (eigvecs * np.maximum(eigvals, floor)) @ eigvecs.T
        cov = 0.5 * (cov + cov.T)
    except np.linalg.LinAlgError:
        cov = np.diag(var)

    if cov.shape != (n, n) or not np.all(np.isfinite(cov)):
        cov = np.eye(n) * fallback_var
    return cov


def build_sector_constraints(
    tickers: List[str],
    sector_map: Dict[str, str],
    bm_weights: np.ndarray,
) -> Dict[str, List[int]]:
    """Return sector -> position-index mapping."""
    sector_groups: Dict[str, List[int]] = {}
    for i, ticker in enumerate(tickers):
        sector = sector_map.get(ticker, "Unknown")
        sector_groups.setdefault(sector, []).append(i)
    return sector_groups


def score_based_weights(
    expected_returns: pd.Series,
    max_weight: float = MAX_WEIGHT,
    min_weight: float = 0.002,
) -> np.ndarray:
    """Simple softmax-based weighting for score-only mode."""
    scores = expected_returns.values.copy()
    scores_shifted = scores - scores.max()
    exp_scores = np.exp(scores_shifted)
    raw_weights = exp_scores / exp_scores.sum()

    for _ in range(10):
        raw_weights = np.maximum(raw_weights, min_weight)
        raw_weights = np.minimum(raw_weights, max_weight)
        raw_weights = raw_weights / raw_weights.sum()
        if raw_weights.max() <= max_weight + 1e-6 and raw_weights.min() >= min_weight - 1e-6:
            break

    return raw_weights


def _init_diagnostics(
    diagnostics: Optional[Dict[str, Any]],
    *,
    mode: str,
) -> Optional[Dict[str, Any]]:
    """Initialise standard diagnostics fields while preserving caller metadata."""
    if diagnostics is None:
        return None
    diagnostics.update({
        "mode": mode,
        "solver": None,
        "status": None,
        "used_fallback": False,
        "solver_fallback": False,
        "fallback_reason": None,
    })
    return diagnostics


def _solve_problem(
    prob: cp.Problem,
    diagnostics: Optional[Dict[str, Any]] = None,
    allow_scs_fallback: bool = False,
) -> bool:
    """Solve a CVXPY problem with ECOS by default.

    Production keeps the single-ECOS protocol unless config explicitly enables
    SCS on ECOS exceptions. Non-optimal ECOS statuses still return to the
    caller, which maps them to the configured fallback weights.
    """
    last_error = None
    solvers = [(cp.ECOS, "ECOS", 500)]
    if allow_scs_fallback:
        solvers.append((cp.SCS, "SCS", 5000))

    for solver, solver_name, max_iters in solvers:
        try:
            prob.solve(solver=solver, max_iters=max_iters)
            if diagnostics is not None:
                diagnostics["solver"] = solver_name
                diagnostics["status"] = prob.status
            return True
        except cp.SolverError as exc:
            last_error = str(exc)
        except ValueError as exc:
            # cvxpy raises ValueError("Problem data contains NaN or Inf.")
            # when mu/cov has non-finite entries. Treat ONLY that case as a
            # solver failure so the caller falls back to bm_weights — matches
            # the pre-2026-04-21 behaviour where cvxpy propagated the same
            # condition as SolverError. Re-raise any other ValueError so a
            # genuine construction bug is not silently masked as a bm fallback.
            if "NaN or Inf" not in str(exc):
                raise
            last_error = f"invalid-input: {exc}"

        if diagnostics is not None:
            diagnostics["solver"] = solver_name
            diagnostics["status"] = "solver_error"
            diagnostics["fallback_reason"] = last_error or "solver_error"
            if solver_name == "ECOS" and allow_scs_fallback:
                diagnostics["solver_fallback"] = True

    if diagnostics is not None:
        diagnostics["status"] = "solver_error"
        diagnostics["fallback_reason"] = last_error or "solver_error"
    return False


def compute_bm_proportional_active_cap(
    bm_weights: np.ndarray,
    cov_matrix: Optional[np.ndarray],
    config: PipelineConfig,
) -> np.ndarray:
    """Return per-name active-cap multipliers derived from BM weight + vol.

    Multiplier semantics:
      - 1.0 → no change vs symmetric base cap.
      - >1.0 → more active room (mega caps).
      - <1.0 → less active room (high-vol names).

    Returns a flat array of length n with the final multiplicative scale.
    Caller applies: cap_i = base_cap × multiplier_i.
    """
    n = len(bm_weights)
    mult = np.ones(n, dtype=float)

    if not getattr(config, "bm_proportional_cap_enabled", False):
        return mult

    # BM-proportional term
    bm_top = float(np.max(bm_weights)) if n > 0 else 1.0
    if bm_top <= 0:
        bm_term = np.ones(n)
    else:
        top_scale = float(getattr(config, "bm_proportional_cap_bm_scale_at_top", 1.5))
        bm_term = 1.0 + (top_scale - 1.0) * (bm_weights / bm_top)

    # Vol-proportional term (inverse): high-vol → smaller cap
    if cov_matrix is not None and cov_matrix.shape[0] == n:
        vols = np.sqrt(np.clip(np.diag(cov_matrix), 1e-12, None))
        med_vol = float(np.median(vols))
        floor = float(getattr(config, "bm_proportional_cap_vol_scale_floor", 0.5))
        vol_term = np.clip(med_vol / vols, floor, 1.0)
    else:
        vol_term = np.ones(n)

    mult = bm_term * vol_term
    return mult


def _build_mvo_constraints(
    w: cp.Variable,
    expected_returns: pd.Series,
    cov_matrix: np.ndarray,
    prev_weights: np.ndarray,
    sector_map: Optional[Dict[str, str]],
    bm_weights: np.ndarray,
    max_te_annual: float,
    sector_deviation: float,
    config: PipelineConfig,
) -> Tuple[cp.Expression, cp.Expression, List[cp.Constraint]]:
    """Build the shared hard constraints used by optimisation and projection."""
    n = len(expected_returns)
    tickers = list(expected_returns.index)
    raw_mu = np.asarray(expected_returns.values, dtype=float)
    invalid_alpha = ~np.isfinite(raw_mu)
    mu = np.where(invalid_alpha, 0.0, raw_mu)

    active = w - bm_weights
    risk = cp.quad_form(active, cp.psd_wrap(cov_matrix))
    turnover = cp.norm1(w - prev_weights)

    max_daily_te_var = max_te_annual ** 2 / 252.0
    single_turnover_limit = config.max_single_turnover

    # Per-name max_weight: relax the uniform cap up to bm_i. In a cap-weighted
    # BM a name with bm_i > config.max_weight would clash with the mega-cap
    # `w>=bm` pin -> infeasible -> whole rebalance falls back to BM. Relaxing
    # the cap to bm_i restores feasibility; it stays inert (== the uniform cap)
    # whenever all bm_i <= config.max_weight.
    max_weight_per = np.maximum(
        np.full(n, config.max_weight, dtype=float), bm_weights
    )
    bm_exceeds_cap = bm_weights > config.max_weight
    if np.any(bm_exceeds_cap):
        logger.warning(
            "per-name max_weight relaxed up to bm for %d name(s) with "
            "bm_i > max_weight (%.4f); max bm_i=%.4f.",
            int(bm_exceeds_cap.sum()), config.max_weight, float(np.max(bm_weights)),
        )

    # Per-name active bounds (default: symmetric at config.max_active_per_stock).
    base_active_cap = config.max_active_per_stock
    max_ow_per = np.full(n, base_active_cap, dtype=float)
    max_uw_per = np.full(n, base_active_cap, dtype=float)

    # BM-proportional cap infrastructure (OFF by default; see config).
    bm_mult = compute_bm_proportional_active_cap(bm_weights, cov_matrix, config)
    max_ow_per = max_ow_per * bm_mult
    max_uw_per = max_uw_per * bm_mult

    megacap_enabled = getattr(config, "mega_cap_protection_enabled", False)

    if megacap_enabled:
        mega_bm_thr = config.mega_cap_bm_threshold
        wide_uw_cap = config.mega_cap_wide_uw_cap
        funding_mode = getattr(config, "mega_cap_funding_mode", False)
        funding_k = int(getattr(config, "mega_cap_funding_k", 0))
        funding_score_max = float(getattr(config, "mega_cap_funding_score_max", 0.0))

        if not (funding_mode and funding_k > 0):
            # Only the funding-mode branch is wired; without it mega-cap
            # protection generates no constraints (silent no-op). Warn so the
            # inert config is not mistaken for active asymmetric protection.
            logger.warning(
                "mega_cap_protection_enabled=True but funding_mode is inactive "
                "(funding_mode=%s, funding_k=%d) — asymmetric protection is "
                "unimplemented, so no mega-cap constraints are generated.",
                funding_mode, funding_k,
            )

        mega_indices = [
            i for i in range(n)
            if bm_weights[i] >= mega_bm_thr
        ]

        if funding_mode and funding_k > 0 and mega_indices:
            scored = [
                (i, mu[i] if np.isfinite(mu[i]) else 0.0)
                for i in mega_indices
            ]
            eligible = [(i, s) for i, s in scored if s < funding_score_max]
            eligible.sort(key=lambda x: x[1])
            funding_set = {i for i, _ in eligible[:funding_k]}

            for i in mega_indices:
                if i in funding_set:
                    max_uw_per[i] = max(max_uw_per[i], wide_uw_cap)
                    max_ow_per[i] = 0.0
                else:
                    max_uw_per[i] = 0.0

    constraints: List[cp.Constraint] = [
        cp.sum(w) == 1,
        w >= 0,
        w <= max_weight_per,
        risk <= max_daily_te_var,
        turnover <= single_turnover_limit,
    ]

    weight_floor = bm_weights * config.bm_weight_floor
    for i in range(n):
        constraints.append(w[i] >= weight_floor[i])

    for i in np.flatnonzero(invalid_alpha):
        constraints.append(w[i] == bm_weights[i])

    for i in range(n):
        constraints.append(w[i] - bm_weights[i] <= max_ow_per[i])
        constraints.append(bm_weights[i] - w[i] <= max_uw_per[i])

    if getattr(config, "enforce_score_gated_ow", False):
        score_threshold = getattr(config, "score_threshold_for_ow", 0.0)
        for i in range(n):
            score_i = mu[i]
            # Use <= so that score == threshold (e.g. value-trap-gated names
            # zeroed by vtg_scale=0.0) cannot be overweighted. Previously
            # `<` allowed score==0.0 to slip through and MVO could still OW
            # them via the diversification term — defeating the gate's intent.
            if not np.isfinite(score_i) or score_i <= score_threshold:
                constraints.append(w[i] <= bm_weights[i])

    constraints.append(cp.norm1(w - bm_weights) <= config.max_active_share)

    sec_dev = sector_deviation
    if sector_map is not None:
        sector_groups = build_sector_constraints(tickers, sector_map, bm_weights)
        for indices in sector_groups.values():
            if not indices:
                continue
            sector_bm = float(np.sum(bm_weights[indices]))
            sector_w = cp.sum(w[indices])
            constraints.append(sector_w >= sector_bm - sec_dev)
            constraints.append(sector_w <= sector_bm + sec_dev)

    return risk, turnover, constraints


def project_capped_weights(
    candidate_weights: np.ndarray,
    max_weight: float = MAX_WEIGHT,
    fallback_weights: Optional[np.ndarray] = None,
    config: PipelineConfig = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Project weights onto the long-only capped simplex."""
    config = config or DEFAULT_CONFIG
    if max_weight == MAX_WEIGHT:
        max_weight = config.max_weight

    candidate = np.asarray(candidate_weights, dtype=float)
    fallback = np.asarray(
        fallback_weights if fallback_weights is not None else candidate_weights,
        dtype=float,
    ).copy()

    diag = _init_diagnostics(diagnostics, mode="projection_capped")
    w = cp.Variable(len(candidate))
    prob = cp.Problem(
        cp.Minimize(cp.sum_squares(w - candidate)),
        [cp.sum(w) == 1, w >= 0, w <= max_weight],
    )
    if (
        not _solve_problem(
            prob,
            diag,
            allow_scs_fallback=getattr(config, "allow_scs_on_ecos_exception", False),
        )
        or prob.status not in ("optimal", "optimal_inaccurate")
        or w.value is None
    ):
        if diag is not None:
            diag["used_fallback"] = True
            diag["fallback_reason"] = diag.get("fallback_reason") or prob.status or "projection_failed"
        return fallback

    projected = np.asarray(w.value, dtype=float).flatten()
    if not np.all(np.isfinite(projected)):
        if diag is not None:
            diag["used_fallback"] = True
            diag["fallback_reason"] = "non_finite_projection"
        return fallback

    return projected


def project_portfolio_weights(
    candidate_weights: np.ndarray,
    expected_returns: pd.Series,
    cov_matrix: np.ndarray,
    prev_weights: Optional[np.ndarray] = None,
    sector_map: Optional[Dict[str, str]] = None,
    bm_weights: Optional[np.ndarray] = None,
    max_te_annual: float = MAX_TE_ANNUAL,
    sector_deviation: float = SECTOR_DEVIATION,
    config: PipelineConfig = None,
    fallback_weights: Optional[np.ndarray] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Project candidate weights back into the existing MVO hard constraints."""
    config = config or DEFAULT_CONFIG
    if max_te_annual == MAX_TE_ANNUAL:
        max_te_annual = config.max_te_annual
    if sector_deviation == SECTOR_DEVIATION:
        sector_deviation = config.sector_deviation

    candidate = np.asarray(candidate_weights, dtype=float)
    n = len(candidate)
    if bm_weights is None:
        bm_weights = np.ones(n) / n
    bm_weights = np.asarray(bm_weights, dtype=float)
    if prev_weights is None:
        prev_weights = bm_weights.copy()
    else:
        prev_weights = np.asarray(prev_weights, dtype=float)

    fallback = np.asarray(
        fallback_weights if fallback_weights is not None else bm_weights,
        dtype=float,
    ).copy()

    diag = _init_diagnostics(diagnostics, mode="projection_mvo")
    w = cp.Variable(n)
    _, _, constraints = _build_mvo_constraints(
        w=w,
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
        prev_weights=prev_weights,
        sector_map=sector_map,
        bm_weights=bm_weights,
        max_te_annual=max_te_annual,
        sector_deviation=sector_deviation,
        config=config,
    )
    prob = cp.Problem(cp.Minimize(cp.sum_squares(w - candidate)), constraints)
    if (
        not _solve_problem(
            prob,
            diag,
            allow_scs_fallback=getattr(config, "allow_scs_on_ecos_exception", False),
        )
        or prob.status not in ("optimal", "optimal_inaccurate")
        or w.value is None
    ):
        if diag is not None:
            diag["used_fallback"] = True
            diag["fallback_reason"] = diag.get("fallback_reason") or prob.status or "projection_failed"
        return fallback

    projected = np.asarray(w.value, dtype=float).flatten()
    if not np.all(np.isfinite(projected)):
        if diag is not None:
            diag["used_fallback"] = True
            diag["fallback_reason"] = "non_finite_projection"
        return fallback

    return projected


def _factor_penalty_expr(w, bm_weights, factor_loadings, config):
    """Soft penalty on active style exposures: sum_squares(L.T @ (w - bm)).

    Non-finite loadings imputed to 0 (inert). Returns int 0 when disabled or
    no usable loadings, so the objective is bit-identical when OFF.
    """
    if not getattr(config, "factor_neutral_enabled", False) or factor_loadings is None:
        return 0
    L = np.asarray(factor_loadings, dtype=float)
    L = np.where(np.isfinite(L), L, 0.0)        # impute-to-0
    if L.ndim != 2 or L.shape[0] != len(bm_weights) or not np.any(L):
        return 0
    return config.factor_neutral_penalty * cp.sum_squares(L.T @ (w - bm_weights))


def _sector_active_risk_penalty_expr(w, bm_weights, cov_matrix, sector_map,
                                     tickers, config):
    """Soft penalty on sector active-risk concentration (§S11.5 candidate).

    Convex proxy Σ_s (m_s∘a)'C(m_s∘a) for the non-DCP guardrail ratio
    (top-sector share of Euler-decomposed active TE). Returns int 0 when
    disabled or no sector map, so the objective is bit-identical when OFF.
    """
    if not getattr(config, "sector_active_risk_penalty_enabled", False):
        return 0
    if not sector_map:
        return 0
    sectors: Dict[str, np.ndarray] = {}
    for i, ticker in enumerate(tickers):
        sector = sector_map.get(ticker)
        if sector is None:
            continue
        sectors.setdefault(str(sector), np.zeros(len(tickers)))[i] = 1.0
    if not sectors:
        return 0
    active = w - bm_weights
    wrapped = cp.psd_wrap(cov_matrix)
    total = sum(
        cp.quad_form(cp.multiply(mask, active), wrapped)
        for mask in sectors.values()
    )
    return config.sector_active_risk_penalty * total


def optimize_portfolio(
    expected_returns: pd.Series,
    cov_matrix: np.ndarray,
    prev_weights: Optional[np.ndarray] = None,
    sector_map: Optional[Dict[str, str]] = None,
    bm_weights: Optional[np.ndarray] = None,
    risk_aversion: float = RISK_AVERSION,
    turnover_penalty: float = TURNOVER_PENALTY,
    max_te_annual: float = MAX_TE_ANNUAL,
    sector_deviation: float = SECTOR_DEVIATION,
    config: PipelineConfig = None,
    diagnostics: Optional[Dict[str, Any]] = None,
    factor_loadings: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Optimise the target portfolio under the configured hard constraints."""
    config = config or DEFAULT_CONFIG
    if risk_aversion == RISK_AVERSION:
        risk_aversion = config.risk_aversion
    if turnover_penalty == TURNOVER_PENALTY:
        turnover_penalty = config.turnover_penalty
    if max_te_annual == MAX_TE_ANNUAL:
        max_te_annual = config.max_te_annual
    if sector_deviation == SECTOR_DEVIATION:
        sector_deviation = config.sector_deviation

    n = len(expected_returns)
    if bm_weights is None:
        bm_weights = np.ones(n) / n
    bm_weights = np.asarray(bm_weights, dtype=float)
    if prev_weights is None:
        prev_weights = bm_weights.copy()
    else:
        prev_weights = np.asarray(prev_weights, dtype=float)

    diag = _init_diagnostics(
        diagnostics,
        mode="score_based" if config.use_score_based else "mvo",
    )

    if config.use_score_based:
        if diag is not None:
            diag["status"] = "score_based"
        return score_based_weights(expected_returns, max_weight=config.max_weight)

    w = cp.Variable(n)
    raw_mu = np.asarray(expected_returns.values, dtype=float)
    invalid_alpha = ~np.isfinite(raw_mu)
    mu = np.where(invalid_alpha, 0.0, raw_mu)
    if diag is not None:
        diag["missing_alpha_count"] = int(invalid_alpha.sum())
        if invalid_alpha.any():
            diag["missing_alpha_tickers"] = [
                str(t) for t in expected_returns.index[invalid_alpha]
            ]
    ret = mu @ w
    risk, turnover, constraints = _build_mvo_constraints(
        w=w,
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
        prev_weights=prev_weights,
        sector_map=sector_map,
        bm_weights=bm_weights,
        max_te_annual=max_te_annual,
        sector_deviation=sector_deviation,
        config=config,
    )
    factor_pen = _factor_penalty_expr(w, bm_weights, factor_loadings, config)
    sector_risk_pen = _sector_active_risk_penalty_expr(
        w, bm_weights, cov_matrix, sector_map,
        list(expected_returns.index), config,
    )
    objective = cp.Maximize(
        ret - risk_aversion * risk - turnover_penalty * turnover
        - factor_pen - sector_risk_pen
    )

    prob = cp.Problem(objective, constraints)
    if not _solve_problem(
        prob,
        diag,
        allow_scs_fallback=getattr(config, "allow_scs_on_ecos_exception", False),
    ):
        if diag is not None:
            diag["used_fallback"] = True
        return bm_weights.copy()

    if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
        opt_w = np.asarray(w.value, dtype=float).flatten()
        if not np.all(np.isfinite(opt_w)):
            if diag is not None:
                diag["used_fallback"] = True
                diag["fallback_reason"] = "non_finite_solution"
            return bm_weights.copy()
        return opt_w

    if diag is not None:
        diag["used_fallback"] = True
        diag["fallback_reason"] = prob.status or "non_optimal_status"
    return bm_weights.copy()
