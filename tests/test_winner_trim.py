# -*- coding: utf-8 -*-
"""S13.14 winner-trim protection — unit semantics.

Inline soft penalty in the optimize_portfolio objective (§4.1 precedent):
lambda * sum(mask * pos(prev_w - w)) discourages trimming trailing-252d
top-quintile names unless mu offers enough conviction. Disabled -> the
penalty term is int 0, objective byte-identical (§2.1 parity).
"""

import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.portfolio_optimizer import (
    _winner_trim_penalty_expr,
    compute_winner_mask,
    optimize_portfolio,
)


def test_flag_is_default_off_and_params_are_preregistered():
    cfg = PipelineConfig()
    assert cfg.winner_trim_protection_enabled is False
    assert cfg.winner_trim_lambda == 1.0
    assert cfg.winner_trim_quantile == 0.8


def test_penalty_expr_is_int_zero_when_disabled_or_maskless():
    import cvxpy as cp

    w = cp.Variable(3)
    prev = np.array([0.3, 0.3, 0.4])
    off = PipelineConfig()
    on = PipelineConfig(winner_trim_protection_enabled=True)
    assert _winner_trim_penalty_expr(w, prev, np.array([1.0, 0, 0]), off) == 0
    assert _winner_trim_penalty_expr(w, prev, None, on) == 0
    assert _winner_trim_penalty_expr(w, prev, np.zeros(3), on) == 0
    expr = _winner_trim_penalty_expr(w, prev, np.array([1.0, 0, 0]), on)
    assert expr is not None and not isinstance(expr, int)


def test_compute_winner_mask_top_quintile_of_trailing_cumret():
    rng = np.random.default_rng(1)
    n_days, names = 300, list("ABCDEFGHIJ")
    base = rng.normal(0, 0.01, size=(n_days, 10))
    base[:, 0] += 0.004   # A: strong winner
    base[:, 1] += 0.002   # B: second
    rets = pd.DataFrame(base, columns=names)
    mask = compute_winner_mask(rets, quantile=0.8)
    assert mask is not None and mask.shape == (10,)
    assert mask[0] == 1.0 and mask[1] == 1.0     # top 20% of 10 = 2 names
    assert mask.sum() == 2.0


def test_compute_winner_mask_guards():
    names = list("ABC")
    short = pd.DataFrame(np.zeros((50, 3)), columns=names)
    assert compute_winner_mask(short) is None      # < min_obs rows

    rets = pd.DataFrame(np.random.default_rng(0).normal(0, 0.01, (300, 3)),
                        columns=names)
    rets.iloc[:, 2] = np.nan                       # C: no data -> never a winner
    mask = compute_winner_mask(rets, quantile=0.5)
    assert mask[2] == 0.0


def _toy_problem():
    """30 names so bm_i (3.3%) sits well under the per-name cap and active
    positions are actually expressible (6 names pinned w == bm)."""
    n = 30
    names = [f"T{i}" for i in range(n)]
    mu = pd.Series(0.0, index=names)
    mu.iloc[0] = -1.0                              # model wants to trim T0
    mu.iloc[1] = 2.0                               # ... and buy T1
    cov = np.eye(n) * 1e-4
    bm = np.ones(n) / n
    prev = bm.copy()
    prev[0] += 0.03                                # T0: overweight winner
    prev[1] -= 0.03
    mask = np.zeros(n)
    mask[0] = 1.0                                  # T0 is the protected winner
    return mu, cov, bm, prev, mask


def test_protection_keeps_winner_weight_that_mu_wants_to_trim():
    mu, cov, bm, prev, mask = _toy_problem()
    off = optimize_portfolio(mu, cov, prev_weights=prev, bm_weights=bm,
                             config=PipelineConfig(use_score_based=False))
    on = optimize_portfolio(
        mu, cov, prev_weights=prev, bm_weights=bm,
        config=PipelineConfig(use_score_based=False,
                              winner_trim_protection_enabled=True,
                              winner_trim_lambda=50.0),
        winner_mask=mask,
    )
    assert on[0] > off[0] + 0.01, (on[0], off[0])  # A trimmed less when protected
    assert abs(on.sum() - 1) < 1e-6


def test_parity_mask_argument_is_inert_when_flag_off():
    mu, cov, bm, prev, mask = _toy_problem()
    cfg = PipelineConfig(use_score_based=False)
    a = optimize_portfolio(mu, cov, prev_weights=prev, bm_weights=bm, config=cfg)
    b = optimize_portfolio(mu, cov, prev_weights=prev, bm_weights=bm, config=cfg,
                           winner_mask=mask)
    assert np.allclose(a, b, atol=0)               # byte-identical path
