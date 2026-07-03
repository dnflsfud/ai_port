"""Acceptance tests for REDESIGN-K structural fixes (C-1, C-2).

Written BEFORE implementation (test-first). New-behavior tests are EXPECTED to
FAIL until the implementer wires the spec; current-behavior tests are pinned so
they must keep passing. Run from ai_port with PYTHONPATH=. :

    <PY> -m pytest tests/acceptance/test_execution_structural_fixes.py -v

Idioms follow the repo house style: plain pytest functions (no fixtures),
synthetic data only, np.allclose(..., atol=...) gates, expected values computed
independently and hardcoded / inline-reproduced (never reusing the impl logic).

------------------------------------------------------------------------------
ACCEPTANCE-CRITERION -> TEST MAPPING
------------------------------------------------------------------------------
(C-1a) PipelineConfig gains `confidence_spread_scale: float = 0.20`
        -> test_config_defaults
(C-1b) compute_signal_confidence gains `spread_scale: float = 0.20` param;
       spread_score = clip(raw_spread / spread_scale, 0.20, 1.00); default 0.20
       must reproduce current behaviour exactly
        -> test_confidence_default_scale_parity   (default == explicit 0.20)
        -> test_confidence_larger_scale_desaturates (param actually rescales)
(C-1c) documents the current saturation bug: z-score preds => spread term pins
       to 1.0 at scale 0.20, so confidence collapses to the IC term
        -> test_confidence_saturation_documented
(C-2a) PipelineConfig gains `projection_fallback_mode: str = "target"`
        -> test_config_defaults
(C-2b) invalid projection_fallback_mode raises ValueError in __post_init__
        -> test_config_rejects_bad_projection_mode
(C-2c) project_portfolio_weights returns the caller-supplied fallback_weights
       verbatim when the projection is infeasible (contract Unit B must not
       regress; simulate_portfolio's mode branch selects which array is passed)
        -> test_projection_fallback_returns_given_fallback

Not covered here (left to verifier at the simulate_portfolio wiring level):
the actual `config.projection_fallback_mode` -> fallback_weights branch and the
`config.confidence_spread_scale` pass-through inside simulate_portfolio. Test 7
in the spec is intentionally omitted (integration-heavy); the function-level
contract in test_projection_fallback_returns_given_fallback fixes the piece the
wiring depends on.
------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import pytest

from src.config import PipelineConfig
from src.backtest import compute_signal_confidence
from src.portfolio_optimizer import project_portfolio_weights


# ---------------------------------------------------------------------------
# Synthetic z-score signal shared by the confidence tests.
# Cross-sectional z-scores => top-bottom raw spread is always ~3.4, which is the
# whole point of the bug: raw_spread / 0.20 saturates the spread term at 1.0.
# ---------------------------------------------------------------------------
_TICKERS = [f"T{i}" for i in range(15)]
_ZSCORES = np.linspace(-2.0, 2.0, 15)


def _z_series() -> pd.Series:
    return pd.Series(_ZSCORES.copy(), index=_TICKERS)


def _expected_raw_spread(raw_values: np.ndarray) -> float:
    """Independently recompute the top-bottom spread the impl uses.

    Mirrors the spec's spread definition (tail_n = max(3, n // 10); mean of the
    top tail_n minus mean of the bottom tail_n) WITHOUT importing the impl. Used
    only to prove saturation/desaturation math, not to assert equality with the
    impl's internal spread.
    """
    n = len(raw_values)
    tail_n = max(3, n // 10)
    srt = np.sort(raw_values)
    bot_mean = srt[:tail_n].mean()
    top_mean = srt[-tail_n:].mean()
    return float(top_mean - bot_mean)


def _ic_score(ic: float) -> float:
    """Inline reproduction of the IC term: clip((ic + 0.01) / 0.04, 0.20, 1.00)."""
    return float(np.clip((ic + 0.01) / 0.04, 0.20, 1.00))


# A spread of ~3.43 comfortably saturates at scale 0.20 and de-saturates at 3.5.
_RAW_SPREAD = _expected_raw_spread(_ZSCORES)
# Sanity guard on our own reference arithmetic (independent of the impl).
assert _RAW_SPREAD > 3.0
assert _RAW_SPREAD / 0.20 > 1.0        # => scale=0.20 saturates spread term to 1.0
assert _RAW_SPREAD / 3.5 < 1.0         # => scale=3.5 leaves spread term below 1.0


# ---------------------------------------------------------------------------
# (C-1a, C-2a) config defaults
# ---------------------------------------------------------------------------
def test_config_defaults():
    """New fields exist with the spec's parity-preserving defaults."""
    c = PipelineConfig()
    assert c.confidence_spread_scale == 0.20
    assert c.projection_fallback_mode == "target"


# ---------------------------------------------------------------------------
# (C-2b) invalid projection_fallback_mode rejected
# ---------------------------------------------------------------------------
def test_config_rejects_bad_projection_mode():
    """Any value outside {'target','prev'} must raise ValueError in __post_init__."""
    with pytest.raises(ValueError):
        PipelineConfig(projection_fallback_mode="bogus")
    # The two documented modes must be accepted.
    assert PipelineConfig(projection_fallback_mode="target").projection_fallback_mode == "target"
    assert PipelineConfig(projection_fallback_mode="prev").projection_fallback_mode == "prev"


# ---------------------------------------------------------------------------
# (C-1b) default spread_scale reproduces current behaviour exactly
# ---------------------------------------------------------------------------
def test_confidence_default_scale_parity():
    """compute_signal_confidence(...) == compute_signal_confidence(..., spread_scale=0.20).

    Exact parity: passing the default explicitly must not change a single bit.
    """
    p = _z_series()
    r = _z_series()
    for ic in (-0.02, -0.01, 0.0, 0.01, 0.03, 0.10):
        default = compute_signal_confidence(p, r, ic)
        explicit = compute_signal_confidence(p, r, ic, spread_scale=0.20)
        assert default == explicit


# ---------------------------------------------------------------------------
# (C-1c) the saturation bug: at scale 0.20 confidence collapses to the IC term
# ---------------------------------------------------------------------------
def test_confidence_saturation_documented():
    """With z-score inputs and scale 0.20, spread term == 1.0 so confidence == IC term.

    Expected value computed independently (inline IC term), not via the impl:
        spread_score = 1.0  (saturated)
        confidence   = clip(1.0 * ic_score, 0.10, 1.00) == ic_score  (ic_score >= 0.20)
    """
    p = _z_series()
    r = _z_series()
    for ic in (-0.02, -0.01, 0.0, 0.01, 0.03, 0.10):
        got = compute_signal_confidence(p, r, ic)      # default scale (0.20)
        expected = float(np.clip(1.0 * _ic_score(ic), 0.10, 1.00))
        assert np.isclose(got, expected, atol=1e-9), (ic, got, expected)


# ---------------------------------------------------------------------------
# (C-1b) a larger scale de-saturates the spread term => lower confidence
# ---------------------------------------------------------------------------
def test_confidence_larger_scale_desaturates():
    """scale=3.5 pushes the spread term below 1.0, so confidence drops vs default.

    At ic=0.03 the IC term is exactly 1.0, so the default confidence is pinned
    at 1.0 (fully saturated) while the scale=3.5 confidence == raw_spread/3.5 < 1.0.
    We assert both the strict decrease and the independently-computed magnitude.
    """
    p = _z_series()
    r = _z_series()
    ic = 0.03

    default = compute_signal_confidence(p, r, ic)                     # scale 0.20
    larger = compute_signal_confidence(p, r, ic, spread_scale=3.5)    # scale 3.5

    # Independent expected values (no impl reuse).
    expected_default = float(np.clip(1.0 * _ic_score(ic), 0.10, 1.00))
    spread_score_35 = float(np.clip(_RAW_SPREAD / 3.5, 0.20, 1.00))
    expected_larger = float(np.clip(spread_score_35 * _ic_score(ic), 0.10, 1.00))

    assert np.isclose(default, expected_default, atol=1e-9)
    assert np.isclose(larger, expected_larger, atol=1e-6)
    # 0.5% tolerance gate on the numeric target, plus the qualitative claim.
    assert abs(larger - expected_larger) <= 0.005 * abs(expected_larger)
    assert larger < default


# ---------------------------------------------------------------------------
# (C-2c) projection returns the supplied fallback_weights verbatim when infeasible
# ---------------------------------------------------------------------------
def test_projection_fallback_returns_given_fallback():
    """Current fixed contract: an infeasible MVO projection returns fallback_weights.

    Infeasibility is engineered to be a pure-arithmetic constraint conflict that
    survives ANY 'relax max_weight up to bm' change Unit B might make:

        bm_weight_floor = 1.0  =>  w[i] >= bm[i]  for every name
        sum(bm) = 2.0          =>  sum(w) >= 2.0
        but the optimiser also enforces  sum(w) == 1

    The floor-sum (2.0) exceeds the budget (1.0) => the feasible set is empty
    regardless of any per-name upper-bound (max_weight) loosening. The projector
    must therefore return the caller-supplied fallback_weights unchanged.
    """
    tickers = [f"T{i}" for i in range(4)]
    bm = np.array([0.5, 0.5, 0.5, 0.5])          # sum 2.0 -> floor sum 2.0 > 1
    mu = pd.Series([0.02, 0.01, 0.03, 0.015], index=tickers)  # finite -> no NaN pins
    cov = np.eye(4) * 1e-4
    prev = np.array([0.25, 0.25, 0.25, 0.25])
    sentinel = np.array([0.4, 0.3, 0.2, 0.1])    # unique, distinct from bm/prev

    config = PipelineConfig(
        portfolio_style="unconstrained",
        bm_weight_floor=1.0,                     # the infeasibility driver
        enforce_score_gated_ow=False,
        mega_cap_protection_enabled=False,
        max_weight=0.80,
        max_active_per_stock=0.80,
        max_active_share=2.0,
        max_single_turnover=2.0,
        max_te_annual=10.0,
        sector_deviation=1.0,
    )

    diag: dict = {}
    result = project_portfolio_weights(
        candidate_weights=prev,
        expected_returns=mu,
        cov_matrix=cov,
        prev_weights=prev,
        bm_weights=bm,
        config=config,
        fallback_weights=sentinel,
        diagnostics=diag,
    )

    # The projection is infeasible, so the exact fallback array comes back.
    assert np.allclose(result, sentinel, atol=1e-6)
    # And it came back via the fallback path (not a coincidental optimum).
    assert diag.get("used_fallback") is True
