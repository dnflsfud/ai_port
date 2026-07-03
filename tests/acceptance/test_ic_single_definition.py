"""Acceptance tests for D2 — IC single definition (targets-only convention).

Written BEFORE implementation (test-first). Run from ai_port with PYTHONPATH=. :

    C:/Users/westl/PycharmProjects/pythonProject/venv_vf_new/Scripts/python.exe \
        -m pytest tests/acceptance/test_ic_single_definition.py -v

House-style idioms: plain pytest functions (no fixtures), synthetic data only,
fast, expected values computed INDEPENDENTLY (scipy spearman) and asserted with a
0.5% relative tolerance gate — never reusing the impl's compute_ic. A trivial
optimizer_fn is injected so the MVO objective is bypassed (the tiny 12-name
projection still runs under ECOS but is cheap and cannot corrupt the IC path).

------------------------------------------------------------------------------
SPEC UNDER TEST (decision log D2, option A)
------------------------------------------------------------------------------
src/backtest.py :: simulate_portfolio, "IC computation at rebalance dates" block.

  Current:
      if t_date in targets.index:            realized = targets.loc[t_date, ...]
      elif t_idx + 20 < len(all_dates):      realized = returns[...] 20d fwd sum   <- FALLBACK
      else:                                  realized = None

  After D2 fix:
      if t_date in targets.index:            realized = targets.loc[t_date, ...]
      else:                                  realized = None   (IC skipped)

  i.e. the `elif` raw-20d-forward-sum fallback is REMOVED. IC has a single
  definition (the targets convention). No other branch / signature changes.

------------------------------------------------------------------------------
ACCEPTANCE-CRITERION -> TEST MAPPING
------------------------------------------------------------------------------
(D2-1) IC is computed at rebalance dates that targets covers.
        -> test_ic_computed_when_targets_cover_rebal_date        [PINNED: passes now & after fix]
(D2-2) When targets does NOT cover a rebalance date (but >20 fwd days exist),
       IC is SKIPPED — no silent raw-forward-sum substitution.
        -> test_ic_skipped_when_targets_miss_rebal_date          [NEW BEHAVIOUR: FAILS now, passes after fix]
(D2-3) IC values at targets-covered dates equal the independent spearman of
       (predictions, targets) — the fix must not perturb the retained branch.
        -> test_ic_values_from_targets_unchanged                 [PINNED: passes now & after fix]
(D2-4) targets=None => no IC series at all.
        -> test_no_ic_when_targets_none                          [PINNED: passes now & after fix]

Pre-fix, test D2-2 FAILS for exactly the right reason: the current `elif`
reproduces the dropped date's IC from the returns 20d-forward-sum, so the date
is still present in ic_series (verified: it equals spearman(pred, fwd_sum)).
------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from src.config import PipelineConfig
from src.backtest import simulate_portfolio


# ---------------------------------------------------------------------------
# Shared synthetic scenario
#   * predictions: dense (no NaN) => pred_row.notna() == 12 >= 10 gate met.
#   * targets:     dense, = predictions + small noise => IC well above 0 and
#                  varying per date (0.66..0.92), never NaN (>=3 valid pairs),
#                  no ties (continuous) => spearman is unambiguous.
#   * returns:     dense continuous => the CURRENT fallback (20d fwd sum) yields
#                  a real, non-NaN IC, which is precisely what D2-2 must remove.
# ---------------------------------------------------------------------------
_N_DATES = 60
_N_TICKERS = 12
_FREQ = 10
_DROP_IDX = 20            # interior rebalance date; _DROP_IDX + 20 < _N_DATES
_TICKERS = [f"T{i}" for i in range(_N_TICKERS)]
_DATES = pd.bdate_range("2020-01-01", periods=_N_DATES)


def _build():
    """Deterministic dense predictions/targets/returns keyed by (_DATES, _TICKERS)."""
    rng = np.random.default_rng(20260702)
    preds = pd.DataFrame(
        rng.normal(0.0, 1.0, (_N_DATES, _N_TICKERS)), index=_DATES, columns=_TICKERS
    )
    noise = rng.normal(0.0, 0.5, (_N_DATES, _N_TICKERS))
    targets = pd.DataFrame(preds.values + noise, index=_DATES, columns=_TICKERS)
    returns = pd.DataFrame(
        rng.normal(0.0, 0.01, (_N_DATES, _N_TICKERS)), index=_DATES, columns=_TICKERS
    )
    return preds, returns, targets


def _passthrough_optimizer(pred_row, hist_returns, prev_w, sector_map, bm_w, diagnostics=None):
    """Trivial optimizer: return the benchmark weights, bypassing the MVO solve."""
    return bm_w


def _run(preds, returns, targets):
    """Invoke simulate_portfolio with defaults + the trivial optimizer/EW benchmark."""
    return simulate_portfolio(
        predictions=preds,
        returns=returns,
        tickers=_TICKERS,
        all_dates=_DATES,
        rebalance_freq=_FREQ,
        optimizer_fn=_passthrough_optimizer,
        targets=targets,
        track_ic=True,
        config=PipelineConfig(),
    )


def _rebal_dates():
    """Rebalance dates derived INDEPENDENTLY of the impl.

    Dense predictions from date 0 => simulation start index is 0. The fixed-period
    rule fires on the first bar and then every _FREQ bars: idx in {0,10,20,...}.
    """
    idx = [i for i in range(_N_DATES) if (i == 0 or i % _FREQ == 0)]
    return [_DATES[i] for i in idx], idx


def _independent_ic(preds, targets, date):
    """Independent spearman IC (scipy) — does NOT reuse the impl's compute_ic."""
    return float(
        spearmanr(preds.loc[date, _TICKERS].values, targets.loc[date, _TICKERS].values).correlation
    )


# ---------------------------------------------------------------------------
# (D2-1) IC computed at covered rebalance dates.  [PINNED]
# ---------------------------------------------------------------------------
def test_ic_computed_when_targets_cover_rebal_date():
    """With dense targets covering every rebalance date, ic_series holds exactly
    those rebalance dates."""
    preds, returns, targets = _build()
    rebal_dates, _ = _rebal_dates()

    res = _run(preds, returns, targets)

    # ic_series index must equal the independently-derived rebalance dates.
    assert set(res.ic_series.index) == set(rebal_dates)
    # Sanity: every produced IC is a finite number in [-1, 1].
    assert len(res.ic_series) == len(rebal_dates)
    assert res.ic_series.notna().all()
    assert res.ic_series.between(-1.0, 1.0).all()


# ---------------------------------------------------------------------------
# (D2-2) IC skipped when targets miss a rebalance date.  [NEW BEHAVIOUR]
# Pre-fix this FAILS: the current `elif` reproduces the dropped date's IC from
# the returns 20d-forward-sum, so the date is still present in ic_series.
# ---------------------------------------------------------------------------
def test_ic_skipped_when_targets_miss_rebal_date():
    """Dropping a covered rebalance date's row from targets must remove that date
    from ic_series (single targets-only definition — no forward-sum fallback)."""
    preds, returns, targets = _build()
    rebal_dates, rebal_idx = _rebal_dates()

    # Premise guards (independent reasoning), so the test can only pass for the
    # intended reason: the dropped date is a genuine rebalance date AND enough
    # forward data exists that the CURRENT code would use its 20d-sum fallback.
    assert _DROP_IDX in rebal_idx
    assert _DROP_IDX + 20 < _N_DATES
    drop_date = _DATES[_DROP_IDX]

    # Control run (full targets): the dropped date IS a real IC-producing date.
    control = _run(preds, returns, targets)
    assert drop_date in control.ic_series.index

    # Modified run: remove ONLY that date's target row. Nothing else changes.
    targets_missing = targets.drop(index=drop_date)
    modified = _run(preds, returns, targets_missing)

    # Core assertion (fails pre-fix, passes post-fix): the uncovered rebalance
    # date must be absent from ic_series — IC is skipped, not back-filled.
    assert drop_date not in modified.ic_series.index

    # The skip must be specific to the uncovered date: every OTHER rebalance date
    # is still present with an unchanged IC value.
    other_dates = [d for d in rebal_dates if d != drop_date]
    for d in other_dates:
        assert d in modified.ic_series.index
        assert np.isclose(modified.ic_series.loc[d], control.ic_series.loc[d], atol=1e-12)


# ---------------------------------------------------------------------------
# (D2-3) IC values at covered dates == independent spearman(preds, targets). [PINNED]
# ---------------------------------------------------------------------------
def test_ic_values_from_targets_unchanged():
    """Each targets-covered IC equals the independently-computed spearman, within
    the 0.5% relative tolerance gate (and to floating-point exactness)."""
    preds, returns, targets = _build()

    res = _run(preds, returns, targets)
    assert len(res.ic_series) > 0

    for date in res.ic_series.index:
        got = float(res.ic_series.loc[date])
        expected = _independent_ic(preds, targets, date)
        # Guard our own reference: keep IC comfortably away from 0 so the
        # relative-tolerance gate is meaningful.
        assert abs(expected) > 0.3, (date, expected)
        # 0.5% relative-tolerance gate (D2 numeric requirement).
        assert abs(got - expected) <= 0.005 * abs(expected), (date, got, expected)
        # And, since both compute the identical spearman, exactness too.
        assert np.isclose(got, expected, atol=1e-9), (date, got, expected)


# ---------------------------------------------------------------------------
# (D2-4) targets=None => empty ic_series.  [PINNED]
# ---------------------------------------------------------------------------
def test_no_ic_when_targets_none():
    """No targets => no IC computed at all, regardless of track_ic."""
    preds, returns, _ = _build()

    res = simulate_portfolio(
        predictions=preds,
        returns=returns,
        tickers=_TICKERS,
        all_dates=_DATES,
        rebalance_freq=_FREQ,
        optimizer_fn=_passthrough_optimizer,
        targets=None,
        track_ic=True,
        config=PipelineConfig(),
    )

    assert len(res.ic_series) == 0
