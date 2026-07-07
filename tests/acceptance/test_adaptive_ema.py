"""Acceptance tests for A3 — trailing-IC adaptive EMA arm (test-first).

Written BEFORE implementation, from spec `spec-a3-adaptive-ema.md`. The arm lives
in `scripts/run_adaptive_ema_arm.py` (src/ production code is NOT touched). These
tests pin the CONTRACT of its two pure functions; they must go RED with a
ModuleNotFoundError until that file exists.

Run from ai_port with PYTHONPATH=. :

    C:/Users/westl/PycharmProjects/pythonProject/venv_vf_new/Scripts/python.exe \
        -m pytest tests/acceptance/test_adaptive_ema.py -v

House-style idioms (mirroring tests/test_prediction_ema.py & tests/acceptance/
test_ic_single_definition.py): plain pytest functions (no fixtures), synthetic
data only, fast (<1s). Every expected value is computed INDEPENDENTLY here (plain
numpy arithmetic + an independent reference recursion) — the impl's own logic is
NEVER reused. Numeric asserts carry a 0.5% relative-tolerance gate on top of a
tight exactness check (atol=1e-12), per the quant test contract.

==============================================================================
PINNED CONTRACT  (authoritative — the implementer must satisfy exactly this)
==============================================================================
Module: `scripts.run_adaptive_ema_arm` (importable as a namespace package when
run from ai_port with PYTHONPATH=.; `scripts/` has no __init__.py, `src/` does).

(1) compute_adaptive_alpha(ic_events, dates, m, iqr) -> pd.Series
    * ic_events : pd.Series indexed by the IC event's REALIZATION-COMPLETION date
        (a Timestamp = prediction_date + forward_horizon), value = realized IC
        (float). The caller (main) applies the horizon shift; this function only
        ever sees realization timestamps, so it is CAUSAL BY CONSTRUCTION — it
        cannot peek at prediction dates. Duplicate index timestamps are allowed
        (>1 event realizing on the same day) and averaged. May be empty (but is
        still a DatetimeIndex-typed Series).
    * dates : ordered pd.DatetimeIndex of the dates for which an alpha is wanted
        (the prediction/rebalance calendar; in production == raw_predictions.index).
    * m, iqr : floats — the D0 trailing-IC distribution anchors (median, IQR).
    * returns : pd.Series of alpha_t, indexed EXACTLY by `dates`, dtype float.

    alpha_t = clip( 0.5 + (tIC_t - m) / (2*iqr), 0.25, 0.75 )
      where tIC_t = mean of ic_events whose realization date lies in the trailing
      window [ dates[max(0, i-63)] , dates[i-1] ] inclusive, for t = dates[i].
      TRAILING window length = 63 trading days. Upper bound dates[i-1] is STRICTLY
      before t (causality). i == 0 (or no events in window) => tIC undefined =>
      alpha_t = 0.5 exactly.

(2) apply_adaptive_ema(raw_predictions, alpha_series) -> pd.DataFrame
    * raw_predictions : date x ticker DataFrame (same shape apply_prediction_ema
        consumes; NaNs allowed).
    * alpha_series : pd.Series indexed by raw_predictions.index, alpha per date.
    * returns : date x ticker DataFrame, same recursion as
        src.model_trainer.apply_prediction_ema but with alpha time-varying per row:
            blended_t[c] = alpha_t * raw_t[c] + (1 - alpha_t) * blended_{t-1}[c]
        over the ticker intersection c = (tickers on t) ∩ (tickers on prev row).
      INITIALIZATION / NaN rules IDENTICAL to apply_prediction_ema:
        - all-NaN rows are skipped (prev untouched);
        - the first non-empty row is passed through unchanged (blended = raw; the
          alpha at that date is not applied — no prev exists);
        - tickers absent from the previous row keep their raw value;
        - NaN mask of the output equals the NaN mask of the input.
      Hence with alpha_series ≡ 0.5, the output is BYTE-IDENTICAL to
      apply_prediction_ema(raw_predictions, 0.5).

==============================================================================
ACCEPTANCE-CRITERION -> TEST MAPPING
==============================================================================
(A3-1) alpha formula exact (independent calc); tIC==median => alpha==0.5
        -> test_alpha_formula_matches_independent
        -> test_alpha_is_half_when_tic_equals_median
        -> test_alpha_unclipped_interior_value
(A3-2) clip bounds: extreme IC pins alpha to 0.25 / 0.75
        -> test_alpha_clips_to_bounds
(A3-3a) causal: an event affects alpha only AFTER its realization date, not at
        its (earlier) prediction date
        -> test_event_affects_alpha_only_after_realization
(A3-3b) no lookahead: adding/altering events at or after t leaves alpha_t unchanged
        -> test_future_events_do_not_change_past_alpha
(A3-3c) no events => alpha ≡ 0.5
        -> test_no_events_gives_half
(A3-4) trailing 63d window: an out-of-window (older) event is excluded from tIC
        -> test_old_event_excluded_from_trailing_window
(A3-5) EMA equivalence: alpha≡0.5 == src.model_trainer.apply_prediction_ema(.,0.5)
        -> test_adaptive_ema_equals_prediction_ema_when_half   (frame with NaN)
(A3-6) time-varying recursion reproduced (hand-computed dense case + independent
        reference); first-row initialization matches apply_prediction_ema
        -> test_time_varying_recursion_handcomputed
        -> test_time_varying_matches_independent_reference
(A3-7) NaN handling matches apply_prediction_ema (covered by A3-5's NaN frame +
        an explicit NaN-mask check)
        -> test_nan_mask_preserved_under_adaptive_ema
Plus an end-to-end pin combining formula+window+causality on a rich scenario:
        -> test_full_alpha_series_matches_reference
"""

import numpy as np
import pandas as pd
import pytest

# Direct import of the target module — RED (ModuleNotFoundError) until the
# implementer creates scripts/run_adaptive_ema_arm.py. This is the intended
# test-first failure.
from scripts.run_adaptive_ema_arm import (
    compute_adaptive_alpha,
    apply_adaptive_ema,
)

# apply_prediction_ema is the production recursion the alpha-varying version must
# generalize; imported so A3-5/A3-7 compare against it directly (not a copy).
from src.model_trainer import apply_prediction_ema


# ---------------------------------------------------------------------------
# Independent references (NEVER call the implementation under test)
# ---------------------------------------------------------------------------
_TRAILING_TD = 63  # pinned trailing trading-day window (spec §사전등록)
_REL_GATE = 0.005  # 0.5% relative-tolerance gate (quant test contract)


def _ref_alpha_scalar(tic, m, iqr):
    """The pre-registered functional form, computed with plain arithmetic."""
    a = 0.5 + (tic - m) / (2.0 * iqr)
    return float(min(max(a, 0.25), 0.75))


def _ref_adaptive_alpha(ic_events, dates, m, iqr, window=_TRAILING_TD):
    """Independent alpha_t series: trailing-window mean of realized ICs -> clip."""
    ev_idx = np.asarray(ic_events.index.values, dtype="datetime64[ns]")
    ev_val = np.asarray(ic_events.values, dtype=float)
    out = []
    for i in range(len(dates)):
        if i == 0:
            out.append(0.5)
            continue
        lo = np.datetime64(dates[max(0, i - window)], "ns")
        hi = np.datetime64(dates[i - 1], "ns")          # STRICTLY before dates[i]
        mask = (ev_idx >= lo) & (ev_idx <= hi)
        if mask.sum() == 0:
            out.append(0.5)
            continue
        tic = float(ev_val[mask].mean())
        out.append(_ref_alpha_scalar(tic, m, iqr))
    return pd.Series(out, index=dates, dtype=float)


def _ref_adaptive_ema(raw, alpha_series):
    """Independent frame recursion mirroring apply_prediction_ema with per-date alpha."""
    out = raw.copy()
    prev = None
    for d in out.index:
        cur = out.loc[d].dropna()
        if len(cur) == 0:
            continue
        if prev is not None:
            common = cur.index.intersection(prev.index)
            if len(common) > 0:
                a = float(alpha_series.loc[d])
                blended = a * cur[common] + (1 - a) * prev[common]
                cur.loc[common] = blended
                out.loc[d, common] = blended.values
        prev = cur
    return out


def _assert_alpha_gate(got, expected):
    """Tight exactness AND 0.5% relative gate, elementwise over an alpha vector."""
    got = np.asarray(got, dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert got.shape == expected.shape
    # Exactness (identical arithmetic path expected).
    np.testing.assert_allclose(got, expected, atol=1e-12, rtol=0.0)
    # 0.5% relative-tolerance gate (alpha ∈ [0.25, 0.75] => denominator nonzero).
    assert np.all(np.abs(got - expected) <= _REL_GATE * np.abs(expected)), (
        got, expected
    )


# ---------------------------------------------------------------------------
# Shared synthetic helpers
# ---------------------------------------------------------------------------
def _bdays(n, start="2020-01-01"):
    return pd.bdate_range(start, periods=n)


def _toy_frame(seed=0, n_days=60, n_tk=8):
    """A raw-predictions-like frame: NaN burn-in head, a churn gap, an all-NaN date."""
    rng = np.random.default_rng(seed)
    dates = _bdays(n_days)
    tks = [f"T{i}" for i in range(n_tk)]
    f = pd.DataFrame(rng.normal(size=(n_days, n_tk)), index=dates, columns=tks)
    f.iloc[:10] = np.nan              # burn-in head (like the LGBM walk-forward)
    f.iloc[25:30, 2] = np.nan         # a ticker disappears then reappears
    f.iloc[40] = np.nan               # an all-NaN gap date
    return f


# Anchors near the D0 report values but SYNTHETIC (tests never load the report).
_M = 0.04
_IQR = 0.075


# ===========================================================================
# (A3-1) alpha formula
# ===========================================================================
def test_alpha_is_half_when_tic_equals_median():
    """tIC exactly at the median m => alpha == 0.5 exactly, at every covered date."""
    dates = _bdays(8)
    # Two events realizing early, averaging exactly m => tIC == m on later dates.
    ic_events = pd.Series({dates[1]: _M - 0.02, dates[2]: _M + 0.02})
    got = compute_adaptive_alpha(ic_events, dates, _M, _IQR)
    # Dates whose trailing window covers both events (i >= 3) must be exactly 0.5.
    covered = got.iloc[3:]
    assert np.all(covered.values == 0.5), covered
    _assert_alpha_gate(got.values, _ref_adaptive_alpha(ic_events, dates, _M, _IQR).values)


def test_alpha_formula_matches_independent():
    """Non-degenerate tIC => alpha equals the independently computed clip formula."""
    dates = _bdays(10)
    # Single event of known IC realizing at dates[2]; from dates[3] on the trailing
    # mean is exactly that IC, so alpha = clip(0.5 + (ic - m)/(2*iqr)).
    ic = 0.10
    ic_events = pd.Series({dates[2]: ic})
    got = compute_adaptive_alpha(ic_events, dates, _M, _IQR)
    # 0.5 + (0.10-0.04)/0.15 = 0.9 -> clipped to 0.75.
    expected_scalar = _ref_alpha_scalar(ic, _M, _IQR)
    for i in range(3, len(dates)):
        assert abs(got.iloc[i] - expected_scalar) <= 1e-12, (i, got.iloc[i], expected_scalar)
    _assert_alpha_gate(got.values, _ref_adaptive_alpha(ic_events, dates, _M, _IQR).values)


def test_alpha_unclipped_interior_value():
    """A tIC that lands strictly inside (0.25, 0.75) reproduces the exact interior alpha."""
    dates = _bdays(6)
    ic = 0.055                       # 0.5 + (0.055-0.04)/0.15 = 0.6 exactly, unclipped
    ic_events = pd.Series({dates[1]: ic})
    got = compute_adaptive_alpha(ic_events, dates, _M, _IQR)
    expected = _ref_alpha_scalar(ic, _M, _IQR)
    assert 0.25 < expected < 0.75    # guard: genuinely interior
    assert abs(expected - 0.6) <= 1e-12
    for i in range(2, len(dates)):
        assert abs(got.iloc[i] - expected) <= _REL_GATE * abs(expected)
        assert abs(got.iloc[i] - expected) <= 1e-12


# ===========================================================================
# (A3-2) clip bounds
# ===========================================================================
def test_alpha_clips_to_bounds():
    """Extreme trailing IC saturates alpha at the symmetric [0.25, 0.75] bounds."""
    dates = _bdays(6)
    hi_events = pd.Series({dates[1]: 5.0})     # huge positive IC -> upper clip
    lo_events = pd.Series({dates[1]: -5.0})    # huge negative IC -> lower clip
    a_hi = compute_adaptive_alpha(hi_events, dates, _M, _IQR)
    a_lo = compute_adaptive_alpha(lo_events, dates, _M, _IQR)
    for i in range(2, len(dates)):
        assert a_hi.iloc[i] == 0.75, (i, a_hi.iloc[i])
        assert a_lo.iloc[i] == 0.25, (i, a_lo.iloc[i])


# ===========================================================================
# (A3-3) causality
# ===========================================================================
def test_event_affects_alpha_only_after_realization():
    """An event 'predicted' at P but realizing at R>P must not move alpha before R;
    it first affects alpha on the date AFTER R (upper window bound is dates[i-1])."""
    dates = _bdays(30)
    pred_pos, real_pos = 5, 15          # prediction date vs realization date
    ic_events = pd.Series({dates[real_pos]: 0.09})   # keyed by REALIZATION date

    got = compute_adaptive_alpha(ic_events, dates, _M, _IQR)

    # At/around the prediction date the event is unrealized -> alpha == 0.5.
    assert got.iloc[pred_pos] == 0.5
    # At the realization date itself, hi = dates[real_pos-1] < R -> still 0.5.
    assert got.iloc[real_pos] == 0.5
    # The first date AFTER realization sees it (hi = dates[real_pos] == R).
    assert got.iloc[real_pos + 1] != 0.5
    assert abs(got.iloc[real_pos + 1] - _ref_alpha_scalar(0.09, _M, _IQR)) <= 1e-12


def test_future_events_do_not_change_past_alpha():
    """Appending/altering events realizing at or after t leaves alpha_t unchanged."""
    dates = _bdays(20)
    base = pd.Series({dates[3]: 0.06})
    t_pos = 10
    a_base = compute_adaptive_alpha(base, dates, _M, _IQR)

    # Add events realizing exactly at t and later (all >= dates[t_pos]).
    future = base.copy()
    future.loc[dates[t_pos]] = 0.9
    future.loc[dates[t_pos + 3]] = -0.9
    future = future.sort_index()
    a_future = compute_adaptive_alpha(future, dates, _M, _IQR)

    # alpha at every date up to and including t is byte-identical.
    for i in range(0, t_pos + 1):
        assert a_future.iloc[i] == a_base.iloc[i], (i, a_future.iloc[i], a_base.iloc[i])


def test_no_events_gives_half():
    """Empty IC event set => alpha ≡ 0.5 on every date."""
    dates = _bdays(12)
    # Empty event set, but still DatetimeIndex-typed (as a real filter would yield),
    # consistent with the pinned ic_events contract.
    empty = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)
    got = compute_adaptive_alpha(empty, dates, _M, _IQR)
    assert list(got.index) == list(dates)
    assert np.all(got.values == 0.5)


# ===========================================================================
# (A3-4) trailing 63d window
# ===========================================================================
def test_old_event_excluded_from_trailing_window():
    """An event older than the 63-trading-day trailing window is dropped from tIC."""
    dates = _bdays(80)
    old_pos, recent_pos, t_pos = 0, 70, 74     # 74 - 63 = 11 => window starts dates[11]
    old = pd.Series({dates[old_pos]: 0.30})            # well before window start
    recent = pd.Series({dates[recent_pos]: 0.30})      # inside window
    both = pd.concat([old, recent]).sort_index()

    a_recent = compute_adaptive_alpha(recent, dates, _M, _IQR)
    a_both = compute_adaptive_alpha(both, dates, _M, _IQR)

    # Premise guards (independent): recent is inside, old is outside the window at t.
    lo = dates[max(0, t_pos - _TRAILING_TD)]
    hi = dates[t_pos - 1]
    assert lo <= dates[recent_pos] <= hi
    assert dates[old_pos] < lo

    # At t, the old event contributes nothing: {old, recent} gives the same alpha
    # as {recent} alone (old is excluded, not averaged in).
    assert a_both.iloc[t_pos] == a_recent.iloc[t_pos]
    # And that alpha reflects ONLY the recent event's IC.
    assert abs(a_both.iloc[t_pos] - _ref_alpha_scalar(0.30, _M, _IQR)) <= 1e-12

    # Cross-check: at an EARLY date where the old event IS in-window, it does move
    # alpha away from 0.5 (so the exclusion above is meaningful, not vacuous).
    early = old_pos + 5
    a_old = compute_adaptive_alpha(old, dates, _M, _IQR)
    assert a_old.iloc[early] != 0.5


def test_full_alpha_series_matches_reference():
    """End-to-end pin: formula + windowing + causality on a rich multi-event scenario
    equals the independent reference for EVERY date."""
    dates = _bdays(120)
    rng = np.random.default_rng(20260706)
    ev_positions = sorted(rng.choice(np.arange(2, 118), size=25, replace=False))
    ic_events = pd.Series(
        rng.normal(0.04, 0.08, size=len(ev_positions)),
        index=dates[ev_positions],
    ).sort_index()

    got = compute_adaptive_alpha(ic_events, dates, _M, _IQR)
    ref = _ref_adaptive_alpha(ic_events, dates, _M, _IQR)

    assert list(got.index) == list(dates)
    _assert_alpha_gate(got.values, ref.values)


# ===========================================================================
# (A3-5) EMA equivalence at alpha ≡ 0.5  (frame WITH NaN => also covers A3-7)
# ===========================================================================
def test_adaptive_ema_equals_prediction_ema_when_half():
    """alpha_series ≡ 0.5 => apply_adaptive_ema is byte-identical to
    src.model_trainer.apply_prediction_ema(raw, 0.5), including NaN handling."""
    raw = _toy_frame(seed=3)
    alpha_series = pd.Series(0.5, index=raw.index)

    got = apply_adaptive_ema(raw, alpha_series)
    ref = apply_prediction_ema(raw, 0.5)

    # Exact (identical 0.5 arithmetic) — check_exact catches any recursion drift.
    pd.testing.assert_frame_equal(got, ref, check_exact=True)


# ===========================================================================
# (A3-6) time-varying recursion
# ===========================================================================
def test_time_varying_recursion_handcomputed():
    """Small dense 4d x 2tk case with per-date alpha reproduced by HAND arithmetic.

    First row is a passthrough (no prev). Then blended_t = a_t*raw_t + (1-a_t)*bl_{t-1}.
    """
    dates = _bdays(4)
    raw = pd.DataFrame(
        {"A": [1.0, 3.0, 0.5, 2.0], "B": [2.0, -1.0, 4.0, 1.0]}, index=dates
    )
    # alpha at D0 is irrelevant (passthrough); D1=0.3, D2=0.8, D3=0.6.
    alpha_series = pd.Series([0.5, 0.3, 0.8, 0.6], index=dates)

    # Hand-computed expected (independent of any impl):
    #  D0: A=1.0,               B=2.0                (passthrough)
    #  D1: A=.3*3 +.7*1 =1.6,   B=.3*-1+.7*2 =1.1
    #  D2: A=.8*.5+.2*1.6=.72,  B=.8*4 +.2*1.1=3.42
    #  D3: A=.6*2 +.4*.72=1.488,B=.6*1 +.4*3.42=1.968
    expected = pd.DataFrame(
        {"A": [1.0, 1.6, 0.72, 1.488], "B": [2.0, 1.1, 3.42, 1.968]}, index=dates
    )

    got = apply_adaptive_ema(raw, alpha_series)

    # Tight exactness + 0.5% relative gate on every (nonzero) cell.
    np.testing.assert_allclose(got.values, expected.values, atol=1e-12, rtol=0.0)
    assert np.all(
        np.abs(got.values - expected.values) <= _REL_GATE * np.abs(expected.values)
    ), (got.values, expected.values)


def test_time_varying_matches_independent_reference():
    """Per-date alpha on a NaN-laden frame equals the independent reference recursion
    (covers churn / all-NaN gap / burn-in interacting with a varying alpha)."""
    raw = _toy_frame(seed=4)
    rng = np.random.default_rng(11)
    alpha_series = pd.Series(rng.uniform(0.25, 0.75, size=len(raw.index)), index=raw.index)

    got = apply_adaptive_ema(raw, alpha_series)
    ref = _ref_adaptive_ema(raw, alpha_series)

    pd.testing.assert_frame_equal(got, ref, check_exact=True)


# ===========================================================================
# (A3-7) NaN handling
# ===========================================================================
def test_nan_mask_preserved_under_adaptive_ema():
    """The NaN mask of the output equals that of the input (rows/tickers without a
    value are never fabricated), for both constant and time-varying alpha."""
    raw = _toy_frame(seed=5)
    for alpha_series in (
        pd.Series(0.5, index=raw.index),
        pd.Series(np.linspace(0.25, 0.75, len(raw.index)), index=raw.index),
    ):
        got = apply_adaptive_ema(raw, alpha_series)
        assert got.isna().equals(raw.isna())
