"""Acceptance tests for A4 — LGBM seed-ensemble (k=5) arm (test-first).

Written BEFORE implementation, from spec `spec-a4-seed-ensemble.md`. The arm
lives in `scripts/run_seed_ensemble_arm.py` (src/ production code is NOT touched).
These tests pin the CONTRACT of its two pure functions; they must go RED with a
ModuleNotFoundError until that file exists.

Run from ai_port with PYTHONPATH=. :

    C:/Users/westl/PycharmProjects/pythonProject/venv_vf_new/Scripts/python.exe \
        -m pytest tests/acceptance/test_seed_ensemble.py -v

House-style idioms (mirroring tests/acceptance/test_adaptive_ema.py &
tests/test_prediction_ema.py): plain pytest functions (no fixtures), synthetic
data only, fast (<1s). Every expected value is computed INDEPENDENTLY here (plain
numpy arithmetic + an independent reference) — the impl's own logic is NEVER
reused. Numeric asserts carry a 0.5% relative-tolerance gate on top of a tight
exactness check (atol=1e-9), per the quant test contract.

==============================================================================
PINNED CONTRACT  (authoritative — the implementer must satisfy exactly this)
==============================================================================
Module: `scripts.run_seed_ensemble_arm` (importable as a namespace package when
run from ai_port with PYTHONPATH=.; `scripts/` has no __init__.py, `src/` does).

Panel orientation (both functions): a "seed panel" is a date x ticker DataFrame
(index = prediction/rebalance dates, columns = tickers), matching the shape that
`src.model_trainer.apply_prediction_ema` consumes and what `backtest_result.pkl`
carries in `raw_predictions`. All k panels are aligned on the SAME (identical)
date index and ticker columns (they come from the same universe/calendar; the
spec requires their NaN masks to agree — see `nan_mask_mismatch_rate`).

(1) combine_seed_panels(panels: list[pd.DataFrame]) -> pd.DataFrame
    Two ordered steps, NO EMA (the α=0.5 EMA is a SEPARATE later stage applied by
    main via src.model_trainer.apply_prediction_ema — it is not part of combine):

    STEP 1 — cell-wise finite-value mean across the k panels:
        for each cell (date, ticker):
          * value = arithmetic mean of the FINITE seed values at that cell
            (NaNs skipped, i.e. pandas/skipna semantics);
          * if EVERY seed is NaN at that cell => the cell is NaN.
        (NaN mask of this intermediate = the cells that are all-NaN across seeds.)

    STEP 2 — per-date cross-sectional RE-STANDARDIZATION, applying EXACTLY the
    `src.model_trainer` z-score idiom (predict_cross_sectional, lines 240-245)
    to each DATE ROW (standardize across tickers):
          mean = row.mean()          # pandas: skipna=True
          std  = row.std()           # pandas: ddof=1 (SAMPLE std), skipna=True
          if std > 0:                # STRICT guard
              row = (row - mean) / std
      GUARD BEHAVIOR (must match model_trainer byte-for-byte):
          * a CONSTANT row (all finite values equal)  => std == 0 => row UNCHANGED;
          * a row with a SINGLE finite value          => std == NaN (ddof=1)
                                                        => (NaN > 0) False => UNCHANGED;
          * an ALL-NaN row                            => std == NaN => UNCHANGED (stays NaN).
      NaN cells stay NaN under the affine transform ((NaN-mean)/std == NaN), so the
      output NaN mask == the STEP-1 NaN mask == "cells all-NaN across seeds".

    returns : date x ticker DataFrame on the same index/columns as the inputs.

(2) nan_mask_mismatch_rate(panels: list[pd.DataFrame]) -> float
    PINNED DEFINITION (I choose this; the implementer must match it exactly):
        Over the shared date x ticker grid, a cell POSITION is "mismatched" iff the
        per-seed NaN status is NOT unanimous there — i.e. it is NaN in >= 1 panel
        AND finite in >= 1 panel. A cell that is NaN in ALL panels is unanimous and
        is therefore NOT a mismatch (it is a legitimately-absent cell).
            rate = (# mismatched cell positions) / (total # cell positions)
        Denominator = total cells in the grid (rows * cols), NOT the count of
        NaN-bearing cells. Range [0, 1]. Fully-consistent masks => 0.0. Example
        (spec §13, gate at 0.1%): 3 mismatched cells out of 30 => 0.1.

==============================================================================
VERIFICATION-ITEM -> TEST MAPPING  (spec §12-13 + team-lead items 1-6)
==============================================================================
(V1) finite-value mean: all-NaN cell -> NaN; some-finite cell -> mean of finite
     (verified PRE-restandardization via constant-row guard, so combine reveals it)
        -> test_finite_mean_via_constant_rows
        -> test_all_nan_cell_yields_nan
(V2) per-date re-standardization == model_trainer z (ddof=1, skipna, std>0 guard);
     row mean~=0 / std~=1; constant-row & single-value guard behavior preserved
        -> test_per_date_z_handcomputed
        -> test_per_date_restandardization_matches_reference
        -> test_restandardized_rows_are_zero_mean_unit_std
        -> test_constant_row_guard_preserved
        -> test_single_finite_value_row_guard
(V3) k identical panels combine to the identity (== per-date z of the shared panel)
        -> test_identical_panels_combine_to_per_date_z
(V4) NaN mask preservation: a cell NaN in every seed is NaN in the output
        -> test_nan_mask_preserved
(V5) nan_mask_mismatch_rate: consistent->0.0; 3-of-30->0.1; all-NaN cell NOT a mismatch
        -> test_mismatch_rate_zero_when_consistent
        -> test_mismatch_rate_exact_ratio_three_of_thirty
        -> test_all_nan_cell_is_not_a_mismatch
(V6) order invariance: shuffling the panel list changes neither combine nor the rate
        -> test_combine_order_invariant
        -> test_mismatch_rate_order_invariant
Plus an end-to-end pin (finite-mean + per-date z on a rich, NaN-laden scenario):
        -> test_combine_matches_full_reference
"""

import numpy as np
import pandas as pd
import pytest

# Direct import of the target module — RED (ModuleNotFoundError) until the
# implementer creates scripts/run_seed_ensemble_arm.py. This is the intended
# test-first failure.
from scripts.run_seed_ensemble_arm import (
    combine_seed_panels,
    nan_mask_mismatch_rate,
)


# ---------------------------------------------------------------------------
# Independent references (NEVER call the implementation under test)
# ---------------------------------------------------------------------------
_REL_GATE = 0.005  # 0.5% relative-tolerance gate (quant test contract)
_ATOL = 1e-9       # tight exactness (z-scores involve sqrt -> not float-exact)


def _ref_finite_mean(panels):
    """STEP 1 independent: cell-wise mean of finite seed values; all-NaN -> NaN.

    Computed WITHOUT np.nanmean (avoids empty-slice warnings and pins the exact
    'sum of finite / count of finite' semantics)."""
    idx, cols = panels[0].index, panels[0].columns
    arr = np.stack(
        [p.reindex(index=idx, columns=cols).values.astype(float) for p in panels],
        axis=0,
    )  # (k, D, T)
    finite = np.isfinite(arr)
    cnt = finite.sum(axis=0)                      # (D, T) number of finite seeds
    s = np.where(finite, arr, 0.0).sum(axis=0)    # (D, T) sum of finite seeds
    mean = np.where(cnt > 0, s / np.where(cnt > 0, cnt, 1.0), np.nan)
    return pd.DataFrame(mean, index=idx, columns=cols)


def _ref_per_date_z(frame):
    """STEP 2 independent: model_trainer z-score idiom applied per DATE ROW.

        mean = row.mean(skipna); std = row.std(ddof=1, skipna); if std>0: (row-mean)/std
    Rows with <2 finite values (single value or all-NaN) have std NaN -> guard
    False -> left unchanged; constant rows have std 0 -> guard False -> unchanged.
    NaN cells stay NaN."""
    vals = frame.values.astype(float).copy()
    for i in range(vals.shape[0]):
        row = vals[i]
        finite = np.isfinite(row)
        if finite.sum() >= 2:                     # need >=2 finite for a defined std
            mean = row[finite].mean()
            std = row[finite].std(ddof=1)         # SAMPLE std -> matches pandas .std()
            if std > 0:                           # strict guard (constant row -> skip)
                vals[i] = (row - mean) / std      # NaNs remain NaN
    return pd.DataFrame(vals, index=frame.index, columns=frame.columns)


def _ref_combine(panels):
    """Full independent combine: finite mean THEN per-date re-standardization."""
    return _ref_per_date_z(_ref_finite_mean(panels))


def _ref_mismatch_rate(panels):
    """Independent: fraction of grid cells whose per-seed NaN status is not unanimous."""
    idx, cols = panels[0].index, panels[0].columns
    masks = np.stack(
        [p.reindex(index=idx, columns=cols).isna().values for p in panels], axis=0
    )  # (k, D, T) bool
    all_nan = masks.all(axis=0)
    any_nan = masks.any(axis=0)
    mismatched = any_nan & ~all_nan               # NaN in some seeds, finite in others
    return float(mismatched.sum()) / float(mismatched.size)


def _assert_frame_gate(got, expected):
    """NaN masks identical + tight exactness + 0.5% relative gate on finite cells."""
    assert list(got.index) == list(expected.index), "index mismatch"
    assert list(got.columns) == list(expected.columns), "columns mismatch"
    g = np.asarray(got.values, dtype=float)
    e = np.asarray(expected.values, dtype=float)
    assert np.array_equal(np.isnan(g), np.isnan(e)), "NaN mask mismatch"
    fin = ~np.isnan(e)
    # Exactness (same arithmetic path expected, tolerant of sqrt rounding).
    np.testing.assert_allclose(g[fin], e[fin], atol=_ATOL, rtol=0.0)
    # 0.5% relative-tolerance gate; where expected==0 fall back to the abs atol.
    denom = np.abs(e[fin])
    ok = np.where(
        denom > 0,
        np.abs(g[fin] - e[fin]) <= _REL_GATE * denom,
        np.abs(g[fin] - e[fin]) <= _ATOL,
    )
    assert np.all(ok), (g[fin], e[fin])


# ---------------------------------------------------------------------------
# Shared synthetic helpers
# ---------------------------------------------------------------------------
def _bdays(n, start="2020-01-01"):
    return pd.bdate_range(start, periods=n)


def _tickers(n):
    return [f"T{i}" for i in range(n)]


def _rand_panel(seed, dates, tks):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(len(dates), len(tks))), index=dates, columns=tks)


# ===========================================================================
# (V1) finite-value mean  (isolated via constant-row guard: STEP 2 is a no-op)
# ===========================================================================
def test_finite_mean_via_constant_rows():
    """STEP-1 finite mean, revealed directly: every date row is engineered so the
    per-cell finite mean is CONSTANT across tickers => STEP-2 std==0 guard leaves
    the row unchanged => combine output == the STEP-1 finite mean, hand-checkable.

    Row target constant = 3.0. Per ticker the seed values differ (and some are NaN)
    but each ticker's finite mean is 3.0; one ticker is all-NaN (-> NaN).
    """
    dates = _bdays(1)
    tks = _tickers(4)
    n = np.nan
    # 3 seed panels; column means (over finite): T0=(2,4)->3, T1=(1,5,3)->3,
    # T2=(3,n,n)->3, T3=(n,n,n)->NaN.
    p0 = pd.DataFrame([[2.0, 1.0, 3.0,   n]], index=dates, columns=tks)
    p1 = pd.DataFrame([[4.0, 5.0,   n,   n]], index=dates, columns=tks)
    p2 = pd.DataFrame([[  n, 3.0,   n,   n]], index=dates, columns=tks)
    got = combine_seed_panels([p0, p1, p2])

    # T0..T2 finite mean == 3.0 (constant row => untouched by re-standardization).
    assert abs(got.iloc[0]["T0"] - 3.0) <= _ATOL
    assert abs(got.iloc[0]["T1"] - 3.0) <= _ATOL
    assert abs(got.iloc[0]["T2"] - 3.0) <= _ATOL
    # All-NaN cell stays NaN.
    assert np.isnan(got.iloc[0]["T3"])
    # Cross-check against the full independent reference too.
    _assert_frame_gate(got, _ref_combine([p0, p1, p2]))


def test_all_nan_cell_yields_nan():
    """A cell that is NaN in EVERY seed is NaN in the output; a cell finite in
    at least one seed is finite in the output."""
    dates = _bdays(3)
    tks = _tickers(5)
    p0 = _rand_panel(1, dates, tks)
    p1 = _rand_panel(2, dates, tks)
    p2 = _rand_panel(3, dates, tks)
    # Make (date0, T4) all-NaN across seeds; make (date1, T4) NaN in 2 of 3 seeds.
    for p in (p0, p1, p2):
        p.iloc[0, 4] = np.nan
    p0.iloc[1, 4] = np.nan
    p1.iloc[1, 4] = np.nan  # p2.iloc[1,4] stays finite

    got = combine_seed_panels([p0, p1, p2])
    assert np.isnan(got.iloc[0]["T4"]), "all-seed-NaN cell must be NaN"
    assert np.isfinite(got.iloc[1]["T4"]), "partially-finite cell must be finite"


# ===========================================================================
# (V2) per-date re-standardization == model_trainer z-score idiom
# ===========================================================================
def test_per_date_z_handcomputed():
    """Single date, tickers [1,2,6]: z with mean=3, std(ddof=1)=sqrt(7).

    Hand-computed literals (independent of any impl): z = (x-3)/sqrt(7)
        1 -> -2/sqrt7 = -0.7559289460184545
        2 -> -1/sqrt7 = -0.3779644730092272
        6 ->  3/sqrt7 =  1.1338934190276817
    Two identical panels so STEP-1 finite mean == the shared panel (integer-exact).
    """
    dates = _bdays(1)
    tks = _tickers(3)
    p = pd.DataFrame([[1.0, 2.0, 6.0]], index=dates, columns=tks)
    got = combine_seed_panels([p, p])

    expected = np.array([-0.7559289460184545, -0.3779644730092272, 1.1338934190276817])
    row = got.iloc[0].values.astype(float)
    np.testing.assert_allclose(row, expected, atol=_ATOL, rtol=0.0)
    assert np.all(np.abs(row - expected) <= _REL_GATE * np.abs(expected)), (row, expected)
    # Sanity: re-standardized row has mean ~0 and sample std ~1.
    assert abs(row.mean()) <= _ATOL
    assert abs(row.std(ddof=1) - 1.0) <= _REL_GATE


def test_per_date_restandardization_matches_reference():
    """Non-constant, NaN-free multi-date panels: combine equals finite-mean then
    the independent per-date z reference, every cell within the 0.5% gate."""
    dates = _bdays(3)
    tks = _tickers(5)
    panels = [_rand_panel(s, dates, tks) for s in (10, 11, 12, 13, 14)]
    got = combine_seed_panels(panels)
    _assert_frame_gate(got, _ref_combine(panels))


def test_restandardized_rows_are_zero_mean_unit_std():
    """Each date row of the combined panel is CS-standardized: over its finite
    cells the mean is ~0 and the SAMPLE std (ddof=1) is ~1 (rows here are
    non-constant with >=2 finite values, so the std>0 branch always fires)."""
    dates = _bdays(4)
    tks = _tickers(6)
    panels = [_rand_panel(s, dates, tks) for s in (20, 21, 22, 23, 24)]
    got = combine_seed_panels(panels)
    for d in got.index:
        row = got.loc[d].values.astype(float)
        fin = row[np.isfinite(row)]
        assert len(fin) >= 2
        assert abs(fin.mean()) <= 1e-9, (d, fin.mean())
        assert abs(fin.std(ddof=1) - 1.0) <= _REL_GATE, (d, fin.std(ddof=1))


def test_constant_row_guard_preserved():
    """A date row whose finite mean is CONSTANT across tickers (variance 0) is left
    UNCHANGED by re-standardization (std==0 => guard False), exactly like
    model_trainer's `if std > 0` guard — NOT divided by zero into NaN/inf."""
    dates = _bdays(2)
    tks = _tickers(4)
    # Row 0 constant (all 7.0), row 1 non-constant.
    p0 = pd.DataFrame([[7.0, 7.0, 7.0, 7.0], [1.0, 2.0, 3.0, 4.0]], index=dates, columns=tks)
    p1 = pd.DataFrame([[7.0, 7.0, 7.0, 7.0], [1.0, 2.0, 3.0, 4.0]], index=dates, columns=tks)
    got = combine_seed_panels([p0, p1])

    # Constant row preserved verbatim (no NaN, no inf, still 7.0).
    assert np.allclose(got.iloc[0].values, 7.0, atol=_ATOL)
    assert np.all(np.isfinite(got.iloc[0].values))
    # Non-constant row IS standardized (independent check).
    _assert_frame_gate(got.iloc[[1]], _ref_per_date_z(_ref_finite_mean([p0, p1])).iloc[[1]])


def test_single_finite_value_row_guard():
    """A date row with a SINGLE finite value has std==NaN (ddof=1) => guard False
    => the value is left UNCHANGED (matches model_trainer; the lone value is NOT
    turned into 0 or NaN)."""
    dates = _bdays(1)
    tks = _tickers(4)
    n = np.nan
    p0 = pd.DataFrame([[5.0, n, n, n]], index=dates, columns=tks)
    p1 = pd.DataFrame([[5.0, n, n, n]], index=dates, columns=tks)
    got = combine_seed_panels([p0, p1])
    assert abs(got.iloc[0]["T0"] - 5.0) <= _ATOL, "lone finite value must be preserved"
    assert np.isnan(got.iloc[0]["T1"])
    assert np.isnan(got.iloc[0]["T2"])
    assert np.isnan(got.iloc[0]["T3"])


# ===========================================================================
# (V3) k identical panels -> identity (== per-date z of the shared panel)
# ===========================================================================
def test_identical_panels_combine_to_per_date_z():
    """Combining k copies of the SAME panel == that panel re-standardized per date
    (STEP-1 finite mean of identical copies is the panel itself). Integer-valued
    data so the mean is float-exact across k=5 copies."""
    dates = _bdays(3)
    tks = _tickers(5)
    rng = np.random.default_rng(777)
    p = pd.DataFrame(
        rng.integers(-5, 6, size=(len(dates), len(tks))).astype(float),
        index=dates, columns=tks,
    )
    got = combine_seed_panels([p, p, p, p, p])
    expected = _ref_per_date_z(p)   # z of the ORIGINAL panel, not of 5x/5
    _assert_frame_gate(got, expected)


# ===========================================================================
# (V4) NaN mask preservation
# ===========================================================================
def test_nan_mask_preserved():
    """Output NaN mask == the 'all-seed-NaN' mask: cells NaN in every seed are the
    ONLY NaNs in the output; every partially/fully-finite cell is finite."""
    dates = _bdays(4)
    tks = _tickers(5)
    panels = [_rand_panel(s, dates, tks) for s in (30, 31, 32, 33, 34)]
    # Engineer some all-seed-NaN cells and some partial-NaN cells.
    all_nan_cells = [(0, 0), (2, 4), (3, 1)]
    for (r, c) in all_nan_cells:
        for p in panels:
            p.iloc[r, c] = np.nan
    panels[0].iloc[1, 2] = np.nan   # partial NaN (finite in seeds 1..4)

    got = combine_seed_panels(panels)
    expected_nan = _ref_finite_mean(panels).isna()   # == all-seed-NaN mask
    assert got.isna().equals(expected_nan)
    for (r, c) in all_nan_cells:
        assert np.isnan(got.iloc[r, c])
    assert np.isfinite(got.iloc[1, 2])               # partial cell survives


# ===========================================================================
# (V5) nan_mask_mismatch_rate  (pinned definition)
# ===========================================================================
def test_mismatch_rate_zero_when_consistent():
    """Panels with IDENTICAL NaN masks (incl. some shared all-seed-NaN cells) =>
    rate 0.0."""
    dates = _bdays(3)
    tks = _tickers(5)
    panels = [_rand_panel(s, dates, tks) for s in (40, 41, 42, 43, 44)]
    for p in panels:                     # same NaN positions in every panel
        p.iloc[0, 0] = np.nan
        p.iloc[2, 3] = np.nan
    assert nan_mask_mismatch_rate(panels) == 0.0


def test_mismatch_rate_exact_ratio_three_of_thirty():
    """6 dates x 5 tickers = 30 cells; introduce exactly 3 mismatched cells (NaN in
    one seed, finite in the rest) => rate == 3/30 == 0.1 (spec §13 example)."""
    dates = _bdays(6)
    tks = _tickers(5)                    # 6 * 5 == 30 cells
    panels = [_rand_panel(s, dates, tks) for s in (50, 51, 52, 53, 54)]  # all finite
    mismatch_cells = [(0, 0), (3, 2), (5, 4)]
    for (r, c) in mismatch_cells:
        panels[0].iloc[r, c] = np.nan    # NaN in seed 0 only -> not unanimous

    rate = nan_mask_mismatch_rate(panels)
    assert abs(rate - 0.1) <= 1e-12, rate
    assert abs(rate - _ref_mismatch_rate(panels)) <= 1e-12


def test_all_nan_cell_is_not_a_mismatch():
    """A cell NaN in ALL seeds is UNANIMOUS => NOT counted as a mismatch. Grid of
    20 cells: 2 all-seed-NaN cells (unanimous) + 1 one-seed-NaN cell (mismatch)
    => rate == 1/20 == 0.05, NOT 3/20."""
    dates = _bdays(4)
    tks = _tickers(5)                    # 4 * 5 == 20 cells
    panels = [_rand_panel(s, dates, tks) for s in (60, 61, 62)]
    for p in panels:                     # 2 unanimous all-NaN cells
        p.iloc[0, 0] = np.nan
        p.iloc[1, 1] = np.nan
    panels[0].iloc[2, 2] = np.nan        # single mismatch

    rate = nan_mask_mismatch_rate(panels)
    assert abs(rate - 0.05) <= 1e-12, rate
    assert abs(rate - _ref_mismatch_rate(panels)) <= 1e-12


# ===========================================================================
# (V6) order invariance
# ===========================================================================
def test_combine_order_invariant():
    """Shuffling the panel list leaves the combined panel unchanged. Integer-valued
    data => per-cell mean is summation-order-independent (float-exact), so the
    whole combine is byte-stable under reordering."""
    dates = _bdays(3)
    tks = _tickers(5)
    rng = np.random.default_rng(1234)
    panels = [
        pd.DataFrame(
            rng.integers(-4, 5, size=(len(dates), len(tks))).astype(float),
            index=dates, columns=tks,
        )
        for _ in range(5)
    ]
    # Introduce a shared all-NaN cell and a couple partial-NaN cells.
    for p in panels:
        p.iloc[0, 0] = np.nan
    panels[1].iloc[2, 3] = np.nan
    panels[3].iloc[2, 3] = np.nan

    base = combine_seed_panels(panels)
    shuffled = [panels[i] for i in (3, 0, 4, 1, 2)]
    other = combine_seed_panels(shuffled)
    _assert_frame_gate(other, base)


def test_mismatch_rate_order_invariant():
    """The mismatch rate is symmetric in the panel list => reordering leaves it
    unchanged."""
    dates = _bdays(5)
    tks = _tickers(5)
    panels = [_rand_panel(s, dates, tks) for s in (70, 71, 72, 73, 74)]
    panels[0].iloc[0, 0] = np.nan
    panels[2].iloc[3, 4] = np.nan
    r1 = nan_mask_mismatch_rate(panels)
    r2 = nan_mask_mismatch_rate([panels[i] for i in (2, 4, 0, 3, 1)])
    assert r1 == r2


# ===========================================================================
# End-to-end pin: finite mean + per-date z on a rich, NaN-laden 5-seed scenario
# ===========================================================================
def test_combine_matches_full_reference():
    """Rich scenario (5 seeds, burn-in head, churn gaps, an all-seed-NaN cell, a
    partial-NaN cell) equals the independent finite-mean + per-date-z reference for
    EVERY cell, within the 0.5% gate, with identical NaN masks."""
    dates = _bdays(12)
    tks = _tickers(7)
    panels = [_rand_panel(s, dates, tks) for s in (80, 81, 82, 83, 84)]
    for p in panels:
        p.iloc[:2] = np.nan            # shared burn-in head (all-seed-NaN rows)
        p.iloc[5, 3] = np.nan          # a shared all-seed-NaN cell
    panels[0].iloc[7, 1] = np.nan      # partial-NaN cells (finite in other seeds)
    panels[2].iloc[9, 6] = np.nan

    got = combine_seed_panels(panels)
    _assert_frame_gate(got, _ref_combine(panels))
