"""Acceptance tests for SPEC D0 — degenerate-retrain diagnosis report.

Written BEFORE implementation (test-first). Run from ai_port with PYTHONPATH=. :

    C:/Users/westl/PycharmProjects/pythonProject/venv_vf_new/Scripts/python.exe \
        -m pytest tests/acceptance/test_degenerate_diagnosis.py -v

House-style idioms: plain pytest functions (no fixtures), synthetic in-memory
inputs only (no file I/O, no walk-forward), fast. Expected values for the
distribution sections are computed INDEPENDENTLY here (a local re-statement of
the tail_n=max(3, n//10) spread formula) and asserted with a 0.5% relative
tolerance gate — never importing the implementation's numbers or reusing
src.backtest.compute_signal_confidence.

------------------------------------------------------------------------------
PINNED IMPORT SURFACE (this is the implementation contract for D0)
------------------------------------------------------------------------------
Module: scripts.diagnose_degenerate_retrains  (imported as a namespace package
under PYTHONPATH=.). It exposes a PURE function (no file I/O) that main() feeds
after gathering artifacts:

    build_report(
        retrain_windows: list[dict],   # per window, keys used below
        raw_predictions: pd.DataFrame, # dates x tickers, raw (pre-EMA) baseline preds
        trailing_ic: pd.Series,        # trailing-63d IC, indexed by date
        subperiods: dict,              # {"P1": (start, end), "P2": (...), "P3": (...)}
                                       #   inclusive pd.Timestamp bounds
    ) -> dict

    Each retrain_windows[i] is a dict with at least:
        "train_date": pd.Timestamp, "degenerate": bool,
        "best_iteration": int, "n_trees": int, "val_score": float

    Return dict MUST contain these top-level keys:
        "windows"             : list, one entry per input window (full census)
        "subperiod_overlap"   : {"P1": int, "P2": int, "P3": int}
                                = count of DEGENERATE windows whose train_date
                                  falls in each sub-period (inclusive).
        "root_cause_evidence" : dict (hypothesis -> supporting/refuting evidence)
        "raw_spread_dist"     : {"median","iqr","min","max","n_dates"}
                                over per-date raw top-bottom spreads, using the
                                SAME definition as compute_signal_confidence
                                (src/backtest.py:857-860): for each date take the
                                raw row, dropna, tail_n = max(3, n_valid // 10),
                                spread = mean(top tail_n) - mean(bottom tail_n).
                                Dates with no valid raw values are excluded.
        "trailing_ic_dist"    : {"median","iqr", ...} over `trailing_ic`.

------------------------------------------------------------------------------
ACCEPTANCE-CRITERION -> TEST MAPPING
------------------------------------------------------------------------------
(D0-1) build_report returns a dict carrying all five required top-level keys.
        -> test_report_has_all_required_sections
(D0-2) raw_spread_dist / trailing_ic_dist carry their required sub-fields with
       the right types.
        -> test_dist_subfields_present_and_typed
(D0-3) raw_spread_dist NUMERICS match the independent tail_n=max(3, n//10)
       definition (median/min/max within the 0.5% gate; n_dates exact).
        -> test_raw_spread_matches_signal_confidence_definition
(D0-4) raw_spread uses per-row dropna (NaN cells ignored; all-NaN dates dropped
       from n_dates) — mirrors compute_signal_confidence's raw_valid handling.
        -> test_raw_spread_ignores_nan_cells_and_empty_rows
(D0-5) trailing_ic_dist median matches the independent median of the series.
        -> test_trailing_ic_median_matches
(D0-6) subperiod_overlap counts degenerate windows per sub-period and surfaces
       the P2 concentration flagged in the spec.
        -> test_subperiod_overlap_counts_degenerate_windows
(D0-7) windows section is a full census (one entry per input window).
        -> test_windows_section_is_full_census

Pre-implementation, EVERY test fails at the deferred import of
scripts.diagnose_degenerate_retrains (module does not exist) — the intended TDD
red state. `pytest --collect-only` stays clean because that import happens
inside the test bodies, never at module import time.
------------------------------------------------------------------------------
"""

import importlib

import numpy as np
import pandas as pd


def _build_report():
    """Deferred import so collection succeeds before the script exists."""
    mod = importlib.import_module("scripts.diagnose_degenerate_retrains")
    return mod.build_report


# ---------------------------------------------------------------------------
# Independent re-statement of the compute_signal_confidence spread definition.
# NOT imported from src — recomputed here so the expected values are genuinely
# independent of the implementation under test.
# ---------------------------------------------------------------------------
def _raw_spread(row_values):
    v = np.asarray([x for x in row_values if x == x], dtype=float)  # x==x drops NaN
    if v.size == 0:
        return None
    v = np.sort(v)
    n = v.size
    tail_n = max(3, n // 10)
    top_mean = float(v[-tail_n:].mean())
    bot_mean = float(v[:tail_n].mean())
    return top_mean - bot_mean


def _iqr(values):
    a = np.asarray(values, dtype=float)
    return float(np.percentile(a, 75) - np.percentile(a, 25))


# ---------------------------------------------------------------------------
# Shared synthetic inputs.
#   * raw_predictions: 9 dates (odd => unambiguous median) x 25 tickers, dense.
#   * trailing_ic:     9-point series (odd => unambiguous median).
#   * retrain_windows: 8 windows placed across P1/P2/P3 with a P2-heavy
#                      degenerate concentration (spec: P2 sub-IR 0.575 window).
# ---------------------------------------------------------------------------
_RAW_DATES = pd.bdate_range("2016-01-04", periods=9)
_RAW_TICKERS = [f"T{i}" for i in range(25)]


def _dense_raw():
    rng = np.random.default_rng(20260706)
    return pd.DataFrame(
        rng.normal(0.0, 0.03, (9, 25)), index=_RAW_DATES, columns=_RAW_TICKERS
    )


def _trailing_ic():
    idx = pd.bdate_range("2016-01-04", periods=9)
    vals = [0.01, 0.03, -0.02, 0.05, 0.02, 0.00, 0.04, -0.01, 0.06]
    return pd.Series(vals, index=idx)


_SUBPERIODS = {
    "P1": (pd.Timestamp("2015-01-01"), pd.Timestamp("2017-12-31")),
    "P2": (pd.Timestamp("2018-01-01"), pd.Timestamp("2020-12-31")),
    "P3": (pd.Timestamp("2021-01-01"), pd.Timestamp("2023-12-31")),
}


def _windows():
    """8 windows: 1 degenerate in P1, 4 degenerate in P2, 0 degenerate in P3.
    (Two non-degenerate windows sit in P1/P3 for census coverage.)"""
    def w(date, degen, best_it, n_trees, val):
        return {
            "train_date": pd.Timestamp(date),
            "degenerate": degen,
            "best_iteration": best_it,
            "n_trees": n_trees,
            "val_score": val,
        }

    return [
        w("2016-06-01", True, 5, 5, -0.30),     # P1 degenerate
        w("2017-06-01", False, 180, 180, 0.12), # P1 healthy
        w("2018-03-01", True, 4, 4, -0.25),     # P2 degenerate
        w("2018-09-01", True, 6, 6, -0.28),     # P2 degenerate
        w("2019-06-01", True, 3, 3, -0.31),     # P2 degenerate
        w("2020-06-01", True, 8, 8, -0.22),     # P2 degenerate
        w("2021-06-01", False, 210, 210, 0.15), # P3 healthy
        w("2022-06-01", False, 160, 160, 0.10), # P3 healthy
    ]


# ===========================================================================
# (D0-1) All five required sections present.
# ===========================================================================
def test_report_has_all_required_sections():
    build_report = _build_report()
    report = build_report(_windows(), _dense_raw(), _trailing_ic(), _SUBPERIODS)
    assert isinstance(report, dict)
    for key in (
        "windows",
        "subperiod_overlap",
        "root_cause_evidence",
        "raw_spread_dist",
        "trailing_ic_dist",
    ):
        assert key in report, f"missing required section '{key}'"


# ===========================================================================
# (D0-2) Distribution sub-fields present with correct types.
# ===========================================================================
def test_dist_subfields_present_and_typed():
    build_report = _build_report()
    report = build_report(_windows(), _dense_raw(), _trailing_ic(), _SUBPERIODS)

    rsd = report["raw_spread_dist"]
    for k in ("median", "iqr", "min", "max", "n_dates"):
        assert k in rsd, f"raw_spread_dist missing '{k}'"
    assert isinstance(rsd["n_dates"], int)
    for k in ("median", "iqr", "min", "max"):
        assert isinstance(rsd[k], float)
    assert rsd["min"] <= rsd["median"] <= rsd["max"]
    assert rsd["iqr"] >= 0.0

    tid = report["trailing_ic_dist"]
    for k in ("median", "iqr"):
        assert k in tid, f"trailing_ic_dist missing '{k}'"
        assert isinstance(tid[k], float)


# ===========================================================================
# (D0-3) raw_spread numerics == independent tail_n=max(3, n//10) definition.
# ===========================================================================
def test_raw_spread_matches_signal_confidence_definition():
    build_report = _build_report()
    raw = _dense_raw()

    # Independent expected distribution over the 9 dense dates.
    spreads = [_raw_spread(raw.loc[d].values) for d in raw.index]
    spreads = [s for s in spreads if s is not None]
    exp_median = float(np.median(spreads))
    exp_min = float(np.min(spreads))
    exp_max = float(np.max(spreads))
    exp_iqr = _iqr(spreads)
    exp_n = len(spreads)

    report = build_report(_windows(), raw, _trailing_ic(), _SUBPERIODS)
    rsd = report["raw_spread_dist"]

    # Guard the reference: spreads are comfortably non-zero so the relative gate
    # is meaningful.
    assert exp_median > 0.01, exp_median
    assert rsd["n_dates"] == exp_n

    # 0.5% relative-tolerance gate on the location/extent statistics. min/max and
    # (odd-n) median are percentile-method-independent, so exactness holds too.
    assert abs(rsd["median"] - exp_median) <= 0.005 * abs(exp_median)
    assert abs(rsd["min"] - exp_min) <= 0.005 * abs(exp_min)
    assert abs(rsd["max"] - exp_max) <= 0.005 * abs(exp_max)
    assert np.isclose(rsd["median"], exp_median, atol=1e-9)
    assert np.isclose(rsd["min"], exp_min, atol=1e-9)
    assert np.isclose(rsd["max"], exp_max, atol=1e-9)
    # IQR: allow a little slack for percentile-interpolation differences.
    assert abs(rsd["iqr"] - exp_iqr) <= 0.02 * abs(exp_iqr) + 1e-9


# ===========================================================================
# (D0-4) raw_spread uses per-row dropna; all-NaN dates excluded from n_dates.
# ===========================================================================
def test_raw_spread_ignores_nan_cells_and_empty_rows():
    build_report = _build_report()
    raw = _dense_raw().copy()

    # Punch NaN holes into one row (still >=5 valid => tail_n recomputed on the
    # surviving values), and blank another row entirely (must be dropped).
    hole_date = raw.index[2]
    raw.loc[hole_date, _RAW_TICKERS[:8]] = np.nan  # 17 valid remain
    empty_date = raw.index[5]
    raw.loc[empty_date, :] = np.nan                # 0 valid => excluded

    # Independent expectation over the surviving rows.
    spreads = []
    for d in raw.index:
        s = _raw_spread(raw.loc[d].values)
        if s is not None:
            spreads.append(s)
    exp_n = len(spreads)                 # 8 (one all-NaN row dropped)
    exp_median = float(np.median(spreads))

    report = build_report(_windows(), raw, _trailing_ic(), _SUBPERIODS)
    rsd = report["raw_spread_dist"]

    assert rsd["n_dates"] == exp_n == 8
    assert np.isclose(rsd["median"], exp_median, atol=1e-9)


# ===========================================================================
# (D0-5) trailing_ic_dist median == independent median of the series.
# ===========================================================================
def test_trailing_ic_median_matches():
    build_report = _build_report()
    ic = _trailing_ic()
    exp_median = float(np.median(ic.values))

    report = build_report(_windows(), _dense_raw(), ic, _SUBPERIODS)
    tid = report["trailing_ic_dist"]

    assert abs(tid["median"] - exp_median) <= 0.005 * abs(exp_median) + 1e-12
    assert np.isclose(tid["median"], exp_median, atol=1e-9)


# ===========================================================================
# (D0-6) subperiod_overlap counts degenerate windows per period (P2-heavy).
# ===========================================================================
def test_subperiod_overlap_counts_degenerate_windows():
    build_report = _build_report()
    report = build_report(_windows(), _dense_raw(), _trailing_ic(), _SUBPERIODS)
    overlap = report["subperiod_overlap"]

    assert set(overlap.keys()) >= {"P1", "P2", "P3"}
    # Ground truth from _windows(): 1 degenerate in P1, 4 in P2, 0 in P3.
    assert int(overlap["P1"]) == 1
    assert int(overlap["P2"]) == 4
    assert int(overlap["P3"]) == 0
    # The spec's headline concern — degeneracy concentrates in P2.
    assert int(overlap["P2"]) > int(overlap["P1"])
    assert int(overlap["P2"]) > int(overlap["P3"])


# ===========================================================================
# (D0-7) windows section is a full census (one entry per input window).
# ===========================================================================
def test_windows_section_is_full_census():
    build_report = _build_report()
    windows_in = _windows()
    report = build_report(windows_in, _dense_raw(), _trailing_ic(), _SUBPERIODS)

    windows_out = report["windows"]
    assert isinstance(windows_out, list)
    assert len(windows_out) == len(windows_in)  # all 8 windows enumerated
