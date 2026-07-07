"""Co-located smoke test for scripts.diagnose_degenerate_retrains.build_report.

The authoritative acceptance suite lives in
tests/acceptance/test_degenerate_diagnosis.py (test-designer owned; not edited
here). This file is a thin smoke check on the pure core so the module has a
name-matching test. Deferred import keeps `pytest --collect-only` green before
the script exists (the intended TDD red state runs inside the test body).
"""

import importlib

import numpy as np
import pandas as pd


def _build_report():
    mod = importlib.import_module("scripts.diagnose_degenerate_retrains")
    return mod.build_report


def _inputs():
    dates = pd.bdate_range("2019-01-02", periods=6)
    raw = pd.DataFrame(
        np.random.default_rng(0).normal(0, 0.03, (6, 25)),
        index=dates,
        columns=[f"T{i}" for i in range(25)],
    )
    ic = pd.Series([0.01, 0.03, -0.02, 0.05, 0.02, 0.04], index=dates)
    windows = [
        {"train_date": pd.Timestamp("2019-06-01"), "degenerate": True,
         "best_iteration": 4, "n_trees": 4, "val_score": None},
        {"train_date": pd.Timestamp("2022-06-01"), "degenerate": False,
         "best_iteration": 200, "n_trees": 200, "val_score": None},
    ]
    subperiods = {
        "P1": (pd.Timestamp("2018-11-23"), pd.Timestamp("2021-05-11")),
        "P2": (pd.Timestamp("2021-05-12"), pd.Timestamp("2023-10-27")),
        "P3": (pd.Timestamp("2023-10-30"), pd.Timestamp("2026-04-13")),
    }
    return windows, raw, ic, subperiods


def test_build_report_smoke():
    build_report = _build_report()
    report = build_report(*_inputs())
    assert isinstance(report, dict)
    for key in ("windows", "subperiod_overlap", "root_cause_evidence",
                "raw_spread_dist", "trailing_ic_dist"):
        assert key in report
    assert isinstance(report["raw_spread_dist"]["n_dates"], int)
    assert len(report["windows"]) == 2
    assert report["subperiod_overlap"]["P1"] == 1
