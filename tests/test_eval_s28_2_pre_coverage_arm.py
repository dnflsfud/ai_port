"""§S28.2 arm judge — pure helpers (the script itself needs the two run directories)."""

from scripts.eval_s28_2_pre_coverage_arm import (
    FLAG,
    NO_HARM_DELTA_IR,
    NO_HARM_T,
    mechanism_mask,
    overrides_differ_only_by_flag,
)


def test_preregistered_bars_and_flag():
    assert FLAG == "pre_coverage_mask_enabled"
    assert NO_HARM_DELTA_IR == -0.10 and NO_HARM_T == -1.0


def test_g0_requires_the_arm_to_be_the_base_plus_exactly_the_flag():
    base = {"train_window": 1260, "kfold_capacity_refit_enabled": True}
    assert overrides_differ_only_by_flag(base, {**base, FLAG: True}) is True
    assert overrides_differ_only_by_flag(base, dict(base)) is False                       # flag missing
    assert overrides_differ_only_by_flag(base, {**base, FLAG: False}) is False            # flag not on
    assert overrides_differ_only_by_flag(base, {**base, FLAG: True, "train_window": 756}) is False
    assert overrides_differ_only_by_flag({**base, FLAG: True}, {**base, FLAG: True}) is False   # base already on


def test_mechanism_needs_masked_cells_in_the_recorded_diagnostics():
    dq = {"pre_coverage_mask": {
        "BEST_GROSS_MARGIN": {"masked_cells": 1200, "tickers": 53, "never_observed_tickers": 2, "top": {"APO": 300}},
        "BEST_EPS": {"masked_cells": 0, "tickers": 0, "never_observed_tickers": 0, "top": {}},
    }}
    out = mechanism_mask(dq)
    assert out["total_masked_cells"] == 1200 and out["sheets_with_masked_cells"] == 1 and out["pass"] is True
    assert out["by_sheet"]["BEST_GROSS_MARGIN"] == {"masked_cells": 1200, "tickers": 53, "never_observed_tickers": 2}
    assert mechanism_mask({"pre_coverage_mask": {"BEST_EPS": {"masked_cells": 0, "tickers": 0}}})["pass"] is False
    assert mechanism_mask({})["pass"] is False and mechanism_mask(None)["pass"] is False
