"""§S28.1 arm judge — pure helpers (the script itself needs the four run directories)."""

import pandas as pd
import pytest

from scripts.eval_s28_1_kfold_arm import (
    DIST_RATIO_MAX,
    MAX_DEGENERATE,
    NO_HARM_DELTA_IR,
    NO_HARM_T,
    book_distance,
    degenerate_flips,
    evaluate_g1,
    evaluate_no_harm,
)

D1, D2, D3 = (pd.Timestamp(x) for x in ("2026-08-06", "2026-09-04", "2026-10-06"))


def test_preregistered_bars():
    assert MAX_DEGENERATE == 1 and DIST_RATIO_MAX == 0.6
    assert NO_HARM_DELTA_IR == -0.10 and NO_HARM_T == -1.0


def test_book_distance_is_the_mean_one_way_l1_on_the_given_dates():
    a = {D1: pd.Series({"A": 0.6, "B": 0.4}), D2: pd.Series({"A": 0.5, "B": 0.5}), D3: pd.Series({"A": 1.0})}
    b = {D1: pd.Series({"A": 0.5, "B": 0.5}), D2: pd.Series({"A": 0.5, "C": 0.5}), D3: pd.Series({"A": 1.0})}
    out = book_distance(a, b, [D1, D2])                 # D3 excluded by the caller's common-date set
    assert out["n"] == 2
    assert out["by_date"] == pytest.approx([0.1, 0.5])   # union of tickers, missing = 0
    assert out["mean"] == pytest.approx(0.3) and out["last"] == pytest.approx(0.5)


def test_degenerate_flips_counts_slots_that_change_state_between_vintages():
    slots_a = ["2025-05-02", "2025-08-04", "2025-10-31", "2026-02-03"]
    slots_b = ["2025-05-02", "2025-08-04", "2025-10-31", "2026-02-03", "2026-05-05"]   # one extra retrain
    out = degenerate_flips(slots_a, ["2025-05-02", "2025-10-31"], slots_b, ["2025-08-04", "2025-10-31", "2026-05-05"])
    assert out["common_slots"] == 4
    assert out["flipped"] == ["2025-05-02", "2025-08-04"]          # 2026-05-05 is not a common slot


def test_g1_needs_both_the_degenerate_bar_and_the_distance_ratio():
    ok = evaluate_g1({"1106": 1, "1021": 0}, dist_arm=0.03, dist_ctl=0.05)
    assert ok["distance_ratio"] == pytest.approx(0.6) and ok["g1_pass"] is True
    assert evaluate_g1({"1106": 2, "1021": 0}, dist_arm=0.01, dist_ctl=0.05)["g1_pass"] is False
    assert evaluate_g1({"1106": 0, "1021": 0}, dist_arm=0.031, dist_ctl=0.05)["g1_pass"] is False
    assert evaluate_g1({"1106": 0, "1021": 0}, dist_arm=0.01, dist_ctl=0.0)["g1_pass"] is False   # undefined ratio


def test_no_harm_uses_the_mean_delta_ir_and_every_paired_t():
    assert evaluate_no_harm([-0.15, 0.00], [-0.5, 0.2]) is True      # mean -0.075 > -0.10
    assert evaluate_no_harm([-0.15, -0.06], [-0.5, 0.2]) is False    # mean -0.105
    assert evaluate_no_harm([0.05, 0.05], [-1.2, 0.9]) is False      # one paired t below -1
    assert evaluate_no_harm([0.05, 0.05], [float("nan"), 0.9]) is False
