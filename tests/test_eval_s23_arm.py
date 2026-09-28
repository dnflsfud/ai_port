"""§S23.1 판정 스크립트(scripts/eval_s23_arm.py)의 순수 함수 테스트."""
import numpy as np
import pandas as pd

from scripts.eval_s23_arm import (
    flip_verdict,
    mechanism_b01,
    mechanism_d04,
    mechanism_m01,
)
from src.features.fwd_sales_slope import FWD_SALES_SLOPE_FEATURES


def test_m01_requires_exactly_the_four_slope_features_removed():
    base = ["a", "b", *FWD_SALES_SLOPE_FEATURES]
    assert mechanism_m01(base, ["a", "b"])["pass"] is True
    assert mechanism_m01(base, ["a"])["pass"] is False                      # extra removal
    assert mechanism_m01(base, ["a", "b", "c"])["pass"] is False            # new feature
    assert mechanism_m01(base, base)["pass"] is False                       # nothing removed


def test_b01_needs_changed_labels_anticorrelated_with_momentum():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2024-01-01", periods=30)
    cols = [f"T{i}" for i in range(40)]
    mom = pd.DataFrame(rng.normal(size=(30, 40)), idx, cols)
    arm = pd.DataFrame(rng.normal(size=(30, 40)), idx, cols)
    base = arm - 0.01 * mom                      # legacy label = fixed - (I-P) mu
    ok = mechanism_b01(base, arm, mom)
    assert ok["pass"] is True and ok["median_spearman_diff_vs_momentum"] < -0.9
    assert mechanism_b01(arm + 0.01 * mom, arm, mom)["pass"] is False      # wrong sign
    assert mechanism_b01(arm.copy(), arm, mom)["pass"] is False            # unchanged


def test_d04_needs_identical_alpha_and_moved_weights():
    idx = pd.bdate_range("2024-01-01", periods=3)
    pre = pd.DataFrame({"A": [0.1, np.nan, 0.3], "B": [0.2, 0.1, np.nan]}, idx)
    w = {idx[0]: pd.Series({"A": 0.6, "B": 0.4})}
    w_moved = {idx[0]: pd.Series({"A": 0.55, "B": 0.45})}
    assert mechanism_d04(pre, pre.copy(), w, w_moved)["pass"] is True
    assert mechanism_d04(pre, pre.copy(), w, {idx[0]: w[idx[0]].copy()})["pass"] is False
    assert mechanism_d04(pre, pre + 1e-9, w, w_moved)["pass"] is False


def test_flip_verdict_is_correctness_plus_do_no_harm():
    assert flip_verdict(True, True, True, True) is True
    for i in range(4):
        args = [True] * 4
        args[i] = False
        assert flip_verdict(*args) is False
