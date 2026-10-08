# -*- coding: utf-8 -*-
"""§S25.4 정확성 arm 판정 스크립트의 순수 함수 테스트 (결정 로그 §S25.4 사전등록 규칙 핀)."""

import numpy as np
import pandas as pd

from scripts.eval_s25_4_arms import (
    ARMS,
    REFERENCE_BASE_IR,
    SPREAD_MEDIAN_MIN,
    SPREAD_SHARE_MIN,
    changed_features,
    mechanism_feature_subset,
    mechanism_label_shift,
    mechanism_spread,
    mechanism_stale_mask,
    verdicts,
)


def _panel(n_dates=6, n_tickers=40, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_dates)
    idx = pd.MultiIndex.from_product([dates, [f"t{i}" for i in range(n_tickers)]], names=["date", "ticker"])
    return pd.DataFrame(rng.normal(size=(len(idx), 3)), index=idx, columns=["eps_rev", "fin_pe_level_z", "momentum_252d"])


def test_arm_table_pins_the_four_single_flag_arms():
    assert set(ARMS) == {"s25_4_b02_no_gradual_mask", "s25_4_b03_winsor_first", "s25_4_b05_label_lag", "s25_4_stale_run_mask"}
    assert ARMS["s25_4_b02_no_gradual_mask"]["flag"] == "revision_gradual_mask_disabled"
    assert ARMS["s25_4_stale_run_mask"]["flag"] == "stale_run_mask_enabled"
    assert REFERENCE_BASE_IR == 1.7218790679024922


def test_changed_features_counts_cells_and_subset_rule():
    base = _panel(); arm = base.copy()
    arm.loc[arm.index[:10], "eps_rev"] += 1.0
    ch = changed_features(base, arm)
    assert ch == {"eps_rev": 10}
    ok = mechanism_feature_subset(ch, lambda f: "rev" in f)
    assert ok["pass"] is True and ok["outside"] == [] and ok["changed"] == ["eps_rev"]
    arm.loc[arm.index[:1], "momentum_252d"] = 9.0
    bad = mechanism_feature_subset(changed_features(base, arm), lambda f: "rev" in f)
    assert bad["pass"] is False and bad["outside"] == ["momentum_252d"]
    assert mechanism_feature_subset({}, lambda f: True)["pass"] is False   # a no-op arm is not a mechanism


def test_mechanism_spread_requires_wider_cross_sections():
    base = _panel(seed=1)
    wider = base * 1.3
    out = mechanism_spread(base, wider)
    assert out["pass"] is True and out["median_ratio"] > 1.0 and out["share_ge_1"] == 1.0
    narrower = base * 0.8
    assert mechanism_spread(base, narrower)["pass"] is False
    assert SPREAD_MEDIAN_MIN == 1.0 and SPREAD_SHARE_MIN == 0.9


def test_mechanism_label_shift_requires_audit_horizon_and_changed_targets():
    dates = pd.bdate_range("2021-01-04", periods=5)
    base_t = pd.DataFrame(np.arange(15, dtype=float).reshape(5, 3), index=dates, columns=list("abc"))
    arm_t = base_t + 0.5
    audit_ok = [{"forward_horizon": 21, "embargo_days": 21, "causal_validation_ok": True}] * 3
    out = mechanism_label_shift(base_t, arm_t, audit_ok, lag=1, base_horizon=20)
    assert out["pass"] is True and out["share_changed"] == 1.0
    audit_bad = [{"forward_horizon": 20, "embargo_days": 20, "causal_validation_ok": True}] * 3
    assert mechanism_label_shift(base_t, arm_t, audit_bad, lag=1, base_horizon=20)["pass"] is False
    assert mechanism_label_shift(base_t, base_t, audit_ok, lag=1, base_horizon=20)["pass"] is False


def test_mechanism_stale_mask_requires_every_sheet_and_named_cases():
    dq = {"stale_run_mask": {
        "BEST_PE_RATIO": {"masked_cells": 1200, "tickers": 3, "top": {"RBLX": 1100, "X": 100}},
        "BEST_PX_BPS_RATIO": {"masked_cells": 3000, "tickers": 10, "top": {"VRSN": 2900}},
        "BEST_PEG_RATIO": {"masked_cells": 50, "tickers": 2, "top": {"A": 50}},
        "BEST_EV_TO_BEST_EBITDA": {"masked_cells": 7, "tickers": 1, "top": {"B": 7}},
    }}
    ok = mechanism_stale_mask(dq)
    assert ok["pass"] is True and ok["total_masked_cells"] == 4257 and ok["missing_sheets"] == [] and ok["named_cases_missing"] == []
    dq2 = {"stale_run_mask": {k: v for k, v in dq["stale_run_mask"].items() if k != "BEST_PEG_RATIO"}}
    assert mechanism_stale_mask(dq2)["missing_sheets"] == ["BEST_PEG_RATIO"]
    dq3 = {"stale_run_mask": dict(dq["stale_run_mask"], BEST_PE_RATIO={"masked_cells": 5, "tickers": 1, "top": {"X": 5}})}
    assert mechanism_stale_mask(dq3)["named_cases_missing"] == ["BEST_PE_RATIO/RBLX"] and mechanism_stale_mask(dq3)["pass"] is False
    assert mechanism_stale_mask({})["pass"] is False


def test_verdicts_follow_the_accuracy_frame():
    v = verdicts(g0=True, mechanism=True, e2=True, d_ir=-0.05, split_deltas=[-0.1, 0.02, -0.03])
    assert v["no_harm_pass"] is True and v["flip_candidate"] is True
    v = verdicts(g0=True, mechanism=True, e2=True, d_ir=-0.10, split_deltas=[-0.1, -0.2, -0.3])
    assert v["no_harm_pass"] is False and v["flip_candidate"] is False
    v = verdicts(g0=True, mechanism=False, e2=True, d_ir=0.5, split_deltas=[0.1, 0.2, 0.3])
    assert v["flip_candidate"] is False and v["e1_formal_pass"] is True
    v = verdicts(g0=False, mechanism=True, e2=True, d_ir=0.0, split_deltas=[0.0, 0.0, 0.0])
    assert v["flip_candidate"] is False
