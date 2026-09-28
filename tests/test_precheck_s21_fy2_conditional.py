# -*- coding: utf-8 -*-
"""§S21 FY2 조건부 사전점검 — plain-function 단위 테스트."""
import numpy as np
import pandas as pd

from scripts.precheck_s21_fy2_conditional import (
    agreement_gate,
    chain_target,
    detect_roll_events,
    fiscal_progress,
    gate_verdict,
    ic_row,
    roll_month,
)


def _sales_levels():
    """ROLL: 매년 7월 1일 1FY←직전 2FY(성장 10%) / FLAT: 성장 2%라 검출 대상 아님."""
    idx = pd.date_range("2018-01-01", "2021-12-31", freq="D")
    s1, s2 = {}, {}
    for name, g in (("ROLL", 1.10), ("FLAT", 1.02)):
        n_rolls = np.array([(d.year - 2018) + (d.month >= 7) for d in idx])
        s1[name] = 100.0 * g ** n_rolls
        s2[name] = 100.0 * g ** (n_rolls + 1)
    return pd.DataFrame(s1, index=idx), pd.DataFrame(s2, index=idx)


def test_roll_detection_finds_july_and_skips_low_growth():
    s1, s2 = _sales_levels()
    ev = detect_roll_events(s1, s2)
    assert [d.month for d in ev["ROLL"]] == [7, 7, 7, 7]
    assert roll_month(ev["ROLL"]) == 7
    assert ev["FLAT"] == [] and roll_month(ev["FLAT"]) is None


def test_fiscal_progress_zero_after_roll_and_max_before():
    idx = pd.DatetimeIndex(["2024-02-15", "2024-01-15", "2024-08-15"])
    p = fiscal_progress(idx, ["A"], {"A": 2})
    assert np.allclose(p["A"].to_numpy(), [0.0, 11 / 12, 6 / 12])


def test_chain_target_is_non_overlapping_sum():
    idx = pd.date_range("2024-01-01", periods=100, freq="B")
    t20 = pd.DataFrame({"A": np.arange(100, dtype=float)}, index=idx)
    t60 = chain_target(t20, 3)
    assert t60["A"].iloc[0] == 0 + 20 + 40
    assert t60["A"].iloc[59] == 59 + 79 + 99
    assert t60["A"].iloc[60:].isna().all()


def test_agreement_gate_keeps_agree_zeroes_disagree_nans_missing():
    f1 = pd.DataFrame({"A": [5.0, 5.0, 5.0, -3.0]})
    f2 = pd.DataFrame({"A": [2.0, -2.0, np.nan, -1.0]})
    out = agreement_gate(f1, f2)["A"].tolist()
    assert out[0] == 5.0 and out[1] == 0.0 and np.isnan(out[2]) and out[3] == -3.0


def test_gate_requires_bonferroni_t_and_consistent_thirds():
    rng = np.random.default_rng(0)
    strong = pd.Series(0.05 + 0.01 * rng.standard_normal(600))
    assert gate_verdict(ic_row(strong, 20)) == "PASS"
    mixed = pd.concat([strong.iloc[:400], -0.2 + 0 * strong.iloc[400:]], ignore_index=True)
    assert ic_row(mixed, 20)["thirds_sign_consistent"] is False
    assert gate_verdict(ic_row(mixed, 20)) != "PASS"
