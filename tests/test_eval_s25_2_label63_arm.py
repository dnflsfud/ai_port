# -*- coding: utf-8 -*-
"""§S25.2 G4-01 63BD 라벨 arm 판정 스크립트의 순수 함수 테스트 (결정 로그 §S25.2 사전등록 규칙 핀)."""

import numpy as np
import pandas as pd

from scripts.eval_s25_2_label63_arm import (
    ARM_HORIZON,
    REFERENCE_BASE_IR,
    TURNOVER_MAX,
    TURNOVER_REDUCED,
    evaluate_g0,
    label_ic_series,
    mechanism_label_ic,
    mechanism_split_audit,
    verdicts,
)

WB = {"data_path": "x/ai_signal_data.xlsx", "data_mtime_utc": "2026-09-30T05:27:35Z", "data_size_bytes": 392851329}
FX_A = {"fx_source_path": "x/Index.xlsx", "fx_mtime_utc": "2026-09-30T02:31:05Z", "fx_size_bytes": 97194523}
FX_B = {"fx_source_path": "x/Index.xlsx", "fx_mtime_utc": "2026-10-02T02:30:43Z", "fx_size_bytes": 97242444}


def test_constants_pin_preregistered_bars():
    assert ARM_HORIZON == 63
    assert REFERENCE_BASE_IR == 1.8327704100987559
    assert TURNOVER_MAX == 1.25
    assert TURNOVER_REDUCED == 0.90


def test_g0_pass_when_same_vintage_and_base_reproduces():
    out = evaluate_g0({**WB, **FX_A}, {**WB, **FX_A}, REFERENCE_BASE_IR)
    assert out["workbook_equal"] and out["fx_equal"] and out["fx_inert_proven"] and out["g0_pass"]


def test_g0_fx_refresh_needs_same_vintage_proof_run_that_reproduces_reference():
    base, arm = {**WB, **FX_A}, {**WB, **FX_B}
    assert evaluate_g0(base, arm, REFERENCE_BASE_IR)["g0_pass"] is False
    ok = evaluate_g0(base, arm, REFERENCE_BASE_IR, proof_vintage={**WB, **FX_B}, proof_ir=REFERENCE_BASE_IR)
    assert ok["fx_equal"] is False and ok["fx_inert_proven"] is True and ok["g0_pass"] is True
    # a proof run on yet another vintage, or one that does not reproduce the reference, proves nothing
    assert evaluate_g0(base, arm, REFERENCE_BASE_IR, proof_vintage={**WB, **FX_A}, proof_ir=REFERENCE_BASE_IR)["g0_pass"] is False
    assert evaluate_g0(base, arm, REFERENCE_BASE_IR, proof_vintage={**WB, **FX_B}, proof_ir=REFERENCE_BASE_IR + 1e-6)["g0_pass"] is False


def test_g0_fails_when_workbook_differs_or_base_does_not_reproduce():
    other_wb = {**WB, "data_mtime_utc": "2026-10-08T05:00:00Z"}
    assert evaluate_g0({**WB, **FX_A}, {**other_wb, **FX_A}, REFERENCE_BASE_IR)["g0_pass"] is False
    assert evaluate_g0({**WB, **FX_A}, {**WB, **FX_A}, REFERENCE_BASE_IR + 1e-6)["g0_pass"] is False


def _audit(n, horizon=63, embargo=63, ok=True):
    return [{"prediction_date": f"2020-01-{i + 1:02d}", "forward_horizon": horizon, "embargo_days": embargo,
             "causal_validation_ok": ok} for i in range(n)]


def test_split_audit_requires_every_retrain_on_the_63d_label_with_embargo_and_causal_ok():
    assert mechanism_split_audit(_audit(5))["pass"] is True
    assert mechanism_split_audit(_audit(5, horizon=20, embargo=20))["pass"] is False
    assert mechanism_split_audit(_audit(5, embargo=20))["pass"] is False
    assert mechanism_split_audit(_audit(5, ok=False))["pass"] is False
    assert mechanism_split_audit(_audit(4) + _audit(1, horizon=20))["pass"] is False
    assert mechanism_split_audit([])["pass"] is False


def test_label_ic_series_is_spearman_per_date_and_skips_thin_dates():
    dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
    cols = [f"t{i}" for i in range(40)]
    rng = np.random.default_rng(0)
    pred = pd.DataFrame(rng.normal(size=(2, 40)), index=dates, columns=cols)
    targ = pred.copy()
    targ.iloc[1, :] = np.nan
    targ.iloc[1, :10] = 1.0  # only 10 valid cells on the second date (< min_n)
    out = label_ic_series(pred, targ, dates, min_n=30)
    assert np.isclose(out.loc[dates[0]], 1.0)
    assert np.isnan(out.loc[dates[1]])


def test_mechanism_label_ic_pass_only_when_arm_explains_the_63d_label_better():
    dates = pd.date_range("2020-01-01", periods=12, freq="21B")
    base = pd.Series(0.05, index=dates)
    arm = pd.Series(np.linspace(0.06, 0.09, 12), index=dates)
    out = mechanism_label_ic(base, arm)
    assert out["pass"] is True and out["delta_mean"] > 0 and out["paired_t"] > 0 and out["n_dates"] == 12
    assert out["share_arm_higher"] == 1.0
    worse = mechanism_label_ic(base, base - 0.01)
    assert worse["pass"] is False and worse["delta_mean"] < 0
    # dates present in only one series are dropped (paired comparison)
    partial = mechanism_label_ic(base, arm.iloc[:6])
    assert partial["n_dates"] == 6


def test_verdict_matrix_pins_the_preregistered_interpretation():
    # formal E1: dIR > +0.36 and all three splits positive -> adoption candidate
    v = verdicts(True, True, True, 0.40, [0.1, 0.2, 0.3], turnover_ratio=0.95)
    assert v["e1_formal_pass"] and v["adoption_candidate"] and not v["two_model_followup"] and not v["axis_closed"]
    # mechanism PASS but E1 missed -> G4-01 not adopted, two-model mu combination is the next pre-registration
    v = verdicts(True, True, True, 0.05, [0.1, -0.1, 0.1], turnover_ratio=0.85)
    assert not v["e1_formal_pass"] and not v["adoption_candidate"] and v["two_model_followup"] and not v["axis_closed"]
    assert v["turnover_reduced"] is True
    # override candidate (S13.50 policy): dIR > 0, all splits positive, gates -- reported only
    v = verdicts(True, True, True, 0.20, [0.1, 0.05, 0.02], turnover_ratio=1.0)
    assert v["override_candidate"] is True and v["adoption_candidate"] is False
    # mechanism FAIL (no incremental 63d explanatory power) -> axis closed: G4-01 and two-model both shelved
    v = verdicts(True, False, True, 0.40, [0.1, 0.2, 0.3], turnover_ratio=1.0)
    assert v["axis_closed"] and not v["adoption_candidate"] and not v["two_model_followup"]
    # G0 FAIL -> nothing can be concluded
    v = verdicts(False, True, True, 0.40, [0.1, 0.2, 0.3], turnover_ratio=1.0)
    assert not v["adoption_candidate"] and not v["two_model_followup"] and not v["axis_closed"]
    assert v["no_harm_pass"] is True
