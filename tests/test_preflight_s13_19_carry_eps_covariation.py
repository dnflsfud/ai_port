# -*- coding: utf-8 -*-
"""§S13.19 사전점검 스크립트 순수 함수 단위테스트 (워크북/CSV I/O 없음)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.preflight_s13_19_carry_eps_covariation import (
    EPS_SPREADS, PRIMARY, assign_terciles, eps_features, half_deltas,
    judge_gates, tercile_delta)


def _synthetic_px(n: int = 320, seed: int = 7) -> pd.DataFrame:
    bd = pd.bdate_range("2020-01-02", periods=n)
    rng = np.random.default_rng(seed)
    cols = ["SPX_FWD_EPS", "NDX_FWD_EPS", "MXWD_FWD_EPS"]
    return pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.005, (n, len(cols))),
                                 axis=0)),
        index=bd, columns=cols)


def test_eps_features_hand_match():
    px = _synthetic_px()
    feats = eps_features(px)
    assert list(feats.columns) == list(EPS_SPREADS)
    assert PRIMARY == "fac_eps_g63"
    eps = {k: np.log(px[f"{k}_FWD_EPS"]) for k in ("SPX", "NDX", "MXWD")}
    np.testing.assert_allclose(
        feats["fac_eps_g63"], eps["MXWD"].diff(63), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        feats["fac_eps_us_lead63"],
        eps["SPX"].diff(63) - eps["MXWD"].diff(63), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        feats["fac_eps_us_lead252"],
        eps["SPX"].diff(252) - eps["MXWD"].diff(252), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        feats["fac_eps_tech_lead63"],
        eps["NDX"].diff(63) - eps["SPX"].diff(63), rtol=0.0, atol=1e-12)


def test_assign_terciles_balanced():
    idx = pd.RangeIndex(96)
    cond = pd.Series(np.arange(96, dtype=float), index=idx)
    b = assign_terciles(cond)
    counts = b.value_counts()
    assert counts["bottom"] == 32 and counts["mid"] == 32 and counts["top"] == 32
    # 단조: 최솟값들은 bottom, 최댓값들은 top
    assert (b.iloc[:32] == "bottom").all() and (b.iloc[-32:] == "top").all()


def test_tercile_delta_hand_match():
    idx = pd.RangeIndex(9)
    cond = pd.Series(np.arange(9, dtype=float), index=idx)
    vals = pd.Series([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 7.0, 8.0, 9.0],
                     index=idx) / 1200.0  # 월간 수익 스케일
    out = tercile_delta(vals, cond)
    # bottom 평균 2/1200, top 평균 8/1200, ×12 연환산
    np.testing.assert_allclose(out["bottom"], 0.02, atol=1e-12)
    np.testing.assert_allclose(out["top"], 0.08, atol=1e-12)
    np.testing.assert_allclose(out["delta"], 0.06, atol=1e-12)


def test_half_deltas_uses_full_sample_buckets():
    idx = pd.RangeIndex(12)
    cond = pd.Series(np.arange(12, dtype=float), index=idx)
    # H1(0..5)은 bottom+mid만, H2(6..11)는 mid+top만 → H1 top 없음 → NaN
    vals = pd.Series(np.ones(12) / 1200.0, index=idx)
    halves = half_deltas(vals, cond)
    assert np.isnan(halves["H1"])
    assert np.isnan(halves["H2"])  # H2에는 bottom 없음


def test_judge_gates_logic():
    full_pass = {"delta": +0.020}
    spec_small = {"delta": +0.005}
    g = judge_gates(full_pass, {"H1": +0.01, "H2": +0.03}, spec_small)
    assert g["G1"] and g["G2"] and g["G3"] and g["verdict"] == "PROCEED"
    # G1 실패: 문턱 1.5%/yr 미달
    g = judge_gates({"delta": +0.010}, {"H1": +0.01, "H2": +0.03}, spec_small)
    assert not g["G1"] and g["verdict"] == "SHELVE"
    # G2 실패: 반분 부호 불일치
    g = judge_gates(full_pass, {"H1": -0.01, "H2": +0.03}, spec_small)
    assert not g["G2"] and g["verdict"] == "SHELVE"
    # G2 실패: 반분 NaN
    g = judge_gates(full_pass, {"H1": np.nan, "H2": +0.03}, spec_small)
    assert not g["G2"] and g["verdict"] == "SHELVE"
    # G3 실패: spec 쪽 델타가 더 큼 (국지화 실패)
    g = judge_gates(full_pass, {"H1": +0.01, "H2": +0.03}, {"delta": -0.025})
    assert not g["G3"] and g["verdict"] == "SHELVE"
