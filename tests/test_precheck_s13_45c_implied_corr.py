# -*- coding: utf-8 -*-
"""§S13.45-C implied correlation 사전점검 — plain-function 단위 테스트."""
import numpy as np

from scripts.precheck_s13_45c_implied_corr import (clip_unit_interval,
                                                   expanding_oos_rmse,
                                                   implied_avg_corr,
                                                   realized_avg_corr,
                                                   renorm_valid_weights)


def test_implied_avg_corr_recovers_equicorrelation():
    rng = np.random.default_rng(5)
    n = 12
    rho = 0.35
    sig = rng.uniform(0.15, 0.45, n)
    w = rng.uniform(0.5, 2.0, n)
    w /= w.sum()
    corr = np.full((n, n), rho)
    np.fill_diagonal(corr, 1.0)
    cov = corr * np.outer(sig, sig)
    sig_idx = float(np.sqrt(w @ cov @ w))
    assert abs(implied_avg_corr(sig_idx, sig, w) - rho) < 1e-12
    # 퇴화 가드: 종목 1개(비대각 항 없음) → denom 0 → NaN
    assert np.isnan(implied_avg_corr(0.2, np.array([0.2]), np.array([1.0])))


def test_realized_identity_matches_pairwise_weighted_average():
    rng = np.random.default_rng(9)
    n_days, n = 126, 8
    r = rng.normal(0.0, 0.02, (n_days, n)) + 0.01 * rng.normal(
        0.0, 1.0, (n_days, 1))
    w = rng.uniform(0.5, 2.0, n)
    w /= w.sum()
    got = realized_avg_corr(r, w)
    # 직접 계산: 표본공분산의 pairwise corr을 wσ-가중 평균
    cov = np.cov(r, rowvar=False, ddof=1)
    sig = np.sqrt(np.diag(cov))
    a = np.outer(w * sig, w * sig)
    off = ~np.eye(n, dtype=bool)
    corr = cov / np.outer(sig, sig)
    expected = float((a * corr)[off].sum() / a[off].sum())
    assert abs(got - expected) < 1e-10
    assert 0.0 < got < 1.0


def test_renorm_valid_weights_coverage_gate():
    w = np.array([0.5, 0.3, 0.15, 0.05])
    cov_lo, out_lo = renorm_valid_weights(w, np.array([False, True, True, True]),
                                          min_coverage=0.60)
    assert abs(cov_lo - 0.5) < 1e-12 and out_lo is None
    cov_hi, out_hi = renorm_valid_weights(w, np.array([True, True, False, True]),
                                          min_coverage=0.60)
    assert abs(cov_hi - 0.85) < 1e-12
    assert abs(out_hi.sum() - 1.0) < 1e-12
    assert abs(out_hi[0] - 0.5 / 0.85) < 1e-12


def test_clip_unit_interval_nan_outside():
    assert clip_unit_interval(0.5) == 0.5
    assert clip_unit_interval(0.0) == 0.0
    assert clip_unit_interval(1.0) == 1.0
    assert np.isnan(clip_unit_interval(1.2))
    assert np.isnan(clip_unit_interval(-0.1))
    assert np.isnan(clip_unit_interval(float("nan")))


def test_expanding_oos_rmse_detects_true_incremental_regressor():
    rng = np.random.default_rng(3)
    n = 400
    trail = rng.normal(0.4, 0.1, n)
    icorr = rng.normal(0.5, 0.1, n)
    y = 0.5 * trail + 0.4 * icorr + rng.normal(0.0, 0.01, n)
    out = expanding_oos_rmse(y, trail.reshape(-1, 1),
                             np.column_stack([trail, icorr]))
    assert out["improvement"] >= 0.05
    assert len(out["per_split"]) == 3
    assert all(s["improvement"] > 0 for s in out["per_split"])
    # expanding 구조: 학습 표본이 단조 증가, eval은 다음 블록 1개
    trains = [s["n_train"] for s in out["per_split"]]
    assert trains == sorted(trains) and trains[0] < trains[-1]
    assert sum(s["n_eval"] for s in out["per_split"]) == n - trains[0]
    # 무정보 증강(노이즈 열)은 5% 개선 게이트를 넘지 못한다
    noise = rng.normal(0.0, 1.0, n)
    out_noise = expanding_oos_rmse(y, trail.reshape(-1, 1),
                                   np.column_stack([trail, noise]))
    assert out_noise["improvement"] < 0.05
