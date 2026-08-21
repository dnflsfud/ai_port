# -*- coding: utf-8 -*-
"""§S13.48-B RMT 벌크 고유값 정제 사전점검 — plain-function 단위 테스트."""
import numpy as np

from scripts.precheck_s13_48b_rmt_denoise import (excess_qlike,
                                                  floor_direction_share,
                                                  mp_lambda_plus, rmt_denoise)


def _synthetic_cov(eigvals, seed):
    """주어진 고유값 스펙트럼을 갖는 무작위 회전 공분산."""
    n = len(eigvals)
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((n, n)))
    return (q * np.asarray(eigvals, dtype=float)) @ q.T


def test_mp_lambda_plus_matches_hand_computation():
    expected = (1.0 + np.sqrt(200.0 / 126.0)) ** 2
    assert np.allclose(mp_lambda_plus(200, 126, 1.0), expected, atol=1e-6)
    assert np.allclose(mp_lambda_plus(200, 126, 0.5), 0.5 * expected, atol=1e-6)


def test_rmt_denoise_preserves_symmetry_and_psd():
    rng = np.random.default_rng(4801)
    x = rng.standard_normal((300, 40))
    cov = np.cov(x, rowvar=False)
    out, diag = rmt_denoise(cov, t_obs=300)
    assert np.allclose(out, out.T, atol=1e-12)
    assert float(np.linalg.eigvalsh(out).min()) >= -1e-10
    assert diag["n_replaced"] >= 2


def test_rmt_denoise_preserves_trace():
    cov = _synthetic_cov(np.concatenate([[50.0], np.linspace(0.5, 1.5, 49)]), 4802)
    out, _ = rmt_denoise(cov, t_obs=126)
    assert np.allclose(np.trace(out), np.trace(cov), atol=1e-8)


def test_rmt_denoise_keeps_spike_and_averages_bulk():
    bulk = np.linspace(0.5, 1.5, 49)          # 산술평균 정확히 1.0
    cov = _synthetic_cov(np.concatenate([[50.0], bulk]), 4803)
    out, diag = rmt_denoise(cov, t_obs=126)
    # lam_plus = (trace/50)*(1+sqrt(50/126))**2 = 1.98*... < 50 → 스파이크 미치환
    assert diag["n_replaced"] == 49
    assert np.allclose(diag["replaced_share"], 49.0 / 50.0, atol=1e-6)
    out_vals = np.sort(np.linalg.eigvalsh(out))
    assert np.allclose(out_vals[-1], 50.0, atol=1e-6)      # 스파이크 보존
    assert np.allclose(out_vals[:-1], 1.0, atol=1e-6)      # 벌크는 평균으로 평탄화


def test_floor_direction_share_tracks_the_active_direction():
    cov = np.diag([1.0, 1e-12])
    floor = 1e-10
    # a가 floor에 닿은(두 번째) 고유방향 → 그 방향이 분산 전부를 설명
    assert np.allclose(floor_direction_share(cov, np.array([0.0, 1.0]), floor),
                       1.0, atol=1e-6)
    # a가 floor 위(첫 번째) 고유방향 → floor 방향 점유율 0
    assert np.allclose(floor_direction_share(cov, np.array([1.0, 0.0]), floor),
                       0.0, atol=1e-6)
    assert np.isnan(floor_direction_share(cov, np.zeros(2), floor))


def test_excess_qlike_zero_at_truth_positive_elsewhere():
    s2 = np.array([0.04, 0.09])
    assert np.allclose(excess_qlike(s2, s2), 0.0, atol=1e-6)
    assert (excess_qlike(s2 * 2, s2) > 0).all()
    assert (excess_qlike(s2 * 0.5, s2) > 0).all()
