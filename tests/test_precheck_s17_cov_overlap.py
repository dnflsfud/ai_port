# -*- coding: utf-8 -*-
"""§S17.1 Σ 채널 기전 사전점검 헬퍼 단위테스트 (경량 — 데이터 로드 없음)."""
import numpy as np

from scripts.precheck_s17_cov_overlap import (
    G1_ASIA_DATE_FRAC_MIN,
    G1_ASIA_RATIO_MIN,
    G1_US_ABS_DEV_MAX,
    block_mean_corr,
    evaluate_g1,
    nw_ols,
)


def test_evaluate_g1_gates():
    assert (G1_ASIA_RATIO_MIN, G1_ASIA_DATE_FRAC_MIN, G1_US_ABS_DEV_MAX) == (2.0, 0.90, 0.10)
    assert evaluate_g1(asia_ratio_frac_ge2=0.97, us_abs_dev_median=0.05, all_psd=True)["g1_pass"] is True
    assert evaluate_g1(0.80, 0.05, True)["g1_pass"] is False     # 회복 날짜 비율 부족
    assert evaluate_g1(0.97, 0.15, True)["g1_pass"] is False     # 미국 블록 왜곡
    assert evaluate_g1(0.97, 0.05, False)["g1_pass"] is False    # PSD 깨짐


def test_block_mean_corr_uses_correlation_not_covariance():
    cov = np.array([[4.0, 1.0, 0.0],
                    [1.0, 1.0, 0.5],
                    [0.0, 0.5, 1.0]])
    # corr(0,1) = 1/(2*1) = 0.5 ; corr(0,2) = 0 → 블록 [0]x[1,2] 평균 0.25
    assert np.isclose(block_mean_corr(cov, [0], [1, 2]), 0.25)
    # 대각 제외 US×US: corr(1,2)=0.5 양방향 → 0.5
    assert np.isclose(block_mean_corr(cov, [1, 2], [1, 2], exclude_diag=True), 0.5)


def test_nw_ols_recovers_exact_linear_relation():
    x = np.linspace(1.0, 2.0, 40)
    y = 0.5 + 2.0 * x
    fit = nw_ols(y, x, lag=3)
    assert np.isclose(fit["beta"], 2.0) and np.isclose(fit["alpha"], 0.5)
    assert fit["se"] < 1e-8
    assert fit["ci95"][0] <= 2.0 <= fit["ci95"][1]
