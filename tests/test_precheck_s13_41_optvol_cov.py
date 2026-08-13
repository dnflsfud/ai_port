# -*- coding: utf-8 -*-
"""§S13.41 옵션 IV → 공분산 대각 사전점검 — plain-function 단위 테스트."""
import numpy as np
import pandas as pd

from scripts.precheck_s13_41_optvol_cov import (
    excess_qlike,
    fit_vol_models,
    predict_scale,
)


def test_excess_qlike_zero_at_truth_positive_elsewhere():
    s2 = np.array([0.04, 0.09])
    assert np.allclose(excess_qlike(s2, s2), 0.0)
    assert (excess_qlike(s2 * 2, s2) > 0).all()
    assert (excess_qlike(s2 * 0.5, s2) > 0).all()


def test_fit_vol_models_recovers_z_coefficient():
    rng = np.random.default_rng(41)
    n = 4000
    lt = rng.normal(np.log(0.3), 0.3, n)
    z = rng.normal(0.0, 1.0, n)
    lf = 0.05 + 0.9 * lt + 0.15 * z + rng.normal(0.0, 0.05, n)
    coef_a, coef_b = fit_vol_models(lt, z, lf)
    assert len(coef_a) == 2 and len(coef_b) == 3
    assert abs(coef_b[2] - 0.15) < 0.02


def test_predict_scale_clip_nan_and_direction():
    coef_a = np.array([0.0, 1.0])
    coef_b = np.array([0.0, 1.0, 0.2])
    trail = pd.Series([0.3, 0.3, 0.3])
    z = pd.Series([0.0, 3.0, -30.0])
    s = predict_scale(coef_a, coef_b, trail, z, 0.8, 1.5)
    assert abs(s.iloc[0] - 1.0) < 1e-12
    # exp(0.2*3)=1.822 -> 상한 1.5 클립
    assert s.iloc[1] == 1.5
    assert s.iloc[2] == 0.8
    # z 결측 -> inert(1.0)
    s2 = predict_scale(coef_a, coef_b, pd.Series([0.3]), pd.Series([np.nan]), 0.8, 1.5)
    assert s2.iloc[0] == 1.0
