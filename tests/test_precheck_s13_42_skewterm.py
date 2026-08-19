# -*- coding: utf-8 -*-
"""§S13.42 skew/term 변동성 모델 증강 사전점검 — plain-function 단위 테스트."""
import numpy as np
import pandas as pd

from scripts.precheck_s13_42_skewterm import fit_ols, ratio_scale


def test_fit_ols_recovers_four_regressor_coefficients():
    rng = np.random.default_rng(42)
    n = 6000
    lt = rng.normal(np.log(0.3), 0.3, n)
    ziv = rng.normal(0, 1, n)
    zsk = rng.normal(0, 1, n)
    ztm = rng.normal(0, 1, n)
    y = 0.05 + 0.9 * lt + 0.12 * ziv + 0.06 * zsk - 0.04 * ztm + rng.normal(0, 0.05, n)
    coef = fit_ols(np.column_stack([lt, ziv, zsk, ztm]), y)
    assert len(coef) == 5  # 절편 + 4
    assert abs(coef[2] - 0.12) < 0.02
    assert abs(coef[3] - 0.06) < 0.02
    assert abs(coef[4] + 0.04) < 0.02


def test_ratio_scale_clip_neutral_and_nan_guard():
    # 분자·분모 동일 모델 → 정확히 1.0
    coef = np.array([0.1, 0.9])
    x = pd.DataFrame({"lt": [np.log(0.3), np.log(0.5)]})
    s = ratio_scale(coef, x[["lt"]], coef, x[["lt"]], 0.8, 1.5)
    assert np.allclose(s, 1.0)
    # 분자 우세 → 상한 클립
    up = np.array([1.0, 0.9])
    s2 = ratio_scale(up, x[["lt"]], coef, x[["lt"]], 0.8, 1.5)
    assert (s2 == 1.5).all()
    # 입력 NaN → inert 1.0
    xn = pd.DataFrame({"lt": [np.nan]})
    s3 = ratio_scale(up, xn[["lt"]], coef, xn[["lt"]], 0.8, 1.5)
    assert float(s3.iloc[0]) == 1.0
