# -*- coding: utf-8 -*-
"""§S13.41 옵션 IV 변동성 예측 → 공분산 대각 스케일 모듈 테스트."""
import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.option_vol_cov import (
    CLIP_HI,
    CLIP_LO,
    FIRST_EST_POS,
    build_option_vol_scale,
)


def _synthetic(n_days=700, n_names=40, seed=7, diverge_after=None):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    cols = [f"T{i:02d}" for i in range(n_names)]
    vols = rng.uniform(0.01, 0.03, n_names)
    ret = pd.DataFrame(rng.normal(0.0, vols, (n_days, n_names)), idx, cols)
    z = pd.DataFrame(rng.normal(0.0, 1.0, (n_days, n_names)), idx, cols)
    if diverge_after is not None:
        ret.iloc[diverge_after:] += 0.05
        z.iloc[diverge_after:] += 3.0
    return ret, z


def test_flag_default_off():
    assert PipelineConfig().option_vol_covariance_enabled is False


def test_scale_panel_shape_clip_and_warmup():
    ret, z = _synthetic()
    s = build_option_vol_scale(ret, z)
    assert s.shape == ret.shape
    assert (s.iloc[:FIRST_EST_POS] == 1.0).all().all()  # 최초 추정 전 inert
    assert float(s.values.min()) >= CLIP_LO and float(s.values.max()) <= CLIP_HI
    # 추정 이후 구간은 실제로 1.0이 아닌 값이 존재해야 함 (모델이 작동)
    assert (s.iloc[FIRST_EST_POS + 63:] != 1.0).any().any()


def test_scale_nan_inputs_are_inert():
    ret, z = _synthetic()
    z.iloc[:, 0] = np.nan  # 첫 종목 z 전결측
    s = build_option_vol_scale(ret, z)
    assert (s.iloc[:, 0] == 1.0).all()


def test_causality_future_data_does_not_change_past_scale():
    ret_a, z_a = _synthetic()
    ret_b, z_b = _synthetic(diverge_after=450)
    s_a = build_option_vol_scale(ret_a, z_a)
    s_b = build_option_vol_scale(ret_b, z_b)
    # 450 이전 행은 (모델·입력 모두 과거 동일이므로) 바이트 동일해야 한다
    pd.testing.assert_frame_equal(s_a.iloc[:450], s_b.iloc[:450])
    # 미래 발산은 미래 행에는 반영된다 (온전성)
    assert not s_a.iloc[500:].equals(s_b.iloc[500:])
