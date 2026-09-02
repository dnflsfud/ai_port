# -*- coding: utf-8 -*-
"""§S17 M2 (Σ 채널) — 시차 비동기 거래 보정: 상관행렬만 K일 겹침 수익률로 추정
(cov_corr_overlap_enabled, K = cov_corr_overlap_days = 5, 사전등록 상수·스윕 금지).

결함(결정 로그 §S17 P3): 아시아 16종은 미국 종가 이후에 거래되므로 일별 수익률의
동시 상관이 구조적으로 과소(ASIA×US 블록 126d Σ 상관 0.044 vs 5일 겹침 0.201,
97/97 리밸일 3.1×). Σ의 대각(일별 분산)은 정합하므로 상관행렬만 K일 겹침 합
수익률로 다시 추정하고 Σ' = D·C_K·D 로 재조립한다(PSD 유지). 분기(LW / pairwise)는
일별 경로와 동일하게 유지. OFF(기본)는 바이트 동일.
"""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src.config import PipelineConfig
from src.portfolio_optimizer import estimate_covariance

N = 200
LOOKBACK = 126


def _lead_lag_panel(seed=0, with_nan=False):
    """A = 충격_t, B = 0.8·충격_{t-1}(하루 시차), C = 독립. 일별 corr(A,B)≈0,
    5일 겹침 corr(A,B)≈0.65 (4/5 겹침)."""
    rng = np.random.default_rng(seed)
    shock = rng.normal(0.0, 0.010, N + 1)
    a = shock[1:] + rng.normal(0.0, 0.003, N)
    b = 0.8 * shock[:-1] + rng.normal(0.0, 0.005, N)
    c = rng.normal(0.0, 0.012, N)
    df = pd.DataFrame({"A": a, "B": b, "C": c},
                      index=pd.bdate_range("2024-01-01", periods=N))
    if with_nan:
        df.iloc[N - LOOKBACK: N - LOOKBACK + 40, 2] = np.nan   # C 늦은 상장 → pairwise 분기
    return df


def _corr(cov):
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)


def _cfg(**kw):
    return PipelineConfig(cov_megacap_vol_shrink_enabled=False, **kw)


def test_s17_cov_corr_overlap_flag_defaults():
    cfg = PipelineConfig()
    assert cfg.cov_corr_overlap_enabled is False
    assert cfg.cov_corr_overlap_days == 5


def test_off_parity_matches_ledoit_wolf_reference():
    df = _lead_lag_panel()
    ref = LedoitWolf().fit(df.iloc[-LOOKBACK:].values).covariance_
    off_default = estimate_covariance(df, config=_cfg())
    off_explicit = estimate_covariance(df, config=_cfg(cov_corr_overlap_enabled=False))
    assert np.array_equal(off_default, ref)
    assert np.array_equal(off_explicit, ref)


def test_on_recovers_lagged_comovement_and_keeps_diagonal():
    df = _lead_lag_panel()
    off = estimate_covariance(df, config=_cfg())
    on = estimate_covariance(df, config=_cfg(cov_corr_overlap_enabled=True))
    assert abs(_corr(off)[0, 1]) < 0.20          # 일별: 시차 공변동이 보이지 않음
    assert _corr(on)[0, 1] > 0.40                # 5일 겹침: 회복
    assert np.allclose(np.diag(on), np.diag(off))  # 대각(일별 분산) 불변
    assert np.linalg.eigvalsh(on).min() >= -1e-12  # PSD
    assert np.allclose(on, on.T)


def test_on_pairwise_branch_with_missing_history():
    df = _lead_lag_panel(with_nan=True)
    off = estimate_covariance(df, config=_cfg())
    on = estimate_covariance(df, config=_cfg(cov_corr_overlap_enabled=True))
    assert on.shape == (3, 3) and np.all(np.isfinite(on))
    assert abs(_corr(off)[0, 1]) < 0.20
    assert _corr(on)[0, 1] > 0.40
    assert np.allclose(np.diag(on), np.diag(off))
    assert np.linalg.eigvalsh(on).min() >= -1e-12


def test_on_with_one_day_window_is_identical_to_off():
    df = _lead_lag_panel()
    off = estimate_covariance(df, config=_cfg())
    on1 = estimate_covariance(
        df, config=_cfg(cov_corr_overlap_enabled=True, cov_corr_overlap_days=1))
    assert np.array_equal(on1, off)
