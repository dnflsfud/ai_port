# -*- coding: utf-8 -*-
"""§S13.41 — 옵션 IV(iv30_z) 변동성 예측의 공분산 대각 반영 (사전등록 고정).

결정 로그 §S13.41 (2026-08-13) 사전등록 파라미터의 정본. 모든 상수는
사전등록 값으로 고정이며 config로 노출하지 않는다(스윕 금지, §2.4).

모델 (풀드 OLS, 63BD마다 워크포워드 재추정, expanding):
    A: log σ_fwd21 = a + b·log σ_trail126
    B: A + c·iv30_z
학습 표본은 5BD 샘플링, 타깃 윈도우가 추정일 이전에 완결된 관측만 사용
(p + 21 < 추정위치 — 엠바고). 스케일 s = clip(σ̂_B/σ̂_A, 0.8, 1.5),
결측/워밍업은 1.0(inert). 소비자는 Σ_new = diag(s) @ Σ @ diag(s)로
대각만 조정한다(상관 불변·PSD 보존 — 메가캡 수축과 동일 관용구).

사전점검 기록: outputs/s13_41_optvol_precheck/summary.json
(P0 excess-QLIKE +19.1%/MAE +9.1% 3분할 일관, P1 median L1 0.0245).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OPTION_VOL_SHEET = "iv30_z"

FWD = 21              # 타깃: 향후 21BD 실현변동성
TRAIL = 126           # 베이스 예측자: trailing 126BD (cov_lookback과 동일)
EST_FREQ = 63         # 재추정 주기
OBS_STEP = 5          # 학습 관측 샘플링 간격
MIN_TRAIN_ROWS = 500  # 추정 최소 풀드 관측 수
FIRST_EST_POS = 252   # 최초 추정 위치 (BD)
CLIP_LO, CLIP_HI = 0.8, 1.5
_ANN = float(np.sqrt(252.0))


def fit_vol_models(log_trail, z, log_fwd):
    """풀드 OLS: A = [1, log_trail], B = [1, log_trail, z]. (coef_a, coef_b)"""
    lt = np.asarray(log_trail, float)
    zz = np.asarray(z, float)
    lf = np.asarray(log_fwd, float)
    xa = np.column_stack([np.ones_like(lt), lt])
    xb = np.column_stack([np.ones_like(lt), lt, zz])
    coef_a, *_ = np.linalg.lstsq(xa, lf, rcond=None)
    coef_b, *_ = np.linalg.lstsq(xb, lf, rcond=None)
    return coef_a, coef_b


def _predict_scale_block(coef_a, coef_b, trail_block: pd.DataFrame,
                         z_block: pd.DataFrame) -> pd.DataFrame:
    lt = np.log(trail_block.where(trail_block > 0))
    log_ratio = ((coef_b[0] - coef_a[0])
                 + (coef_b[1] - coef_a[1]) * lt
                 + coef_b[2] * z_block)
    s = np.exp(log_ratio).clip(CLIP_LO, CLIP_HI)
    return s.where(np.isfinite(s), 1.0)


def build_option_vol_scale(returns: pd.DataFrame,
                           iv_z: pd.DataFrame) -> pd.DataFrame:
    """워크포워드 대각 스케일 패널 (dates×tickers). 커버리지 밖은 1.0."""
    iv_z = iv_z.reindex(index=returns.index, columns=returns.columns)
    trail = returns.rolling(TRAIL, min_periods=60).std() * _ANN
    fwd = (returns.rolling(FWD).std() * _ANN).shift(-FWD)
    lt_panel = np.log(trail.where(trail > 0))
    lf_panel = np.log(fwd.where(fwd > 0))

    scale = pd.DataFrame(1.0, index=returns.index, columns=returns.columns)
    n = len(returns.index)
    last_models = None
    for e in range(FIRST_EST_POS, n, EST_FREQ):
        train_pos = [p for p in range(0, e - FWD, OBS_STEP) if p + FWD < e]
        if train_pos:
            sub = pd.DataFrame({
                "lt": lt_panel.iloc[train_pos].values.ravel(),
                "z": iv_z.iloc[train_pos].values.ravel(),
                "lf": lf_panel.iloc[train_pos].values.ravel(),
            }).dropna()
            if len(sub) >= MIN_TRAIN_ROWS:
                last_models = fit_vol_models(sub["lt"], sub["z"], sub["lf"])
        if last_models is None:
            continue
        end = min(e + EST_FREQ, n)
        coef_a, coef_b = last_models
        block = _predict_scale_block(
            coef_a, coef_b, trail.iloc[e:end], iv_z.iloc[e:end])
        scale.iloc[e:end] = block.values
    return scale
