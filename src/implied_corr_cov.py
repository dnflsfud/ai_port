# -*- coding: utf-8 -*-
"""§S13.46 — implied correlation → 공분산 비대각 스칼라 스케일 (사전등록 고정).

결정 로그 §S13.46 (2026-08-20) 사전등록 파라미터의 정본. 모든 상수는
사전등록 값으로 고정이며 config로 노출하지 않는다(스윕 금지).

리밸런싱 시점별 스칼라 s_t = clip(ICorr_t / ρ̄_trail,t, 0.5, 2.0):
    ICorr_t   = (σ_idx² − Σw²σ²) / ((Σwσ)² − Σw²σ²), σ = iv30 시트의 당일 값
                (정확 날짜 일치만, ffill 금지), idx = SPX 열, w = 당일 bm
                비중을 IV 유효 종목으로 renorm. coverage ≥ 0.60 게이트,
                [0,1] 밖·denom ≤ 1e-12 → 무효.
    ρ̄_trail,t = 동일 항등식을 optimizer에 전달된 hist_returns 창(공분산이
                소비하는 trailing look-ahead-free 창)의 실현 σᵢ(ddof=1)와
                실현 포트 σ로 계산. 종목 집합 = IV 유효 ∩ 창내 dense 교집합,
                동일 renorm·coverage 게이트.
무효·워밍업·커버리지 미달·비유한·ρ̄_trail ≤ 0 → s_t = 1.0 (inert).
소비자는 Σ′_ij = s_t·Σ_ij (i≠j), 대각 정확히 불변으로 적용하고, s_t ≠ 1.0일
때만 고유값 하한 EIG_FLOOR로 PSD 수리한다(eigh → clip → 재구성 → 대칭화).

사전점검 기록: outputs/s13_45c_implied_corr/summary.json
(§S13.45-C: P0' NW t +8.43, P1' OOS RMSE +11.6% 3분할 일관 — PASS).
헬퍼 시맨틱은 scripts/precheck_s13_45c_implied_corr.py의 검증된 정의와 동일
(src는 scripts에 의존할 수 없어 재구현).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

IV_SHEET = "iv30"
MIN_COVERAGE = 0.60
CLIP_LO, CLIP_HI = 0.5, 2.0
EIG_FLOOR = 1e-8


def implied_avg_corr(sig_idx: float, sig: np.ndarray, w: np.ndarray) -> float:
    """사전등록 항등식 ρ̄ = (σ_idx² − Σw²σ²)/((Σwσ)² − Σw²σ²). denom≤0 → NaN."""
    sig = np.asarray(sig, float)
    w = np.asarray(w, float)
    diag = float(np.sum(w ** 2 * sig ** 2))
    denom = float(np.sum(w * sig)) ** 2 - diag
    if not np.isfinite(denom) or denom <= 1e-12:
        return float("nan")
    return (float(sig_idx) ** 2 - diag) / denom


def realized_avg_corr(window_returns: np.ndarray, w: np.ndarray) -> float:
    """동일 항등식의 실현판: σᵢ = 창내 std(ddof=1), σ_idx = (창수익률@w)의 std."""
    r = np.asarray(window_returns, float)
    sig = r.std(axis=0, ddof=1)
    sig_p = float((r @ np.asarray(w, float)).std(ddof=1))
    return implied_avg_corr(sig_p, sig, w)


def renorm_valid_weights(w: np.ndarray, valid: np.ndarray,
                         min_coverage: float = MIN_COVERAGE):
    """(coverage, renormed w|None) — 유효 종목 bm 비중 합이 게이트 미달이면 None."""
    w = np.asarray(w, float)
    valid = np.asarray(valid, bool)
    coverage = float(w[valid].sum())
    if not np.isfinite(coverage) or coverage < min_coverage:
        return coverage, None
    return coverage, w[valid] / coverage


def clip_unit_interval(x: float) -> float:
    """사전등록: [0,1] 밖 값은 NaN (경계 포함 유지)."""
    return float(x) if np.isfinite(x) and 0.0 <= x <= 1.0 else float("nan")


def build_iv_panel(data, dates, tickers):
    """iv30 시트 → (iv_panel dates×tickers, spx_iv Series). 불가하면 None.

    접근 관용은 precheck와 동일: raw 시트 → to_numeric, 종목 열은
    'TICKER ...' 접두 매핑, SPX 열은 'SPX' 포함 열. reindex는 정확 날짜
    일치만 남긴다(ffill 금지). % 표기(중앙값 > 3)면 소수로 통일 —
    항등식은 스케일 불변이므로 결과에는 영향 없음."""
    raw = getattr(data, "raw", None)
    if raw is None or IV_SHEET not in raw:
        return None
    iv_raw = raw[IV_SHEET].apply(pd.to_numeric, errors="coerce")
    iv_raw.index = pd.to_datetime(iv_raw.index)
    spx_cols = [c for c in iv_raw.columns if "SPX" in str(c)]
    if not spx_cols:
        return None
    stock_cols = {str(c).split(" ")[0]: c for c in iv_raw.columns
                  if c not in spx_cols}
    iv_panel = pd.DataFrame(
        {t: iv_raw[stock_cols[t]] for t in tickers if t in stock_cols}
    ).reindex(index=dates, columns=list(tickers))
    spx_iv = iv_raw[spx_cols[0]].reindex(dates)
    if float(np.nanmedian(iv_panel.values)) > 3.0:
        iv_panel /= 100.0
        spx_iv /= 100.0
    return iv_panel, spx_iv


def compute_icorr_scale(date, iv_panel: pd.DataFrame, spx_iv: pd.Series,
                        hist_returns: pd.DataFrame, bm_w: np.ndarray) -> float:
    """리밸런싱 시점 스칼라 s_t. 모든 무효 조건은 1.0 (inert)으로 수렴."""
    if date not in iv_panel.index:
        return 1.0
    spx = spx_iv.loc[date]
    if not np.isfinite(spx):
        return 1.0
    iv_row = iv_panel.loc[date].reindex(hist_returns.columns).to_numpy(float)
    w_full = np.asarray(bm_w, float)
    valid_iv = np.isfinite(iv_row) & (w_full > 0)
    _, w_iv = renorm_valid_weights(w_full, valid_iv)
    if w_iv is None:
        return 1.0
    icorr = clip_unit_interval(
        implied_avg_corr(float(spx), iv_row[valid_iv], w_iv))
    if not np.isfinite(icorr):
        return 1.0
    ret = hist_returns.to_numpy(float)
    if ret.shape[0] < 2:  # ddof=1 워밍업
        return 1.0
    valid_c = valid_iv & np.isfinite(ret).all(axis=0)
    _, w_c = renorm_valid_weights(w_full, valid_c)
    if w_c is None:
        return 1.0
    rho_trail = realized_avg_corr(ret[:, valid_c], w_c)
    if not np.isfinite(rho_trail) or rho_trail <= 0:
        return 1.0
    s = icorr / rho_trail
    if not np.isfinite(s):
        return 1.0
    return float(min(max(s, CLIP_LO), CLIP_HI))


def scale_off_diagonal(cov_matrix: np.ndarray, s: float) -> np.ndarray:
    """Σ′_ij = s·Σ_ij (i≠j), 대각 원소는 정확히 불변.

    s == 1.0이면 입력 객체를 그대로 반환(동일 객체 경로 — 구조적 parity).
    s ≠ 1.0일 때만 PSD 수리: 최소 고유값이 EIG_FLOOR 미만이면 eigh →
    eigvals clip → 재구성 → 0.5(A+Aᵀ) 대칭화. 위반이 없으면 스케일 결과를
    그대로 두어 대각이 바이트 동일하게 유지된다."""
    if s == 1.0:
        return cov_matrix
    cov = np.asarray(cov_matrix, float)
    out = cov * float(s)
    np.fill_diagonal(out, np.diag(cov))
    eigvals, eigvecs = np.linalg.eigh(out)
    if float(eigvals.min()) < EIG_FLOOR:
        out = (eigvecs * np.maximum(eigvals, EIG_FLOOR)) @ eigvecs.T
        out = 0.5 * (out + out.T)
    return out
