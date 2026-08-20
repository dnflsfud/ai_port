# -*- coding: utf-8 -*-
"""§S13.46 implied correlation → 공분산 비대각 스칼라 스케일 모듈 테스트."""
import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.implied_corr_cov import (
    CLIP_HI,
    CLIP_LO,
    EIG_FLOOR,
    MIN_COVERAGE,
    compute_icorr_scale,
    implied_avg_corr,
    realized_avg_corr,
    scale_off_diagonal,
)


def _equicorr_cov(vols, rho):
    v = np.asarray(vols, float)
    corr = np.full((len(v), len(v)), rho)
    np.fill_diagonal(corr, 1.0)
    return corr * np.outer(v, v)


def _sig_idx_for(target_icorr, sig, w):
    """항등식을 역산해 목표 ICorr를 만드는 σ_idx."""
    sig = np.asarray(sig, float)
    w = np.asarray(w, float)
    diag = np.sum(w ** 2 * sig ** 2)
    denom = np.sum(w * sig) ** 2 - diag
    return float(np.sqrt(target_icorr * denom + diag))


def _synthetic_inputs(n_days=126, n_names=8, rho=0.4, seed=11):
    """equicorrelated 수익률 창 + 균등 bm + IV 패널(당일 1행) 생성."""
    rng = np.random.default_rng(seed)
    common = rng.normal(size=(n_days, 1))
    idio = rng.normal(size=(n_days, n_names))
    ret = 0.01 * (np.sqrt(rho) * common + np.sqrt(1.0 - rho) * idio)
    idx = pd.bdate_range("2024-01-01", periods=n_days)
    cols = [f"T{i:02d}" for i in range(n_names)]
    hist = pd.DataFrame(ret, index=idx, columns=cols)
    date = pd.Timestamp("2024-08-01")
    bm_w = np.full(n_names, 1.0 / n_names)
    iv = np.full(n_names, 0.25)
    iv_panel = pd.DataFrame([iv], index=[date], columns=cols)
    return date, iv_panel, hist, bm_w, iv


def test_flag_default_off():
    assert PipelineConfig().implied_corr_covariance_enabled is False


def test_identity_recovers_equicorrelation():
    vols = np.array([0.15, 0.22, 0.30, 0.18, 0.25])
    w = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    for rho in (0.1, 0.45, 0.8):
        cov = _equicorr_cov(vols, rho)
        sig_idx = float(np.sqrt(w @ cov @ w))
        assert abs(implied_avg_corr(sig_idx, vols, w) - rho) < 1e-6


def test_realized_identity_matches_weighted_pairwise_corr():
    rng = np.random.default_rng(3)
    r = rng.normal(0.0, 0.01, (126, 5))
    w = np.array([0.4, 0.2, 0.2, 0.1, 0.1])
    corr = np.corrcoef(r, rowvar=False)
    sig = r.std(axis=0, ddof=1)
    ws = w * sig
    off = np.sum(np.outer(ws, ws) * corr) - np.sum(ws ** 2)
    expected = off / (np.sum(ws) ** 2 - np.sum(ws ** 2))
    assert abs(realized_avg_corr(r, w) - expected) < 1e-10


def test_clip_boundaries_exact():
    date, iv_panel, hist, bm_w, iv = _synthetic_inputs()
    rho_trail = realized_avg_corr(hist.to_numpy(float), bm_w)
    assert 0.0 < rho_trail < 1.0
    # 비율 0.1 → 정확히 CLIP_LO, 비율 3.0(ICorr는 [0,1] 내) → 정확히 CLIP_HI
    for target, expected in ((0.1 * rho_trail, CLIP_LO),
                             (min(3.0 * rho_trail, 0.99), CLIP_HI)):
        spx = pd.Series([_sig_idx_for(target, iv, bm_w)], index=[date])
        s = compute_icorr_scale(date, iv_panel, spx, hist, bm_w)
        assert s == expected


def test_invalid_inputs_are_inert():
    date, iv_panel, hist, bm_w, iv = _synthetic_inputs()
    good_spx = pd.Series([_sig_idx_for(0.5, iv, bm_w)], index=[date])
    # SPX 결측 → 1.0
    nan_spx = pd.Series([np.nan], index=[date])
    assert compute_icorr_scale(date, iv_panel, nan_spx, hist, bm_w) == 1.0
    # 당일 IV 전결측 → 1.0
    nan_panel = iv_panel * np.nan
    assert compute_icorr_scale(date, nan_panel, good_spx, hist, bm_w) == 1.0
    # 패널에 없는 날짜(정확 일치만, ffill 금지) → 1.0
    other = pd.Timestamp("2024-08-04")
    assert compute_icorr_scale(other, iv_panel, good_spx, hist, bm_w) == 1.0
    # coverage < MIN_COVERAGE: 유효 IV 종목의 bm 비중 합 0.5 < 0.60 → 1.0
    sparse = iv_panel.copy()
    sparse.iloc[0, 4:] = np.nan  # 균등 8종목 중 4종만 유효 = coverage 0.5
    assert float(np.full(8, 0.125)[:4].sum()) < MIN_COVERAGE
    assert compute_icorr_scale(date, sparse, good_spx, hist, bm_w) == 1.0
    # ICorr가 [0,1] 밖(σ_idx 과대) → 무효 → 1.0
    big_spx = pd.Series([10.0], index=[date])
    assert compute_icorr_scale(date, iv_panel, big_spx, hist, bm_w) == 1.0


def test_offdiag_scaling_preserves_diagonal_exactly():
    cov = _equicorr_cov([0.15, 0.22, 0.30, 0.18], 0.3)
    for s in (0.5, 0.8, 1.2):
        out = scale_off_diagonal(cov, s)
        assert out is not cov
        assert np.array_equal(np.diag(out), np.diag(cov))  # 대각 바이트 불변
        offmask = ~np.eye(4, dtype=bool)
        assert np.allclose(out[offmask], s * cov[offmask], atol=1e-12)
    # s == 1.0 → 입력 객체 완전 무변(동일 객체 반환)
    assert scale_off_diagonal(cov, 1.0) is cov


def test_psd_repair_floors_eigenvalues():
    # 고상관(0.9) 행렬에 s=2.0 → 유효상관 1.8, PSD 위반 → 수리 발동
    cov = _equicorr_cov([0.20, 0.25, 0.18, 0.22, 0.30], 0.9)
    out = scale_off_diagonal(cov, 2.0)
    min_eig = float(np.linalg.eigvalsh(out).min())
    assert min_eig >= EIG_FLOOR - 1e-12
    assert np.allclose(out, out.T)  # 대칭화 확인
