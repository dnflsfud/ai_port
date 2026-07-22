# -*- coding: utf-8 -*-
"""compute_specific_returns PIT eligibility 계약 테스트 (§S11.7).

- dense 파리티: NaN 없는 패널에서는 기존 알고리즘(전 열 PCA 기저)과 항등.
- ghost 제외: 창에 상장 전 NaN이 있는 열은 기저·타깃에서 제외되고,
  실상장 열의 타깃은 eligible-only 기저로 계산되며 날짜 스킵이 없어야 한다.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.target_engine import compute_specific_returns


def _panel(n_dates=60, n_tickers=12, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_dates)
    cols = [f"T{i}" for i in range(n_tickers)]
    return pd.DataFrame(
        rng.normal(0.0005, 0.02, size=(n_dates, n_tickers)), index=dates, columns=cols
    )


def _reference_full_basis(returns, n_components, n_remove, lookback, horizon):
    """§S11.7 이전 알고리즘의 inline 참조 구현 (전 열 PCA 기저)."""
    dates = returns.index
    tickers = returns.columns
    cum = (1 + returns).cumprod()
    fwd = cum.shift(-horizon) / cum - 1
    out = pd.DataFrame(np.nan, index=dates, columns=tickers)
    for t in range(lookback, len(dates) - horizon):
        hist = returns.iloc[t - lookback: t]
        hist_clean = hist.loc[hist.notna().all(axis=1)]
        if len(hist_clean) < lookback // 2:
            continue
        actual_n = min(n_components, len(tickers) - 1)
        pca = PCA(n_components=actual_n)
        pca.fit(hist_clean.values)
        fwd_t = fwd.iloc[t].values.reshape(1, -1)
        if np.any(np.isnan(fwd_t)):
            continue
        factors = pca.transform(fwd_t)
        if n_remove < actual_n:
            fp = factors.copy()
            fp[:, n_remove:] = 0
            common = pca.inverse_transform(fp)
        else:
            common = pca.inverse_transform(factors)
        out.iloc[t] = (fwd_t - common).flatten()
    return out


KW = dict(n_components=3, n_remove=1, lookback=10, horizon=2)


def test_dense_parity_with_previous_algorithm():
    returns = _panel()
    got = compute_specific_returns(returns, **KW)
    ref = _reference_full_basis(returns, **KW)
    assert np.allclose(got.values, ref.values, atol=1e-12, equal_nan=True)


def test_ghost_column_excluded_and_reals_use_eligible_basis():
    returns = _panel()
    ghost_end = 30  # T0는 dates[0:30] 상장 전(NaN)
    returns.iloc[:ghost_end, 0] = np.nan

    got = compute_specific_returns(returns, **KW)
    lookback, horizon = KW["lookback"], KW["horizon"]
    n = len(returns)

    real_cols = list(returns.columns[1:])
    ref_eligible = _reference_full_basis(returns[real_cols], **KW)
    ref_full = _reference_full_basis(returns, **KW)

    for t in range(lookback, n - horizon):
        window_has_ghost = t - lookback < ghost_end
        row = got.iloc[t]
        if window_has_ghost:
            # 유령 열은 타깃 없음, 실상장 열은 eligible-only 기저로 계산(스킵 금지)
            assert np.isnan(row.iloc[0])
            assert np.allclose(
                row[real_cols].values, ref_eligible.iloc[t].values, atol=1e-12
            ), f"t={t}: real-col targets should match eligible-only basis"
        else:
            # 창 전체가 실데이터 -> 전 열 기저, 기존 알고리즘과 항등
            assert np.allclose(
                row.values, ref_full.iloc[t].values, atol=1e-12, equal_nan=True
            ), f"t={t}: fully-listed window should match full-basis reference"


def test_thin_eligible_universe_is_skipped_not_crashed():
    returns = _panel(n_tickers=5)
    returns.iloc[:55, 1:] = np.nan  # 유령 4열 -> eligible 1열뿐
    got = compute_specific_returns(returns, **KW)
    # eligible < 2인 구간은 예외 없이 전부 NaN이어야 한다
    assert got.iloc[KW["lookback"]: 40].isna().all().all()


# ---------------------------------------------------------------------------
# §S12.5: pca_vol_standardize — OFF 파리티 + ON은 inline 참조 구현과 항등
# ---------------------------------------------------------------------------
from src.config import PipelineConfig  # noqa: E402


def _reference_vol_standardized(returns, n_components, n_remove, lookback, horizon):
    """§S12.5 표준화 PCA의 inline 참조 구현 (σ 표준화 후 σ 복원)."""
    dates = returns.index
    tickers = returns.columns
    cum = (1 + returns).cumprod()
    fwd = cum.shift(-horizon) / cum - 1
    out = pd.DataFrame(np.nan, index=dates, columns=tickers)
    for t in range(lookback, len(dates) - horizon):
        hist = returns.iloc[t - lookback: t]
        hist_clean = hist.loc[hist.notna().all(axis=1)]
        if len(hist_clean) < lookback // 2:
            continue
        sigma = hist_clean.values.std(axis=0, ddof=1)
        sigma = np.where(np.isfinite(sigma) & (sigma > 1e-12), sigma, 1.0)
        actual_n = min(n_components, len(tickers) - 1)
        pca = PCA(n_components=actual_n)
        pca.fit(hist_clean.values / sigma)
        fwd_t = fwd.iloc[t].values.reshape(1, -1)
        if np.any(np.isnan(fwd_t)):
            continue
        fwd_scaled = fwd_t / sigma
        factors = pca.transform(fwd_scaled)
        if n_remove < actual_n:
            fp = factors.copy()
            fp[:, n_remove:] = 0
            common = pca.inverse_transform(fp)
        else:
            common = pca.inverse_transform(factors)
        out.iloc[t] = ((fwd_scaled - common) * sigma).flatten()
    return out


def test_pca_vol_standardize_off_is_flag_inert():
    returns = _panel()
    got_default = compute_specific_returns(returns, **KW)
    got_off = compute_specific_returns(
        returns, config=PipelineConfig(pca_vol_standardize=False), **KW
    )
    assert np.allclose(
        got_default.values, got_off.values, atol=0.0, equal_nan=True
    )


def test_pca_vol_standardize_matches_inline_reference():
    returns = _panel()
    got = compute_specific_returns(
        returns, config=PipelineConfig(pca_vol_standardize=True), **KW
    )
    ref = _reference_vol_standardized(returns, **KW)
    assert np.allclose(got.values, ref.values, atol=1e-12, equal_nan=True)
    # 표준화 결과는 raw PCA와 실제로 달라야 한다(무의미한 no-op 방지)
    raw = _reference_full_basis(returns, **KW)
    finite = ~np.isnan(ref.values)
    assert not np.allclose(ref.values[finite], raw.values[finite], atol=1e-10)


def test_pca_vol_standardize_guards_zero_vol_column():
    returns = _panel()
    returns["T0"] = 0.0  # 상수(무변동) 열 -> sigma 0 -> 1.0 폴백, 크래시 금지
    got = compute_specific_returns(
        returns, config=PipelineConfig(pca_vol_standardize=True), **KW
    )
    assert np.isfinite(
        got.iloc[KW["lookback"]: len(returns) - KW["horizon"]].values
    ).all()
