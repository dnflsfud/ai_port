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
