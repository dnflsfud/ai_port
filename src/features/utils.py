"""Shared utility functions for feature engineering."""

import pandas as pd
import numpy as np


def cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional Z-score 정규화 (Round 4 방식).
    각 날짜에서 종목 간 평균/표준편차로 표준화.
    Scale 정보를 보존하여 실제 수익률 예측에 유리.
    """
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0)


def safe_pct_change(df: pd.DataFrame, periods: int) -> pd.DataFrame:
    shifted = df.shift(periods)
    denom = shifted.replace(0, np.nan)
    return (df - shifted) / denom.abs()


def clip_outliers(df: pd.DataFrame, n_std: float = 5.0) -> pd.DataFrame:
    return df.clip(lower=-n_std, upper=n_std)


def cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank."""
    return df.rank(axis=1, pct=True)


def rolling_tsz(
    df: pd.DataFrame,
    window: int = 756,
    min_periods: int = 252,
) -> pd.DataFrame:
    """Per-ticker rolling time-series z-score.

    (x_t - rolling_mean(x, w)) / rolling_std(x, w)

    Each column gets its OWN history-based standardization, so the
    resulting value answers "how far is this ticker from its own
    recent norm" rather than "how does this ticker compare to the
    universe today". Intended for valuation metrics (PE, PEG, P/B,
    EV/EBITDA) where cross-sectional comparison of raw levels
    disproportionately penalizes structurally high-multiple names
    (e.g. NVDA) even when they are cheap relative to their own
    history. Follow with cross_sectional_zscore to get a final
    cross-section ranking in "history-relative" space.

    Window defaults to 756 business days (~3 years); min_periods=252
    (~1 year) lets early-history features ramp up gradually instead
    of being fully masked for 3 years.
    """
    mean = df.rolling(window, min_periods=min_periods).mean()
    std = df.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
    return (df - mean) / std
