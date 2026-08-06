# -*- coding: utf-8 -*-
"""price 피처의 PIT returns 소비 계약 (§S11.7).

build_price_features는 dense `data.returns`(유령=median-fill)가 아니라
`data.returns_masked`(상장 전 NaN)를 소비해야 한다. 유령의 합성 수익률이
횡단면 순위·시장평균·비율형 피처에 참여하면 안 된다.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.features.price import (
    build_price_features,
    lagged_cap_weighted_market_return,
    rolling_market_model_idio_vol,
)


def _stub(n_dates=40, listing_pos=30, seed=1):
    dates = pd.bdate_range("2022-01-03", periods=n_dates)
    rng = np.random.default_rng(seed)
    dense = pd.DataFrame(
        rng.normal(0.0, 0.02, size=(n_dates, 3)),
        index=dates,
        columns=["AAA", "BBB", "NEW"],
    )
    masked = dense.copy()
    masked.iloc[:listing_pos, 2] = np.nan  # NEW는 listing_pos부터 상장
    prices = (1 + dense).cumprod() * 100.0
    mktcap = prices * 10.0
    data = SimpleNamespace(
        returns=dense, returns_masked=masked, prices=prices, market_cap=mktcap
    )
    return data, dates, listing_pos


def test_mom_rank_excludes_pre_listing_ghost():
    data, dates, listing_pos = _stub()
    features = build_price_features(data)
    d = dates[listing_pos - 1]  # 상장 전 마지막 날 (>= 21 obs)
    row = features["mom_rank_21d"].loc[d]
    assert np.isnan(row["NEW"])
    # 순위는 상장 2종 사이에서만: pct rank == {0.5, 1.0}
    assert set(np.round(row[["AAA", "BBB"]].values, 6)) == {0.5, 1.0}


def test_relative_momentum_market_mean_excludes_ghost():
    data, dates, listing_pos = _stub()
    features = build_price_features(data)
    d = dates[listing_pos - 1]
    ew = data.returns_masked.mean(axis=1)
    expected = (
        data.returns_masked.rolling(21, min_periods=21).sum()
        .sub(ew.rolling(21, min_periods=21).sum(), axis=0)
    )
    assert features["rel_mom_21d"].loc[d, "AAA"] == pytest.approx(
        expected.loc[d, "AAA"], abs=1e-12
    )


def test_comparison_features_are_nan_not_zero_pre_listing():
    """NaN>0=False가 0.0으로 새면 안 된다 (pos_ret_ratio·trend_consist 가드)."""
    data, dates, listing_pos = _stub()
    features = build_price_features(data)
    d = dates[listing_pos - 1]
    assert np.isnan(features["pos_ret_ratio_21d"].loc[d, "NEW"])
    assert np.isnan(features["trend_consist_63d"].loc[d, "NEW"])


def test_lagged_cap_weighted_market_return_uses_prior_day_caps():
    dates = pd.bdate_range("2026-01-01", periods=3)
    returns = pd.DataFrame(
        {"A": [0.01, 0.10, 0.03], "B": [0.02, 0.00, 0.04]}, index=dates
    )
    caps = pd.DataFrame(
        {"A": [100.0, 900.0, 900.0], "B": [300.0, 100.0, 100.0]},
        index=dates,
    )

    market = lagged_cap_weighted_market_return(returns, caps)

    assert np.isnan(market.iloc[0])
    assert market.iloc[1] == pytest.approx(0.10 * 0.25 + 0.00 * 0.75)
    assert market.iloc[2] == pytest.approx(0.03 * 0.90 + 0.04 * 0.10)


def test_market_model_idio_vol_matches_window_ols_residual_std():
    rng = np.random.default_rng(19)
    dates = pd.bdate_range("2025-01-02", periods=80)
    market = pd.Series(rng.normal(0.0004, 0.012, len(dates)), index=dates)
    eps = rng.normal(0.0, 0.007, len(dates))
    stock = 0.0002 + 1.35 * market.to_numpy() + eps
    returns = pd.DataFrame({"A": stock}, index=dates)

    result = rolling_market_model_idio_vol(
        returns, market, window=63, min_periods=63
    )

    x = market.iloc[-63:].to_numpy()
    y = returns["A"].iloc[-63:].to_numpy()
    design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coef
    expected = np.sqrt(np.sum(residual ** 2) / (len(x) - 2)) * np.sqrt(252.0)
    assert result.iloc[-1, 0] == pytest.approx(expected, rel=1e-10, abs=1e-12)
    assert result["A"].iloc[:62].isna().all()


def test_price_features_expose_standard_capm_idio_vol():
    data, _dates, _listing_pos = _stub(n_dates=100, listing_pos=30)
    features = build_price_features(data)
    assert "idio_vol_capm_63d" in features
    assert np.isfinite(features["idio_vol_capm_63d"]["AAA"].iloc[-1])
