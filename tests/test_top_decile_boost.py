# -*- coding: utf-8 -*-
"""§S25.6 top-decile boost (decision log §S25.5 precheck PASS → §S25.6 arm): μ' = μ + κ·sd_t(μ)·1[μ ≥ q_t] per date,
applied after the overlays and before the optimizer. OFF returns the input object (byte-identical)."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import apply_top_decile_boost
from src.config import PipelineConfig


def _preds(n_dates=3, n=40, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_dates)
    df = pd.DataFrame(rng.normal(size=(n_dates, n)), index=dates, columns=[f"t{i}" for i in range(n)])
    df.iloc[1, :5] = np.nan     # a few missing names on the second date
    return df


def test_config_defaults_off_and_validation():
    cfg = PipelineConfig()
    assert cfg.top_decile_boost_enabled is False
    assert cfg.top_decile_boost_kappa == 0.5 and cfg.top_decile_boost_quantile == 0.9
    with pytest.raises(ValueError):
        PipelineConfig(top_decile_boost_kappa=0.0)
    with pytest.raises(ValueError):
        PipelineConfig(top_decile_boost_quantile=1.0)


def test_off_returns_the_same_object():
    p = _preds()
    assert apply_top_decile_boost(p, PipelineConfig()) is p


def test_on_boosts_exactly_the_top_decile_by_kappa_sd_per_date_and_keeps_nan():
    p = _preds()
    out = apply_top_decile_boost(p, PipelineConfig(top_decile_boost_enabled=True))
    assert out is not p and out.shape == p.shape
    assert out.isna().equals(p.isna())
    for d in p.index:
        row = p.loc[d]; thr = row.quantile(0.9); sd = row.std(ddof=1)
        top = row.ge(thr) & row.notna()
        delta = (out.loc[d] - row)
        assert np.allclose(delta[top].to_numpy(), 0.5 * sd)          # boosted cells
        assert np.allclose(delta[~top & row.notna()].to_numpy(), 0.0)  # everyone else untouched
        assert int(top.sum()) in (4, 5)                                  # ~10% of 35..40 valid names
    # ranking inside the top decile is preserved and the top decile stays on top
    d = p.index[0]
    assert (out.loc[d].rank() == p.loc[d].rank()).all()


def test_kappa_and_quantile_are_honoured():
    p = _preds()
    out = apply_top_decile_boost(p, PipelineConfig(top_decile_boost_enabled=True, top_decile_boost_kappa=1.0,
                                                   top_decile_boost_quantile=0.5))
    d = p.index[0]; row = p.loc[d]
    top = row.ge(row.quantile(0.5))
    assert np.allclose((out.loc[d] - row)[top].to_numpy(), row.std(ddof=1))
    assert int(top.sum()) == 20
