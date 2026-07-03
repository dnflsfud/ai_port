"""Task 3 — CS-DR-Alpha true walk-forward (src/rl/dr_walkforward.py).

Toys use shrunken train_window/retrain_freq (via config) so folds actually
train on a 400-day panel; otherwise the production TRAIN_WINDOW=1260 would mean
no training happens and the tests would pass trivially.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from src.rl.dr_walkforward import run_walkforward
from src.config import PipelineConfig


def _toy(n_days=400, n_tk=10, F=4, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    tks = [f"T{i}" for i in range(n_tk)]
    idx = pd.MultiIndex.from_product([dates, tks], names=["date", "ticker"])
    fcols = [f"f{i}" for i in range(F)]
    panel = pd.DataFrame(rng.normal(size=(len(idx), F)), index=idx, columns=fcols)
    fwd = pd.DataFrame(rng.normal(size=(n_days, n_tk)), index=dates, columns=tks)
    prior = pd.DataFrame(rng.normal(size=(n_days, n_tk)), index=dates, columns=tks)
    return panel, fwd, prior, fcols, dates, tks


def _toy_cfg():
    c = PipelineConfig()
    c.train_window = 100
    c.retrain_freq = 50
    c.rebalance_freq = 10
    c.dr_alpha_epochs = 30
    c.dr_alpha_embargo = 20
    c.dr_alpha_val_months = 1
    c.dr_alpha_min_train_rebal = 3   # toy pools yield ~10 sampled dates max
    return c


def test_output_grid_matches_prior():
    panel, fwd, prior, fn, dates, tks = _toy()
    out = run_walkforward(panel, fwd, prior, fn, _toy_cfg())
    assert list(out.columns) == list(prior.columns)
    assert out.index.equals(prior.index)
    tail = out.iloc[-50:]
    assert tail.notna().values.mean() >= 0.99


def test_no_future_label_leak():
    panel, fwd, prior, fn, dates, tks = _toy()
    cfg = _toy_cfg()
    base = run_walkforward(panel, fwd, prior, fn, cfg)
    fwd2 = fwd.copy()
    fwd2.iloc[-30:] = fwd2.iloc[-30:] + 10.0   # corrupt far-future labels
    pert = run_walkforward(panel, fwd2, prior, fn, cfg)
    head = base.index[: len(base) // 3]
    assert np.allclose(base.loc[head].fillna(0).values,
                       pert.loc[head].fillna(0).values, atol=1e-6)


def test_gamma_zero_returns_prior_ranks():
    panel, fwd, prior, fn, dates, tks = _toy()
    cfg = _toy_cfg()
    cfg.dr_alpha_gamma = 0.0
    out = run_walkforward(panel, fwd, prior, fn, cfg)
    tail = out.index[-30:]
    for d in tail:
        a = out.loc[d].dropna()
        b = prior.loc[d].reindex(a.index)
        if len(a) > 3:
            assert spearmanr(a, b).correlation > 0.999


def test_embargo_must_cover_forward_horizon():
    # OOS-by-construction requires dr_alpha_embargo >= forward_horizon. An
    # embargo smaller than the label window must fail loudly, not leak silently.
    panel, fwd, prior, fn, dates, tks = _toy()
    cfg = _toy_cfg()
    cfg.forward_horizon = 20
    cfg.dr_alpha_embargo = 10          # < forward_horizon -> leak risk
    with pytest.raises(ValueError, match="embargo"):
        run_walkforward(panel, fwd, prior, fn, cfg)


def test_some_folds_actually_train():
    # With train_window=100 + embargo=20, the tail must be RL-modified
    # (different from prior) when gamma>0 — proves training fired.
    panel, fwd, prior, fn, dates, tks = _toy()
    cfg = _toy_cfg()
    out = run_walkforward(panel, fwd, prior, fn, cfg)
    tail = out.index[-20:]
    # zscore(prior) on the same set would preserve ranks; a trained gamma>0
    # residual should break the perfect rank-corr on at least one tail date.
    broke = 0
    for d in tail:
        a = out.loc[d].dropna()
        b = prior.loc[d].reindex(a.index)
        if len(a) > 3 and spearmanr(a, b).correlation < 0.999:
            broke += 1
    assert broke >= 1


def test_nan_head_prior_does_not_delay_activation():
    # Regression: the production prior (LightGBM walk-forward z) has a
    # ~train_window NaN burn-in head. Folds whose lookback pool reached into
    # that head used to be silently skipped (empty fold universe), delaying DR
    # activation by a full train_window beyond the prediction start. With pool
    # clipping, activation must begin within ~(min_train_rebal rebalances +
    # embargo + one retrain block) of prior coverage, not train_window later.
    panel, fwd, prior, fn, dates, tks = _toy(n_days=400)
    cfg = _toy_cfg()
    warmup = 150
    prior_nan_head = prior.copy()
    prior_nan_head.iloc[:warmup] = np.nan

    out = run_walkforward(panel, fwd, prior_nan_head, fn, cfg)
    diff = (out - prior_nan_head).abs()
    both = out.notna() & prior_nan_head.notna()
    mod = diff.where(both).max(axis=1)
    mod_dates = mod[mod > 1e-12].index
    assert len(mod_dates) > 0, "DR never modified any date with a NaN-head prior"

    first_mod_pos = list(dates).index(mod_dates.min())
    # Earliest trainable boundary: enough embargoed, prior-covered history for
    # min_train_rebal samples; plus up to one retrain_freq of boundary
    # quantisation. Before the fix, first_mod_pos was >= warmup + train_window
    # + embargo (i.e. 270+); assert it activates well before that.
    min_hist = cfg.dr_alpha_min_train_rebal * cfg.rebalance_freq
    latest_ok = warmup + min_hist + cfg.dr_alpha_embargo + cfg.retrain_freq
    assert first_mod_pos <= latest_ok, (
        f"DR activation at pos {first_mod_pos}, expected <= {latest_ok} "
        f"(prior coverage starts at {warmup})"
    )
    assert first_mod_pos < warmup + cfg.train_window
