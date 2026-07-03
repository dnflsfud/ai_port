"""apply_prediction_ema — frame version of walk_forward_train's in-loop EMA.

The CS-DR-Alpha 2-pass path feeds precomputed predictions into run_backtest,
bypassing walk_forward_train's inline blending. apply_prediction_ema must
reproduce that recursion exactly, otherwise the DR variant trades a
differently-smoothed signal than the LightGBM baseline (not apples-to-apples).
"""
import numpy as np
import pandas as pd

from src.model_trainer import apply_prediction_ema


def _inline_reference(frame: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Simulate walk_forward_train's blending loop on a date x ticker frame."""
    out = frame.copy()
    prev_pred = None
    for d in out.index:
        pred = out.loc[d].dropna()          # tickers predicted on d
        if len(pred) == 0:
            continue
        if prev_pred is not None:
            common = pred.index.intersection(prev_pred.index)
            blended = alpha * pred[common] + (1 - alpha) * prev_pred[common]
            pred[common] = blended
        prev_pred = pred.copy()
        out.loc[d, pred.index] = pred.values
    return out


def _toy_frame(seed=0, n_days=60, n_tk=8):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    tks = [f"T{i}" for i in range(n_tk)]
    f = pd.DataFrame(rng.normal(size=(n_days, n_tk)), index=dates, columns=tks)
    # NaN burn-in head (like the LGBM walk-forward frame)
    f.iloc[:10] = np.nan
    # a ticker that disappears and reappears (coverage churn)
    f.iloc[25:30, 2] = np.nan
    # an all-NaN gap date
    f.iloc[40] = np.nan
    return f


def test_matches_inline_recursion():
    f = _toy_frame()
    for alpha in (0.3, 0.5, 0.8):
        got = apply_prediction_ema(f, alpha)
        ref = _inline_reference(f, alpha)
        pd.testing.assert_frame_equal(got, ref)


def test_alpha_one_is_identity():
    f = _toy_frame(seed=1)
    got = apply_prediction_ema(f, 1.0)
    pd.testing.assert_frame_equal(got, f)


def test_nan_pattern_preserved():
    f = _toy_frame(seed=2)
    got = apply_prediction_ema(f, 0.5)
    assert got.isna().equals(f.isna())
