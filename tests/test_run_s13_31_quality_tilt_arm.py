"""§S13.31 quality-tilt transform — plain-function unit tests (no fixtures)."""
import numpy as np
import pandas as pd

from scripts.run_s13_31_quality_tilt_arm import (
    _max_drawdown,
    _subperiod_irs,
    _zscore_row,
    apply_quality_tilt,
)


def _panels(n_days=3, n_tk=60):
    idx = pd.bdate_range("2024-01-02", periods=n_days)
    tickers = [f"T{i:02d}" for i in range(n_tk)]
    preds = pd.DataFrame(
        np.tile(np.linspace(-1.0, 1.0, n_tk), (n_days, 1)), index=idx, columns=tickers)
    # first 20 tickers form the high-vol tercile on every date
    vol = np.concatenate([np.full(20, 5.0), np.linspace(0.5, 1.0, n_tk - 20)])
    vol_panel = pd.DataFrame(np.tile(vol, (n_days, 1)), index=idx, columns=tickers)
    rng = np.random.default_rng(1)
    q_panel = pd.DataFrame(rng.normal(size=(n_days, n_tk)), index=idx, columns=tickers)
    return idx, tickers, preds, vol_panel, q_panel


def test_lambda_zero_is_identity():
    _, _, preds, vol_panel, q_panel = _panels()
    out, diag = apply_quality_tilt(preds, vol_panel, q_panel, lam=0.0)
    pd.testing.assert_frame_equal(out, preds)
    assert diag["tilted_dates"] == len(preds.index)


def test_below_min_names_passes_through():
    _, _, preds, vol_panel, q_panel = _panels(n_tk=60)
    sparse = preds.copy()
    sparse.iloc[0, 10:] = np.nan  # only 10 scored names on date 0
    out, diag = apply_quality_tilt(sparse, vol_panel, q_panel, lam=0.25)
    pd.testing.assert_series_equal(out.iloc[0], sparse.iloc[0])
    assert diag["tilted_dates"] == len(preds.index) - 1


def test_tilt_touches_only_top_vol_tercile_with_known_delta():
    idx, tickers, preds, vol_panel, q_panel = _panels()
    lam = 0.25
    out, _ = apply_quality_tilt(preds, vol_panel, q_panel, lam=lam)
    dt = idx[0]
    s = preds.loc[dt]
    vz = _zscore_row(vol_panel.loc[dt].reindex(s.dropna().index))
    top = vz.sort_values(ascending=False).head(len(vz) // 3).index
    assert set(top) == {f"T{i:02d}" for i in range(20)}
    # outside the tercile: byte-identical
    rest = [t for t in tickers if t not in set(top)]
    pd.testing.assert_series_equal(out.loc[dt, rest], preds.loc[dt, rest])
    # inside: delta equals lam * sd(scored) * z_q(within tercile)
    zq = _zscore_row(q_panel.loc[dt].reindex(top))
    sd = float(s.std(ddof=1))
    expected = s[zq.index] + lam * sd * zq
    pd.testing.assert_series_equal(out.loc[dt, zq.index], expected, check_names=False)


def test_nan_quality_cell_is_unchanged():
    idx, _, preds, vol_panel, q_panel = _panels()
    q_panel.loc[idx[0], "T05"] = np.nan
    out, _ = apply_quality_tilt(preds, vol_panel, q_panel, lam=0.25)
    assert out.loc[idx[0], "T05"] == preds.loc[idx[0], "T05"]
    assert out.loc[idx[0], "T06"] != preds.loc[idx[0], "T06"]


def test_max_drawdown_known_path():
    r = pd.Series([0.10, -0.50, 0.10])
    assert abs(_max_drawdown(r) - (-0.50)) < 1e-12


def test_subperiod_irs_shape_and_sign():
    rng = np.random.default_rng(2)
    act = pd.Series(rng.normal(0.001, 0.01, 252),
                    index=pd.bdate_range("2024-01-02", periods=252))
    irs = _subperiod_irs(act)
    assert len(irs) == 3
    assert all(np.isfinite(v) for v in irs)
