"""§S13.30 preflight helpers — plain-function unit tests (no fixtures)."""
import numpy as np
import pandas as pd

from scripts.preflight_s13_30_vol_quality import (
    _fwd_return,
    _summarise,
    _tstat,
    _zscore_row,
    run_metric,
)


def _bdays(n, start="2024-01-02"):
    return pd.bdate_range(start, periods=n)


def test_zscore_row_basic():
    z = _zscore_row(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=list("abcde")))
    assert abs(z.mean()) < 1e-12
    assert abs(z.std(ddof=1) - 1.0) < 1e-6
    assert z["e"] > z["a"]


def test_zscore_row_degenerate():
    assert _zscore_row(pd.Series([1.0, 1.0, 1.0], index=list("abc"))).empty
    assert _zscore_row(pd.Series([1.0, np.nan], index=list("ab"))).empty


def test_fwd_return_compounds_window_after_date():
    idx = _bdays(30)
    rets = pd.DataFrame(0.01, index=idx, columns=["X", "Y"])
    fwd = _fwd_return(rets, idx[3], 21)
    assert fwd is not None
    expected = 1.01 ** 21 - 1
    assert abs(fwd["X"] - expected) < 1e-12
    # a window that cannot fill the horizon is dropped
    assert _fwd_return(rets, idx[15], 21) is None


def test_tstat_known_value():
    x = np.array([1.0, 1.0, 1.0, 1.0])
    assert np.isnan(_tstat(np.array([1.0])))
    assert np.isnan(_tstat(x))  # zero variance -> nan
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(_tstat(y) - y.mean() / y.std(ddof=1) * 2) < 1e-12


def test_summarise_subperiods():
    rows = [{"v": 1.0}, {"v": 2.0}, {"v": -1.0}, {"v": 3.0}, {"v": 4.0},
            {"v": -2.0}]
    s = _summarise(rows, "v")
    assert s["n"] == 6
    assert abs(s["mean"] - (7.0 / 6)) < 1e-12
    assert len(s["subperiod_means"]) == 3
    assert s["subperiod_means"][0] == 1.5   # rows 0-1
    assert s["subperiod_means"][1] == 1.0   # rows 2-3
    assert s["subperiod_means"][2] == 1.0   # rows 4-5
    assert s["subperiod_pos"] == 3


def test_run_metric_perfect_quality_discrimination():
    """Quality == forward return inside the top-vol tercile -> IC ~ 1."""
    n_days, n_tk = 120, 60
    idx = _bdays(n_days)
    tickers = [f"T{i:02d}" for i in range(n_tk)]
    rng = np.random.default_rng(0)

    # constant per-ticker daily return, increasing in ticker id
    daily = np.linspace(-0.004, 0.004, n_tk)
    rets = pd.DataFrame(np.tile(daily, (n_days, 1)), index=idx, columns=tickers)

    # vol: first 20 tickers are the high-vol tercile on every date
    vol = np.concatenate([np.full(20, 5.0), rng.uniform(0.5, 1.0, n_tk - 20)])
    vol_panel = pd.DataFrame(np.tile(vol, (n_days, 1)), index=idx, columns=tickers)

    # quality == daily return -> perfect rank agreement with forward return
    q_panel = pd.DataFrame(np.tile(daily, (n_days, 1)), index=idx, columns=tickers)

    bm_ret = pd.Series(-0.001, index=idx)  # every window is a BM-down window
    dates = [idx[10], idx[40], idx[70]]

    out = run_metric(dates, vol_panel, q_panel, rets, bm_ret, horizon=21)
    assert out["windows"] == 3
    assert out["ic"]["mean"] > 0.99
    assert out["spread"]["mean"] > 0
    assert out["spread_bm_down"]["n"] == 3
    assert out["spread_bm_up"]["n"] == 0
