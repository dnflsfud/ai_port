"""§S13.36 preflight helpers — plain-function unit tests (no fixtures)."""
import numpy as np
import pandas as pd

from scripts.preflight_s13_36_turnover_tilt import (
    compute_window_row,
    evaluate_gates,
    residualize,
)


def test_residualize_removes_linear_component():
    idx = list("abcdefgh")
    x = pd.Series(np.arange(8, dtype=float), index=idx)
    y = 2.0 + 3.0 * x
    r = residualize(y, x)
    assert list(r.index) == idx
    assert np.allclose(r.to_numpy(), 0.0, atol=1e-10)


def test_residualize_keeps_orthogonal_part():
    rng = np.random.default_rng(1)
    idx = [f"t{i}" for i in range(200)]
    x = pd.Series(rng.normal(size=200), index=idx)
    orth = pd.Series(rng.normal(size=200), index=idx)
    y = 0.5 * x + orth
    r = residualize(y, x)
    assert abs(r.corr(x)) < 1e-8   # OLS residual is orthogonal to x
    assert r.corr(orth) > 0.9      # the incremental part survives


def test_residualize_degenerate():
    assert residualize(pd.Series([1.0, 2.0], index=list("ab")),
                       pd.Series([1.0, 2.0], index=list("ab"))).empty
    assert residualize(pd.Series([np.nan, np.nan, np.nan], index=list("abc")),
                       pd.Series([1.0, 2.0, 3.0], index=list("abc"))).empty


def test_compute_window_row_incremental_signal_survives():
    """fwd = score + orthogonal turnover part -> residual IC stays high."""
    rng = np.random.default_rng(2)
    idx = [f"t{i}" for i in range(120)]
    score = pd.Series(rng.normal(size=120), index=idx)
    orth = pd.Series(rng.normal(size=120), index=idx)
    z_st = 0.8 * score + 0.6 * orth   # turnover shares a component with score
    fwd = score + orth                # ... but its orthogonal part is priced
    z_vol = z_st + 0.1 * pd.Series(rng.normal(size=120), index=idx)
    row = compute_window_row(z_st, score, z_vol, fwd)
    assert row is not None and row["n"] == 120
    assert row["raw_ic"] > 0.5
    assert row["resid_ic"] > 0.5
    assert row["vol_overlap"] > 0.8


def test_compute_window_row_redundant_signal_dies():
    """z_st ~ monotone(score) -> raw IC ~ 1 but residual IC collapses."""
    rng = np.random.default_rng(3)
    idx = [f"t{i}" for i in range(100)]
    score = pd.Series(np.linspace(-2.0, 2.0, 100), index=idx)
    z_st = 1.7 * score + 0.05 * pd.Series(rng.normal(size=100), index=idx)
    fwd = score.copy()
    z_vol = pd.Series(rng.normal(size=100), index=idx)
    row = compute_window_row(z_st, score, z_vol, fwd)
    assert row["raw_ic"] > 0.99
    assert abs(row["resid_ic"]) < 0.3   # residual is de-signalled noise


def test_compute_window_row_too_few_names():
    idx = list("abc")
    s = pd.Series([1.0, 2.0, 3.0], index=idx)
    assert compute_window_row(s, s, s, s, min_names=30) is None


def test_evaluate_gates_thresholds():
    good = {"mean": 0.03, "t": 2.5, "subperiod_pos": 3}
    g = evaluate_gates(good, overlap_mean=0.30, expo_mean=0.10)
    assert g["P1_residual_ic"] and g["P2_vol_overlap"] and g["P3_room_in_book"]
    assert g["PROCEED"]
    assert not evaluate_gates({"mean": 0.03, "t": 1.9, "subperiod_pos": 3},
                              0.30, 0.10)["PROCEED"]
    assert not evaluate_gates({"mean": -0.01, "t": 2.5, "subperiod_pos": 3},
                              0.30, 0.10)["P1_residual_ic"]
    assert not evaluate_gates({"mean": 0.03, "t": 2.5, "subperiod_pos": 1},
                              0.30, 0.10)["P1_residual_ic"]
    assert not evaluate_gates(good, 0.65, 0.10)["P2_vol_overlap"]
    assert not evaluate_gates(good, 0.30, 0.30)["P3_room_in_book"]
    assert not evaluate_gates(good, float("nan"), 0.10)["P2_vol_overlap"]
