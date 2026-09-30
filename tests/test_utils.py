"""§S22 stage-4 (decision log §S24): tests for the two ``utils`` modules.

A-07 ``src.utils.compute_performance_metrics`` — a benchmark day missing from
the portfolio calendar was forward-filled (a fabricated repeat of yesterday's
benchmark return). Active metrics are now computed on the common dates only.

B-03 ``src.features.utils`` — cross-sectional winsorisation BEFORE the z-score
(default-OFF ``zscore_winsor_first_enabled``; OFF path byte-identical).
"""
import logging

import numpy as np
import pandas as pd
import pytest

from src.config import PipelineConfig


# ------------------------------------------------------------------ A-07
def _port_and_bm():
    idx = pd.bdate_range("2026-01-05", periods=4)
    port = pd.Series([0.01, 0.00, 0.00, 0.00], index=idx)
    bm_full = pd.Series([0.01, -0.05, 0.00, 0.00], index=idx)
    return idx, port, bm_full


def test_missing_benchmark_day_is_excluded_not_forward_filled(caplog):
    from src.utils import compute_performance_metrics

    idx, port, bm_full = _port_and_bm()
    bm_missing = bm_full.drop(idx[1])
    with caplog.at_level(logging.WARNING, logger="src.utils"):
        with_gap = compute_performance_metrics(port, bm_missing, 252)
    aligned = compute_performance_metrics(port.drop(idx[1]), bm_missing, 252)
    for key in ("active_return", "tracking_error", "information_ratio"):
        assert with_gap[key] == pytest.approx(aligned[key])
    # portfolio-only metrics still use every portfolio day
    full = compute_performance_metrics(port, bm_full, 252)
    assert with_gap["annual_return"] == pytest.approx(full["annual_return"])
    assert any("benchmark" in rec.getMessage() for rec in caplog.records)


def test_complete_benchmark_is_parity(caplog):
    from src.utils import compute_performance_metrics

    _, port, bm_full = _port_and_bm()
    with caplog.at_level(logging.WARNING, logger="src.utils"):
        got = compute_performance_metrics(port, bm_full, 252)
    active = port - bm_full
    assert got["tracking_error"] == pytest.approx(active.std() * np.sqrt(252))
    assert not caplog.records


# ------------------------------------------------------------------ B-03
def _outlier_panel():
    rng = np.random.default_rng(0)
    n = 250
    dates = pd.bdate_range("2023-01-02", periods=5)
    values = rng.normal(0.08, 0.20, (len(dates), n))
    df = pd.DataFrame(values, index=dates, columns=[f"T{i}" for i in range(n)])
    df.iloc[-1, 0] = 49.0          # one near-zero-base pct_change blow-up
    df.iloc[-1, 1] = np.nan        # NaN must survive untouched
    return df


def test_flag_default_off_and_off_path_is_bytewise_zscore():
    from src.features.utils import cross_sectional_zscore, standardise_feature

    assert PipelineConfig().zscore_winsor_first_enabled is False
    df = _outlier_panel()
    assert standardise_feature(df, winsor_first=False).equals(cross_sectional_zscore(df))


def test_winsorize_cross_section_clips_to_per_date_quantiles_and_keeps_nan():
    from src.features.utils import WINSOR_QUANTILE, winsorize_cross_section

    df = _outlier_panel()
    out = winsorize_cross_section(df)
    lo = df.quantile(WINSOR_QUANTILE, axis=1)
    hi = df.quantile(1 - WINSOR_QUANTILE, axis=1)
    assert out.shape == df.shape
    assert bool(np.isnan(out.iloc[-1, 1]))
    assert out.isna().equals(df.isna())
    assert (out.max(axis=1) <= hi + 1e-12).all()
    assert (out.min(axis=1) >= lo - 1e-12).all()
    assert out.iloc[-1, 0] == pytest.approx(hi.iloc[-1])


def test_winsor_first_stops_one_outlier_compressing_the_rest():
    from src.features.utils import clip_outliers, standardise_feature

    df = _outlier_panel()
    off = clip_outliers(standardise_feature(df, winsor_first=False)).iloc[-1].drop(["T0", "T1"])
    on = clip_outliers(standardise_feature(df, winsor_first=True)).iloc[-1].drop(["T0", "T1"])
    spread_off = off.quantile(0.9) - off.quantile(0.1)
    spread_on = on.quantile(0.9) - on.quantile(0.1)
    assert spread_off < 0.5           # legacy: 248 names squeezed by one outlier
    assert spread_on > 4 * spread_off


def test_assembly_routes_zscore_through_standardise_feature():
    import inspect

    from src.features import assembly

    src = inspect.getsource(assembly)
    assert "zscore_winsor_first_enabled" in src
    assert "standardise_feature(" in src
