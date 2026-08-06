"""§S13.32 pipeline vol-quality tilt — plain-function unit tests (no fixtures).

Parity contract: disabled -> the input object is returned unchanged (byte
parity is structural). Enabled -> the pipeline function must reproduce the
§S13.31 diagnostic transform (scripts/run_s13_31_quality_tilt_arm.py)
cell-for-cell, since gate E0 asserts the full run reproduces the re-MVO arm.
"""
import numpy as np
import pandas as pd

from scripts.run_s13_31_quality_tilt_arm import apply_quality_tilt
from src.backtest import apply_vol_quality_tilt
from src.config import PipelineConfig


def _make_inputs(n_days=3, n_tk=60):
    idx = pd.bdate_range("2024-01-02", periods=n_days)
    tickers = [f"T{i:02d}" for i in range(n_tk)]
    rng = np.random.default_rng(7)
    preds = pd.DataFrame(rng.normal(size=(n_days, n_tk)), index=idx, columns=tickers)
    vol = np.concatenate([np.full(20, 5.0), np.linspace(0.5, 1.0, n_tk - 20)])
    vol_panel = pd.DataFrame(np.tile(vol, (n_days, 1)), index=idx, columns=tickers)
    q_panel = pd.DataFrame(rng.normal(size=(n_days, n_tk)), index=idx, columns=tickers)
    q_panel.loc[idx[1], "T03"] = np.nan  # one missing-quality cell in tercile
    panel = pd.concat(
        {"idio_vol_63d": vol_panel.stack(), "best_roe_level_z": q_panel.stack()},
        axis=1,
    )
    panel.index.names = ["date", "ticker"]
    return preds, vol_panel, q_panel, panel


def test_config_defaults_off_and_lambda_committed():
    cfg = PipelineConfig()
    assert cfg.vol_quality_tilt_enabled is False
    assert cfg.vol_quality_tilt_lambda == 0.25
    assert cfg.vol_quality_tilt_vol_feature == "idio_vol_63d"
    assert cfg.standard_idio_vol_feature_enabled is False


def test_disabled_returns_the_same_object():
    preds, _, _, panel = _make_inputs()
    cfg = PipelineConfig()
    out = apply_vol_quality_tilt(preds, panel, cfg)
    assert out is preds  # structural byte parity when OFF


def test_enabled_reproduces_s13_31_diagnostic_transform():
    preds, vol_panel, q_panel, panel = _make_inputs()
    cfg = PipelineConfig()
    cfg.vol_quality_tilt_enabled = True
    out = apply_vol_quality_tilt(preds, panel, cfg)
    expected, diag = apply_quality_tilt(preds, vol_panel, q_panel,
                                        lam=cfg.vol_quality_tilt_lambda)
    assert diag["tilted_dates"] == len(preds.index)
    pd.testing.assert_frame_equal(out, expected)
    # and it actually changed something inside the tercile
    assert not out.equals(preds)


def test_enabled_passthrough_below_min_names():
    preds, _, _, panel = _make_inputs()
    sparse = preds.copy()
    sparse.iloc[0, 10:] = np.nan  # 10 scored names < 30 on date 0
    cfg = PipelineConfig()
    cfg.vol_quality_tilt_enabled = True
    out = apply_vol_quality_tilt(sparse, panel, cfg)
    pd.testing.assert_series_equal(out.iloc[0], sparse.iloc[0])


def test_enabled_uses_configured_standard_idio_vol_feature():
    preds, vol_panel, q_panel, panel = _make_inputs()
    standard_vol = vol_panel.iloc[:, ::-1].copy()
    standard_vol.columns = vol_panel.columns
    standard_series = standard_vol.stack()
    standard_series.index.names = ["date", "ticker"]
    panel["idio_vol_capm_63d"] = standard_series

    cfg = PipelineConfig(
        vol_quality_tilt_enabled=True,
        vol_quality_tilt_vol_feature="idio_vol_capm_63d",
    )
    out = apply_vol_quality_tilt(preds, panel, cfg)
    expected, _ = apply_quality_tilt(
        preds, standard_vol, q_panel, lam=cfg.vol_quality_tilt_lambda
    )
    pd.testing.assert_frame_equal(out, expected)
