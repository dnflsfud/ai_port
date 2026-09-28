"""§S22 stage-1 guards (decision log §S23): output-invariant fixes for the
2026-09-28 structural audit findings D-02, A-02 and the production-gate
halves of D-01 / D-02. The batch-file halves of D-01 / D-05 live in
test_portfolio_automation.py, A-04 in test_tg_basis_guard.py.
"""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.config import PipelineConfig
from src.data_loader import UniverseData
from src.features.fwd_sales_slope import (
    FWD_SALES_SLOPE_SHEET,
    NL_CONFIRM_PARENTS,
    build_fwd_sales_slope_features,
)
from src.tg_basis_guard import annotate_basis_guard


# --------------------------------------------------------------- D-02 stub run
_DATES = pd.bdate_range("2021-01-01", periods=200)
_TICKERS = [f"T{i:02d}" for i in range(20)]
_RNG = np.random.default_rng(3)
_RETS = pd.DataFrame(_RNG.normal(0.0003, 0.01, (200, 20)), index=_DATES, columns=_TICKERS)
_PANEL = pd.DataFrame(
    {"f0": _RNG.normal(size=200 * 20)},
    index=pd.MultiIndex.from_product([_DATES, _TICKERS], names=["date", "ticker"]),
)
_PREDS = pd.DataFrame(_RNG.normal(size=(200, 20)), index=_DATES, columns=_TICKERS)
_PREDS.iloc[:130] = np.nan
_BASE = dict(listing_mask_enabled=False, vol_quality_tilt_enabled=False,
             growth_tilt_enabled=False, value_trap_gate_enabled=False,
             execution_signal_lag_days=1, static_execution_enabled=True,
             sp500_benchmark_enabled=False, max_te_annual=0.035,
             sector_deviation=0.5, mega_cap_protection_enabled=False)


class _StubData:
    """Toy UniverseData: every sheet absent unless passed in `sheets`."""

    def __init__(self, sheets=None):
        self.tickers, self.dates = _TICKERS, _DATES
        self.returns, self.raw_returns = _RETS, _RETS.copy()
        self.market_cap = pd.DataFrame(
            np.tile(np.linspace(50, 150, 20), (200, 1)), index=_DATES, columns=_TICKERS)
        self.meta = pd.DataFrame({"sector": [f"S{i % 4}" for i in range(20)]}, index=_TICKERS)
        self.listing_dates, self.data_quality = {}, {}
        self.earnings_timeline = None
        self.factor_data, self.factor_prices = {}, None
        self._sheets = sheets or {}

    def has_factor_data(self):
        return False

    def get_sheet(self, name):
        if name not in self._sheets:
            raise KeyError(name)
        return self._sheets[name]


def _run_stub(cfg, data):
    from src.backtest import run_backtest
    return run_backtest(data, precomputed_panel=_PANEL, precomputed_feature_names=["f0"],
                        precomputed_feature_groups={}, precomputed_targets=_RETS.rolling(20).sum().shift(-20),
                        precomputed_models={}, precomputed_predictions=_PREDS, config=cfg)


def test_pead_on_without_earnings_timeline_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # run_backtest writes ./outputs/progress.md
    cfg = PipelineConfig(**_BASE, pead_boost_enabled=True)
    with pytest.raises(ValueError, match="earnings_timeline"):
        _run_stub(cfg, _StubData())


def test_option_vol_on_without_iv30_z_sheet_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = PipelineConfig(**_BASE, pead_boost_enabled=False, option_vol_covariance_enabled=True)
    with pytest.raises(KeyError, match="iv30_z"):
        _run_stub(cfg, _StubData())


def test_option_vol_scale_fix_without_observed_mask_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    iv = pd.DataFrame(0.0, index=_DATES, columns=_TICKERS)
    cfg = PipelineConfig(**_BASE, pead_boost_enabled=False, option_vol_covariance_enabled=True,
                         option_vol_scale_fix_enabled=True)
    with pytest.raises(ValueError, match="observed mask"):
        _run_stub(cfg, _StubData({"iv30_z": iv}))  # stub has no raw_sheet_observed_mask


class _SlopeData:
    def __init__(self, sheets):
        self._sheets, self.tickers = sheets, ["AAA", "BBB"]
        self.dates = pd.bdate_range("2024-01-01", periods=80)

    def get_sheet(self, name):
        if name not in self._sheets:
            raise KeyError(name)
        return self._sheets[name]


def _slope_inputs():
    data = _SlopeData({FWD_SALES_SLOPE_SHEET: pd.DataFrame(
        np.linspace(0.0, 0.2, 160).reshape(80, 2),
        index=pd.bdate_range("2024-01-01", periods=80), columns=["AAA", "BBB"])})
    parents = {p: pd.DataFrame(1.0, index=data.dates, columns=data.tickers)
               for p in NL_CONFIRM_PARENTS.values()}
    return data, parents


def test_slope_on_requires_sheet_and_parents_off_keeps_skipping():
    on = SimpleNamespace(fwd_sales_slope_features_enabled=True)
    off = SimpleNamespace(fwd_sales_slope_features_enabled=False)
    data, parents = _slope_inputs()
    assert len(build_fwd_sales_slope_features(parents, data, config=on)) == 4
    empty = _SlopeData({})
    with pytest.raises(KeyError, match=FWD_SALES_SLOPE_SHEET):
        build_fwd_sales_slope_features(parents, empty, config=on)
    with pytest.raises(KeyError, match="sales_rev_ma_63d"):
        build_fwd_sales_slope_features(
            {k: v for k, v in parents.items() if k != "sales_rev_ma_63d"}, data, config=on)
    assert build_fwd_sales_slope_features(parents, empty, config=off) == {}
    assert len(build_fwd_sales_slope_features({}, data, config=off)) == 2


# ------------------------------------------------ D-01 / D-02 production gate
def test_production_gate_requires_clean_tree():
    from scripts.validate_portfolio_bundles import evaluate_production
    dirty = evaluate_production({"git_dirty": True})
    assert dirty["checks"]["clean_tree_ok"] is False
    assert dirty["status"] == "HOLD"
    assert evaluate_production({"git_dirty": False})["checks"]["clean_tree_ok"] is True
    assert evaluate_production({})["checks"]["clean_tree_ok"] is None  # fail-closed


def test_production_gate_requires_enabled_option_vol_to_be_applied():
    from scripts.validate_portfolio_bundles import evaluate_production

    def check(enabled, applied):
        record = {"_option_vol_cov": {"enabled": enabled, "applied": applied}}
        return evaluate_production(record)["checks"]["option_vol_applied_ok"]

    assert check(True, False) is False
    assert check(True, True) is True
    assert check(False, False) is True
    assert check(None, None) is None
    assert evaluate_production({})["checks"]["option_vol_applied_ok"] is None


# ------------------------------------------------------------------ A-02 guard
def _spin_off_shell(nominal_source):
    """Distribution spin-off 60 rows before the end (RTX/T class, §S18 P1 class 3)."""
    idx = pd.bdate_range(end="2026-09-14", periods=300)
    event = idx[-60]
    shell = UniverseData.__new__(UniverseData)
    shell.sheets = {"Factset_TG_Price": pd.DataFrame({"XYZ": 69.0, "CTL": 110.0}, index=idx)}
    shell.local_prices = pd.DataFrame({"XYZ": 60.0, "CTL": 100.0}, index=idx)  # back-adjusted
    shell.local_prices_nominal = pd.DataFrame(
        {"XYZ": np.where(idx < event, 100.0, 60.0), "CTL": 100.0}, index=idx)
    shell.data_quality = {"currency": {}}
    shell.config = SimpleNamespace(nominal_price_source=nominal_source)
    return shell


def test_tg_guard_divides_by_the_consumer_denominator(tmp_path):
    shell = _spin_off_shell("PX_LAST_UNADJ")
    shell._check_target_price_unit_ratio()
    medians = shell.data_quality["currency"]["tg_px_ratio_median"]
    consumer = (shell.sheets["Factset_TG_Price"] / shell.local_prices_nominal).tail(252).median()
    assert medians["XYZ"] == pytest.approx(float(consumer["XYZ"]))  # 0.69, not 1.15
    assert medians["CTL"] == pytest.approx(1.10)
    (tmp_path / "tg_basis_state.json").write_text(json.dumps(
        {"schema_version": 1, "baseline": {"XYZ": 1.15, "CTL": 1.10}, "pending": {}}))
    assert set(annotate_basis_guard(tmp_path, shell.data_quality)) == {"XYZ"}
    assert shell.data_quality["currency"]["tg_basis_guard_ok"] is False


def test_tg_guard_without_nominal_source_keeps_adjusted_denominator():
    shell = _spin_off_shell(None)
    shell._check_target_price_unit_ratio()
    assert shell.data_quality["currency"]["tg_px_ratio_median"]["XYZ"] == pytest.approx(1.15)

