"""Regression cases from the 2026-09-14 structural audit."""
import json
import dataclasses
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from run_variant import annotate_tg_ratio_jump
from scripts.export_operating_data import build_rebalance_metadata, validate_cached_result_compatibility
from src.config import PipelineConfig
from src.data_loader import UniverseData, restrict_to_business_days


def _quality(value):
    return {"currency": {"tg_px_ratio_median": {"AAA": value}, "tg_px_ratio_suspect": {}}}


def test_jump_survives_metrics_overwrite_and_clears_only_on_recovery(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"data_quality": _quality(1.0)}))
    for value in (1.5, 1.5, 1.49):
        quality = _quality(value)
        assert "AAA" in annotate_tg_ratio_jump(tmp_path, quality)
        path.write_text(json.dumps({"data_quality": quality}))
    assert annotate_tg_ratio_jump(tmp_path, _quality(1.01)) == {}


def test_legacy_jump_is_migrated_without_adopting_bad_baseline(tmp_path):
    quality = _quality(1.5)
    quality["currency"]["tg_px_ratio_jump_vs_prev"] = {"AAA": {"previous": 1.0, "now": 1.5}}
    (tmp_path / "metrics.json").write_text(json.dumps({"data_quality": quality}))
    assert "AAA" in annotate_tg_ratio_jump(tmp_path, _quality(1.5))


@pytest.mark.parametrize("sheet_name", ["Earnings_Timeline", "Earnings_Date"])
def test_holiday_earnings_survive_loader_and_do_not_move_backwards(sheet_name):
    dates = pd.to_datetime(["2026-07-02", "2026-07-03", "2026-07-06", "2026-07-07"])
    sessions = dates[[0, 2]]
    px = pd.DataFrame({"AAA": [100., 100., 102., 103.]}, index=dates)
    raw = {"BusinessDays": pd.DataFrame(index=sessions), "PX_LAST": px,
           "Daily_Returns": px.pct_change(fill_method=None),
           sheet_name: pd.DataFrame({"AAA": [0, 1, 0, 1]}, index=dates)}
    shell = UniverseData.__new__(UniverseData)
    shell.config = PipelineConfig(business_day_calendar_enabled=True)
    shell.raw, _ = restrict_to_business_days(raw)
    shell.dates = sessions
    shell.full_universe = ["AAA"]
    shell.data_quality = {}
    result = shell._load_earnings_timeline()
    assert result.loc[sessions[0], "AAA"] == 0
    assert result.loc[sessions[1], "AAA"] == 1
    assert int(result.to_numpy().sum()) == 1  # future 7/7 must not land on 7/6


def test_schedule_uses_supplied_trading_dates_not_weekdays():
    dates = pd.to_datetime(["2026-07-02", "2026-07-06", "2026-07-07"])
    result = build_rebalance_metadata([dates[0]], dates[:1], 1,
                                     calendar_dates=dates, calendar_name="XNYS")
    assert result["next_expected_rebalance_date"] == "2026-07-06"


def test_schedule_can_extend_past_workbook_without_counting_holidays():
    dates = pd.DatetimeIndex(["2025-12-03"])
    result = build_rebalance_metadata(dates, dates, 21,
                                     calendar_dates=dates, calendar_name="XNYS")
    assert result["next_expected_rebalance_date"] == "2026-01-05"


def test_cache_rejects_changed_interior_calendar_even_if_asof_matches():
    old = pd.to_datetime(["2026-07-02", "2026-07-03", "2026-07-06"])
    w = pd.Series({"AAA": 1.0})
    res = SimpleNamespace(portfolio_returns=pd.Series(0., index=old),
                          portfolio_weights={old[-1]: w}, daily_weights={d: w for d in old})
    current = pd.DataFrame({"AAA": [0., .21]}, index=old[[0, 2]])
    with pytest.raises(ValueError, match="calendar"):
        validate_cached_result_compatibility(res, ["AAA"], current)


def test_pending_jump_survives_missing_observation_and_missing_sidecar(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({"data_quality": _quality(1.0)}))
    quality = _quality(1.5)
    annotate_tg_ratio_jump(tmp_path, quality)
    (tmp_path / "metrics.json").write_text(json.dumps({"data_quality": quality}))
    (tmp_path / "tg_basis_state.json").unlink()
    assert "AAA" in annotate_tg_ratio_jump(tmp_path, {"currency": {"tg_px_ratio_median": {}}})


def test_invalid_guard_state_is_not_silently_reset(tmp_path):
    (tmp_path / "tg_basis_state.json").write_text("broken json")
    with pytest.raises(ValueError, match="guard state"):
        annotate_tg_ratio_jump(tmp_path, _quality(1.5))


def test_guard_recovery_allows_normal_run_and_missing_values_hold(tmp_path):
    from scripts.validate_portfolio_bundles import evaluate_production
    quality = _quality(1.0)
    annotate_tg_ratio_jump(tmp_path, quality)
    assert evaluate_production({"performance": {"data_quality": quality}})["checks"]["tg_px_ratio_ok"] is True
    quality = {"currency": {"tg_px_ratio_median": {}, "tg_px_ratio_suspect": {}}}
    annotate_tg_ratio_jump(tmp_path, quality)
    assert evaluate_production({"performance": {"data_quality": quality}})["checks"]["tg_px_ratio_ok"] is False


def test_event_collisions_keep_binary_flags_and_preserve_accounting():
    from src.trading_calendar import align_earnings_events
    events = pd.DataFrame({"AAA": [1, 1, 1, 1]}, index=pd.to_datetime(
        ["2026-07-02", "2026-07-03", "2026-07-04", "2026-07-07"]))
    result, diag = align_earnings_events(events, pd.to_datetime(["2026-07-02", "2026-07-06"]))
    assert result["AAA"].tolist() == [1, 1]
    assert diag["represented_events"] == 3
    assert diag["shifted_events"] == 2
    assert diag["outside_calendar_events"] == 1


def test_holiday_event_produces_pead_only_after_it_is_observed(monkeypatch):
    from src.backtest import apply_pead_boost
    from src.trading_calendar import align_earnings_events
    from src.features import sellside
    dates = pd.bdate_range("2026-05-01", "2026-07-08").difference(pd.to_datetime(["2026-07-03"]))
    events = pd.DataFrame({"AAA": [1]}, index=pd.to_datetime(["2026-07-03"]))
    aligned, _ = align_earnings_events(events, dates)
    data = SimpleNamespace(earnings_timeline=aligned)
    monkeypatch.setattr(sellside, "get_cleaned_revision", lambda *a, **k: pd.DataFrame(50., index=dates, columns=["AAA"]))
    pred = pd.DataFrame(0., index=dates, columns=["AAA"])
    cfg = PipelineConfig(pead_boost_enabled=True, pead_boost_weight=.3, s15_fixpack_enabled=True)
    boosted = apply_pead_boost(pred, data, cfg)
    assert boosted.loc[:"2026-07-02", "AAA"].eq(0).all()
    assert boosted.loc["2026-07-06", "AAA"] == pytest.approx(.15)
    assert 0 < boosted.loc["2026-07-07", "AAA"] < .15


def test_schedule_keeps_original_grid_after_a_skipped_trade():
    from src.trading_calendar import exchange_sessions
    days = exchange_sessions("2025-12-01", "2026-04-01")
    result = build_rebalance_metadata([days[0]], days[:26], 21,
                                     calendar_dates=days, calendar_name="XNYS")
    assert result["rebalance_overdue"] is True
    assert result["next_expected_rebalance_date"] == str(days[42])[:10]


def _contract_fixture(tmp_path, monkeypatch):
    from src import run_integrity
    data_path = tmp_path / "data.xlsx"
    fx_path = tmp_path / "fx.xlsx"
    data_path.write_bytes(b"fundamental-v1")
    fx_path.write_bytes(b"fx-v1")
    monkeypatch.setattr(run_integrity, "code_fingerprint", lambda: "code-v1")
    cfg = PipelineConfig(data_path=str(data_path), fx_source_path=str(fx_path))
    returns = pd.DataFrame({"AAA": [0., .01]}, index=pd.bdate_range("2026-07-01", periods=2))
    data = SimpleNamespace(returns=returns, tickers=["AAA"])
    inputs = run_integrity.capture_run_inputs(cfg)
    contract = run_integrity.build_run_contract(cfg, data, inputs)
    return cfg, data, inputs, contract


def test_same_size_same_mtime_fundamental_change_invalidates_contract(tmp_path, monkeypatch):
    from pathlib import Path
    from src.run_integrity import capture_run_inputs, build_run_contract, validate_run_contract, verify_run_inputs
    cfg, data, inputs, saved = _contract_fixture(tmp_path, monkeypatch)
    path = Path(cfg.data_path)
    stat = path.stat()
    path.write_bytes(b"fundamental-v2")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="changed during"):
        verify_run_inputs(cfg, inputs)
    current = build_run_contract(cfg, data, capture_run_inputs(cfg))
    with pytest.raises(ValueError, match="inputs"):
        validate_run_contract(saved, current)


def test_model_config_change_invalidates_contract_but_output_dir_does_not(tmp_path, monkeypatch):
    from src.run_integrity import build_run_contract, validate_run_contract
    cfg, data, inputs, saved = _contract_fixture(tmp_path, monkeypatch)
    moved = dataclasses.replace(cfg, output_dir="elsewhere")
    validate_run_contract(saved, build_run_contract(moved, data, inputs))
    changed = dataclasses.replace(cfg, prediction_ema_alpha=.25)
    with pytest.raises(ValueError, match="config_sha256"):
        validate_run_contract(saved, build_run_contract(changed, data, inputs))


def test_cached_export_requires_run_contract_even_with_matching_dates():
    dates = pd.to_datetime(["2026-07-02"])
    w = pd.Series({"AAA": 1.})
    result = SimpleNamespace(portfolio_returns=pd.Series(0., index=dates),
                             portfolio_weights={dates[0]: w}, daily_weights={dates[0]: w})
    with pytest.raises(ValueError, match="provenance"):
        validate_cached_result_compatibility(result, ["AAA"], pd.DataFrame(0., index=dates, columns=["AAA"]))


def test_reviewed_basis_change_requires_exact_value_and_keeps_audit_history(tmp_path):
    from scripts.acknowledge_tg_basis import acknowledge
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"data_quality": _quality(1.)}))
    quality = _quality(1.5)
    annotate_tg_ratio_jump(tmp_path, quality)
    previous_metrics = json.dumps({"data_quality": quality})
    path.write_text(previous_metrics)
    with pytest.raises(ValueError, match="match"):
        acknowledge(tmp_path, "AAA", 1.4, "reviewed")
    with pytest.raises(ValueError, match="reason"):
        acknowledge(tmp_path, "AAA", 1.5, "")
    acknowledge(tmp_path, "AAA", 1.5, "verified vendor basis")
    assert path.read_text() == previous_metrics
    assert annotate_tg_ratio_jump(tmp_path, _quality(1.5)) == {}
    history = json.loads((tmp_path / "tg_basis_acknowledgements.json").read_text())
    assert history["reviews"][0]["reason"] == "verified vendor basis"
