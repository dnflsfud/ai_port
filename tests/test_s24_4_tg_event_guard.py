# -*- coding: utf-8 -*-
"""§S24.4 (b) 등록 이벤트 정합 가드 — 결정 로그 §S24.4.

`tg_basis_events` 에 등록된 각 (종목, 이벤트일, factor) 에 대해 raw TG/가격 비율의 이벤트 전후 중앙값 step 을 재고
`log(step × factor)` 가 허용오차 안인지 본다. 벤더가 기저를 바꿔 단절이 사라지면(step ≈ 1) 등록 factor 는 이중 보정이
되므로 `inconsistent` → 로더 진단 `tg_basis_events_consistent_ok=False` → production 게이트 HOLD(fail-closed).
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.validate_portfolio_bundles import evaluate_production
from src.data_loader import UniverseData
from src.tg_basis_guard import (
    EVENT_GAP,
    EVENT_MIN_OBS,
    EVENT_TOL_LOG,
    EVENT_WINDOW,
    event_consistency,
)

IDX = pd.bdate_range(end="2026-09-29", periods=600)
EVENT = IDX[-200]


def _ratio(step_before_over_after: float, ticker: str = "XYZ", noise: float = 0.0):
    """Raw TG/price ratio: `after` level 1.10, `before` level = 1.10 * step."""
    rng = np.random.default_rng(0)
    base = np.where(IDX < EVENT, 1.10 * step_before_over_after, 1.10)
    return pd.DataFrame({ticker: base * (1 + noise * rng.standard_normal(len(IDX)))}, index=IDX)


def test_params_pinned():
    assert (EVENT_WINDOW, EVENT_GAP, EVENT_MIN_OBS, EVENT_TOL_LOG) == (60, 5, 20, 0.20)


def test_consistent_when_raw_step_matches_registered_factor():
    # Split-class event (DELL/DHR shape): raw before/after = 1/factor.
    out = event_consistency(_ratio(1 / 0.506, noise=0.02), {"XYZ": {str(EVENT.date()): 0.506}})
    ev = out["events"][0]
    assert ev["status"] == "consistent" and abs(ev["log_residual"]) < 0.05
    assert ev["n_before"] >= EVENT_MIN_OBS and ev["n_after"] >= EVENT_MIN_OBS
    assert out["ok"] is True and out["n_inconsistent"] == 0


def test_inconsistent_when_vendor_removed_the_step():
    # RTX/T after the 2026-09-29 basis switch: raw step ~1.0 while factor 1.696 is still registered.
    out = event_consistency(_ratio(1.0, noise=0.02), {"XYZ": {str(EVENT.date()): 1.696}})
    ev = out["events"][0]
    assert ev["status"] == "inconsistent" and ev["log_residual"] > EVENT_TOL_LOG
    assert out["ok"] is False and out["n_inconsistent"] == 1


def test_insufficient_window_is_none_not_pass():
    ratio = _ratio(2.0)
    ratio.iloc[: len(IDX) - 210] = np.nan  # only ~10 valid rows before the event
    out = event_consistency(ratio, {"XYZ": {str(EVENT.date()): 0.5}})
    assert out["events"][0]["status"] == "insufficient"
    assert out["ok"] is None and out["n_insufficient"] == 1


def test_ticker_outside_universe_is_skipped_and_no_events_is_ok():
    out = event_consistency(_ratio(2.0), {"ZZZ": {str(EVENT.date()): 0.5}})
    assert out["events"][0]["status"] == "not_in_universe" and out["ok"] is True
    assert event_consistency(_ratio(2.0), {})["ok"] is True
    assert event_consistency(_ratio(2.0), None)["ok"] is True


def test_invalid_registration_is_none():
    out = event_consistency(_ratio(2.0), {"XYZ": {"not-a-date": 0.5}})
    assert out["events"][0]["status"] == "invalid_date" and out["ok"] is None
    out = event_consistency(_ratio(2.0), {"XYZ": {str(EVENT.date()): 0.0}})
    assert out["events"][0]["status"] == "invalid_factor" and out["ok"] is None


# ------------------------------------------------------------------ loader wiring
def _shell(events, step: float):
    shell = UniverseData.__new__(UniverseData)
    ratio = _ratio(step)
    shell.sheets = {"Factset_TG_Price": ratio * 100.0}
    shell.local_prices = pd.DataFrame({"XYZ": 100.0}, index=IDX)
    shell.local_prices_nominal = pd.DataFrame({"XYZ": 100.0}, index=IDX)
    shell.data_quality = {"currency": {}}
    shell.config = SimpleNamespace(nominal_price_source="PX_LAST_UNADJ", tg_basis_events=events)
    return shell


def test_loader_publishes_event_check_and_flag():
    shell = _shell({"XYZ": {str(EVENT.date()): 0.5}}, step=2.0)
    shell._check_target_price_unit_ratio()
    cur = shell.data_quality["currency"]
    assert cur["tg_basis_events_consistent_ok"] is True
    assert cur["tg_basis_events_check"]["events"][0]["status"] == "consistent"
    # existing diagnostics untouched
    assert "tg_px_ratio_median" in cur and "tg_px_ratio_suspect" in cur


def test_loader_flags_double_correction():
    shell = _shell({"XYZ": {str(EVENT.date()): 1.696}}, step=1.0)
    shell._check_target_price_unit_ratio()
    assert shell.data_quality["currency"]["tg_basis_events_consistent_ok"] is False


def test_loader_without_events_is_ok_true():
    shell = _shell({}, step=1.0)
    shell.config = SimpleNamespace(nominal_price_source="PX_LAST_UNADJ")  # no attribute at all
    shell._check_target_price_unit_ratio()
    assert shell.data_quality["currency"]["tg_basis_events_consistent_ok"] is True


# ------------------------------------------------------------------ production gate
def _record(currency):
    return {"performance": {"data_quality": {"currency": currency}}}


def test_gate_check_is_fail_closed_on_missing_and_binds_on_false():
    assert evaluate_production(_record({"tg_px_ratio_suspect": {}}))["checks"]["tg_basis_events_consistent_ok"] is None
    checks = evaluate_production(_record({"tg_px_ratio_suspect": {}, "tg_basis_events_consistent_ok": False,
                                          "tg_basis_events_check": {"events": [
                                              {"ticker": "RTX", "date": "2020-04-03", "status": "inconsistent"}]}}))
    assert checks["checks"]["tg_basis_events_consistent_ok"] is False
    assert checks["status"] == "HOLD"
    assert checks["values"]["tg_basis_events_inconsistent"] == ["RTX@2020-04-03"]
    ok = evaluate_production(_record({"tg_px_ratio_suspect": {}, "tg_basis_events_consistent_ok": True}))
    assert ok["checks"]["tg_basis_events_consistent_ok"] is True
    assert ok["values"]["tg_basis_events_inconsistent"] == []


@pytest.mark.parametrize("value", ["yes", 1, None])
def test_gate_check_rejects_non_bool_values(value):
    assert evaluate_production(_record({"tg_basis_events_consistent_ok": value}))["checks"][
        "tg_basis_events_consistent_ok"] is None
