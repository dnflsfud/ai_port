"""§S22 stage-4 (decision log §S24): remaining Medium / Low audit findings.

Output-invariant fixes (guards, diagnostics, non-production paths):
  A-03 guard median on observed target-price cells only
  A-05 BusinessDays tail truncation is reported and, in the loader, fatal
  A-06 calendar_type diagnostic follows the configured calendar
  C-01 tuning holdout stops P&L at the cutoff
  C-03 run_backtest keeps the effective one_way_tc
  C-04 the inf guard no longer fires on ordinary NaN predictions
  C-05 bagging inertness pinned (subsample without subsample_freq)
  C-06 DR overlay gets the pre-lag prior (unreachable in production)
  D-03 production gate sees a benchmark-fallback / over-cap rebalance
  D-06 TE audit uses the conditioned cap when conditioning is on
  D-07 currency reconciliation has an independent FX identity check
  D-08 name-risk breach uses the optimizer's tolerance
  D-09 scheduled runs keep a dated log and a history line
  D-10 dashboard universe chip compares to the loaded funnel, not 150

Default-OFF research flags (parity pinned, arms pre-registered in §S24):
  B-02 revision_gradual_mask_disabled
  B-03 zscore_winsor_first_enabled (tests/test_utils.py)
  B-05 label_execution_lag_enabled
"""
import copy
import inspect
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.config import DEFAULT_CONFIG, PipelineConfig

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ A-03
def _tg_guard_quality(with_observed_mask: bool) -> dict:
    from src.data_loader import UniverseData, align_dates, preprocess_sheets

    idx = pd.bdate_range(end="2026-09-14", periods=300)
    px = pd.DataFrame({"AAA": 100.0, "BBB": 500.0, "NEW": 20.0}, index=idx)
    tg = pd.DataFrame({"AAA": 110.0, "BBB": 550.0, "NEW": np.nan}, index=idx)
    tg.iloc[-100:, tg.columns.get_loc("NEW")] = 22.0   # coverage starts 100 rows before the end
    sheets = preprocess_sheets({"PX_LAST": px, "Factset_TG_Price": tg}, tickers=["AAA", "BBB", "NEW"])
    sheets = align_dates(sheets, config=PipelineConfig(), diagnostics={})
    stub = SimpleNamespace(sheets=sheets, local_prices=sheets["PX_LAST"], data_quality={"currency": {}})
    if with_observed_mask:
        stub.raw_sheet_observed_mask = lambda name: tg.notna()
    UniverseData._check_target_price_unit_ratio(stub)
    return stub.data_quality["currency"]


def test_tg_guard_ignores_loader_imputed_pre_coverage_cells():
    legacy = _tg_guard_quality(with_observed_mask=False)
    assert legacy["tg_px_ratio_median"]["NEW"] > 5.0        # other names' levels leak in
    fixed = _tg_guard_quality(with_observed_mask=True)
    assert fixed["tg_px_ratio_median"]["NEW"] == pytest.approx(1.10, abs=1e-6)
    assert "NEW" not in fixed["tg_px_ratio_suspect"]
    for name in ("AAA", "BBB"):
        assert fixed["tg_px_ratio_median"][name] == pytest.approx(legacy["tg_px_ratio_median"][name])


# ------------------------------------------------------------------ A-05
def _short_business_days_raw():
    px_days = pd.bdate_range("2026-08-03", "2026-09-14")
    bdays = px_days[px_days <= "2026-09-10"]
    px = pd.DataFrame({"AAA": np.linspace(100, 110, len(px_days))}, index=px_days)
    return {
        "PX_LAST": px,
        "Daily_Returns": px.pct_change(),
        "BusinessDays": pd.DataFrame(index=pd.Index(bdays, name="BusinessDay")),
    }


def test_business_day_tail_truncation_is_reported_and_optionally_fatal():
    from src.data_loader import restrict_to_business_days

    out, diag = restrict_to_business_days(_short_business_days_raw())
    assert diag["tail_rows_dropped"] == 2
    assert diag["tail_dropped_from"] == "2026-09-11"
    assert diag["price_last_date"] == "2026-09-14"
    assert out["PX_LAST"].index.max() == pd.Timestamp("2026-09-10")
    with pytest.raises(ValueError, match="BusinessDays"):
        restrict_to_business_days(_short_business_days_raw(), fail_on_tail_truncation=True)


def test_business_day_tail_diagnostic_is_zero_when_calendar_covers_prices():
    from src.data_loader import restrict_to_business_days

    raw = _short_business_days_raw()
    raw["BusinessDays"] = pd.DataFrame(index=pd.Index(raw["PX_LAST"].index, name="BusinessDay"))
    _, diag = restrict_to_business_days(raw, fail_on_tail_truncation=True)
    assert diag["tail_rows_dropped"] == 0
    assert diag["tail_dropped_from"] is None


def test_universe_loader_fails_closed_on_tail_truncation():
    from src.data_loader import UniverseData

    assert "fail_on_tail_truncation=True" in inspect.getsource(UniverseData.__init__)


# ------------------------------------------------------------------ A-06
def test_calendar_type_diagnostic_follows_config():
    from src.data_loader import align_dates, preprocess_sheets

    idx = pd.bdate_range("2026-01-05", periods=30)
    px = pd.DataFrame({"AAA": 100.0, "BBB": 50.0}, index=idx)
    for enabled, expected in ((False, "weekday_index"), (True, "business_day")):
        diag = {}
        align_dates(
            preprocess_sheets({"PX_LAST": px, "Daily_Returns": px.pct_change()}, tickers=["AAA", "BBB"]),
            config=PipelineConfig(business_day_calendar_enabled=enabled),
            diagnostics=diag,
        )
        assert diag["calendar_type"] == expected


# ------------------------------------------------------------------ B-02
def _ramp_revision():
    dates = pd.bdate_range("2024-01-01", "2024-06-28")

    def ramp(start, lvl0, lvl1, days):
        s = pd.Series(float(lvl0), index=dates)
        i0 = dates.get_indexer([pd.Timestamp(start)])[0]
        step = (lvl1 - lvl0) / days
        for k in range(1, days + 1):
            s.iloc[i0 + k - 1] = lvl0 + step * k
        s.iloc[i0 + days:] = float(lvl1)
        return s

    return pd.DataFrame({
        "DOWN_FEB": ramp("2024-02-05", 40, -40, 10),
        "UP_FEB": ramp("2024-02-05", -40, 40, 10),
    })


class _RevStub:
    def __init__(self, rev):
        self._rev = rev

    def get_sheet(self, name):
        return self._rev


def _production_like_cfg(**overrides):
    cfg = copy.copy(DEFAULT_CONFIG)
    cfg.s15_fixpack_enabled = True
    cfg.revision_extension_max_days = 21
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_gradual_mask_flag_default_off_and_off_path_parity():
    from src.features.sellside import clean_revision_spikes

    assert PipelineConfig().revision_gradual_mask_disabled is False
    rev = _ramp_revision()
    legacy = clean_revision_spikes(rev, threshold=15.0, mode="reversion_gated")
    explicit = clean_revision_spikes(rev, threshold=15.0, mode="reversion_gated", gradual_mask_enabled=True)
    assert legacy.equals(explicit)
    # the legacy calendar-month mask freezes a genuine steady downgrade ...
    assert not np.allclose(legacy["DOWN_FEB"], rev["DOWN_FEB"])
    # ... and passes the mirror-image upgrade untouched
    assert np.allclose(legacy["UP_FEB"], rev["UP_FEB"])


def test_gradual_mask_disabled_lets_the_downgrade_flow():
    from src.features.sellside import get_cleaned_revision

    rev = _ramp_revision()
    off = get_cleaned_revision(_RevStub(rev), "Factset_EPS_Revision", config=_production_like_cfg())
    on = get_cleaned_revision(
        _RevStub(rev), "Factset_EPS_Revision",
        config=_production_like_cfg(revision_gradual_mask_disabled=True),
    )
    assert not np.allclose(off["DOWN_FEB"], rev["DOWN_FEB"])
    assert np.allclose(on["DOWN_FEB"], rev["DOWN_FEB"])
    assert np.allclose(on["UP_FEB"], rev["UP_FEB"])


# ------------------------------------------------------------------ B-05
def test_forward_return_lag_and_label_start_lag_helpers():
    from src.model_trainer import effective_label_horizon
    from src.target_engine import compute_forward_returns, label_start_lag

    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2024-01-01", periods=60)
    rets = pd.DataFrame(rng.normal(0, 0.01, (60, 3)), index=dates, columns=list("ABC"))
    cum = (1 + rets).cumprod()
    assert compute_forward_returns(rets, 5).equals(compute_forward_returns(rets, 5, lag=0))
    lagged = compute_forward_returns(rets, 5, lag=1)
    expected = cum.shift(-6) / cum.shift(-1) - 1
    assert np.allclose(lagged.fillna(0), expected.fillna(0))
    assert lagged.isna().equals(expected.isna())

    assert PipelineConfig().label_execution_lag_enabled is False
    off = PipelineConfig(execution_signal_lag_days=1)
    on = PipelineConfig(execution_signal_lag_days=1, label_execution_lag_enabled=True)
    assert label_start_lag(off) == 0
    assert label_start_lag(on) == 1
    assert label_start_lag(PipelineConfig(label_execution_lag_enabled=True)) == 0
    assert effective_label_horizon(off) == 20
    assert effective_label_horizon(on) == 21


def test_target_engine_passes_label_lag_to_both_forward_return_sites():
    from src import target_engine

    src = inspect.getsource(target_engine)
    assert src.count("compute_forward_returns(returns, horizon, lag=label_start_lag(config))") == 2


# ------------------------------------------------------------------ C-01 / C-04 / D-03 (simulate_portfolio)
def _toy_sim(n_days=420, n_tickers=20, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    rets = pd.DataFrame(rng.normal(0, 0.01, (n_days, n_tickers)), index=dates, columns=tickers)
    preds = pd.DataFrame(rng.normal(size=(n_days, n_tickers)), index=dates, columns=tickers)
    return dates, tickers, rets, preds


def _tilt_optimizer(pred_row, hist, prev_w, s_map, bm_w):
    w = bm_w + 0.02 * np.sign(pred_row.fillna(0).values)
    w = np.clip(w, 0, None)
    return w / w.sum()


def test_tuning_holdout_stops_pnl_at_the_cutoff():
    from src.backtest import simulate_portfolio

    dates, tickers, rets, preds = _toy_sim()
    cutoff = dates[300]
    tuning = PipelineConfig(enforce_oos_holdout=True, train_cutoff_date=str(cutoff.date()))
    res = simulate_portfolio(preds, rets, tickers, dates, rebalance_freq=21,
                             optimizer_fn=_tilt_optimizer, config=tuning)
    assert res.portfolio_returns.index.max() <= cutoff
    assert res.turnover.index.max() <= cutoff
    assert len(res.portfolio_returns) > 0

    prod = PipelineConfig(enforce_oos_holdout=False, train_cutoff_date=str(cutoff.date()))
    full = simulate_portfolio(preds, rets, tickers, dates, rebalance_freq=21,
                              optimizer_fn=_tilt_optimizer, config=prod)
    assert full.portfolio_returns.index.max() == dates[-1]
    # identical up to the cutoff
    pre = full.portfolio_returns.loc[:cutoff]
    assert np.allclose(pre.values, res.portfolio_returns.reindex(pre.index).values)


def test_inf_guard_is_silent_on_nan_and_fires_on_inf(caplog):
    from src.backtest import simulate_portfolio

    dates, tickers, rets, preds = _toy_sim(n_days=90, n_tickers=15, seed=0)
    preds["T14"] = np.nan                         # a not-yet-listed name
    bm_only = lambda pred_row, hist, prev_w, s_map, bm_w: bm_w  # noqa: E731
    with caplog.at_level(logging.WARNING, logger="src.backtest"):
        simulate_portfolio(preds, rets, tickers, dates, rebalance_freq=21, optimizer_fn=bm_only)
    assert not [r for r in caplog.records if "non-finite" in r.getMessage()]

    caplog.clear()
    preds.iloc[0, 0] = np.inf
    with caplog.at_level(logging.WARNING, logger="src.backtest"):
        simulate_portfolio(preds, rets, tickers, dates, rebalance_freq=21, optimizer_fn=bm_only)
    hits = [r for r in caplog.records if "non-finite" in r.getMessage()]
    assert len(hits) == 1


def test_simulation_records_fallback_rebalance_dates():
    from src.backtest import simulate_portfolio

    dates, tickers, rets, preds = _toy_sim(n_days=120, n_tickers=12, seed=2)
    bm_only = lambda pred_row, hist, prev_w, s_map, bm_w: bm_w  # noqa: E731
    res = simulate_portfolio(preds, rets, tickers, dates, rebalance_freq=21, optimizer_fn=bm_only)
    assert res.optimizer_failures == len(res.turnover)
    assert list(res.optimizer_fallback_dates) == list(res.turnover.index)

    clean = simulate_portfolio(preds, rets, tickers, dates, rebalance_freq=21, optimizer_fn=_tilt_optimizer)
    assert clean.optimizer_failures == 0
    assert list(clean.optimizer_fallback_dates) == []


def test_run_backtest_copies_one_way_tc_and_fallback_dates():
    from src.backtest import run_backtest

    src = inspect.getsource(run_backtest)
    assert "result.one_way_tc = sim_result.one_way_tc" in src
    assert "result.optimizer_fallback_dates = sim_result.optimizer_fallback_dates" in src


# ------------------------------------------------------------------ C-05
def test_bagging_is_documented_inert_in_default_and_production_params():
    import yaml

    params = DEFAULT_CONFIG.lgbm_params
    assert params.get("subsample") == 0.8
    assert int(params.get("subsample_freq", 0)) == 0      # bagging_freq 0 -> subsample inert
    manifest = yaml.safe_load((ROOT / "variants" / "codex_causal_rank_65.yaml").read_text(encoding="utf-8"))
    prod = manifest["overrides"]["lgbm_params"]
    assert prod.get("subsample") == 0.8
    assert int(prod.get("subsample_freq", 0)) == 0
    assert "C-05" in inspect.getsource(PipelineConfig)
    assert "C-05" in (ROOT / "variants" / "codex_causal_rank_65.yaml").read_text(encoding="utf-8")


# ------------------------------------------------------------------ C-06
def test_dr_overlay_uses_pre_lag_prior_and_docstring_is_current():
    import src.rl as rl
    import run_variant
    from src.backtest import run_backtest

    assert "pre_execution_raw_predictions" in inspect.getsource(run_backtest)
    assert 'getattr(base, "pre_execution_raw_predictions"' in inspect.getsource(run_variant)
    assert "iter15_65tkr_reb21_vtg" not in (rl.__doc__ or "")


# ------------------------------------------------------------------ D-03 gate
def test_production_gate_sees_fallback_and_over_cap_turnover():
    from scripts.validate_portfolio_bundles import evaluate_production

    bad = evaluate_production({"_operations": {
        "latest_rebalance_used_fallback": True,
        "turnover_two_way_latest": 0.60, "max_two_way_turnover": 0.15,
    }})
    assert bad["checks"]["latest_rebalance_no_fallback_ok"] is False
    assert bad["checks"]["turnover_within_cap_ok"] is False
    assert bad["status"] == "HOLD"
    assert bad["values"]["turnover_two_way_latest"] == 0.60

    good = evaluate_production({"_operations": {
        "latest_rebalance_used_fallback": False,
        "turnover_two_way_latest": 0.05, "max_two_way_turnover": 0.15,
    }})
    assert good["checks"]["latest_rebalance_no_fallback_ok"] is True
    assert good["checks"]["turnover_within_cap_ok"] is True

    missing = evaluate_production({})
    assert missing["checks"]["latest_rebalance_no_fallback_ok"] is None   # fail-closed
    assert missing["checks"]["turnover_within_cap_ok"] is None


def test_exporter_fallback_operations_fields():
    from scripts.export_operating_data import fallback_operations_fields

    last = pd.Timestamp("2026-09-04")
    res = SimpleNamespace(optimizer_fallback_dates=[pd.Timestamp("2025-01-06"), last])
    cfg = PipelineConfig(max_single_turnover=0.15)
    fields = fallback_operations_fields(res, last, cfg)
    assert fields["latest_rebalance_used_fallback"] is True
    assert fields["optimizer_fallback_dates"] == ["2025-01-06", "2026-09-04"]
    assert fields["max_two_way_turnover"] == 0.15

    clean = fallback_operations_fields(SimpleNamespace(optimizer_fallback_dates=[]), last, cfg)
    assert clean["latest_rebalance_used_fallback"] is False
    legacy = fallback_operations_fields(SimpleNamespace(), last, cfg)   # pre-fix pickle
    assert legacy["latest_rebalance_used_fallback"] is None
    assert legacy["optimizer_fallback_dates"] is None


# ------------------------------------------------------------------ D-06
def test_te_audit_limit_uses_conditioned_cap_only_when_enabled(monkeypatch):
    import src.carry_te_conditioning as cte
    from scripts.export_operating_data import effective_te_limit

    as_of = pd.Timestamp("2026-09-04")
    off = PipelineConfig(max_te_annual=0.035, carry_te_conditioning_enabled=False)
    assert effective_te_limit(off, SimpleNamespace(factor_prices=None), as_of) == (0.035, 0.035, None)

    on = PipelineConfig(max_te_annual=0.035, carry_te_conditioning_enabled=True)
    idx = pd.bdate_range("2026-08-01", "2026-09-10")
    monkeypatch.setattr(cte, "build_te_cap_multipliers",
                        lambda factor_px, config: pd.Series(0.8, index=idx))
    limit, base, mult = effective_te_limit(on, SimpleNamespace(factor_prices=pd.DataFrame(index=idx)), as_of)
    assert (limit, base, mult) == (pytest.approx(0.028), 0.035, 0.8)


# ------------------------------------------------------------------ D-07
def _currency_inputs(seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-01-01", periods=60)
    tickers = ["AAA", "JPX"]
    local = pd.DataFrame(rng.normal(0, 0.01, (60, 2)), index=dates, columns=tickers)
    fx_ret = pd.DataFrame({"JPX": rng.normal(0, 0.005, 60)}, index=dates)
    fx_rate = (1 + fx_ret).cumprod() * 0.0067
    usd = local.copy()
    usd["JPX"] = (1 + local["JPX"]) * (1 + fx_ret["JPX"]) - 1
    w = pd.DataFrame(0.5, index=dates, columns=tickers)
    return dict(
        tickers=tickers, dates=dates, usd_returns=usd, local_returns=local,
        fx_returns=fx_ret, fx_rates_usd_per_local=fx_rate,
        currency_map={"AAA": "USD", "JPX": "JPY"},
        portfolio_entering_weights=w, benchmark_entering_weights=w,
        latest_portfolio_weights=w.iloc[-1], latest_benchmark_weights=w.iloc[-1],
        data_quality={"fx_data_as_of": str(dates[-1].date())},
    )


def test_currency_reconciliation_can_now_fail_on_inconsistent_local_returns():
    from scripts.export_operating_data import build_currency_attribution

    ok = build_currency_attribution(**_currency_inputs())
    rec = ok["reconciliation"]
    assert rec["passed"] is True
    assert rec["independent_fx_passed"] is True
    assert rec["independent_fx_cells_checked"] == 120
    assert rec["independent_fx_max_abs_residual"] <= rec["independent_fx_tolerance"]

    bad_inputs = _currency_inputs()
    rng = np.random.default_rng(9)
    bad_inputs["local_returns"] = pd.DataFrame(
        rng.normal(0, 0.05, (60, 2)), index=bad_inputs["dates"], columns=bad_inputs["tickers"])
    bad = build_currency_attribution(**bad_inputs)
    assert bad["reconciliation"]["independent_fx_passed"] is False
    assert bad["reconciliation"]["passed"] is False


def test_currency_independent_check_skips_unobserved_usd_cells():
    from scripts.export_operating_data import build_currency_attribution

    inputs = _currency_inputs()
    inputs["usd_returns"] = inputs["usd_returns"].copy()
    inputs["usd_returns"].iloc[:5, 1] = 0.0                    # loader fillna(0) on masked cells
    mask = pd.DataFrame(True, index=inputs["dates"], columns=inputs["tickers"])
    mask.iloc[:5, 1] = False
    assert build_currency_attribution(**inputs)["reconciliation"]["independent_fx_passed"] is False
    out = build_currency_attribution(**inputs, usd_observed_mask=mask)
    assert out["reconciliation"]["independent_fx_passed"] is True
    assert out["reconciliation"]["independent_fx_cells_checked"] == 115


# ------------------------------------------------------------------ D-08
def test_name_risk_breach_uses_optimizer_tolerance():
    from scripts.export_operating_data import risk_share_breached
    from src.portfolio_optimizer import NAME_RISK_CAP_TOL

    assert NAME_RISK_CAP_TOL == 0.01
    assert risk_share_breached(0.355, 0.35, NAME_RISK_CAP_TOL) is False
    assert risk_share_breached(0.361, 0.35, NAME_RISK_CAP_TOL) is True
    assert risk_share_breached(None, 0.35, NAME_RISK_CAP_TOL) is False
    assert risk_share_breached(0.76, 0.75, 0.0) is True
    src = inspect.getsource(__import__("scripts.export_operating_data", fromlist=["main"]).main)
    assert "risk_share_breached(top_name_active_risk_share, max_name_risk, NAME_RISK_CAP_TOL)" in src


# ------------------------------------------------------------------ D-09
def test_scheduled_wrapper_keeps_dated_log_and_history():
    text = (ROOT / "run_and_upload_scheduled.bat").read_text(encoding="utf-8")
    assert 'set "AI_PORT_NO_DASHBOARD=1"' in text
    assert 'call "%~dp0run_and_upload.bat"' in text
    assert "scheduled_run_%RUN_TS%.log" in text
    assert "scheduled_run_history.log" in text
    assert "scheduled_run_last.log" in text          # unchanged path for readers
    assert "exit /b %RC%" in text


# ------------------------------------------------------------------ D-10
def test_dashboard_universe_chip_compares_to_loaded_funnel():
    import streamlit_app

    ok = {"universe_funnel": {"full_universe_count": 250, "loaded_ticker_count": 250, "missing_tickers": []}}
    assert streamlit_app.universe_chip_ok(250, ok) is True
    assert streamlit_app.universe_chip_ok(150, ok) is False
    short = {"universe_funnel": {"full_universe_count": 250, "loaded_ticker_count": 249, "missing_tickers": ["X"]}}
    assert streamlit_app.universe_chip_ok(249, short) is False
    assert streamlit_app.universe_chip_ok(0, ok) is False
    assert streamlit_app.universe_chip_ok(250, {}) is False
    assert "universe_size == 150" not in inspect.getsource(streamlit_app)
