#!/usr/bin/env python
"""Validate operating bundles and publish the dashboard portfolio registry."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "portfolio.json",
    "performance.json",
    "holdings.json",
    "features.json",
    "operations.json",
    "contribution.json",
    "risk.json",
    "monitoring.json",
    "feature_attribution.json",
    "currency.json",
    "returns.csv",
)

CURRENCY_RETURN_COLUMNS = (
    "portfolio_local_gross_ret",
    "portfolio_fx_effect",
    "portfolio_usd_gross_ret",
    "benchmark_local_ret",
    "benchmark_fx_effect",
    "benchmark_usd_gross_ret",
    "active_local_gross_ret",
    "active_fx_effect",
    "active_usd_gross_ret",
)


def _rooted(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        value = json.load(fh)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _finite_float(value, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not pd.notna(result) or result in (float("inf"), float("-inf")):
        raise ValueError(f"{label} must be finite")
    return result


def _validate_currency_payload(
    bundle_dir: Path,
    currency: dict,
    meta: dict,
    returns: pd.DataFrame,
    returns_as_of: str,
) -> None:
    if currency.get("base_currency") != "USD" or meta.get("base_currency") != "USD":
        raise ValueError(f"{bundle_dir}: base currency must be USD")
    if currency.get("as_of") != returns_as_of:
        raise ValueError(f"{bundle_dir}: currency/returns as_of mismatch")
    if not currency.get("method"):
        raise ValueError(f"{bundle_dir}: currency attribution method is missing")
    fx_data_as_of = currency.get("fx_data_as_of")
    if not fx_data_as_of or pd.Timestamp(fx_data_as_of) > pd.Timestamp(returns_as_of):
        raise ValueError(f"{bundle_dir}: invalid fx_data_as_of")

    coverage = currency.get("coverage")
    if not isinstance(coverage, dict):
        raise ValueError(f"{bundle_dir}: currency coverage is missing")
    required_coverage = ("total", "mapped", "missing", "missing_fx", "stale")
    missing_fields = [field for field in required_coverage if field not in coverage]
    if missing_fields:
        raise ValueError(f"{bundle_dir}: currency coverage missing fields: {missing_fields}")
    counts = {}
    for field in required_coverage:
        value = coverage[field]
        if isinstance(value, bool):
            raise ValueError(f"{bundle_dir}: currency coverage {field} must be a count")
        try:
            counts[field] = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{bundle_dir}: currency coverage {field} must be a count") from exc
        if counts[field] < 0:
            raise ValueError(f"{bundle_dir}: currency coverage {field} is negative")
    universe_size = int(meta.get("universe_size", 0))
    if counts["total"] != universe_size or counts["mapped"] + counts["missing"] != universe_size:
        raise ValueError(f"{bundle_dir}: currency mapping coverage does not match universe")
    if counts["mapped"] != universe_size or counts["missing"] != 0:
        raise ValueError(f"{bundle_dir}: incomplete currency mapping coverage")
    if counts["missing_fx"] != 0:
        raise ValueError(f"{bundle_dir}: missing FX series in currency coverage")
    if counts["stale"] != 0:
        raise ValueError(f"{bundle_dir}: stale FX series in currency coverage")

    by_ticker = currency.get("by_ticker")
    if not isinstance(by_ticker, list) or len(by_ticker) != universe_size:
        raise ValueError(f"{bundle_dir}: currency by_ticker coverage mismatch")
    tickers = [row.get("ticker") for row in by_ticker if isinstance(row, dict)]
    if len(tickers) != len(set(tickers)) or set(tickers) != set(meta.get("universe") or []):
        raise ValueError(f"{bundle_dir}: currency by_ticker universe mismatch")
    if any(not row.get("currency") for row in by_ticker):
        raise ValueError(f"{bundle_dir}: currency by_ticker contains unmapped names")

    attribution_fields = (
        "target_weight", "benchmark_weight", "active_weight",
        "portfolio_local_contribution", "portfolio_fx_contribution",
        "portfolio_usd_contribution", "benchmark_local_contribution",
        "benchmark_fx_contribution", "benchmark_usd_contribution",
        "active_local_contribution", "active_fx_contribution",
        "active_usd_contribution",
    )
    for row in by_ticker:
        missing_attribution = [field for field in attribution_fields if field not in row]
        if missing_attribution:
            raise ValueError(
                f"{bundle_dir}: currency by_ticker missing fields: {missing_attribution}"
            )
        for field in attribution_fields:
            _finite_float(row[field], f"currency by_ticker {field}")

    by_currency = currency.get("by_currency")
    if not isinstance(by_currency, list) or not by_currency:
        raise ValueError(f"{bundle_dir}: currency by_currency is missing")
    for row in by_currency:
        missing_attribution = [field for field in attribution_fields if field not in row]
        if missing_attribution:
            raise ValueError(
                f"{bundle_dir}: currency by_currency missing fields: {missing_attribution}"
            )
    target_weight = sum(_finite_float(row.get("target_weight"), "currency target weight") for row in by_currency)
    benchmark_weight = sum(
        _finite_float(row.get("benchmark_weight"), "currency benchmark weight")
        for row in by_currency
    )
    if abs(target_weight - 1.0) > 1e-6 or abs(benchmark_weight - 1.0) > 1e-6:
        raise ValueError(f"{bundle_dir}: currency exposure weights do not sum to one")

    summary = currency.get("summary")
    required_summary = (
        "non_usd_target_weight", "non_usd_benchmark_weight",
        "portfolio_fx_arithmetic_contribution",
        "benchmark_fx_arithmetic_contribution",
        "active_fx_arithmetic_contribution",
    )
    if not isinstance(summary, dict) or any(field not in summary for field in required_summary):
        raise ValueError(f"{bundle_dir}: currency summary schema is incomplete")
    for field in required_summary:
        _finite_float(summary[field], f"currency summary {field}")
    non_usd_rows = [row for row in by_currency if row.get("currency") not in {"USD", "UNKNOWN"}]
    expected_non_usd_target = sum(float(row["target_weight"]) for row in non_usd_rows)
    expected_non_usd_benchmark = sum(float(row["benchmark_weight"]) for row in non_usd_rows)
    if (
        abs(float(summary["non_usd_target_weight"]) - expected_non_usd_target) > 1e-8
        or abs(float(summary["non_usd_benchmark_weight"]) - expected_non_usd_benchmark) > 1e-8
    ):
        raise ValueError(f"{bundle_dir}: currency non-USD summary exposure mismatch")

    reconciliation = currency.get("reconciliation")
    if not isinstance(reconciliation, dict) or reconciliation.get("passed") is not True:
        raise ValueError(f"{bundle_dir}: currency reconciliation did not pass")
    tolerance = _finite_float(reconciliation.get("tolerance", 1e-10), "currency tolerance")
    tolerance = max(tolerance, 1e-12)
    reconciliation_fields = (
        "max_daily_portfolio_error", "max_daily_benchmark_error", "max_daily_active_error",
        "period_portfolio_error", "period_benchmark_error", "period_active_error",
    )
    for field in reconciliation_fields:
        if abs(_finite_float(reconciliation.get(field), f"currency {field}")) > tolerance:
            raise ValueError(f"{bundle_dir}: currency reconciliation {field} exceeds tolerance")

    daily = currency.get("daily")
    if not isinstance(daily, list) or len(daily) != len(returns):
        raise ValueError(f"{bundle_dir}: currency daily rows do not match returns.csv")
    daily_df = pd.DataFrame(daily)
    required_daily = {
        "date", "portfolio_local_return", "portfolio_fx_effect",
        "portfolio_usd_gross_return", "benchmark_local_return",
        "benchmark_fx_effect", "benchmark_usd_gross_return",
        "active_local_return", "active_fx_effect", "active_usd_gross_return",
    }
    if not required_daily.issubset(daily_df.columns):
        raise ValueError(f"{bundle_dir}: currency daily schema is incomplete")
    daily_df["date"] = pd.to_datetime(daily_df["date"], errors="coerce")
    if daily_df["date"].isna().any() or set(daily_df["date"]) != set(returns["date"]):
        raise ValueError(f"{bundle_dir}: currency daily dates do not match returns.csv")
    daily_tolerance = max(tolerance, 5e-9)
    equations = (
        ("portfolio_local_return", "portfolio_fx_effect", "portfolio_usd_gross_return"),
        ("benchmark_local_return", "benchmark_fx_effect", "benchmark_usd_gross_return"),
        ("active_local_return", "active_fx_effect", "active_usd_gross_return"),
    )
    for local_col, fx_col, usd_col in equations:
        values = daily_df[[local_col, fx_col, usd_col]].apply(pd.to_numeric, errors="coerce")
        if values.isna().any().any():
            raise ValueError(f"{bundle_dir}: currency daily values are non-numeric")
        if (values[local_col] + values[fx_col] - values[usd_col]).abs().max() > daily_tolerance:
            raise ValueError(f"{bundle_dir}: currency daily local + FX != USD")

    missing_return_columns = [col for col in CURRENCY_RETURN_COLUMNS if col not in returns.columns]
    if missing_return_columns:
        raise ValueError(
            f"{bundle_dir}: returns.csv missing currency columns: {missing_return_columns}"
        )
    csv_values = returns[list(CURRENCY_RETURN_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if csv_values.isna().any().any():
        raise ValueError(f"{bundle_dir}: returns.csv currency columns are non-numeric")
    csv_equations = (
        ("portfolio_local_gross_ret", "portfolio_fx_effect", "portfolio_usd_gross_ret"),
        ("benchmark_local_ret", "benchmark_fx_effect", "benchmark_usd_gross_ret"),
        ("active_local_gross_ret", "active_fx_effect", "active_usd_gross_ret"),
    )
    for local_col, fx_col, usd_col in csv_equations:
        if (csv_values[local_col] + csv_values[fx_col] - csv_values[usd_col]).abs().max() > daily_tolerance:
            raise ValueError(f"{bundle_dir}: returns.csv local + FX != USD gross")


def _validate_operations_payload(
    bundle_dir: Path,
    operations: dict,
    meta: dict,
    returns_as_of: str,
) -> None:
    required = (
        "data_as_of", "base_currency", "trade_data_as_of", "returns_data_as_of",
        "trade_data_as_of_valid", "turnover_two_way_latest",
        "turnover_two_way_reconstructed", "turnover_reconciliation_error",
        "turnover_reconciled", "one_way_transaction_cost_rate",
        "one_way_transaction_cost_bps", "expected_transaction_cost",
        "n_trades", "trade_list",
    )
    missing = [field for field in required if field not in operations]
    if missing:
        raise ValueError(f"{bundle_dir}: operations missing fields: {missing}")
    if operations.get("base_currency") != "USD" or operations.get("data_as_of") != returns_as_of:
        raise ValueError(f"{bundle_dir}: operations data/base currency mismatch")
    if operations.get("trade_data_as_of") != meta.get("last_rebalance_date"):
        raise ValueError(f"{bundle_dir}: operations trade_data_as_of mismatch")
    if operations.get("returns_data_as_of") != returns_as_of:
        raise ValueError(f"{bundle_dir}: operations returns_data_as_of mismatch")
    if operations.get("trade_data_as_of_valid") is not True:
        raise ValueError(f"{bundle_dir}: operations trade data is not valid as of rebalance")
    if operations.get("turnover_reconciled") is not True:
        raise ValueError(f"{bundle_dir}: operations turnover did not reconcile")

    reported = _finite_float(operations.get("turnover_two_way_latest"), "operations turnover")
    reconstructed = _finite_float(
        operations.get("turnover_two_way_reconstructed"), "operations reconstructed turnover"
    )
    reconciliation_error = _finite_float(
        operations.get("turnover_reconciliation_error"), "operations turnover error"
    )
    tolerance = 1e-8
    if abs(reported - reconstructed) > tolerance or abs(reconciliation_error) > tolerance:
        raise ValueError(f"{bundle_dir}: operations turnover reconciliation mismatch")
    trades = operations.get("trade_list")
    if not isinstance(trades, list) or int(operations.get("n_trades")) != len(trades):
        raise ValueError(f"{bundle_dir}: operations trade count mismatch")
    delta_l1 = 0.0
    for trade in trades:
        if not isinstance(trade, dict):
            raise ValueError(f"{bundle_dir}: invalid operations trade row")
        pre_trade = _finite_float(trade.get("pre_trade"), "trade pre_trade")
        if "prev" in trade and abs(_finite_float(trade.get("prev"), "trade prev") - pre_trade) > tolerance:
            raise ValueError(f"{bundle_dir}: trade prev/pre_trade alias mismatch")
        target = _finite_float(trade.get("target"), "trade target")
        delta = _finite_float(trade.get("delta"), "trade delta")
        if abs(target - pre_trade - delta) > tolerance:
            raise ValueError(f"{bundle_dir}: trade delta does not use drifted pre-trade weight")
        delta_l1 += abs(delta)
    if abs(delta_l1 - reported) > tolerance:
        raise ValueError(f"{bundle_dir}: trade-list L1 does not match turnover")
    cost_rate = _finite_float(
        operations.get("one_way_transaction_cost_rate"), "operations cost rate"
    )
    cost_bps = _finite_float(
        operations.get("one_way_transaction_cost_bps"), "operations cost bps"
    )
    expected_cost = _finite_float(
        operations.get("expected_transaction_cost"), "operations expected cost"
    )
    if abs(cost_bps - cost_rate * 10000.0) > 1e-6:
        raise ValueError(f"{bundle_dir}: operations cost bps mismatch")
    if abs(expected_cost - reported * cost_rate) > tolerance:
        raise ValueError(f"{bundle_dir}: operations expected transaction cost mismatch")

    # New operating additions are validated when present (every freshly exported
    # bundle carries them); a legacy-schema bundle without them is left untouched.
    drift = operations.get("current_drift")
    if isinstance(drift, dict):
        drift_rows = drift.get("by_ticker")
        if not isinstance(drift_rows, list) or not drift_rows:
            raise ValueError(f"{bundle_dir}: current_drift by_ticker is missing")
        drift_l1 = _finite_float(drift.get("drift_l1"), "current_drift drift_l1")
        sum_abs_drift = sum(
            abs(_finite_float(row.get("drift"), "current_drift drift")) for row in drift_rows
        )
        if abs(drift_l1 - sum_abs_drift) > 1e-8:
            raise ValueError(f"{bundle_dir}: current_drift L1 does not match by_ticker")
        weight_sum = _finite_float(drift.get("weight_sum"), "current_drift weight_sum")
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(f"{bundle_dir}: current_drift weight_sum is not one")
        if int(drift.get("days_since_rebalance")) < 0:
            raise ValueError(f"{bundle_dir}: current_drift days_since_rebalance is negative")
        if drift.get("last_rebalance_date") != meta.get("last_rebalance_date"):
            raise ValueError(f"{bundle_dir}: current_drift last_rebalance_date mismatch")

    sector_active = operations.get("sector_active")
    if sector_active is not None:
        limit = _finite_float(operations.get("sector_deviation_limit"), "sector_deviation_limit")
        if not isinstance(sector_active, list) or not sector_active:
            raise ValueError(f"{bundle_dir}: operations sector_active is empty")
        for row in sector_active:
            row_limit = _finite_float(row.get("limit"), "sector_active limit")
            if abs(row_limit - limit) > 1e-12:
                raise ValueError(f"{bundle_dir}: sector_active limit mismatch")
            active = _finite_float(row.get("active"), "sector_active active")
            if bool(row.get("binding")) != bool(abs(active) >= limit - 1e-9):
                raise ValueError(f"{bundle_dir}: sector_active binding logic mismatch")


def run_config_drift(meta: dict, run_dir: Path):
    """Report-only §S16 O4 check: was this run produced by today's variant?

    The bundle names the run_dir it was exported from; that run's
    experiment_manifest carries the resolved config. Comparing it with the
    variant yaml's overrides surfaces a bundle published from a pre-flip run
    (e.g. a dashboard still showing a retired baseline). Report-only by
    design: a flip adopted after the last full backtest is a legitimate,
    temporary drift and must not break operational validation.
    Returns None when either side is unavailable.
    """
    import yaml

    variant_rel = meta.get("variant_path")
    manifest_path = run_dir / "experiment_manifest.json"
    if not variant_rel or not manifest_path.exists():
        return None
    variant_path = _rooted(Path(variant_rel)).resolve()
    if not variant_path.exists():
        return None
    config = _read_json(manifest_path).get("config") or {}
    variant = yaml.safe_load(variant_path.read_text(encoding="utf-8")) or {}
    overrides = variant.get("overrides") or {}
    # Canonical-JSON normalisation: yaml and the manifest's JSON round-trip
    # differ in dict key ORDER and in int/float typing, not in meaning. Plain
    # str() reports every dict-valued override (lgbm_params) as drifted, which
    # would make the warning noise and get it ignored.
    def _canon(value):
        return json.dumps(value, sort_keys=True, default=str)

    mismatched = sorted(
        str(key) for key, value in overrides.items()
        if _canon(config.get(key)) != _canon(value)
    )
    return {
        "mismatched_keys": mismatched,
        "manifest_path": str(manifest_path),
        "variant_path": str(variant_path),
    }


def validate_bundle(bundle_dir: Path) -> dict:
    """Validate one bundle and return registry-ready metadata."""
    bundle_dir = _rooted(bundle_dir).resolve()
    missing = [name for name in REQUIRED_FILES if not (bundle_dir / name).exists()]
    if missing:
        raise ValueError(f"{bundle_dir}: missing required files: {missing}")

    meta = _read_json(bundle_dir / "portfolio.json")
    perf = _read_json(bundle_dir / "performance.json")
    holdings = _read_json(bundle_dir / "holdings.json")
    operations = _read_json(bundle_dir / "operations.json")
    currency = _read_json(bundle_dir / "currency.json")
    risk = _read_json(bundle_dir / "risk.json")
    for name in REQUIRED_FILES:
        if name.endswith(".json"):
            _read_json(bundle_dir / name)

    returns = pd.read_csv(bundle_dir / "returns.csv", parse_dates=["date"])
    if returns.empty or returns["date"].isna().all():
        raise ValueError(f"{bundle_dir}: returns.csv has no dated observations")
    if returns["date"].duplicated().any():
        raise ValueError(f"{bundle_dir}: returns.csv has duplicate dates")
    weekend_rows = returns.loc[returns["date"].dt.dayofweek >= 5, "date"]
    if not weekend_rows.empty:
        sample = weekend_rows.dt.strftime("%Y-%m-%d").head(5).tolist()
        raise ValueError(f"{bundle_dir}: returns.csv contains weekend dates: {sample}")
    returns_as_of = returns["date"].max().strftime("%Y-%m-%d")
    if perf.get("as_of") != returns_as_of or meta.get("data_as_of") != returns_as_of:
        raise ValueError(
            f"{bundle_dir}: as_of mismatch "
            f"performance={perf.get('as_of')} meta={meta.get('data_as_of')} "
            f"returns={returns_as_of}"
        )
    if pd.Timestamp(holdings.get("as_of")) > pd.Timestamp(returns_as_of):
        raise ValueError(f"{bundle_dir}: holdings date is after performance date")
    _validate_currency_payload(bundle_dir, currency, meta, returns, returns_as_of)

    data_quality = perf.get("data_quality") or {}
    tail_days = data_quality.get("tail_ffill_days")
    max_tail_days = data_quality.get("max_tail_ffill_days")
    if tail_days is not None and max_tail_days is not None and int(tail_days) > int(max_tail_days):
        fail_on = bool(data_quality.get("fail_on_stale_tail_ffill", False))
        if fail_on:
            raise ValueError(
                f"{bundle_dir}: stale tail exceeds limit: {tail_days} > {max_tail_days}"
            )
        print(
            f"[bundle-validator] WARN: {bundle_dir}: stale tail {tail_days} > {max_tail_days} "
            f"(fail_on_stale_tail_ffill=false)",
            file=sys.stderr,
        )

    rebalance_fields = (
        "last_rebalance_date",
        "previous_rebalance_date",
        "next_expected_rebalance_date",
        "rebalance_freq_days",
        "rebalance_calendar",
        "rows_since_last_rebalance",
        "rows_until_next_rebalance",
        "is_rebalance_data_as_of",
        "next_rebalance_is_estimate",
    )
    missing_rebalance = [field for field in rebalance_fields if field not in meta]
    if missing_rebalance:
        raise ValueError(f"{bundle_dir}: missing rebalance metadata: {missing_rebalance}")
    for field in rebalance_fields:
        if operations.get(field) != meta.get(field):
            raise ValueError(f"{bundle_dir}: operations/portfolio {field} mismatch")
    last_rebalance = pd.Timestamp(meta["last_rebalance_date"])
    next_rebalance = pd.Timestamp(meta["next_expected_rebalance_date"])
    if (
        holdings.get("as_of") != meta["last_rebalance_date"]
        or operations.get("as_of") != meta["last_rebalance_date"]
    ):
        raise ValueError(f"{bundle_dir}: latest holdings/operations rebalance date mismatch")
    if next_rebalance <= last_rebalance or next_rebalance.dayofweek >= 5:
        raise ValueError(f"{bundle_dir}: invalid next expected rebalance date")
    freq = int(meta["rebalance_freq_days"])
    rows_since = int(meta["rows_since_last_rebalance"])
    rows_until = int(meta["rows_until_next_rebalance"])
    # §S15 FIX4: 스킵된 리밸 뒤에는 rows_since >= freq 인 overdue 상태가
    # rebalance_overdue 플래그와 함께 정당하게 export된다 — 그 경우 카운터는
    # 원 그리드 기준 (since + until) % freq == 0 이어야 한다.
    counters_ok = rows_since + rows_until == freq
    if bool(meta.get("rebalance_overdue", False)):
        counters_ok = (
            rows_since >= freq
            and 0 < rows_until <= freq
            and (rows_since + rows_until) % freq == 0
        )
    if freq <= 0 or rows_since < 0 or rows_until <= 0 or not counters_ok:
        raise ValueError(f"{bundle_dir}: inconsistent rebalance row counters")
    expected_is_rebalance = returns_as_of == meta["last_rebalance_date"]
    if bool(meta["is_rebalance_data_as_of"]) != expected_is_rebalance:
        raise ValueError(f"{bundle_dir}: is_rebalance_data_as_of mismatch")
    _validate_operations_payload(bundle_dir, operations, meta, returns_as_of)

    # transaction_costs is validated when present (every fresh export carries it).
    monitoring = _read_json(bundle_dir / "monitoring.json")
    tc = monitoring.get("transaction_costs")
    if isinstance(tc, dict):
        cum_cost = _finite_float(tc.get("cumulative_transaction_cost"), "transaction_costs cumulative cost")
        cum_turnover = _finite_float(tc.get("cumulative_two_way_turnover"), "transaction_costs turnover")
        tc_rate = _finite_float(tc.get("one_way_transaction_cost_rate"), "transaction_costs rate")
        if abs(cum_cost - cum_turnover * tc_rate) > 1e-10:
            raise ValueError(f"{bundle_dir}: transaction_costs cumulative cost identity mismatch")
        tc_series = tc.get("series")
        if tc_series:
            last_cum = _finite_float(
                tc_series[-1].get("cumulative_cost"), "transaction_costs series cumulative"
            )
            if abs(last_cum - cum_cost) > 1e-10:
                raise ValueError(f"{bundle_dir}: transaction_costs series tail mismatch")

    if int(meta.get("universe_size", 0)) != len(meta.get("universe") or []):
        raise ValueError(f"{bundle_dir}: universe_size does not match universe list")
    expected_universe_hash = hashlib.sha256(
        "\n".join(meta.get("universe") or []).encode("utf-8")
    ).hexdigest()
    if meta.get("universe_hash") != expected_universe_hash:
        raise ValueError(f"{bundle_dir}: universe hash mismatch")
    universe_funnel = meta.get("universe_funnel")
    if isinstance(universe_funnel, dict):
        loaded_count = universe_funnel.get("loaded_ticker_count")
        if loaded_count is not None and int(loaded_count) != int(meta.get("universe_size", 0)):
            raise ValueError(f"{bundle_dir}: universe funnel loaded count mismatch")
        if operations.get("universe_funnel") != universe_funnel:
            raise ValueError(f"{bundle_dir}: operations/portfolio universe funnel mismatch")
    if meta.get("data_freshness") is not None and operations.get("data_freshness") != meta.get("data_freshness"):
        raise ValueError(f"{bundle_dir}: operations/portfolio data freshness mismatch")

    run_dir = _rooted(Path(meta.get("run_dir", ""))).resolve()
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise ValueError(f"{bundle_dir}: source metrics missing: {metrics_path}")
    if meta.get("source_metrics_sha256") != _sha256(metrics_path):
        raise ValueError(f"{bundle_dir}: source metrics hash mismatch")
    exported_at = datetime.fromisoformat(str(meta["exported_at_utc"]).replace("Z", "+00:00"))
    metrics_mtime = datetime.fromtimestamp(metrics_path.stat().st_mtime, timezone.utc)
    # 1s grace: st_mtime has 100ns precision vs exported_at's 1us truncation,
    # so a same-microsecond write can make mtime appear later than now().
    if exported_at < metrics_mtime - timedelta(seconds=1):
        raise ValueError(f"{bundle_dir}: bundle predates its source metrics")

    try:
        config_drift = run_config_drift(meta, run_dir)
    except Exception:
        config_drift = None
    if config_drift and config_drift["mismatched_keys"]:
        print(
            f"[bundle-validator] WARN: {bundle_dir}: run config drift vs "
            f"{meta.get('variant_path')}: {config_drift['mismatched_keys']}",
            file=sys.stderr,
        )

    try:
        expected_operating = str(bundle_dir.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        expected_operating = str(bundle_dir).replace("\\", "/")
    if meta.get("operating_dir") != expected_operating:
        raise ValueError(
            f"{bundle_dir}: operating_dir metadata is {meta.get('operating_dir')!r}, "
            f"expected {expected_operating!r}"
        )
    if meta.get("causal_validation_enabled") and meta.get("causal_validation_ok") is not True:
        raise ValueError(f"{bundle_dir}: causal validation audit failed")

    risk_guardrails = risk.get("guardrails") or {}
    if not isinstance(risk_guardrails, dict):
        risk_guardrails = {}
    try:
        model_quality = _read_json(metrics_path).get("model_quality") or {}
    except Exception:
        model_quality = {}
    if not isinstance(model_quality, dict):
        model_quality = {}

    return {
        **meta,
        "performance": perf,
        "_risk_guardrails": risk_guardrails,
        "_model_quality": model_quality,
        "_run_config_drift": config_drift,
    }


def evaluate_challenger(production: dict, challenger: dict) -> dict:
    """Apply the pre-registered, non-blocking investment performance gate."""
    base = production["performance"]
    arm = challenger["performance"]
    base_sub = base.get("sub_period_ir") or {}
    arm_sub = arm.get("sub_period_ir") or {}
    sub_wins = sum(
        float(arm_sub.get(k, float("-inf"))) >= float(base_sub.get(k, float("inf")))
        for k in ("P1_ir", "P2_ir", "P3_ir")
    )
    checks = {
        "information_ratio_improved": float(arm.get("information_ratio")) > float(base.get("information_ratio")),
        "active_return_improved": float(arm.get("active_return")) > float(base.get("active_return")),
        "tracking_error_within_4_5pct": float(arm.get("tracking_error")) <= 0.045,
        "beta_between_0_95_and_1_05": 0.95 <= float(arm.get("realized_beta")) <= 1.05,
        "turnover_within_1_25x": float(arm.get("avg_annual_turnover")) <= 1.25 * float(base.get("avg_annual_turnover")),
        "max_drawdown_not_worse_by_3pp": float(arm.get("max_drawdown")) >= float(base.get("max_drawdown")) - 0.03,
        "subperiod_ir_wins_at_least_2": sub_wins >= 2,
    }
    passed = all(checks.values())
    return {
        "status": "PASS" if passed else "RESEARCH/FAIL",
        "passed": passed,
        "checks": checks,
        "subperiod_ir_wins": sub_wins,
    }


# S12.1 (2026-07-22): pre-registered consecutive-stale-depth limit. Runs of
# <= 7 degenerate retrains are the empirically cleared range (S11.10 reuse
# diagnosis: no damage detected); deeper runs are untested territory.
MAX_CONSECUTIVE_STALE_RETRAINS = 7

# S16.3 (2026-08-31): the time axis MAX_CONSECUTIVE_STALE_RETRAINS lacks. That
# limit counts retrain SLOTS, so at retrain_freq=63BD it tolerates a live model
# roughly 7*63 = 441BD ~ 1.75 years old. Measured on the certified S0' run: the
# live 2026-08-17 model was fit on 2025-08-28 (354 days, 4 slots) and the depth
# gate still passed at 4. 3 slots ~ 189BD ~ 9 months is the pre-registered
# ceiling. REPORT-ONLY: this check is published in ``checks``/``values`` but is
# deliberately excluded from the fail-closed status computation, so it cannot
# flip an existing bundle's PRODUCTION/HOLD verdict (§S16 O1 is a diagnostic
# until the s16_3_fresh_fixed arm is judged).
MAX_LIVE_MODEL_AGE_RETRAINS = 3
_REPORT_ONLY_CHECKS = ("live_model_age_ok",)


def max_stale_depth(model_quality: dict):
    """Longest run of consecutive degenerate retrains, derived from the
    retrain order (split_audit) and the degenerate event dates (events).
    Returns None when either input is missing (fail-closed upstream)."""
    audit = model_quality.get("split_audit")
    events = model_quality.get("events")
    if not isinstance(audit, list) or not audit or not isinstance(events, list):
        return None
    try:
        degenerate_dates = {event["date"] for event in events}
        order = [entry["prediction_date"] for entry in audit]
    except (KeyError, TypeError):
        return None
    longest = current = 0
    for date in order:
        current = current + 1 if date in degenerate_dates else 0
        longest = max(longest, current)
    return longest


def evaluate_production(record: dict) -> dict:
    """Report-only production HOLD gate. Never raises; never blocks validation."""
    guardrails = record.get("_risk_guardrails") or {}
    model_quality = record.get("_model_quality") or {}
    perf = record.get("performance") or {}
    data_quality = perf.get("data_quality") or {}

    def _num(value):
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        if result != result or result in (float("inf"), float("-inf")):
            return None
        return result

    def _not_breached(key):
        breached = guardrails.get(key)
        return None if breached is None else not bool(breached)

    degenerate_rate = _num(model_quality.get("degenerate_rate"))
    stale_depth = max_stale_depth(model_quality)
    live_model_age = _num(model_quality.get("live_model_age_retrains"))
    tracking_error = _num(perf.get("tracking_error"))
    tail_days = _num(data_quality.get("tail_ffill_days"))
    max_tail_days = _num(data_quality.get("max_tail_ffill_days"))
    # §S18.1 (decision log §S18 P1): FactSet target-price basis guard. The
    # loader publishes names whose 252d median TG/price sits outside the
    # suspect band, and run_variant the names whose median jumped vs the
    # previous run (APH 1.16 -> 2.33 on the 2026-09-03 workbook went silently
    # into the certified baseline). Fail-closed: missing suspect list -> None.
    currency = data_quality.get("currency") if isinstance(data_quality.get("currency"), dict) else {}
    tg_suspect = currency.get("tg_px_ratio_suspect")
    tg_jump = currency.get("tg_px_ratio_jump_vs_prev") or {}

    checks = {
        "estimated_te_ok": _not_breached("estimated_te_breached"),
        "name_active_risk_ok": _not_breached("top_name_active_risk_breached"),
        "sector_active_risk_ok": _not_breached("top_sector_active_risk_breached"),
        # S12.1: overall degenerate rate is observation-only; the gate binds
        # on consecutive stale depth (S11.10 reuse diagnosis).
        "stale_depth_ok": (
            None if stale_depth is None
            else stale_depth <= MAX_CONSECUTIVE_STALE_RETRAINS
        ),
        # S16.3: report-only (see _REPORT_ONLY_CHECKS) — model AGE, not run
        # length, so a reused model that never hits a long consecutive run is
        # still caught.
        "live_model_age_ok": (
            None if live_model_age is None
            else live_model_age <= MAX_LIVE_MODEL_AGE_RETRAINS
        ),
        "realized_te_within_guard": None if tracking_error is None else tracking_error <= 0.045,
        "stale_tail_ok": (
            None if tail_days is None or max_tail_days is None else tail_days <= max_tail_days
        ),
        "tg_px_ratio_ok": (
            None if not isinstance(tg_suspect, dict)
            else (len(tg_suspect) == 0 and len(tg_jump) == 0)
        ),
    }
    # Fail-closed (2026-07-21): PRODUCTION requires every check explicitly
    # True — a missing input (None) is not evidence of passing. S16.3 report-
    # only checks are published but never bind (existing thresholds unchanged).
    status = "PRODUCTION" if all(
        v is True for k, v in checks.items() if k not in _REPORT_ONLY_CHECKS
    ) else "HOLD"
    values = {
        "top_sector": guardrails.get("top_sector"),
        "top_sector_active_risk_share": guardrails.get("top_sector_active_risk_share"),
        "top_name": guardrails.get("top_name"),
        "top_name_active_risk_share": guardrails.get("top_name_active_risk_share"),
        "degenerate_rate": degenerate_rate,
        "max_stale_depth": stale_depth,
        "max_consecutive_stale_retrains": MAX_CONSECUTIVE_STALE_RETRAINS,
        "live_model_age_retrains": live_model_age,
        "max_live_model_age_retrains": MAX_LIVE_MODEL_AGE_RETRAINS,
        "tracking_error": tracking_error,
        "tail_ffill_days": tail_days,
        "max_tail_ffill_days": max_tail_days,
        "tg_px_ratio_suspect": tg_suspect if isinstance(tg_suspect, dict) else None,
        "tg_px_ratio_jump_vs_prev": tg_jump,
    }
    return {"status": status, "checks": checks, "values": values}


def build_registry(bundle_dirs: list[Path]) -> dict:
    records = [validate_bundle(path) for path in bundle_dirs]
    ids = [row["id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate portfolio ids: {ids}")

    common_fields = (
        "data_as_of",
        "base_currency",
        "universe_hash",
        "benchmark_type",
        "last_rebalance_date",
        "previous_rebalance_date",
        "next_expected_rebalance_date",
        "rebalance_freq_days",
        "rebalance_calendar",
        "rows_since_last_rebalance",
        "rows_until_next_rebalance",
        "is_rebalance_data_as_of",
    )
    for field in common_fields:
        values = {row.get(field) for row in records}
        if len(values) != 1:
            raise ValueError(f"portfolio {field} mismatch: {sorted(values)}")

    production = next((r for r in records if r.get("portfolio_role") == "production"), None)
    if production is None:
        raise ValueError("exactly one production portfolio is required")
    challengers = [r for r in records if r.get("portfolio_role") == "challenger"]
    if len(challengers) != 1:
        raise ValueError("exactly one challenger portfolio is required")
    gate = evaluate_challenger(production, challengers[0])
    production_gate = evaluate_production(production)

    ordered = sorted(records, key=lambda r: (r.get("portfolio_role") != "production", r["id"]))
    entries = []
    for row in ordered:
        item = {k: v for k, v in row.items() if k != "performance" and not k.startswith("_")}
        item["status"] = production_gate["status"] if row is production else gate["status"]
        entries.append(item)
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stale_after_hours": 96,
        "data_as_of": production["data_as_of"],
        "comparison_gate": gate,
        "production_gate": production_gate,
        "portfolios": entries,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", action="append", type=Path, required=True)
    parser.add_argument(
        "--registry", type=Path, default=ROOT / "outputs" / "portfolio_registry.json"
    )
    args = parser.parse_args(argv)
    try:
        registry = build_registry(args.bundle)
        target = _rooted(args.registry).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, target)
    except Exception as exc:
        print(f"[bundle-validator] ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"[bundle-validator] OK: {len(registry['portfolios'])} portfolios, "
        f"as_of={registry['data_as_of']}, gate={registry['comparison_gate']['status']}"
    )
    print(f"[bundle-validator] wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
