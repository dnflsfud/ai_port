import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from scripts.validate_portfolio_bundles import build_registry, validate_bundle


def _write_bundle(root: Path, name: str, role: str, *, universe=("AAA", "BBB"), as_of="2026-06-11"):
    run_dir = root / f"run_{name}"
    bundle = root / f"bundle_{name}"
    run_dir.mkdir()
    bundle.mkdir()
    metrics = run_dir / "metrics.json"
    metrics.write_text(json.dumps({"label": name, "metrics": {"information_ratio": 1.0}}), encoding="utf-8")
    metrics_hash = hashlib.sha256(metrics.read_bytes()).hexdigest()
    universe_hash = hashlib.sha256("\n".join(universe).encode()).hexdigest()
    as_of_ts = pd.Timestamp(as_of)
    next_rebalance = (as_of_ts + pd.offsets.BDay(21)).strftime("%Y-%m-%d")
    perf = {
        "as_of": as_of, "annual_return": 0.2, "active_return": 0.03,
        "information_ratio": 1.0, "tracking_error": 0.03, "realized_beta": 1.0,
        "avg_annual_turnover": 1.0, "max_drawdown": -0.2, "avg_ic": 0.04,
        "sub_period_ir": {"P1_ir": 1.0, "P2_ir": 1.0, "P3_ir": 1.0},
        "data_quality": {"tail_ffill_days": 1, "max_tail_ffill_days": 10},
    }
    meta = {
        "schema_version": 1, "id": name, "display_name": name,
        "portfolio_role": role, "model_type": "regression",
        "run_dir": str(run_dir).replace("\\", "/"),
        "operating_dir": str(bundle).replace("\\", "/"),
        "benchmark_type": "cap_weighted", "base_currency": "USD",
        "universe_size": len(universe),
        "universe": list(universe), "universe_hash": universe_hash,
        "universe_funnel": {
            "full_universe_count": len(universe),
            "loaded_ticker_count": len(universe),
            "missing_tickers": [],
        },
        "data_freshness": {
            "data_as_of": as_of,
            "fx_data_as_of": as_of,
            "fx_missing_currencies": [],
            "fx_stale_currencies": [],
            "tail_ffill_days": 1,
            "max_tail_ffill_days": 10,
        },
        "data_as_of": as_of, "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "last_rebalance_date": as_of,
        "previous_rebalance_date": as_of,
        "next_expected_rebalance_date": next_rebalance,
        "rebalance_freq_days": 21,
        "rebalance_calendar": "weekday_index",
        "rows_since_last_rebalance": 0,
        "rows_until_next_rebalance": 21,
        "is_rebalance_data_as_of": True,
        "next_rebalance_is_estimate": True,
        "source_metrics_sha256": metrics_hash, "causal_validation_enabled": role == "challenger",
        "causal_validation_ok": True if role == "challenger" else None,
        "execution_signal_lag_days": 1 if role == "challenger" else 0,
    }
    (bundle / "portfolio.json").write_text(json.dumps(meta), encoding="utf-8")
    (bundle / "performance.json").write_text(json.dumps(perf), encoding="utf-8")
    (bundle / "holdings.json").write_text(json.dumps({"as_of": as_of}), encoding="utf-8")
    operations = {
        "as_of": as_of,
        "data_as_of": as_of,
        "base_currency": "USD",
        "last_rebalance_date": as_of,
        "previous_rebalance_date": as_of,
        "next_expected_rebalance_date": next_rebalance,
        "rebalance_freq_days": 21,
        "rebalance_calendar": "weekday_index",
        "rows_since_last_rebalance": 0,
        "rows_until_next_rebalance": 21,
        "is_rebalance_data_as_of": True,
        "next_rebalance_is_estimate": True,
        "trade_data_as_of": as_of,
        "returns_data_as_of": as_of,
        "trade_data_as_of_valid": True,
        "turnover_two_way_latest": 0.0,
        "turnover_two_way_reconstructed": 0.0,
        "turnover_reconciliation_error": 0.0,
        "turnover_reconciled": True,
        "one_way_transaction_cost_rate": 0.001,
        "one_way_transaction_cost_bps": 10.0,
        "expected_transaction_cost": 0.0,
        "n_trades": 0,
        "trade_list": [],
        "universe_funnel": meta["universe_funnel"],
        "data_freshness": meta["data_freshness"],
    }
    (bundle / "operations.json").write_text(json.dumps(operations), encoding="utf-8")
    ticker_weight = 1.0 / len(universe)
    zero_contributions = {
        "portfolio_local_contribution": 0.0,
        "portfolio_fx_contribution": 0.0,
        "portfolio_usd_contribution": 0.0,
        "benchmark_local_contribution": 0.0,
        "benchmark_fx_contribution": 0.0,
        "benchmark_usd_contribution": 0.0,
        "active_local_contribution": 0.0,
        "active_fx_contribution": 0.0,
        "active_usd_contribution": 0.0,
    }
    currency = {
        "schema_version": 1,
        "base_currency": "USD",
        "as_of": as_of,
        "fx_data_as_of": as_of,
        "method": "Arithmetic entering-day attribution; exact FX is USD minus local.",
        "coverage": {
            "total": len(universe), "mapped": len(universe), "missing": 0,
            "missing_fx": 0, "stale": 0,
            "missing_tickers": [], "missing_fx_tickers": [], "stale_tickers": [],
        },
        "summary": {
            "non_usd_target_weight": 0.0,
            "non_usd_benchmark_weight": 0.0,
            "portfolio_fx_arithmetic_contribution": 0.0,
            "benchmark_fx_arithmetic_contribution": 0.0,
            "active_fx_arithmetic_contribution": 0.0,
        },
        "daily": [{
            "date": as_of,
            "portfolio_local_return": 0.0,
            "portfolio_fx_effect": 0.0,
            "portfolio_usd_gross_return": 0.0,
            "benchmark_local_return": 0.0,
            "benchmark_fx_effect": 0.0,
            "benchmark_usd_gross_return": 0.0,
            "active_local_return": 0.0,
            "active_fx_effect": 0.0,
            "active_usd_gross_return": 0.0,
        }],
        "by_currency": [{
            "currency": "USD", "ticker_count": len(universe),
            "target_weight": 1.0, "benchmark_weight": 1.0, "active_weight": 0.0,
            **zero_contributions,
        }],
        "by_ticker": [{
            "ticker": ticker, "currency": "USD",
            "target_weight": ticker_weight,
            "benchmark_weight": ticker_weight,
            "active_weight": 0.0,
            **zero_contributions,
        } for ticker in universe],
        "reconciliation": {
            "tolerance": 1e-10, "passed": True,
            "max_daily_portfolio_error": 0.0,
            "max_daily_benchmark_error": 0.0,
            "max_daily_active_error": 0.0,
            "period_portfolio_error": 0.0,
            "period_benchmark_error": 0.0,
            "period_active_error": 0.0,
        },
    }
    (bundle / "currency.json").write_text(json.dumps(currency), encoding="utf-8")
    for fname in ("features.json", "contribution.json", "risk.json", "monitoring.json", "feature_attribution.json"):
        (bundle / fname).write_text("{}", encoding="utf-8")
    returns = {"date": [as_of], "portfolio_ret": [0.0]}
    for column in (
        "portfolio_local_gross_ret", "portfolio_fx_effect", "portfolio_usd_gross_ret",
        "benchmark_local_ret", "benchmark_fx_effect", "benchmark_usd_gross_ret",
        "active_local_gross_ret", "active_fx_effect", "active_usd_gross_ret",
    ):
        returns[column] = [0.0]
    pd.DataFrame(returns).to_csv(bundle / "returns.csv", index=False)
    return bundle


def test_registry_requires_matching_source_and_common_contract(tmp_path):
    production = _write_bundle(tmp_path, "prod", "production")
    challenger = _write_bundle(tmp_path, "causal", "challenger")
    registry = build_registry([production, challenger])
    assert [p["portfolio_role"] for p in registry["portfolios"]] == ["production", "challenger"]
    assert registry["data_as_of"] == "2026-06-11"
    assert registry["comparison_gate"]["status"] in {"PASS", "RESEARCH/FAIL"}


def test_registry_rejects_universe_mismatch(tmp_path):
    production = _write_bundle(tmp_path, "prod", "production")
    challenger = _write_bundle(tmp_path, "causal", "challenger", universe=("AAA", "CCC"))
    with pytest.raises(ValueError, match="universe_hash mismatch"):
        build_registry([production, challenger])


def test_bundle_rejects_missing_or_tampered_files(tmp_path):
    bundle = _write_bundle(tmp_path, "prod", "production")
    (bundle / "risk.json").unlink()
    with pytest.raises(ValueError, match="missing required files"):
        validate_bundle(bundle)


def test_bundle_requires_currency_json(tmp_path):
    bundle = _write_bundle(tmp_path, "prod", "production")
    (bundle / "currency.json").unlink()
    with pytest.raises(ValueError, match="currency.json"):
        validate_bundle(bundle)


def test_bundle_rejects_non_usd_base_currency(tmp_path):
    bundle = _write_bundle(tmp_path, "prod", "production")
    currency_path = bundle / "currency.json"
    currency = json.loads(currency_path.read_text(encoding="utf-8"))
    currency["base_currency"] = "KRW"
    currency_path.write_text(json.dumps(currency), encoding="utf-8")
    with pytest.raises(ValueError, match="base currency must be USD"):
        validate_bundle(bundle)


def test_bundle_rejects_incomplete_currency_mapping(tmp_path):
    bundle = _write_bundle(tmp_path, "prod", "production")
    currency_path = bundle / "currency.json"
    currency = json.loads(currency_path.read_text(encoding="utf-8"))
    currency["coverage"].update({"mapped": 1, "missing": 1})
    currency_path.write_text(json.dumps(currency), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete currency mapping coverage"):
        validate_bundle(bundle)


def test_bundle_rejects_currency_reconciliation_error(tmp_path):
    bundle = _write_bundle(tmp_path, "prod", "production")
    currency_path = bundle / "currency.json"
    currency = json.loads(currency_path.read_text(encoding="utf-8"))
    currency["daily"][0]["portfolio_fx_effect"] = 0.01
    currency_path.write_text(json.dumps(currency), encoding="utf-8")
    with pytest.raises(ValueError, match=r"local \+ FX != USD"):
        validate_bundle(bundle)


def test_bundle_rejects_trade_list_turnover_mismatch(tmp_path):
    bundle = _write_bundle(tmp_path, "prod", "production")
    operations_path = bundle / "operations.json"
    operations = json.loads(operations_path.read_text(encoding="utf-8"))
    operations.update({
        "n_trades": 1,
        "trade_list": [{
            "ticker": "AAA", "pre_trade": 0.4, "prev": 0.4,
            "target": 0.5, "delta": 0.1,
        }],
    })
    operations_path.write_text(json.dumps(operations), encoding="utf-8")
    with pytest.raises(ValueError, match="trade-list L1"):
        validate_bundle(bundle)


def test_bundle_rejects_weekend_return_rows(tmp_path):
    bundle = _write_bundle(tmp_path, "prod", "production")
    pd.DataFrame({"date": ["2026-06-11", "2026-06-13"], "portfolio_ret": [0.0, 0.0]}).to_csv(
        bundle / "returns.csv", index=False
    )
    with pytest.raises(ValueError, match="weekend dates"):
        validate_bundle(bundle)


def test_bundle_rejects_stale_tail_when_fail_on_true(tmp_path):
    bundle = _write_bundle(tmp_path, "prod", "production")
    perf_path = bundle / "performance.json"
    perf = json.loads(perf_path.read_text(encoding="utf-8"))
    perf["data_quality"] = {
        "tail_ffill_days": 11, "max_tail_ffill_days": 10, "fail_on_stale_tail_ffill": True,
    }
    perf_path.write_text(json.dumps(perf), encoding="utf-8")
    with pytest.raises(ValueError, match="stale tail exceeds limit"):
        validate_bundle(bundle)


def test_bundle_warns_stale_tail_when_fail_on_false(tmp_path):
    bundle = _write_bundle(tmp_path, "prod", "production")
    perf_path = bundle / "performance.json"
    perf = json.loads(perf_path.read_text(encoding="utf-8"))
    perf["data_quality"] = {
        "tail_ffill_days": 11, "max_tail_ffill_days": 10, "fail_on_stale_tail_ffill": False,
    }
    perf_path.write_text(json.dumps(perf), encoding="utf-8")
    # Policy is warn-only: bundle validates despite exceeding the limit.
    assert validate_bundle(bundle)["id"] == "prod"


def test_registry_rejects_rebalance_schedule_mismatch(tmp_path):
    production = _write_bundle(tmp_path, "prod", "production")
    challenger = _write_bundle(tmp_path, "causal", "challenger")
    meta_path = challenger / "portfolio.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["next_expected_rebalance_date"] = "2026-07-13"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    ops_path = challenger / "operations.json"
    ops = json.loads(ops_path.read_text(encoding="utf-8"))
    ops["next_expected_rebalance_date"] = "2026-07-13"
    ops_path.write_text(json.dumps(ops), encoding="utf-8")
    with pytest.raises(ValueError, match="next_expected_rebalance_date mismatch"):
        build_registry([production, challenger])
