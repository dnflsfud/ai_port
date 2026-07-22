import json

import pandas as pd

from streamlit_app import (
    build_comparison_returns,
    collect_operating_alerts,
    load_operating_bundle,
    load_portfolio_registry,
    summarize_currency_period,
)


def test_registry_falls_back_to_legacy_only(tmp_path):
    registry = load_portfolio_registry(tmp_path)
    assert registry["schema_version"] == 0
    assert len(registry["portfolios"]) == 1
    assert registry["portfolios"][0]["id"] == "iter15_65tkr_reb21_vtg"


def test_operating_bundle_loads_registry_path_without_streamlit(tmp_path):
    operating = tmp_path / "outputs" / "operating_test"
    operating.mkdir(parents=True)
    (operating / "portfolio.json").write_text(
        json.dumps({"id": "test", "display_name": "Test"}), encoding="utf-8"
    )
    (operating / "performance.json").write_text(json.dumps({"information_ratio": 1.2}), encoding="utf-8")
    (operating / "currency.json").write_text(
        json.dumps({"base_currency": "USD", "coverage": {"total": 100, "mapped": 100}}),
        encoding="utf-8",
    )
    pd.DataFrame({"date": ["2026-01-02"], "portfolio_cum": [1.1]}).to_csv(
        operating / "returns.csv", index=False
    )
    bundle = load_operating_bundle(
        {"id": "test", "operating_dir": "outputs/operating_test"}, tmp_path
    )
    assert bundle["meta"]["display_name"] == "Test"
    assert bundle["perf"]["information_ratio"] == 1.2
    assert bundle["currency"]["base_currency"] == "USD"
    assert bundle["returns"].iloc[0]["portfolio_cum"] == 1.1


def test_comparison_returns_aligns_both_portfolios_and_one_benchmark():
    dates = pd.bdate_range("2026-01-01", periods=3)
    production = pd.DataFrame(
        {
            "portfolio_cum": [1.0, 1.1, 1.2],
            "benchmark_cum": [1.0, 1.05, 1.1],
            "sp500_cum": [1.0, 1.03, 1.08],
        },
        index=dates,
    )
    challenger = pd.DataFrame(
        {"portfolio_cum": [1.0, 1.12, 1.25], "benchmark_cum": [1.0, 1.05, 1.1]},
        index=dates,
    )
    out = build_comparison_returns(production, challenger, "Legacy S0", "Causal Rank 65")
    assert list(out.columns) == [
        "Legacy S0", "Benchmark", "S&P 500", "Causal Rank 65"
    ]
    assert out.iloc[-1]["Causal Rank 65"] == 1.25
    assert out.iloc[-1]["S&P 500"] == 1.08


def test_summarize_currency_period_filters_and_sums_exact_effects():
    currency = {
        "daily": [
            {
                "date": "2026-01-02",
                "portfolio_local_return": 0.010,
                "portfolio_fx_effect": 0.002,
                "benchmark_local_return": 0.008,
                "benchmark_fx_effect": 0.001,
                "active_fx_effect": 0.001,
            },
            {
                "date": "2026-01-05",
                "portfolio_local_return": -0.004,
                "portfolio_fx_effect": -0.001,
                "benchmark_local_return": -0.003,
                "benchmark_fx_effect": -0.0005,
                "active_fx_effect": -0.0005,
            },
        ]
    }
    out = summarize_currency_period(currency, "2026-01-05", "2026-01-05")
    assert out["observations"] == 1
    assert out["portfolio_local_return"] == -0.004
    assert out["portfolio_fx_effect"] == -0.001
    assert out["active_fx_effect"] == -0.0005


def test_collect_operating_alerts_surfaces_existing_breaches_and_fx_gaps():
    monitoring = {
        "guardrails": {
            "estimated_te_breached": True,
            "latest_estimated_te": 0.036,
            "te_constraint_breached": True,
            "max_rebalance_estimated_te": 0.0351,
            "top_name_active_risk_breached": True,
            "top_name_active_risk_share": 0.41,
            "top_sector_active_risk_breached": False,
            "model_degenerate_rate_breached": True,
            "model_degenerate_rate": 0.50,
        }
    }
    currency = {"coverage": {"missing": ["ABC"], "missing_fx": ["GBP"]}}
    alerts = collect_operating_alerts(monitoring, currency)
    keys = {row["key"] for row in alerts}
    assert "top_name_active_risk_breached" in keys
    assert "estimated_te_breached" in keys
    assert "te_constraint_breached" in keys
    assert "model_degenerate_rate_breached" in keys
    assert "currency_mapping_missing" in keys
    assert "fx_series_missing" in keys
