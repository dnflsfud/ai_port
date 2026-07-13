import json

import pandas as pd

from streamlit_app import (
    build_comparison_returns,
    load_operating_bundle,
    load_portfolio_registry,
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
    pd.DataFrame({"date": ["2026-01-02"], "portfolio_cum": [1.1]}).to_csv(
        operating / "returns.csv", index=False
    )
    bundle = load_operating_bundle(
        {"id": "test", "operating_dir": "outputs/operating_test"}, tmp_path
    )
    assert bundle["meta"]["display_name"] == "Test"
    assert bundle["perf"]["information_ratio"] == 1.2
    assert bundle["returns"].iloc[0]["portfolio_cum"] == 1.1


def test_comparison_returns_aligns_both_portfolios_and_one_benchmark():
    dates = pd.bdate_range("2026-01-01", periods=3)
    production = pd.DataFrame(
        {"portfolio_cum": [1.0, 1.1, 1.2], "benchmark_cum": [1.0, 1.05, 1.1]},
        index=dates,
    )
    challenger = pd.DataFrame(
        {"portfolio_cum": [1.0, 1.12, 1.25], "benchmark_cum": [1.0, 1.05, 1.1]},
        index=dates,
    )
    out = build_comparison_returns(production, challenger, "Legacy S0", "Causal Rank 65")
    assert list(out.columns) == ["Legacy S0", "Benchmark", "Causal Rank 65"]
    assert out.iloc[-1]["Causal Rank 65"] == 1.25
