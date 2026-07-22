"""Unit smoke — build_feature_attribution (satisfies the filename TDD guard).

This file exists so the guard that requires ``tests/test_<module>.py`` for edits
to ``scripts/export_operating_data.py`` is satisfied. It is a MINIMAL smoke of
``build_feature_attribution`` (schema + SHAP additivity) — the exhaustive
coverage lives in ``tests/acceptance/test_feature_attribution_tab.py``.

Written by test-designer BEFORE implementation; copied verbatim by the
implementer (sha256-verified). Target imported INSIDE test bodies so
``pytest --collect-only`` stays clean; RED now (build_feature_attribution
does not exist yet -> ImportError).
"""

import json
import types

import numpy as np
import pandas as pd
import pytest


_TICKERS = ["AAA", "BBB", "CCC", "DDD"]
_ACTIVE = [f"f{i}" for i in range(6)]           # model subset
_PANEL_COLS = _ACTIVE + ["f6"]                  # panel superset (f6 extra)
_AS_OF = pd.Timestamp("2026-05-21")


def _result():
    import lightgbm as lgb

    rng = np.random.default_rng(0)
    n = 300
    Xtr = pd.DataFrame(rng.normal(size=(n, len(_ACTIVE))), columns=_ACTIVE)
    ytr = 0.4 * Xtr["f0"] - 0.25 * Xtr["f2"] + rng.normal(0, 0.1, n)
    model = lgb.LGBMRegressor(n_estimators=30, num_leaves=8,
                              min_child_samples=5, verbose=-1)
    model.fit(Xtr, ytr)
    model._active_features = list(_ACTIVE)

    idx = pd.MultiIndex.from_product([[_AS_OF], _TICKERS], names=["date", "ticker"])
    panel = pd.DataFrame(rng.normal(size=(len(idx), len(_PANEL_COLS))),
                         index=idx, columns=_PANEL_COLS)
    weights = pd.Series(1.0 / len(_TICKERS), index=_TICKERS)

    return types.SimpleNamespace(
        models={_AS_OF: model},
        panel=panel,
        portfolio_weights={_AS_OF: weights},
        feature_groups={"G1": ["f0", "f1", "f2"], "G2": ["f3", "f4", "f5"]},
        sector_map={t: "S" for t in _TICKERS},
        bm_weights={t: 1.0 / len(_TICKERS) for t in _TICKERS},
    )


def test_smoke_schema():
    from scripts.export_operating_data import build_feature_attribution

    out = build_feature_attribution(_result())
    assert {"as_of", "model_date", "feature_groups", "tickers"} <= set(out)
    assert set(out["tickers"]) == set(_TICKERS)
    for rec in out["tickers"].values():
        assert set(rec["shap"]) == set(_ACTIVE)      # sliced to model features


def test_smoke_additivity():
    from scripts.export_operating_data import build_feature_attribution

    out = build_feature_attribution(_result())
    for rec in out["tickers"].values():
        recon = float(rec["base_value"]) + float(sum(rec["shap"].values()))
        assert abs(recon - float(rec["mu"])) <= 1e-3 * abs(float(rec["mu"])) + 1e-9
    assert out.get("additivity_ok") is True


def test_rebalance_metadata_for_non_rebalance_and_rebalance_asof():
    from scripts.export_operating_data import build_rebalance_metadata

    dates = pd.bdate_range("2026-01-02", periods=25)
    non_rebalance = build_rebalance_metadata(
        [dates[0], dates[21]],
        dates,
        21,
    )
    assert non_rebalance["last_rebalance_date"] == dates[21].strftime("%Y-%m-%d")
    assert non_rebalance["rows_since_last_rebalance"] == 3
    assert non_rebalance["rows_until_next_rebalance"] == 18
    assert non_rebalance["is_rebalance_data_as_of"] is False
    assert pd.Timestamp(non_rebalance["next_expected_rebalance_date"]).dayofweek < 5

    rebalance_asof = build_rebalance_metadata(
        [dates[0], dates[21]],
        dates[:22],
        21,
    )
    assert rebalance_asof["is_rebalance_data_as_of"] is True
    assert rebalance_asof["rows_since_last_rebalance"] == 0
    assert rebalance_asof["rows_until_next_rebalance"] == 21


def test_rebalance_metadata_rejects_weekend_calendar():
    from scripts.export_operating_data import build_rebalance_metadata

    dates = pd.to_datetime(["2026-01-02", "2026-01-03"])
    with np.testing.assert_raises_regex(ValueError, "weekend"):
        build_rebalance_metadata([dates[0]], dates, 21)


def _fx_fixture():
    dates = pd.bdate_range("2026-06-10", periods=2)
    tickers = ["AAA", "BBB"]
    local = pd.DataFrame(
        {"AAA": [0.01, 0.02], "BBB": [0.02, -0.01]}, index=dates
    )
    raw_fx = pd.DataFrame(
        {"AAA": [0.0, 0.0], "BBB": [0.01, -0.005]}, index=dates
    )
    usd = (1.0 + local) * (1.0 + raw_fx) - 1.0
    rates = pd.DataFrame(
        {"AAA": [1.0, 1.0], "BBB": [0.00075, 0.00074625]}, index=dates
    )
    portfolio_entering = pd.DataFrame(
        {"AAA": [0.60, 0.55], "BBB": [0.40, 0.45]}, index=dates
    )
    benchmark_entering = pd.DataFrame(
        {"AAA": [0.50, 0.50], "BBB": [0.50, 0.50]}, index=dates
    )
    return {
        "tickers": tickers,
        "dates": dates,
        "usd_returns": usd,
        "local_returns": local,
        "fx_returns": raw_fx,
        "fx_rates_usd_per_local": rates,
        "currency_map": {"AAA": "USD", "BBB": "KRW"},
        "portfolio_entering_weights": portfolio_entering,
        "benchmark_entering_weights": benchmark_entering,
        "latest_portfolio_weights": pd.Series({"AAA": 0.55, "BBB": 0.45}),
        "latest_benchmark_weights": pd.Series({"AAA": 0.50, "BBB": 0.50}),
        "data_quality": {
            "fx_data_as_of": dates[-1].strftime("%Y-%m-%d"),
            "fx": {
                "latest_source_date_by_currency": {
                    "KRW": dates[-1].strftime("%Y-%m-%d")
                },
                "missing_currencies": [],
                "stale_currencies": [],
            },
        },
    }


def test_currency_attribution_exactly_reconciles_local_fx_and_usd():
    from scripts.export_operating_data import build_currency_attribution

    kwargs = _fx_fixture()
    out = build_currency_attribution(**kwargs)
    assert out["base_currency"] == "USD"
    assert out["coverage"] == {
        "total": 2,
        "mapped": 2,
        "missing": 0,
        "missing_fx": 0,
        "stale": 0,
        "missing_tickers": [],
        "missing_fx_tickers": [],
        "stale_tickers": [],
    }
    assert out["summary"]["non_usd_target_weight"] == pytest.approx(0.45)
    assert out["stress"]["plus_1pct"]["portfolio"] == pytest.approx(0.0045)
    assert out["reconciliation"]["passed"] is True
    for row in out["daily"]:
        assert row["portfolio_local_return"] + row["portfolio_fx_effect"] == pytest.approx(
            row["portfolio_usd_gross_return"], abs=2e-10
        )
        assert row["benchmark_local_return"] + row["benchmark_fx_effect"] == pytest.approx(
            row["benchmark_usd_gross_return"], abs=2e-10
        )
    expected_fx = (
        kwargs["portfolio_entering_weights"]
        * (kwargs["usd_returns"] - kwargs["local_returns"])
    ).sum().sum()
    assert out["summary"]["portfolio_fx_arithmetic_contribution"] == pytest.approx(expected_fx)
    bbb = next(row for row in out["by_ticker"] if row["ticker"] == "BBB")
    # Day 1 proves the local/FX interaction is in the exact effect: .0302-.02=.0102.
    assert bbb["portfolio_fx_contribution"] == pytest.approx(
        0.40 * 0.0102 + 0.45 * ((1 - 0.01) * (1 - 0.005) - 1 + 0.01)
    )


def test_currency_freshness_respects_configured_staleness_window():
    from scripts.export_operating_data import build_currency_attribution

    kwargs = _fx_fixture()
    prior_day = kwargs["dates"][-2].strftime("%Y-%m-%d")
    kwargs["data_quality"]["fx_data_as_of"] = prior_day
    kwargs["data_quality"]["fx"]["latest_source_date_by_currency"]["KRW"] = prior_day
    kwargs["data_quality"]["data_freshness"] = {"max_fx_staleness_days": 2}

    within_window = build_currency_attribution(**kwargs)
    krw = next(
        row for row in within_window["freshness_by_currency"]
        if row["currency"] == "KRW"
    )
    assert krw["staleness_days"] == 1
    assert krw["allowed_staleness_days"] == 2
    assert krw["stale"] is False
    assert within_window["coverage"]["stale"] == 0

    kwargs["data_quality"]["data_freshness"]["max_fx_staleness_days"] = 0
    outside_window = build_currency_attribution(**kwargs)
    assert outside_window["coverage"]["stale"] == 1


def test_latest_trade_plan_uses_drifted_pre_trade_weights_and_reconciles_cost():
    from scripts.export_operating_data import build_latest_trade_plan

    entering = pd.Series({"AAA": 0.5, "BBB": 0.5})
    latest_returns = pd.Series({"AAA": 0.10, "BBB": 0.0})
    pre_trade = pd.Series({"AAA": 0.55 / 1.05, "BBB": 0.50 / 1.05})
    target = pd.Series({"AAA": 0.60, "BBB": 0.40})
    reported_turnover = float((target - pre_trade).abs().sum())
    plan = build_latest_trade_plan(
        target_weights=target,
        entering_weights=entering,
        latest_returns=latest_returns,
        reported_turnover=reported_turnover,
        one_way_transaction_cost=0.001,
        as_of="2026-06-11",
        returns_data_as_of="2026-06-12",
    )
    trades = {row["ticker"]: row for row in plan["trade_list"]}
    assert trades["AAA"]["pre_trade"] == pytest.approx(pre_trade["AAA"])
    assert trades["AAA"]["prev"] == trades["AAA"]["pre_trade"]
    assert sum(abs(row["delta"]) for row in trades.values()) == pytest.approx(
        reported_turnover, abs=2e-10
    )
    assert plan["expected_transaction_cost"] == pytest.approx(reported_turnover * 0.001)
    assert plan["trade_data_as_of_valid"] is True


def test_latest_trade_plan_rejects_turnover_from_previous_targets():
    from scripts.export_operating_data import build_latest_trade_plan

    with pytest.raises(ValueError, match="does not match backtest turnover"):
        build_latest_trade_plan(
            target_weights=pd.Series({"AAA": 0.60, "BBB": 0.40}),
            entering_weights=pd.Series({"AAA": 0.50, "BBB": 0.50}),
            latest_returns=pd.Series({"AAA": 0.10, "BBB": 0.0}),
            reported_turnover=0.20,  # old-target comparison, not drifted pre-trade
            one_way_transaction_cost=0.001,
            as_of="2026-06-11",
            returns_data_as_of="2026-06-11",
        )


def test_production_label_keeps_legacy_id_but_displays_150():
    from scripts.export_operating_data import _LABEL_DEFAULTS

    assert _LABEL_DEFAULTS["codex_causal_rank_65"] == ("Causal Rank 150", "production")


def _cached_result(tickers, as_of="2026-06-11"):
    weights = pd.Series(1.0 / len(tickers), index=tickers)
    date = pd.Timestamp(as_of)
    return types.SimpleNamespace(
        portfolio_weights={date: weights.copy()},
        daily_weights={date: weights.copy()},
        portfolio_returns=pd.Series([0.0], index=[date]),
    )


def test_cached_result_compatibility_rejects_65_name_result_for_100_name_data():
    from scripts.export_operating_data import validate_cached_result_compatibility

    cached_tickers = [f"T{i:03d}" for i in range(65)]
    current_tickers = [f"T{i:03d}" for i in range(100)]
    data_returns = pd.DataFrame(
        0.0, index=[pd.Timestamp("2026-06-11")], columns=current_tickers
    )
    with pytest.raises(ValueError, match="--no-cache"):
        validate_cached_result_compatibility(
            _cached_result(cached_tickers),
            current_tickers,
            data_returns,
            variant_path="variants/codex_causal_rank_65.yaml",
        )


def test_cached_result_compatibility_rejects_stale_result_asof():
    from scripts.export_operating_data import validate_cached_result_compatibility

    tickers = ["AAA", "BBB"]
    data_returns = pd.DataFrame(
        0.0, index=[pd.Timestamp("2026-06-12")], columns=tickers
    )
    with pytest.raises(ValueError, match="as_of=2026-06-11"):
        validate_cached_result_compatibility(
            _cached_result(tickers, as_of="2026-06-11"),
            tickers,
            data_returns,
        )


def test_cached_result_compatibility_accepts_exact_universe_and_asof():
    from scripts.export_operating_data import validate_cached_result_compatibility

    tickers = ["AAA", "BBB"]
    data_returns = pd.DataFrame(
        0.0, index=[pd.Timestamp("2026-06-11")], columns=tickers
    )
    validate_cached_result_compatibility(
        _cached_result(tickers), tickers, data_returns
    )


def test_provenance_meta_copies_run_manifest_git_and_checksums(tmp_path):
    from scripts.export_operating_data import (
        PORTFOLIO_VERSION,
        _sha256,
        build_provenance_meta,
    )

    manifest_path = tmp_path / "experiment_manifest.json"
    manifest_path.write_text(
        json.dumps({"git_hash": "a" * 40, "git_dirty": False}),
        encoding="utf-8",
    )
    meta = build_provenance_meta(tmp_path)
    assert PORTFOLIO_VERSION == "universe150-usd-pit-sp500-v2"
    assert meta["portfolio_version"] == "universe150-usd-pit-sp500-v2"
    assert meta["git_hash"] == "a" * 40
    assert meta["git_dirty"] is False
    assert meta["source_manifest_sha256"] == _sha256(manifest_path)


def test_provenance_meta_tolerates_missing_manifest(tmp_path):
    from scripts.export_operating_data import build_provenance_meta

    meta = build_provenance_meta(tmp_path)  # no experiment_manifest.json present
    assert meta["portfolio_version"] == "universe150-usd-pit-sp500-v2"
    assert meta["git_hash"] is None
    assert meta["git_dirty"] is None
    assert meta["source_manifest_sha256"] is None
