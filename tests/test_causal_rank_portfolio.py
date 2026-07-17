import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.backtest import apply_execution_signal_lag
from src.config import PipelineConfig
from src.harness import build_override_config
from src.model_trainer import build_walk_forward_split, prepare_rank_data, train_model
from src.portfolio_optimizer import optimize_portfolio, project_portfolio_weights


def _production_config():
    variant = Path(__file__).resolve().parents[1] / "variants" / "codex_causal_rank_65.yaml"
    manifest = yaml.safe_load(variant.read_text(encoding="utf-8"))
    return build_override_config(dict(manifest["overrides"]))


def test_production_100_fund_uses_usd_cap_weighted_benchmark_and_35pct_te_limit():
    config = _production_config()
    assert config.base_currency == "USD"
    assert config.convert_returns_to_usd is True
    assert config.benchmark_type == "cap_weighted"
    assert config.max_te_annual == pytest.approx(0.035)


def test_production_te_limit_survives_optimizer_and_execution_projection():
    config = _production_config()
    config.portfolio_style = "unconstrained"
    config.max_active_share = 2.0
    config.max_active_per_stock = 1.0
    config.max_weight = 1.0
    config.max_single_turnover = 2.0
    config.turnover_penalty = 0.0
    config.risk_aversion = 0.0
    config.mega_cap_protection_enabled = False
    config.enforce_score_gated_ow = False

    tickers = [f"T{i:02d}" for i in range(20)]
    benchmark = np.full(len(tickers), 1.0 / len(tickers))
    covariance = np.eye(len(tickers)) * 0.0004
    alpha = pd.Series(np.linspace(1.0, -1.0, len(tickers)), index=tickers)

    target = optimize_portfolio(
        alpha,
        covariance,
        prev_weights=benchmark,
        bm_weights=benchmark,
        config=config,
    )
    projected = project_portfolio_weights(
        0.5 * benchmark + 0.5 * target,
        alpha,
        covariance,
        prev_weights=benchmark,
        bm_weights=benchmark,
        config=config,
    )
    for weights in (target, projected):
        active = weights - benchmark
        estimated_te = np.sqrt(active @ covariance @ active) * np.sqrt(252.0)
        assert estimated_te <= 0.035 + 1e-5


def test_causal_split_purges_realization_overlap_and_future_labels():
    dates = pd.bdate_range("2018-01-01", periods=1400)
    split = build_walk_forward_split(dates, 1260, 1260, 126, 20)
    audit = split["audit"]
    assert audit["causal_validation_ok"] is True
    assert audit["embargo_days"] == 20
    assert pd.Timestamp(audit["latest_validation_label_realization"]) <= dates[1260]
    assert pd.Timestamp(audit["latest_train_label_realization"]) < pd.Timestamp(audit["validation_start"])
    assert len(split["val_dates"]) == 126


def test_rank_data_is_date_grouped_sorted_and_integer_relevance():
    dates = pd.bdate_range("2025-01-01", periods=3)
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    panel = pd.DataFrame({"f": np.arange(len(index), dtype=float)}, index=index)
    targets = pd.DataFrame(
        np.tile(np.arange(4, dtype=float), (3, 1)), index=dates, columns=tickers
    )
    X, y, groups = prepare_rank_data(panel, targets, ["f"], dates, 10)
    assert groups == [4, 4, 4]
    assert X.shape == (12, 1)
    assert y.dtype.kind in "iu"
    assert y[:4].tolist() == [0, 2, 5, 7]
    assert sum(groups) == len(y)


def test_ranker_fits_with_date_groups():
    import lightgbm as lgb

    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-01", periods=18)
    tickers = [f"T{i:02d}" for i in range(12)]
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    panel = pd.DataFrame(rng.normal(size=(len(index), 3)), index=index, columns=["a", "b", "c"])
    target_values = panel["a"].unstack("ticker") + 0.1 * rng.normal(size=(len(dates), len(tickers)))
    config = PipelineConfig(
        model_objective="cross_sectional_rank",
        rank_relevance_levels=10,
        rank_eval_at=[5, 10],
        early_stopping_rounds=5,
        lgbm_params={
            "objective": "rank_xendcg", "metric": "ndcg", "n_estimators": 20,
            "learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 2,
            "verbose": -1, "random_state": 42,
        },
    )
    model = train_model(panel, target_values, ["a", "b", "c"], dates[:12], dates[12:], config)
    assert isinstance(model, lgb.LGBMRanker)
    assert np.isfinite(model.predict(panel.xs(dates[-1])[['a', 'b', 'c']])).all()


def test_execution_lag_uses_only_previous_row_and_zero_is_identity():
    dates = pd.bdate_range("2026-01-01", periods=3)
    pred = pd.DataFrame({"AAA": [1.0, 2.0, 3.0]}, index=dates)
    raw = pred * 10
    delayed, delayed_raw = apply_execution_signal_lag(pred, raw, 1)
    assert np.isnan(delayed.iloc[0, 0])
    assert delayed.iloc[1, 0] == 1.0
    assert delayed_raw.iloc[2, 0] == 20.0
    same, same_raw = apply_execution_signal_lag(pred, raw, 0)
    assert same is pred and same_raw is raw


def test_ranker_feature_attribution_schema_and_additivity():
    import lightgbm as lgb
    from scripts.export_operating_data import build_feature_attribution

    rng = np.random.default_rng(9)
    features = ["a", "b", "c"]
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    X = rng.normal(size=(40, len(features)))
    y = np.tile(np.arange(4, dtype=np.int32), 10)
    model = lgb.LGBMRanker(
        objective="rank_xendcg", metric="ndcg", n_estimators=12,
        num_leaves=7, min_child_samples=2, verbose=-1, random_state=42,
    )
    model.fit(X, y, group=[4] * 10)
    model._active_features = features
    as_of = pd.Timestamp("2026-05-21")
    index = pd.MultiIndex.from_product([[as_of], tickers], names=["date", "ticker"])
    panel = pd.DataFrame(rng.normal(size=(4, 3)), index=index, columns=features)
    result = types.SimpleNamespace(
        models={as_of: model}, panel=panel,
        portfolio_weights={as_of: pd.Series(0.25, index=tickers)},
        feature_groups={"Signal": features},
        sector_map={ticker: "Test" for ticker in tickers},
        bm_weights={ticker: 0.25 for ticker in tickers},
    )
    out = build_feature_attribution(result)
    assert out["additivity_ok"] is True
    assert set(out["tickers"]) == set(tickers)
