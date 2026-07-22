import types

import numpy as np
import pandas as pd

from src.backtest import (
    BacktestResult,
    apply_oof_alpha_calibration,
    make_capweight_bm_fn,
    make_ew_bm_fn,
)
from src.config import PipelineConfig
from src.data_loader import resolve_listing_dates
from src.model_trainer import (
    EWMAFeatureTracker,
    prepare_symmetric_rank_data,
    train_model,
)


def test_future_universe_addition_inherits_listing_inference():
    dates = pd.bdate_range("2024-01-01", periods=8)
    meta = pd.DataFrame(
        {"Name": ["Old", "New"]}, index=pd.Index(["OLD", "NEW"], name="ticker")
    )
    raw = {
        "PX_LAST": pd.DataFrame(
            {
                "OLD": np.arange(100.0, 108.0),
                "NEW": [10.0] * 5 + [12.0, 12.5, 13.0],
            },
            index=dates,
        )
    }
    cfg = PipelineConfig(listing_dates={}, listing_flat_min_run=5)

    resolved, sources = resolve_listing_dates(meta, raw, cfg)

    assert resolved["OLD"] == dates[0].strftime("%Y-%m-%d")
    assert resolved["NEW"] == dates[5].strftime("%Y-%m-%d")
    assert sources["NEW"] == "px_last_inferred"


def test_metadata_listing_date_overrides_price_inference():
    dates = pd.bdate_range("2024-01-01", periods=4)
    meta = pd.DataFrame(
        {"Listing_Date": ["2024-01-03"]},
        index=pd.Index(["NEW"], name="ticker"),
    )
    raw = {"PX_LAST": pd.DataFrame({"NEW": [1.0, 2.0, 3.0, 4.0]}, index=dates)}
    resolved, sources = resolve_listing_dates(meta, raw, PipelineConfig(listing_dates={}))
    assert resolved["NEW"] == "2024-01-03"
    assert sources["NEW"] == "meta:Listing_Date"


def test_equal_weight_benchmark_excludes_prelisting_names():
    tickers = ["OLD", "NEW"]
    fn = make_ew_bm_fn(tickers, {"NEW": "2024-01-03"})
    assert np.allclose(fn(pd.Timestamp("2024-01-02"), tickers, 2), [1.0, 0.0])
    assert np.allclose(fn(pd.Timestamp("2024-01-03"), tickers, 2), [0.5, 0.5])


def test_cap_weight_benchmark_excludes_dense_vendor_backfill():
    dates = pd.bdate_range("2024-01-01", periods=4)
    caps = pd.DataFrame({"OLD": 100.0, "NEW": 900.0}, index=dates)
    data = types.SimpleNamespace(
        market_cap=caps,
        listing_dates={"NEW": "2024-01-03"},
    )
    cfg = PipelineConfig(listing_dates={}, listing_mask_enabled=True)
    fn = make_capweight_bm_fn(data, ["OLD", "NEW"], config=cfg)
    assert np.allclose(fn(pd.Timestamp("2024-01-02"), ["OLD", "NEW"], 2), [1.0, 0.0])
    assert np.allclose(fn(pd.Timestamp("2024-01-03"), ["OLD", "NEW"], 2), [0.1, 0.9])


def test_symmetric_rank_labels_weight_both_tails_equally():
    dates = pd.bdate_range("2024-01-01", periods=2)
    index = pd.MultiIndex.from_product(
        [dates, ["A", "B", "C"]], names=["date", "ticker"]
    )
    panel = pd.DataFrame({"f": np.arange(6.0)}, index=index)
    targets = pd.DataFrame(
        [[-2.0, 0.0, 5.0], [7.0, 1.0, -3.0]],
        index=dates,
        columns=["A", "B", "C"],
    )
    _x, y = prepare_symmetric_rank_data(panel, targets, ["f"], dates)
    assert np.allclose(y[:3], [-1.0, 0.0, 1.0])
    assert np.allclose(y[3:], [1.0, 0.0, -1.0])


def test_ewma_candidate_disables_scaling_and_periodically_refreshes_all_features():
    cfg = PipelineConfig(
        ewma_min_retrains=1,
        ewma_feature_scaling_enabled=False,
        ewma_full_refresh_interval=2,
    )
    tracker = EWMAFeatureTracker(cfg)
    features = ["a", "b", "c"]
    tracker.init_full_features(features)
    tracker.n_updates = 2
    tracker.ewma_importance = np.array([0.9, 0.09, 0.01])
    assert tracker.get_active_features(features) == features
    assert tracker.get_feature_weights(features) is None


def test_ewma_permutation_importance_override_is_used():
    cfg = PipelineConfig(ewma_alpha=1.0, ewma_importance_type="permutation")
    tracker = EWMAFeatureTracker(cfg)
    tracker.init_full_features(["a", "b"])
    model = types.SimpleNamespace(
        _ewma_permutation_importance=np.array([0.0, 2.0])
    )
    tracker.update(model, ["a", "b"], pd.Timestamp("2024-01-01"))
    assert np.allclose(tracker.ewma_importance, [0.0, 1.0])


def test_ewma_stability_penalizes_features_with_unstable_gain():
    class Booster:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=float)

        def feature_importance(self, importance_type):
            assert importance_type == "gain"
            return self.values

    cfg = PipelineConfig(
        ewma_alpha=1.0,
        ewma_importance_type="stability",
        ewma_stability_window=2,
    )
    tracker = EWMAFeatureTracker(cfg)
    tracker.init_full_features(["stable", "flip_a", "flip_b"])
    tracker.update(
        types.SimpleNamespace(booster_=Booster([0.5, 0.4, 0.1])),
        ["stable", "flip_a", "flip_b"],
        pd.Timestamp("2024-01-01"),
    )
    tracker.update(
        types.SimpleNamespace(booster_=Booster([0.5, 0.1, 0.4])),
        ["stable", "flip_a", "flip_b"],
        pd.Timestamp("2024-01-02"),
    )
    assert tracker.ewma_importance[0] > tracker.ewma_importance[1]
    assert tracker.ewma_importance[0] > tracker.ewma_importance[2]


def test_symmetric_model_computes_validation_permutation_importance():
    dates = pd.bdate_range("2024-01-01", periods=12)
    tickers = ["A", "B", "C", "D"]
    index = pd.MultiIndex.from_product(
        [dates, tickers], names=["date", "ticker"]
    )
    x1 = np.tile([-1.5, -0.5, 0.5, 1.5], len(dates))
    panel = pd.DataFrame(
        {"signal": x1, "noise": np.sin(np.arange(len(index)))}, index=index
    )
    targets = pd.DataFrame(
        np.tile([-0.03, -0.01, 0.01, 0.03], (len(dates), 1)),
        index=dates,
        columns=tickers,
    )
    cfg = PipelineConfig(
        model_objective="symmetric_rank",
        ewma_importance_type="permutation",
        ewma_permutation_repeats=1,
        ewma_permutation_max_samples=100,
        early_stopping_rounds=3,
        lgbm_params={
            "objective": "regression",
            "metric": "mse",
            "learning_rate": 0.1,
            "num_leaves": 7,
            "min_child_samples": 2,
            "n_estimators": 12,
            "verbose": -1,
            "random_state": 42,
        },
    )
    model = train_model(
        panel, targets, ["signal", "noise"], dates[:8], dates[8:], config=cfg
    )
    importance = getattr(model, "_ewma_permutation_importance")
    assert importance.shape == (2,)
    assert np.isfinite(importance).all()


def test_oof_alpha_calibration_does_not_use_unmatured_future_targets():
    dates = pd.bdate_range("2024-01-01", periods=14)
    columns = ["A", "B", "C", "D"]
    predictions = pd.DataFrame(
        np.tile([-1.5, -0.5, 0.5, 1.5], (len(dates), 1)),
        index=dates,
        columns=columns,
    )
    targets = predictions.copy()
    changed_targets = targets.copy()
    changed_targets.loc[dates[8]] = changed_targets.loc[dates[8], ::-1].to_numpy()
    rng = np.random.default_rng(7)
    risk = pd.DataFrame(
        rng.normal(0.0, 0.01, size=predictions.shape),
        index=dates,
        columns=columns,
    )
    cfg = PipelineConfig(
        forward_horizon=2,
        alpha_calibration_enabled=True,
        alpha_calibration_lookback=5,
        alpha_calibration_min_observations=2,
        alpha_calibration_prior_observations=0,
    )
    base = apply_oof_alpha_calibration(predictions, targets, risk, cfg)
    changed = apply_oof_alpha_calibration(predictions, changed_targets, risk, cfg)

    pd.testing.assert_frame_equal(base.loc[: dates[9]], changed.loc[: dates[9]])


def test_sp500_is_reported_as_full_comparison_benchmark():
    dates = pd.bdate_range("2024-01-01", periods=260)
    result = BacktestResult()
    result.portfolio_returns = pd.Series(0.0005, index=dates)
    result.benchmark_returns = pd.Series(0.0003, index=dates)
    result.spx_returns = pd.Series(0.0002, index=dates)
    metrics = result.compute_metrics()
    required = {
        "sp500_annual_return",
        "sp500_annual_vol",
        "sp500_sharpe",
        "sp500_active_return",
        "sp500_tracking_error",
        "sp500_information_ratio",
        "sp500_beta",
    }
    assert required.issubset(metrics)
    assert result.cumulative_sp500.equals(result.cumulative_spx)
