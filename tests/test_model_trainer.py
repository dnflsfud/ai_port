"""§S11.8(b): effective_label_horizon — causal split이 블렌드 타깃의 최장
horizon으로 purge/embargo를 계산해야 한다(multi-horizon 활성 시 누수 방지).
OFF(기본)면 forward_horizon 그대로 → 기존 분할과 바이트 동일 파리티.

§S12.3: ewma_full_refresh_interval — 주기적 full refresh가 켜졌을 때만
model_quality에 refresh/재진입 증거를 기록한다(OFF면 키 부재 = 파리티).
"""

import numpy as np
import pandas as pd
import pytest

from src.config import PipelineConfig
from src.model_trainer import (
    build_walk_forward_split,
    effective_label_horizon,
    predict_cross_sectional,
    walk_forward_train,
)


def test_effective_label_horizon_default_is_forward_horizon():
    assert effective_label_horizon(PipelineConfig()) == 20


def test_effective_label_horizon_uses_max_blend_horizon():
    cfg = PipelineConfig(
        multi_horizon_targets_enabled=True,
        multi_horizon_weights={20: 0.7, 63: 0.3},
    )
    assert effective_label_horizon(cfg) == 63


def test_effective_label_horizon_ignores_weights_when_disabled():
    cfg = PipelineConfig(
        multi_horizon_targets_enabled=False,
        multi_horizon_weights={20: 0.7, 63: 0.3},
    )
    assert effective_label_horizon(cfg) == 20


def test_effective_label_horizon_never_below_forward_horizon():
    cfg = PipelineConfig(
        multi_horizon_targets_enabled=True,
        multi_horizon_weights={5: 1.0},
    )
    assert effective_label_horizon(cfg) == 20


def test_walk_forward_split_stays_causal_at_63d_horizon():
    dates = pd.bdate_range("2018-01-01", periods=900)
    split = build_walk_forward_split(
        all_dates=dates,
        prediction_idx=800,
        train_window=756,
        val_window=126,
        forward_horizon=63,
    )
    audit = split["audit"]
    assert audit["causal_validation_ok"] is True
    # embargo must cover the full 63d label realization window
    assert audit["embargo_days"] == 63


def test_prediction_excludes_pre_listing_name_before_cross_sectional_zscore():
    class _Model:
        @staticmethod
        def predict(values):
            first = values[:, 0]
            return np.where(np.isnan(first), 100.0, first)

    date = pd.Timestamp("2026-01-05")
    index = pd.MultiIndex.from_product(
        [[date], ["A", "B", "NEW"]], names=["date", "ticker"]
    )
    panel = pd.DataFrame({"f": [-1.0, 1.0, np.nan]}, index=index)

    pred = predict_cross_sectional(
        _Model(),
        panel,
        ["f"],
        date,
        listing_dates={"NEW": date.strftime("%Y-%m-%d")},
    )

    assert list(pred.index) == ["A", "B"]
    assert pred.loc["A"] == pytest.approx(-1.0 / np.sqrt(2.0))
    assert pred.loc["B"] == pytest.approx(1.0 / np.sqrt(2.0))


def _run_synthetic_walk_forward(**config_kwargs):
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-01", periods=90)
    tickers = ["A", "B", "C", "D", "E"]
    features = [f"f{i}" for i in range(6)]
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    panel = pd.DataFrame(
        rng.normal(size=(len(index), len(features))), index=index, columns=features
    )
    targets = pd.DataFrame(
        rng.normal(size=(len(dates), len(tickers))), index=dates, columns=tickers
    )
    config = PipelineConfig(
        train_window=40,
        retrain_freq=10,
        val_window=10,
        ewma_enabled=True,
        ewma_min_retrains=1,
        ewma_drop_pct=0.34,
        ewma_min_features=4,
        min_model_trees=1,
        early_stopping_rounds=5,
        lgbm_params={
            "objective": "regression", "metric": "mse", "n_estimators": 20,
            "min_child_samples": 5, "num_leaves": 7, "verbose": -1,
            "random_state": 0,
        },
        **config_kwargs,
    )
    models, _pred, _raw, tracker = walk_forward_train(
        panel, targets, features, dates, config=config
    )
    return models, tracker, features


def test_full_refresh_off_leaves_model_quality_without_refresh_keys():
    _models, tracker, _features = _run_synthetic_walk_forward()
    assert "ewma_full_refresh" not in tracker.model_quality


def test_full_refresh_records_refresh_dates_and_trains_on_full_set():
    models, tracker, features = _run_synthetic_walk_forward(
        ewma_full_refresh_interval=2
    )
    refresh = tracker.model_quality["ewma_full_refresh"]
    assert refresh["interval"] == 2
    assert len(refresh["refresh_dates"]) >= 1
    assert isinstance(refresh["reentry_events"], list)
    first_refresh = pd.Timestamp(refresh["refresh_dates"][0])
    assert list(models[first_refresh]._active_features) == features


def test_walk_forward_train_annotated_as_4_tuple():
    """실반환은 4-tuple(models, predictions, raw_predictions, ewma_tracker) —
    annotation/docstring이 3-tuple로 드리프트하면 안 된다."""
    import typing

    from src.model_trainer import EWMAFeatureTracker

    ret = typing.get_type_hints(walk_forward_train)["return"]
    args = typing.get_args(ret)
    assert len(args) == 4
    assert args[3] is EWMAFeatureTracker
    assert "ewma_tracker" in walk_forward_train.__doc__


# ---------------------------------------------------------------------------
# §S15.1 — monotone constraints (마진 축 3피처 arm 인프라)
# ---------------------------------------------------------------------------

def test_monotone_vector_none_when_disabled_or_empty():
    from src.model_trainer import _monotone_constraints_vector
    assert _monotone_constraints_vector(PipelineConfig(), ["a", "b"]) is None
    cfg = PipelineConfig(monotone_constraints_enabled=True)
    assert _monotone_constraints_vector(cfg, ["a", "b"]) is None  # 빈 맵


def test_monotone_vector_aligns_to_feature_order():
    from src.model_trainer import _monotone_constraints_vector
    cfg = PipelineConfig(
        monotone_constraints_enabled=True,
        monotone_constraints_map={"b": 1, "d": -1},
    )
    assert _monotone_constraints_vector(cfg, ["a", "b", "c", "d"]) == [0, 1, 0, -1]


def _monotone_train_setup(enabled):
    from src.model_trainer import train_model
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2024-01-01", periods=60)
    tickers = [f"T{i}" for i in range(10)]
    features = ["f0", "f1"]
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    panel = pd.DataFrame(
        rng.normal(size=(len(index), 2)), index=index, columns=features
    )
    signal = panel["f0"].unstack("ticker")
    targets = signal + 0.3 * pd.DataFrame(
        rng.normal(size=signal.shape), index=signal.index, columns=signal.columns
    )
    cfg = PipelineConfig(
        model_objective="cross_sectional_rank",
        rank_relevance_levels=5,
        early_stopping_rounds=10,
        monotone_constraints_enabled=enabled,
        monotone_constraints_map={"f0": 1},
        lgbm_params={
            "objective": "rank_xendcg", "metric": "ndcg", "n_estimators": 40,
            "min_child_samples": 5, "num_leaves": 7, "learning_rate": 0.1,
            "verbose": -1, "random_state": 0,
        },
    )
    model = train_model(panel, targets, features, dates[:45], dates[45:], config=cfg)
    return model


def test_train_model_monotone_off_params_untouched():
    model = _monotone_train_setup(enabled=False)
    assert "monotone_constraints" not in model.get_params()


def test_train_model_monotone_on_enforces_direction():
    model = _monotone_train_setup(enabled=True)
    assert model.get_params().get("monotone_constraints") == [1, 0]
    grid = np.linspace(-3, 3, 41)
    x_grid = np.zeros((41, 2))
    x_grid[:, 0] = grid
    preds = model.predict(x_grid)
    assert np.all(np.diff(preds) >= -1e-12)
