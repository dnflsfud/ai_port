"""§S11.8(b): effective_label_horizon — causal split이 블렌드 타깃의 최장
horizon으로 purge/embargo를 계산해야 한다(multi-horizon 활성 시 누수 방지).
OFF(기본)면 forward_horizon 그대로 → 기존 분할과 바이트 동일 파리티.

§S12.3: ewma_full_refresh_interval — 주기적 full refresh가 켜졌을 때만
model_quality에 refresh/재진입 증거를 기록한다(OFF면 키 부재 = 파리티).
"""

import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.model_trainer import (
    build_walk_forward_split,
    effective_label_horizon,
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
