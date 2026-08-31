"""§S16.3 (T1-1): degenerate-retrain fallback policy.

`walk_forward_train`의 퇴화 폴백은 이전 모델과 그 피처 세트를 그대로 재사용해서
(reuse_prev), 인증 기준선 런에서 33회 재훈련 중 고유 모델이 19개뿐이고 라이브
모델이 354일 묵는 결과를 냈다. stale-depth 게이트는 재훈련 '슬롯 수'만 세므로
이를 잡지 못한다. `fresh_fixed` 모드는 조기종료 없이 사전등록 고정 용량으로
재적합해 라이브 모델이 묵는 경로 자체를 없앤다.

OFF(기본 reuse_prev·fixed_trees=None)이면 기존 훈련 경로와 동일하다.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src.config import PipelineConfig
from src.model_trainer import _prepare_train_data, train_model, walk_forward_train


def _synthetic_panel():
    """소형 합성 패널 — 라벨이 순수 노이즈라 조기종료가 실제로 퇴화한다."""
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2024-01-01", periods=90)
    tickers = [f"T{i}" for i in range(12)]
    features = [f"f{i}" for i in range(4)]
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    panel = pd.DataFrame(
        rng.normal(size=(len(index), len(features))), index=index, columns=features
    )
    targets = pd.DataFrame(
        rng.normal(size=(len(dates), len(tickers))), index=dates, columns=tickers
    )
    return panel, targets, features, dates


def _config(**overrides):
    base = dict(
        train_window=40,
        retrain_freq=10,
        val_window=10,
        ewma_enabled=False,
        min_model_trees=30,          # 노이즈 라벨 + 조기종료 -> 항상 퇴화
        early_stopping_rounds=5,
        lgbm_params={
            "objective": "regression", "metric": "mse", "n_estimators": 20,
            "min_child_samples": 5, "num_leaves": 7, "verbose": -1,
            "random_state": 0,
        },
    )
    base.update(overrides)
    return PipelineConfig(**base)


def _run(**overrides):
    panel, targets, features, dates = _synthetic_panel()
    config = _config(**overrides)
    models, _pred, _raw, tracker = walk_forward_train(
        panel, targets, features, dates, config=config
    )
    return models, tracker


# ---------------------------------------------------------------------------
# 1. 기본값 (사전등록 단일값)
# ---------------------------------------------------------------------------

def test_degenerate_fallback_defaults_are_reuse_prev_and_67():
    config = PipelineConfig()
    assert config.degenerate_fallback_mode == "reuse_prev"
    assert config.degenerate_fresh_trees == 67


# ---------------------------------------------------------------------------
# 2. 파리티 — fixed_trees=None은 인자 추가 이전 경로와 동일
# ---------------------------------------------------------------------------

def _inline_reference_model(panel, targets, features, train_dates, val_dates, config):
    """S16.3 이전 train_model의 default(regression) 경로를 인라인 재현."""
    X_train, y_train = _prepare_train_data(panel, targets, features, train_dates)
    X_val, y_val = _prepare_train_data(panel, targets, features, val_dates)
    model = lgb.LGBMRegressor(**config.lgbm_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(config.early_stopping_rounds, verbose=False),
                   lgb.log_evaluation(0)],
    )
    return model


def test_train_model_without_fixed_trees_matches_inline_reference():
    panel, targets, features, dates = _synthetic_panel()
    config = _config()
    train_dates, val_dates = dates[:30], dates[30:40]

    model = train_model(
        panel, targets, features, train_dates, val_dates, config=config
    )
    reference = _inline_reference_model(
        panel, targets, features, train_dates, val_dates, config
    )

    grid = panel.loc[(dates[40], slice(None)), features].values
    assert model.n_estimators_ == reference.n_estimators_
    assert np.allclose(model.predict(grid), reference.predict(grid), atol=1e-12)


# ---------------------------------------------------------------------------
# 3. 고정 용량 경로 — 조기종료 없음
# ---------------------------------------------------------------------------

def test_fixed_trees_pins_capacity_and_skips_early_stopping():
    panel, targets, features, dates = _synthetic_panel()
    config = _config()
    train_dates, val_dates = dates[:30], dates[30:40]

    fixed = train_model(
        panel, targets, features, train_dates, val_dates,
        config=config, fixed_trees=5,
    )
    stopped = train_model(
        panel, targets, features, train_dates, val_dates, config=config
    )

    assert fixed.n_estimators_ == 5
    assert fixed.booster_.num_trees() == 5
    # 조기종료 콜백이 붙지 않았다는 증거: best_iteration 미설정 + 캡까지 학습.
    assert not fixed.best_iteration_
    assert stopped.n_estimators_ < 5


# ---------------------------------------------------------------------------
# 4~5. 폴백 모드별 모델 객체 동일성
# ---------------------------------------------------------------------------

def test_fresh_fixed_adopts_a_new_model_on_every_degenerate_retrain():
    models, tracker = _run(
        degenerate_fallback_mode="fresh_fixed", degenerate_fresh_trees=40
    )
    dates = sorted(models)
    assert tracker.model_quality["degenerate_retrains"] == len(dates)
    assert len({id(m) for m in models.values()}) == len(dates)
    for earlier, later in zip(dates, dates[1:]):
        assert models[later] is not models[earlier]
    assert all(models[d].n_estimators_ == 40 for d in dates)
    assert {e["fallback"] for e in tracker.model_quality["events"]} == {"fresh_fixed"}


def test_reuse_prev_keeps_serving_the_same_model_object():
    models, tracker = _run()
    dates = sorted(models)
    assert tracker.model_quality["degenerate_retrains"] == len(dates)
    for later in dates[1:]:
        assert models[later] is models[dates[0]]
    assert {e["fallback"] for e in tracker.model_quality["events"]} == {"reuse_prev"}


def test_fresh_fixed_falls_back_to_reuse_prev_when_refit_stays_degenerate():
    # 고정 용량이 min_model_trees 미만이면 새 모델도 퇴화 -> 기존 동작으로 복귀.
    models, tracker = _run(
        degenerate_fallback_mode="fresh_fixed", degenerate_fresh_trees=5
    )
    dates = sorted(models)
    for later in dates[1:]:
        assert models[later] is models[dates[0]]
    assert {e["fallback"] for e in tracker.model_quality["events"]} == {
        "fresh_fixed_failed"
    }


# ---------------------------------------------------------------------------
# 6. model_quality 신선도 기록
# ---------------------------------------------------------------------------

def test_model_quality_records_live_model_freshness():
    models, tracker = _run()
    quality = tracker.model_quality
    dates = sorted(models)

    assert quality["unique_models"] == 1          # 첫 적합본만 실제로 훈련됨
    assert quality["live_model_fit_date"] == dates[0].strftime("%Y-%m-%d")
    # 첫 슬롯에서 적합된 모델이 남은 재훈련 슬롯 전부를 서빙했다.
    assert quality["live_model_age_retrains"] == quality["total_retrains"] - 1
    assert quality["live_model_age_retrains"] == len(dates) - 1

    fresh_quality = _run(
        degenerate_fallback_mode="fresh_fixed", degenerate_fresh_trees=40
    )[1].model_quality
    assert fresh_quality["unique_models"] == fresh_quality["total_retrains"]
    assert fresh_quality["live_model_age_retrains"] == 0


# ---------------------------------------------------------------------------
# 7. 잘못된 모드
# ---------------------------------------------------------------------------

def test_unknown_fallback_mode_raises():
    panel, targets, features, dates = _synthetic_panel()
    config = _config(degenerate_fallback_mode="refit_always")
    with pytest.raises(ValueError, match="degenerate_fallback_mode"):
        walk_forward_train(panel, targets, features, dates, config=config)
