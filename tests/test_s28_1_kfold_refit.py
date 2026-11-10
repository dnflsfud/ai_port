"""§S28.1 (decision log §S28 H2·H3): purged K-fold capacity selection + full-window refit.

The production split stops a retrain's capacity on ONE 126-day validation window; a
quarter of the retrains land under ``min_model_trees`` and that set flips between data
vintages (§S28), and the training rows end 168 sessions before the prediction date.
``kfold_capacity_refit_enabled`` instead (i) scores every capacity 1..n_estimators on K
contiguous validation blocks covering the whole label-realised window (training rows
purged by the label horizon on both sides of each block), (ii) takes the first argmax of
the fold-mean curve and (iii) refits one fixed-capacity model on the whole window.

OFF (default) executes none of the new code.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import PipelineConfig
from src.model_trainer import (
    build_purged_kfold_splits,
    capacity_from_fold_curves,
    prepare_rank_data,
    train_model,
    walk_forward_train,
)

H = 5


def _synthetic_panel(n_dates=170, n_tickers=14, seed=5):
    """Small panel with a real (weak) linear signal in f0 so rank training has something to fit."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    features = [f"f{i}" for i in range(4)]
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    panel = pd.DataFrame(rng.normal(size=(len(index), len(features))), index=index, columns=features)
    signal = panel["f0"].unstack("ticker")
    targets = 0.5 * signal + pd.DataFrame(rng.normal(size=signal.shape), index=dates, columns=tickers)
    return panel, targets, features, dates


def _config(**overrides):
    base = dict(
        model_objective="cross_sectional_rank",
        causal_validation_enabled=True,
        rank_relevance_levels=4,
        rank_eval_at=[3],
        forward_horizon=H,
        train_window=100,
        retrain_freq=30,
        val_window=20,
        ewma_enabled=False,
        min_model_trees=1,
        early_stopping_rounds=5,
        lgbm_params={
            "objective": "rank_xendcg", "metric": "ndcg", "n_estimators": 12,
            "learning_rate": 0.1, "min_child_samples": 5, "num_leaves": 7,
            "verbose": -1, "random_state": 0,
        },
    )
    base.update(overrides)
    return PipelineConfig(**base)


def _positions(dates, subset):
    return np.asarray(pd.DatetimeIndex(dates).get_indexer(pd.DatetimeIndex(subset)))


# ---------------------------------------------------------------------------
# 1. defaults
# ---------------------------------------------------------------------------

def test_kfold_capacity_refit_is_off_by_default_with_five_folds():
    config = PipelineConfig()
    assert config.kfold_capacity_refit_enabled is False
    assert config.kfold_n_splits == 5


# ---------------------------------------------------------------------------
# 2. split geometry
# ---------------------------------------------------------------------------

def _splits(n_splits=5, t=250, window=200):
    dates = pd.bdate_range("2020-01-01", periods=300)
    return dates, build_purged_kfold_splits(dates, t, window, H, n_splits)


def test_kfold_validation_blocks_partition_the_label_realised_window():
    dates, kf = _splits()
    window = kf["window_dates"]
    # rows [t - window, t - H]: the last row's label realises ON the prediction date
    assert window[0] == dates[50] and window[-1] == dates[250 - H]
    blocks = [fold["val_dates"] for fold in kf["folds"]]
    assert len(blocks) == 5
    assert pd.DatetimeIndex(np.concatenate([b.values for b in blocks])).equals(window)
    sizes = [len(b) for b in blocks]
    assert max(sizes) - min(sizes) <= 1


def test_kfold_training_rows_are_purged_by_the_label_horizon_on_both_sides():
    dates, kf = _splits()
    window_pos = set(_positions(dates, kf["window_dates"]))
    for k, fold in enumerate(kf["folds"]):
        train = _positions(dates, fold["train_dates"])
        val = _positions(dates, fold["val_dates"])
        assert len(train) > 0 and not set(train) & set(val)
        # no training label window overlaps a validation label window: |i - j| > H
        gap = np.abs(train[:, None] - val[None, :]).min()
        assert gap == H + 1
        purged = window_pos - set(train) - set(val)
        expected_purged = H * (2 if 0 < k < len(kf["folds"]) - 1 else 1)
        assert len(purged) == expected_purged
    middle = _positions(dates, kf["folds"][2]["train_dates"])
    mid_val = _positions(dates, kf["folds"][2]["val_dates"])
    assert middle.min() < mid_val.min() and middle.max() > mid_val.max()     # trains on both sides


def test_kfold_audit_row_is_causal_and_names_the_refit_window():
    dates, kf = _splits()
    audit = kf["audit"]
    assert audit["prediction_date"] == str(dates[250].date())
    assert audit["train_start"] == str(dates[50].date())
    assert audit["train_end"] == str(dates[250 - H].date())
    assert audit["latest_train_label_realization"] == str(dates[250].date())
    assert audit["causal_validation_ok"] is True
    assert audit["forward_horizon"] == H and audit["kfold_n_splits"] == 5


def test_kfold_split_requires_enough_history_for_every_fold():
    dates = pd.bdate_range("2020-01-01", periods=300)
    with pytest.raises(ValueError, match="insufficient history"):
        build_purged_kfold_splits(dates, 12, 200, H, 5)       # 8 usable rows, 5 folds, H=5 purge
    with pytest.raises(ValueError, match="n_splits"):
        build_purged_kfold_splits(dates, 250, 200, H, 1)


# ---------------------------------------------------------------------------
# 3. capacity rule
# ---------------------------------------------------------------------------

def test_capacity_is_the_first_argmax_of_the_fold_mean_curve():
    # fold means: [0.50, 0.50, 0.575, 0.575] -> first maximum at iteration 3
    assert capacity_from_fold_curves([[0.5, 0.6, 0.55, 0.60], [0.5, 0.4, 0.60, 0.55]]) == 3
    assert capacity_from_fold_curves([[0.9, 0.1], [0.9, 0.1]]) == 1


def test_capacity_rejects_ragged_or_empty_curves():
    with pytest.raises(ValueError):
        capacity_from_fold_curves([[0.5, 0.6], [0.5]])
    with pytest.raises(ValueError):
        capacity_from_fold_curves([])


def test_prepare_rank_data_can_report_the_date_of_every_group():
    panel, targets, features, dates = _synthetic_panel()
    X, y, groups = prepare_rank_data(panel, targets, features, dates[:30], 4)
    X2, y2, groups2, group_dates = prepare_rank_data(panel, targets, features, dates[:30], 4,
                                                     return_group_dates=True)
    assert np.array_equal(X, X2) and np.array_equal(y, y2) and groups == groups2
    assert list(group_dates) == list(dates[:30]) and len(group_dates) == len(groups)


# ---------------------------------------------------------------------------
# 4. walk-forward integration
# ---------------------------------------------------------------------------

def _run(**overrides):
    panel, targets, features, dates = _synthetic_panel()
    config = _config(**overrides)
    models, pred, _raw, tracker = walk_forward_train(panel, targets, features, dates, config=config)
    return panel, targets, features, dates, config, models, pred, tracker


def test_flag_off_leaves_the_single_window_path_untouched():
    *_, models_a, pred_a, tracker_a = _run()
    *_, models_b, pred_b, tracker_b = _run(kfold_n_splits=3)          # inert while the flag is OFF
    assert np.allclose(pred_a.fillna(0).values, pred_b.fillna(0).values, atol=0, rtol=0)
    assert [m.n_estimators_ for m in models_a.values()] == [m.n_estimators_ for m in models_b.values()]
    assert "kfold" not in tracker_a.model_quality["split_audit"][0]
    assert "validation_start" in tracker_a.model_quality["split_audit"][0]


def test_flag_on_refits_the_whole_realised_window_at_the_selected_capacity():
    panel, targets, features, dates, config, models, _pred, tracker = _run(kfold_capacity_refit_enabled=True)
    quality = tracker.model_quality
    assert quality["causal_validation_ok"] is True
    assert quality["unique_models"] == quality["total_retrains"] == len(models)
    for audit, (t_date, model) in zip(quality["split_audit"], sorted(models.items())):
        t = dates.get_loc(t_date)
        assert audit["prediction_date"] == str(t_date.date())
        assert audit["train_end"] == str(dates[t - H].date())            # 0 rows held out for early stopping
        info = audit["kfold"]
        assert 1 <= info["n_star"] <= 12 and info["max_rounds"] == 12 and info["n_folds"] == 5
        assert len(info["fold_argmax"]) == 5
        assert model.n_estimators_ == info["n_star"]
        window = dates[t - config.train_window: t - H + 1]
        reference = train_model(panel, targets, features, window, window[:0], config=config,
                                fixed_trees=info["n_star"])
        grid = panel.loc[(t_date, slice(None)), features].values
        assert np.allclose(model.predict(grid), reference.predict(grid), atol=1e-12)


def test_flag_on_keeps_the_degenerate_rule_and_reuse_fallback():
    *_, models, _pred, tracker = _run(kfold_capacity_refit_enabled=True, min_model_trees=10 ** 6)
    quality = tracker.model_quality
    order = sorted(models)
    assert quality["degenerate_retrains"] == quality["total_retrains"] == len(order)
    assert all(models[d] is models[order[0]] for d in order[1:])
    assert {e["fallback"] for e in quality["events"]} == {"reuse_prev"}


def test_flag_on_requires_the_causal_rank_pipeline():
    panel, targets, features, dates = _synthetic_panel()
    with pytest.raises(ValueError, match="kfold_capacity_refit_enabled"):
        walk_forward_train(panel, targets, features, dates,
                           config=_config(kfold_capacity_refit_enabled=True, causal_validation_enabled=False))
    with pytest.raises(ValueError, match="kfold_capacity_refit_enabled"):
        walk_forward_train(panel, targets, features, dates,
                           config=_config(kfold_capacity_refit_enabled=True, model_objective="regression",
                                          lgbm_params={"objective": "regression", "n_estimators": 12,
                                                       "verbose": -1, "random_state": 0}))
