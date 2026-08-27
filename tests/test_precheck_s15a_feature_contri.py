# -*- coding: utf-8 -*-
"""§S15 후보 A 사전점검(feature_contri) — plain-function 단위 테스트."""
import numpy as np
import pandas as pd

from scripts.precheck_s15a_feature_contri import (
    consecutive_persistence, map_importance, per_date_ic_matrix,
    reconstruct_ewma_states, spearman, unique_models_in_order,
    window_alignment)


def test_spearman_perfect_and_inverse():
    assert np.isclose(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
    assert np.isclose(spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
    assert np.isnan(spearman([1, np.nan, np.nan, np.nan], [1, 2, 3, 4]))


def test_unique_models_in_order_dedupes_reused_objects():
    a, b = object(), object()
    models = {
        pd.Timestamp("2020-01-01"): a,
        pd.Timestamp("2020-04-01"): a,   # 퇴화 재사용
        pd.Timestamp("2020-07-01"): b,
    }
    uniq = unique_models_in_order(models)
    assert [d for d, _ in uniq] == [pd.Timestamp("2020-01-01"),
                                    pd.Timestamp("2020-07-01")]
    assert [m for _, m in uniq] == [a, b]


def test_map_importance_normalizes_and_maps_to_full_space():
    full = ["f0", "f1", "f2", "f3"]
    out = map_importance([3.0, 1.0], ["f2", "f0"], full)
    assert np.allclose(out, [0.25, 0.0, 0.75, 0.0])
    assert np.isclose(out.sum(), 1.0)


def test_reconstruct_ewma_states_matches_manual_recursion():
    imp1 = np.array([0.8, 0.2, 0.0])
    imp2 = np.array([0.0, 0.5, 0.5])
    states = reconstruct_ewma_states([imp1, imp2], alpha=0.3, n_features=3)
    s0 = 0.3 * imp1 + 0.7 * np.ones(3) / 3
    s0 = s0 / s0.sum()
    s1 = 0.3 * imp2 + 0.7 * s0
    s1 = s1 / s1.sum()
    assert np.allclose(states[0], s0)
    assert np.allclose(states[1], s1)


def test_consecutive_persistence_uses_intersection():
    imp_a = np.array([0.5, 0.3, 0.2, 0.0])
    imp_b = np.array([0.6, 0.25, 0.15, 0.0])
    vals = consecutive_persistence([imp_a, imp_b], [{0, 1, 2}, {0, 1, 2}])
    assert np.isclose(vals[0], 1.0)


def _tiny_panel(n_dates=6, n_tickers=40, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    tickers = [f"T{i}" for i in range(n_tickers)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    tgt = pd.DataFrame(rng.normal(size=(n_dates, n_tickers)),
                       index=dates, columns=tickers)
    stacked = tgt.stack()
    panel = pd.DataFrame({
        "pos": stacked.values,                      # 타깃과 동일 -> IC=1
        "neg": -stacked.values,                     # 역방향 -> IC=-1
        "noise": rng.normal(size=len(stacked)),
    }, index=idx)
    return panel, tgt


def test_per_date_ic_matrix_signs():
    panel, tgt = _tiny_panel()
    ic = per_date_ic_matrix(panel, tgt, min_tickers=10)
    assert ic.shape[0] == 6
    assert np.allclose(ic["pos"], 1.0)
    assert np.allclose(ic["neg"], -1.0)
    assert ic["noise"].abs().max() < 1.0


def test_window_alignment_positive_when_state_matches_ic():
    dates = pd.bdate_range("2020-01-01", periods=30)
    ic = pd.DataFrame({
        "f0": [0.30] * 30, "f1": [0.10] * 30, "f2": [0.01] * 30,
    }, index=dates)
    state = np.array([0.6, 0.3, 0.1])   # |IC| 순위와 동일
    out = window_alignment([state], [dates[4]], ic)
    assert len(out) == 1
    assert np.isclose(out[0]["spearman"], 1.0)
