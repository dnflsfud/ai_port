"""Unit and causal acceptance tests for the S13.16 residual sleeve."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.config import PipelineConfig
from src.residual_sleeve import (
    BASE_SIGNAL_FEATURE,
    build_residual_feature_panel,
    build_residual_labels,
    compose_fixed_risk_sleeve,
    orthogonalize_residual_scores,
    smooth_residual_scores,
    symmetric_cross_sectional_rank,
    walk_forward_residual_scores,
)


def _panel(dates, tickers, values):
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    return pd.DataFrame(values, index=index)


def _small_config(**updates):
    cfg = PipelineConfig(
        expected_universe_size=None,
        execution_signal_lag_days=1,
        residual_sleeve_min_names=12,
        residual_sleeve_train_window=80,
        residual_sleeve_retrain_freq=20,
        residual_sleeve_sample_freq=5,
        residual_sleeve_min_train_dates=5,
        residual_sleeve_include_confirmation_features=False,
        residual_sleeve_lgbm_params={
            "objective": "huber",
            "learning_rate": 0.05,
            "n_estimators": 20,
            "max_depth": 2,
            "num_leaves": 4,
            "min_child_samples": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 43,
            "n_jobs": 1,
            "verbosity": -1,
        },
    )
    for key, value in updates.items():
        setattr(cfg, key, value)
    return cfg


def test_default_off_and_variant_has_one_semantic_delta():
    cfg = PipelineConfig(expected_universe_size=None)
    assert cfg.residual_sleeve_enabled is False
    assert cfg.dr_alpha_enabled is False

    root = Path(__file__).resolve().parents[1]
    base = yaml.safe_load((root / "variants/codex_causal_rank_65.yaml").read_text(encoding="utf-8"))
    arm = yaml.safe_load(
        (root / "variants/arm_s13_16_residual_nonlinear_sleeve.yaml").read_text(
            encoding="utf-8"
        )
    )
    base_overrides = base["overrides"]
    arm_overrides = arm["overrides"]
    for key, value in base_overrides.items():
        if key == "dr_alpha_enabled":
            continue
        assert arm_overrides[key] == value
    semantic_changes = {
        key: value
        for key, value in arm_overrides.items()
        if value != getattr(cfg, key)
        and key not in base_overrides
    }
    assert semantic_changes == {"residual_sleeve_enabled": True}
    assert arm_overrides["dr_alpha_enabled"] is False
    assert arm_overrides["nonlinear_confirmation_features_enabled"] is False
    assert arm_overrides["interaction_features_enabled"] is False


def test_symmetric_rank_is_bounded_and_zero_mean():
    frame = pd.DataFrame(
        [[4.0, 1.0, 3.0, 2.0], [1.0, np.nan, 1.0, 3.0]],
        columns=list("ABCD"),
    )
    ranked = symmetric_cross_sectional_rank(frame)
    assert np.isclose(ranked.iloc[0].min(), -1.0)
    assert np.isclose(ranked.iloc[0].max(), 1.0)
    assert np.isclose(ranked.iloc[0].mean(), 0.0)
    assert np.isclose(ranked.iloc[1].dropna().mean(), 0.0)


def test_residual_labels_remove_base_styles_size_and_sector():
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2020-01-01", periods=8)
    tickers = [f"T{i:02d}" for i in range(40)]
    base = pd.DataFrame(rng.normal(size=(len(dates), len(tickers))), index=dates, columns=tickers)
    momentum = rng.normal(size=(len(dates), len(tickers)))
    beta = rng.normal(size=(len(dates), len(tickers)))
    panel = _panel(
        dates,
        tickers,
        {
            "momentum_252d": momentum.reshape(-1),
            "beta_63d": beta.reshape(-1),
        },
    )
    sectors = {t: ("Tech" if i < 20 else "Banks") for i, t in enumerate(tickers)}
    sector_effect = np.array([1.0 if sectors[t] == "Tech" else -1.0 for t in tickers])
    cap = pd.DataFrame(
        np.exp(rng.normal(10.0, 1.0, size=(len(dates), len(tickers)))),
        index=dates,
        columns=tickers,
    )
    target = 2.0 * base.to_numpy() + 0.8 * momentum + 0.4 * beta + sector_effect
    target += 0.25 * momentum * beta + rng.normal(scale=0.05, size=target.shape)
    targets = pd.DataFrame(target, index=dates, columns=tickers)
    cfg = _small_config(
        execution_signal_lag_days=0,
        residual_sleeve_min_names=20,
        residual_sleeve_orthogonal_features=["momentum_252d", "beta_63d"],
    )
    labels, diagnostics = build_residual_labels(
        targets, base, panel, cap, sectors, cfg
    )
    assert diagnostics["label_dates"] == len(dates)
    for date in dates:
        assert abs(labels.loc[date].corr(base.loc[date])) < 1e-10
        p = panel.xs(date, level="date")
        assert abs(labels.loc[date].corr(p["momentum_252d"])) < 1e-10
        assert abs(labels.loc[date].corr(p["beta_63d"])) < 1e-10


def test_prediction_orthogonalizer_handles_collinearity_and_keeps_grid():
    rng = np.random.default_rng(13)
    dates = pd.bdate_range("2021-01-01", periods=5)
    tickers = [f"T{i:02d}" for i in range(30)]
    base = pd.DataFrame(rng.normal(size=(5, 30)), index=dates, columns=tickers)
    style = rng.normal(size=(5, 30))
    panel = _panel(
        dates,
        tickers,
        {
            "momentum_252d": style.reshape(-1),
            "beta_63d": (2.0 * style).reshape(-1),
        },
    )
    raw = pd.DataFrame(
        3.0 * base.to_numpy() + 2.0 * style + 0.4 * style**2,
        index=dates,
        columns=tickers,
    )
    cap = pd.DataFrame(1.0, index=dates, columns=tickers)
    sectors = {t: ("A" if i % 2 == 0 else "B") for i, t in enumerate(tickers)}
    cfg = _small_config(
        execution_signal_lag_days=0,
        residual_sleeve_min_names=15,
        residual_sleeve_orthogonal_features=["momentum_252d", "beta_63d"],
        residual_sleeve_clip_quantile=0.0,
    )
    scores, diagnostics = orthogonalize_residual_scores(
        raw, base, panel, cap, sectors, cfg
    )
    assert scores.shape == raw.shape
    assert diagnostics["active_score_dates"] == len(dates)
    for date in dates:
        p = panel.xs(date, level="date")
        assert abs(scores.loc[date].corr(base.loc[date])) < 1e-10
        assert abs(scores.loc[date].corr(p["momentum_252d"])) < 1e-10


def test_residual_score_smoothing_is_past_only():
    dates = pd.bdate_range("2022-01-01", periods=30)
    raw = pd.DataFrame(
        {"A": np.arange(30, dtype=float), "B": np.arange(30, dtype=float)[::-1]},
        index=dates,
    )
    baseline = smooth_residual_scores(raw, span=21)
    changed = raw.copy()
    changed.loc[dates[20]:] += 10_000.0
    perturbed = smooth_residual_scores(changed, span=21)
    pd.testing.assert_frame_equal(baseline.loc[: dates[19]], perturbed.loc[: dates[19]])


def test_walk_forward_embargo_and_future_label_invariance():
    rng = np.random.default_rng(19)
    dates = pd.bdate_range("2019-01-01", periods=180)
    tickers = [f"T{i:02d}" for i in range(24)]
    f1 = rng.normal(size=(len(dates), len(tickers)))
    f2 = rng.normal(size=(len(dates), len(tickers)))
    base = pd.DataFrame(rng.normal(size=f1.shape), index=dates, columns=tickers)
    panel = _panel(
        dates,
        tickers,
        {"f1": f1.reshape(-1), "f2": f2.reshape(-1)},
    )
    cfg = _small_config()
    feature_panel, feature_names = build_residual_feature_panel(
        panel, ["f1", "f2"], base, data=None, config=cfg
    )
    assert BASE_SIGNAL_FEATURE in feature_names
    labels = pd.DataFrame(
        0.5 * f1 * f2 + rng.normal(scale=0.1, size=f1.shape),
        index=dates,
        columns=tickers,
    )
    raw_a, _models_a, diag_a = walk_forward_residual_scores(
        feature_panel, labels, base, feature_names, cfg
    )
    mutated = labels.copy()
    mutated.loc[dates[100]:] += 1000.0 * rng.normal(
        size=mutated.loc[dates[100]:].shape
    )
    raw_b, _models_b, diag_b = walk_forward_residual_scores(
        feature_panel, mutated, base, feature_names, cfg
    )
    # Fold start 120 can use labels only through position 99 (H=20,L=1), so
    # predictions through the end of that block remain exactly unchanged.
    pd.testing.assert_frame_equal(raw_a.loc[: dates[139]], raw_b.loc[: dates[139]])
    for fold in diag_a["fold_audit"]:
        if fold.get("trained"):
            assert fold["latest_label_realisation_position"] <= fold["fold_position"]
    assert diag_a["embargo_days"] == cfg.forward_horizon + cfg.execution_signal_lag_days
    assert diag_b["embargo_days"] == diag_a["embargo_days"]


def test_fixed_risk_composition_is_covariance_orthogonal_90_10():
    rng = np.random.default_rng(23)
    n = 12
    A = rng.normal(size=(n, n))
    cov = A.T @ A + np.eye(n) * 0.2
    bm = np.ones(n) / n
    a = rng.normal(scale=0.01, size=n)
    a -= a.mean()
    r = rng.normal(scale=0.01, size=n)
    r -= r.mean()
    w_base = bm + a
    w_residual = bm + r
    b, c, info = compose_fixed_risk_sleeve(w_base, w_residual, bm, cov, 0.10)
    base_var = float(a @ cov @ a)
    b_active = b - bm
    residual_component = c - b
    c_active = c - bm
    assert np.isclose(float(b_active @ cov @ b_active), 0.90 * base_var, rtol=1e-9)
    assert abs(float(b_active @ cov @ residual_component)) < 1e-10
    assert np.isclose(float(c_active @ cov @ c_active), base_var, rtol=1e-9)
    assert np.isclose(info["pre_projection_residual_variance_share"], 0.10, atol=1e-9)

    b0, c0, _info0 = compose_fixed_risk_sleeve(
        w_base, w_residual, bm, cov, 0.0
    )
    np.testing.assert_array_equal(b0, w_base)
    np.testing.assert_array_equal(c0, w_base)


def test_invalid_residual_budget_rejected():
    try:
        PipelineConfig(expected_universe_size=None, residual_sleeve_risk_fraction=1.0)
    except ValueError as exc:
        assert "risk_fraction" in str(exc)
    else:
        raise AssertionError("invalid residual risk fraction was accepted")
