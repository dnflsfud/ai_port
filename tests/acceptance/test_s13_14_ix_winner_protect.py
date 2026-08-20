# -*- coding: utf-8 -*-
"""S13.14 combined arm — interaction block + winner-trim protection.

Penalty semantics live in tests/test_winner_trim.py, block semantics in
tests/test_interactions.py. This file pins the COMBINED arm contract:
both flags default-OFF, the production optimizer path computes the mask
look-ahead-free, the penalty parameters are the pre-registered values
(lambda 1.0, quantile 0.8 — single values, no sweep), and the variant
differs from production by exactly the two flags.
"""

import inspect
from dataclasses import fields
from pathlib import Path

import yaml

from src.config import PipelineConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ARM_YAML = _REPO_ROOT / "variants" / "arm_s13_14_ix_winner_protect.yaml"
_PROD_YAML = _REPO_ROOT / "variants" / "codex_causal_rank_65.yaml"

_FLAGS = {"interaction_features_enabled", "winner_trim_protection_enabled"}


def test_both_flags_default_off():
    cfg = PipelineConfig()
    assert cfg.interaction_features_enabled is False
    assert cfg.winner_trim_protection_enabled is False


def test_penalty_params_are_the_preregistered_singles():
    cfg = PipelineConfig()
    assert cfg.winner_trim_lambda == 1.0
    assert cfg.winner_trim_quantile == 0.8


def test_production_optimizer_path_wires_the_mask():
    from src import backtest

    source = inspect.getsource(backtest.run_backtest)
    assert "compute_winner_mask" in source
    assert "winner_trim_protection_enabled" in source


def test_optimize_portfolio_accepts_the_mask_and_penalises_in_objective():
    from src import portfolio_optimizer as po

    sig = inspect.signature(po.optimize_portfolio)
    assert "winner_mask" in sig.parameters
    source = inspect.getsource(po.optimize_portfolio)
    assert "_winner_trim_penalty_expr" in source
    assert "- winner_pen" in source.replace("  ", " ")


def test_arm_variant_differs_from_production_by_exactly_the_two_flags():
    arm = yaml.safe_load(_ARM_YAML.read_text(encoding="utf-8"))["overrides"]
    prod = yaml.safe_load(_PROD_YAML.read_text(encoding="utf-8"))["overrides"]

    assert arm["interaction_features_enabled"] is True
    assert arm["winner_trim_protection_enabled"] is True
    delta = {k: v for k, v in arm.items() if prod.get(k, "<absent>") != v}
    # S13.47 promotion (2026-08-20): production rank_eval_at [5, 10] -> [20];
    # historical arms pin the pre-promotion value.
    assert set(delta) - {"rank_eval_at"} == _FLAGS, delta
    post_arm_production_flags = {
        "fwd_sales_slope_features_enabled",
        "vol_quality_tilt_enabled",
        "vol_quality_tilt_lambda",
        "option_vol_covariance_enabled",  # S13.41 promotion (2026-08-13)
    }
    assert not [
        k for k in prod
        if k not in arm and k not in post_arm_production_flags
    ]


def test_arm_variant_keys_are_all_real_config_fields():
    known = {f.name for f in fields(PipelineConfig)}
    arm = yaml.safe_load(_ARM_YAML.read_text(encoding="utf-8"))["overrides"]
    assert not [k for k in arm if k not in known]
    cfg = PipelineConfig(**arm)
    assert cfg.interaction_features_enabled and cfg.winner_trim_protection_enabled
