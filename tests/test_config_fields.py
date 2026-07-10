"""Config fields: active-share derivation + post-init parity.

(The RL-hardening field tests that lived here were removed with the Gen-1 PPO
layer on 2026-06-05; CS-DR-Alpha config is covered by test_dr_alpha_config.py.)
"""
import pytest

from src.config import PipelineConfig


def test_forward_horizon_default_is_20():
    # LightGBM target horizon must stay 20.
    assert PipelineConfig().forward_horizon == 20


def test_causal_rank_defaults_preserve_legacy_path():
    c = PipelineConfig()
    assert c.model_objective == "regression"
    assert c.causal_validation_enabled is False
    assert c.execution_signal_lag_days == 0
    assert c.rank_relevance_levels == 10
    assert c.rank_eval_at == [5, 10]


def test_causal_rank_config_validation():
    with pytest.raises(ValueError):
        PipelineConfig(model_objective="classifier")
    with pytest.raises(ValueError):
        PipelineConfig(execution_signal_lag_days=-1)
    with pytest.raises(ValueError):
        PipelineConfig(rank_relevance_levels=1)
    with pytest.raises(ValueError):
        PipelineConfig(rank_eval_at=[])


def test_max_active_share_ceiling_default_none():
    assert PipelineConfig().max_active_share_ceiling is None


def test_max_active_share_ceiling_out_of_range_raises():
    with pytest.raises(ValueError):
        PipelineConfig(max_active_share_ceiling=0.0)
    with pytest.raises(ValueError):
        PipelineConfig(max_active_share_ceiling=2.5)
    # None (default) and in-range are accepted.
    assert PipelineConfig(max_active_share_ceiling=0.5).max_active_share_ceiling == 0.5


def test_derive_max_active_share_core_satellite_two_way():
    # core_satellite: cap = 2*satellite_budget, BOTH directions (unlike __post_init__
    # which only tightens). 0.45 baseline -> loosen on sb up, tighten on sb down.
    assert PipelineConfig.derive_max_active_share(0.30, "core_satellite", 0.45) == 0.60
    assert PipelineConfig.derive_max_active_share(0.10, "core_satellite", 0.45) == 0.20


def test_derive_max_active_share_non_core_passthrough():
    assert PipelineConfig.derive_max_active_share(0.30, "unconstrained", 0.50) == 0.50


def test_derive_max_active_share_respects_ceiling():
    assert PipelineConfig.derive_max_active_share(
        0.30, "core_satellite", 0.45, ceiling=0.50) == 0.50
    # ceiling above the implied cap is inert.
    assert PipelineConfig.derive_max_active_share(
        0.30, "core_satellite", 0.45, ceiling=0.80) == 0.60


def test_post_init_max_active_share_parity():
    # Default core_satellite, satellite_budget=0.225 -> cs_l1=0.45; 0.50>0.45 ->
    # tightened to 0.45. Must stay identical after the __post_init__ refactor.
    assert PipelineConfig().max_active_share == 0.45


def test_operating_guardrail_defaults():
    c = PipelineConfig()
    assert c.allow_scs_on_ecos_exception is False
    assert c.min_model_trees == 10
    assert c.max_degenerate_model_rate == 0.25
    assert c.fail_on_degenerate_model_rate is False
    assert c.max_tail_ffill_days == 10
    assert c.fail_on_stale_tail_ffill is False
    assert c.max_name_active_risk_share == 0.35
    assert c.max_sector_active_risk_share == 0.75


def test_operating_guardrail_validation():
    with pytest.raises(ValueError):
        PipelineConfig(min_model_trees=0)
    with pytest.raises(ValueError):
        PipelineConfig(max_degenerate_model_rate=1.1)
    with pytest.raises(ValueError):
        PipelineConfig(max_tail_ffill_days=-1)
    with pytest.raises(ValueError):
        PipelineConfig(max_name_active_risk_share=0.0)
    with pytest.raises(ValueError):
        PipelineConfig(max_sector_active_risk_share=0.0)
