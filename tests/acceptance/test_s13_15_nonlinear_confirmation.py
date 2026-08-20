# -*- coding: utf-8 -*-
"""S13.15 nonlinear confirmation challenger contract."""

import inspect
from dataclasses import fields
from pathlib import Path

import yaml

from src.config import NONLINEAR_CONFIRMATION_FEATURES, PipelineConfig


_REPO_ROOT = Path(__file__).resolve().parents[2]
_ARM_YAML = _REPO_ROOT / "variants" / "arm_s13_15_nonlinear_confirmation.yaml"
_PROD_YAML = _REPO_ROOT / "variants" / "codex_causal_rank_65.yaml"


def test_flag_is_default_off():
    assert PipelineConfig().nonlinear_confirmation_features_enabled is False


def test_block_matches_builder_and_has_six_features():
    from src.features.nonlinear_confirmation import CONFIRMATION_PARENTS

    assert len(NONLINEAR_CONFIRMATION_FEATURES) == 6
    assert set(NONLINEAR_CONFIRMATION_FEATURES) == (
        set(CONFIRMATION_PARENTS) | {"nl_trend_efficiency_252"}
    )


def test_every_confirmation_parent_is_already_core():
    from src.features.assembly import CORE_FEATURE_WHITELIST
    from src.features.nonlinear_confirmation import CONFIRMATION_PARENTS

    parents = {parent for pair in CONFIRMATION_PARENTS.values() for parent in pair}
    assert not (parents - CORE_FEATURE_WHITELIST)
    assert not (set(NONLINEAR_CONFIRMATION_FEATURES) & CORE_FEATURE_WHITELIST)


def test_assembly_builds_and_gates_the_block():
    from src.features import assembly

    source = inspect.getsource(assembly.build_all_features)
    assert "build_nonlinear_confirmation_features" in source
    assert "nonlinear_confirmation_features_enabled" in source
    assert "NONLINEAR_CONFIRMATION_FEATURES" in source


def test_block_is_built_after_lean_momentum_composites():
    """nl_mom_accel_confirm needs mom_accel_63_252 to exist first."""
    from src.features import assembly

    source = inspect.getsource(assembly.build_all_features)
    assert source.index("mom_extras = build_lean_momentum_composites") < source.index(
        "nonlinear_confirmation = build_nonlinear_confirmation_features"
    )


def test_arm_variant_differs_from_production_by_exactly_the_flag():
    arm = yaml.safe_load(_ARM_YAML.read_text(encoding="utf-8"))["overrides"]
    prod = yaml.safe_load(_PROD_YAML.read_text(encoding="utf-8"))["overrides"]

    assert arm["nonlinear_confirmation_features_enabled"] is True
    delta = {key: value for key, value in arm.items() if prod.get(key, "<absent>") != value}
    # S13.47 promotion (2026-08-20): production rank_eval_at [5, 10] -> [20];
    # historical arms pin the pre-promotion value.
    assert set(delta) - {"rank_eval_at"} == {"nonlinear_confirmation_features_enabled"}, delta
    post_arm_production_flags = {
        "fwd_sales_slope_features_enabled",
        "vol_quality_tilt_enabled",
        "vol_quality_tilt_lambda",
        "option_vol_covariance_enabled",  # S13.41 promotion (2026-08-13)
    }
    assert not [
        key for key in prod
        if key not in arm and key not in post_arm_production_flags
    ]


def test_arm_variant_keys_are_real_config_fields():
    known = {field.name for field in fields(PipelineConfig)}
    arm = yaml.safe_load(_ARM_YAML.read_text(encoding="utf-8"))["overrides"]
    assert not [key for key in arm if key not in known]
    assert PipelineConfig(**arm).nonlinear_confirmation_features_enabled is True
