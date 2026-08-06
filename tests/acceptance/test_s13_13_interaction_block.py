# -*- coding: utf-8 -*-
"""S13.13 interaction block — arm contract.

Value semantics live in tests/test_interaction_features.py. This file pins
the arm wiring: default-OFF, admitted through the same extra_whitelist path
as S8/S13.9/S13.10, and a variant that differs from production by exactly
the one flag.

The arm's premise is "zero new information axes": every parent of every
interaction must ALREADY be in CORE_FEATURE_WHITELIST, and no interaction
may already be there — otherwise the block would smuggle in a new axis or
overstate its delta.
"""

import inspect
from pathlib import Path

import yaml

from src.config import INTERACTION_FEATURES, PipelineConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ARM_YAML = _REPO_ROOT / "variants" / "arm_s13_13_interaction_block.yaml"
_PROD_YAML = _REPO_ROOT / "variants" / "codex_causal_rank_65.yaml"


def test_flag_is_default_off():
    assert PipelineConfig().interaction_features_enabled is False


def test_block_is_four_products_plus_consistency_and_matches_the_builder():
    """§S13.13 amendment (2026-07-27): user widened the block to all five
    structurally-passing candidates BEFORE any arm result was observed."""
    from src.features.interactions import INTERACTION_PARENTS

    assert len(INTERACTION_FEATURES) == 5
    assert set(INTERACTION_FEATURES) == set(INTERACTION_PARENTS) | {"mom_consistency_252"}


def test_every_parent_is_already_core_and_no_product_is():
    from src.features.assembly import CORE_FEATURE_WHITELIST
    from src.features.interactions import INTERACTION_PARENTS

    parents = {p for pair in INTERACTION_PARENTS.values() for p in pair}
    not_core = parents - CORE_FEATURE_WHITELIST
    assert not not_core, f"parent outside core whitelist breaks the zero-new-axis premise: {not_core}"

    overlap = set(INTERACTION_FEATURES) & CORE_FEATURE_WHITELIST
    assert not overlap, overlap
    assert "mom_consistency_252" not in CORE_FEATURE_WHITELIST


def test_assembly_builds_and_gates_the_block():
    from src.features import assembly

    source = inspect.getsource(assembly.build_all_features)
    assert "build_interaction_features" in source
    assert "interaction_features_enabled" in source
    assert "INTERACTION_FEATURES" in source


def test_arm_variant_differs_from_production_by_exactly_the_flag():
    arm = yaml.safe_load(_ARM_YAML.read_text(encoding="utf-8"))["overrides"]
    prod = yaml.safe_load(_PROD_YAML.read_text(encoding="utf-8"))["overrides"]

    assert arm["interaction_features_enabled"] is True
    delta = {k: v for k, v in arm.items() if prod.get(k, "<absent>") != v}
    assert set(delta) == {"interaction_features_enabled"}, delta
    post_arm_production_flags = {
        "fwd_sales_slope_features_enabled",
        "vol_quality_tilt_enabled",
        "vol_quality_tilt_lambda",
    }
    assert not [
        k for k in prod
        if k not in arm and k not in post_arm_production_flags
    ]


def test_arm_variant_keys_are_all_real_config_fields():
    from dataclasses import fields

    known = {f.name for f in fields(PipelineConfig)}
    arm = yaml.safe_load(_ARM_YAML.read_text(encoding="utf-8"))["overrides"]
    assert not [k for k in arm if k not in known]
    assert PipelineConfig(**arm).interaction_features_enabled is True
