# -*- coding: utf-8 -*-
"""S13.10 peer earnings cascade — arm contract.

Value semantics live in tests/test_peer_earnings.py. This file pins the arm
wiring: default-OFF, admitted through the same extra_whitelist path as S8/S13.4,
and a variant that differs from production by exactly the one flag.

The arm's premise is that every existing core feature is a single-stock
attribute, so the whitelist must not already contain a relational feature —
otherwise the "first relational block" claim is false.
"""

import inspect
from pathlib import Path

import yaml

from src.config import PEER_EARNINGS_FEATURES, PipelineConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ARM_YAML = _REPO_ROOT / "variants" / "arm_s13_10_peer_earnings.yaml"
_PROD_YAML = _REPO_ROOT / "variants" / "codex_causal_rank_65.yaml"


def test_flag_is_default_off():
    assert PipelineConfig().peer_earnings_cascade_feature_enabled is False


def test_block_is_three_relational_features():
    assert len(PEER_EARNINGS_FEATURES) == 3
    from src.features import peer_earnings

    source = inspect.getsource(peer_earnings)
    for name in PEER_EARNINGS_FEATURES:
        assert f'features["{name}"]' in source


def test_none_of_the_block_is_already_in_the_core_whitelist():
    from src.features.assembly import CORE_FEATURE_WHITELIST

    overlap = set(PEER_EARNINGS_FEATURES) & CORE_FEATURE_WHITELIST
    assert not overlap, overlap


def test_surprise_sheet_is_not_a_source():
    """S13.4 sank on Factset_EPS_Surprise's bank coverage gap.

    Re-importing it here would confound this arm with that defect, so the
    module must reach only for Earnings_Timeline / returns / sector meta.
    """
    from src.features import peer_earnings

    source = inspect.getsource(peer_earnings)
    assert "Factset_EPS_Surprise" not in source.split('"""', 2)[2]
    assert "earnings_timeline" in source
    assert "returns" in source


def test_assembly_builds_and_gates_the_block():
    from src.features import assembly

    source = inspect.getsource(assembly.build_all_features)
    assert "build_peer_earnings_features" in source
    assert "peer_earnings_cascade_feature_enabled" in source
    assert "PEER_EARNINGS_FEATURES" in source


def test_arm_variant_differs_from_production_by_exactly_the_flag():
    arm = yaml.safe_load(_ARM_YAML.read_text(encoding="utf-8"))["overrides"]
    prod = yaml.safe_load(_PROD_YAML.read_text(encoding="utf-8"))["overrides"]

    assert arm["peer_earnings_cascade_feature_enabled"] is True
    delta = {k: v for k, v in arm.items() if prod.get(k, "<absent>") != v}
    assert set(delta) == {"peer_earnings_cascade_feature_enabled"}, delta
    # These settings were adopted into production after this historical arm.
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
    assert PipelineConfig(**arm).peer_earnings_cascade_feature_enabled is True
