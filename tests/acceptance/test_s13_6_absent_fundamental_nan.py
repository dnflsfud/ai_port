# -*- coding: utf-8 -*-
"""§S13.6 구조적 결측 펀더멘털 NaN 보존 arm 수용 테스트.

은행권은 BEST_CALCULATED_FCF / BEST_CAPEX / BEST_EV_TO_BEST_EBITDA /
BEST_GROSS_MARGIN 컬럼이 워크북에 아예 없다. 현재 assembly 패널은 그 칸을
per-date 횡단면 median 으로 채워 17종에 "정확히 시장 중앙값"을 매일 주장한다.

구현 계약:
  * config.py — default-OFF 플래그 absent_fundamental_nan_enabled,
    ABSENT_FUNDAMENTAL_SHEET_FEATURES(4시트 -> 8피처), NAN_TOLERANT_FEATURES.
  * assembly.py — apply_absent_fundamental_nan 이 median fill 직후 실측 부재
    (티커 × 피처) 칸만 되돌린다. OFF -> 무연산(바이트 동일).
  * model_trainer.py — _valid_rows 가 관용 피처의 NaN 은 남기고 나머지 필수
    피처의 NaN 행만 제거한다. 이게 없으면 17종이 학습에서 통째로 탈락하면서
    predict_cross_sectional 은 계속 점수를 매기는 비대칭이 생긴다.
  * variants/arm_s13_6_absent_fundamental_nan.yaml — S0(200) 정본 + 플래그 1개.

값 계약은 tests/test_absent_fundamental_nan.py 가 담당한다.
"""

import inspect
from pathlib import Path

import yaml

from src.config import (
    ABSENT_FUNDAMENTAL_SHEET_FEATURES,
    NAN_TOLERANT_FEATURES,
    PipelineConfig,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ARM_YAML = _REPO_ROOT / "variants" / "arm_s13_6_absent_fundamental_nan.yaml"
_PROD_YAML = _REPO_ROOT / "variants" / "codex_causal_rank_65.yaml"


def test_flag_is_default_off():
    assert PipelineConfig().absent_fundamental_nan_enabled is False


def test_masked_features_are_all_in_core_whitelist():
    """마스킹 대상 8개는 전부 실제 프로덕션 피처여야 의미가 있다."""
    from src.features import assembly

    source = inspect.getsource(assembly)
    start = source.index("CORE_FEATURE_WHITELIST")
    block = source[start:source.index("\n}", start)]
    for name in NAN_TOLERANT_FEATURES:
        assert f'"{name}"' in block, f"{name} is not in CORE_FEATURE_WHITELIST"


def test_out_of_scope_optional_sheets_are_not_masked():
    """이번 범위는 은행권 구조적 부재 4시트뿐. PEG/PX_BPS 는 단일 티커 저커버리지."""
    assert "BEST_PEG_RATIO" not in ABSENT_FUNDAMENTAL_SHEET_FEATURES
    assert "BEST_PX_BPS_RATIO" not in ABSENT_FUNDAMENTAL_SHEET_FEATURES


def test_trainer_tolerates_only_the_declared_features():
    from src.model_trainer import _valid_rows

    source = inspect.getsource(_valid_rows)
    assert "NAN_TOLERANT_FEATURES" in source
    # 필수 피처의 NaN 행 제거는 유지되어야 한다.
    assert "notna()" in source


def test_arm_variant_differs_from_production_by_exactly_the_flag():
    arm = yaml.safe_load(_ARM_YAML.read_text(encoding="utf-8"))["overrides"]
    prod = yaml.safe_load(_PROD_YAML.read_text(encoding="utf-8"))["overrides"]

    assert arm["absent_fundamental_nan_enabled"] is True
    delta = {k: v for k, v in arm.items() if prod.get(k, "<absent>") != v}
    assert set(delta) == {"absent_fundamental_nan_enabled"}, delta
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
    from dataclasses import fields

    known = {f.name for f in fields(PipelineConfig)}
    arm = yaml.safe_load(_ARM_YAML.read_text(encoding="utf-8"))["overrides"]
    assert not [k for k in arm if k not in known]
    assert PipelineConfig(**arm).absent_fundamental_nan_enabled is True
