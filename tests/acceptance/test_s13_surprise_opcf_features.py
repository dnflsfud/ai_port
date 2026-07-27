# -*- coding: utf-8 -*-
"""§S13.4 사전등록 피처 arm 3종 수용 테스트 (S8 news_trend 청사진 미러).

구현 계약 (spec):
  * config.py — PipelineConfig에 default-OFF 플래그 3종:
      eps_surprise_feature_enabled / sales_surprise_feature_enabled /
      fwd_opcf_feature_enabled  (모두 False 기본)
  * sellside.py — 무조건 빌드(S8 관용구), 플래그는 whitelist 승인만 제어:
      eps_surprise    = Factset_EPS_Surprise 원 레벨(분기 서프라이즈 %, PEAD)
      sales_surprise  = Factset_Sales_Surprise 원 레벨
      fwd_opcf_yield  = Factset_Fwd_OpCashflow / local_prices (fwd CF yield;
                        가격 0 -> NaN, tg_upside와 동일한 로컬 통화 단위 계약)
  * assembly.py — core filter에서 extra_whitelist로 조건부 승인. 전 플래그
    OFF -> extra None -> 레거시와 바이트 동일 (OFF-parity, 불변식 1).
  * run_variant.py 무변경 — 플래그 3종은 SAFE_FOR_CACHE_REUSE에 없어야 한다
    (피처 패널 변경 = 전체 재실행 강제).
  * variants/arm_s13_4{a,b,c}_*.yaml — codex_causal_rank_65(S0(200) 정본)
    클론 + 해당 플래그 1개만 true. 후보당 단일 사전등록, 스윕 금지.
"""

import inspect
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PipelineConfig

# (flag, feature key, source sheet) — 사전등록 3종.
ARMS = [
    ("eps_surprise_feature_enabled", "eps_surprise", "Factset_EPS_Surprise"),
    ("sales_surprise_feature_enabled", "sales_surprise", "Factset_Sales_Surprise"),
    ("fwd_opcf_feature_enabled", "fwd_opcf_yield", "Factset_Fwd_OpCashflow"),
]
NEW_KEYS = {key for _, key, _ in ARMS}

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VARIANT_YAMLS = {
    "eps_surprise_feature_enabled":
        _REPO_ROOT / "variants" / "arm_s13_4a_eps_surprise.yaml",
    "sales_surprise_feature_enabled":
        _REPO_ROOT / "variants" / "arm_s13_4b_sales_surprise.yaml",
    "fwd_opcf_feature_enabled":
        _REPO_ROOT / "variants" / "arm_s13_4c_fwd_opcf.yaml",
}

_WL_MEMBERS = ["beta_63d", "momentum_252d", "eps_rev"]
_DUMMIES = ["dummy_junk_a", "dummy_junk_b"]


def _tiny_df():
    return pd.DataFrame({"x": [0.0, 1.0]})


def _synthetic_panel():
    names = _WL_MEMBERS + sorted(NEW_KEYS) + _DUMMIES
    features = {n: _tiny_df() for n in names}
    feature_groups = {
        "Price": ["beta_63d", "momentum_252d"],
        "Sellside": ["eps_rev"] + sorted(NEW_KEYS),
        "Junk": list(_DUMMIES),
    }
    return features, feature_groups


def _assert_premise(whitelist):
    for key in NEW_KEYS:
        assert key not in whitelist, (
            f"premise broken: {key!r} must NOT be a CORE_FEATURE_WHITELIST "
            "member (pre-registered as conditional extra only)"
        )
    for m in _WL_MEMBERS:
        assert m in whitelist, f"test fixture stale: {m!r} not in whitelist"


# ---------------------------------------------------------------------------
# 1. default-OFF flags
# ---------------------------------------------------------------------------
def test_config_flags_default_off():
    cfg = PipelineConfig()
    for flag, _, _ in ARMS:
        assert hasattr(cfg, flag), f"PipelineConfig missing S13.4 flag {flag!r}"
        assert getattr(cfg, flag) is False


# ---------------------------------------------------------------------------
# 2. OFF parity — new keys pruned, survivors == whitelist 교집합
# ---------------------------------------------------------------------------
def test_off_parity_prunes_new_keys():
    from src.features.assembly import apply_core_filter, CORE_FEATURE_WHITELIST

    _assert_premise(CORE_FEATURE_WHITELIST)
    features, feature_groups = _synthetic_panel()
    original = set(features.keys())

    apply_core_filter(features, feature_groups)  # OFF path

    assert set(features.keys()) == (original & set(CORE_FEATURE_WHITELIST))
    for key in NEW_KEYS:
        assert key not in features
    assert feature_groups.get("Sellside") == ["eps_rev"]
    assert "Junk" not in feature_groups


# ---------------------------------------------------------------------------
# 3. ON behaviour — 각 extra 키가 정확히 자기 피처 1개만 추가
# ---------------------------------------------------------------------------
def test_on_adds_exactly_one_key_each():
    from src.features.assembly import apply_core_filter, CORE_FEATURE_WHITELIST

    _assert_premise(CORE_FEATURE_WHITELIST)
    for _, key, _ in ARMS:
        features, feature_groups = _synthetic_panel()
        off_survivors = set(features.keys()) & set(CORE_FEATURE_WHITELIST)

        apply_core_filter(features, feature_groups, extra_whitelist={key})

        assert set(features.keys()) == off_survivors | {key}
        others = NEW_KEYS - {key}
        for o in others:
            assert o not in features, (
                f"extra_whitelist={{{key!r}}} must not admit {o!r}"
            )


# ---------------------------------------------------------------------------
# 4. assembly config wiring — 플래그 -> extra_whitelist 매핑
# ---------------------------------------------------------------------------
def test_assembly_flag_wiring():
    """assemble의 extra 구성 로직 계약: 소스 텍스트에 플래그 3종이 모두
    등장하고, news_trend 배선도 유지된다(회귀 가드)."""
    import src.features.assembly as assembly

    src_text = inspect.getsource(assembly)
    for flag, key, _ in ARMS:
        assert flag in src_text, f"assembly wiring missing for {flag}"
        assert key in src_text, f"assembly wiring missing key {key}"
    assert "news_trend_feature_enabled" in src_text  # S8 유지


# ---------------------------------------------------------------------------
# 5. builder 값 계약은 tests/test_sellside.py 로 이관 (TDD 가드 규약).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 6. cache safety — 플래그 3종은 SAFE_FOR_CACHE_REUSE 밖
# ---------------------------------------------------------------------------
def test_flags_not_cache_safe():
    import run_variant

    src_text = inspect.getsource(run_variant.run)
    marker = "SAFE_FOR_CACHE_REUSE = frozenset({"
    assert marker in src_text
    start = src_text.index(marker)
    end = src_text.index("})", start)
    region = src_text[start:end]
    assert "cov_lookback" in region  # positive control
    for flag, _, _ in ARMS:
        assert flag not in region, (
            f"{flag} must NOT be cache-reuse-safe (feature-panel change)"
        )


# ---------------------------------------------------------------------------
# 7. variant yamls — 존재·검증 통과·자기 플래그 1개만 true
# ---------------------------------------------------------------------------
def test_variant_yamls_load_and_pin_single_flag():
    import run_variant

    all_flags = {flag for flag, _, _ in ARMS}
    for flag, path in _VARIANT_YAMLS.items():
        assert path.exists(), f"variant manifest missing: {path}"
        manifest = run_variant.load_manifest(path)
        overrides = manifest.get("overrides") or {}

        valid_fields = run_variant._valid_config_fields()
        assert set(overrides.keys()) <= valid_fields, (
            f"{path.name}: unknown override fields: "
            f"{sorted(set(overrides) - valid_fields)}"
        )
        assert overrides.get(flag) is True, f"{path.name} must set {flag}: true"
        for other in all_flags - {flag}:
            assert other not in overrides, (
                f"{path.name} must pin exactly ONE arm flag (found {other})"
            )
        # 유니버스 가드는 S0(200) 정본과 동일해야 한다.
        assert overrides.get("expected_universe_size") == 200
