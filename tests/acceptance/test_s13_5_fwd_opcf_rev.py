# -*- coding: utf-8 -*-
"""§S13.5 fwd_opcf 리비전 arm 3종(63d/126d/252d) 수용 테스트.

사용자 지시(2026-07-28)에 따른 3윈도우 동시 시험 — 불변식 4의 단일 파라미터
원칙과 충돌하므로 결정 로그 §S13.5에 사용자 지시 스윕으로 선언하고, 채택은
게이트 + 3-trial DSR 해킷 통과 시에만 허용(최대-IR 선택 금지).

구현 계약 (S13.4/S8 청사진 동일):
  * config.py — default-OFF 플래그 3종:
      fwd_opcf_rev_63d_feature_enabled / fwd_opcf_rev_126d_feature_enabled /
      fwd_opcf_rev_252d_feature_enabled
  * sellside.py — 무조건 빌드:
      fwd_opcf_rev_{63,126,252}d = safe_pct_change(Factset_Fwd_OpCashflow, N)
      (tg_mom 관용구 — |기저값| 분모로 부호 보존, 0 -> NaN)
  * assembly.py — 플래그별 extra_whitelist 승인. 전 OFF -> 바이트 동일.
  * variants/arm_s13_5{a,b,c}_*.yaml — S0(200) 정본 + 플래그 1개.
"""

import inspect
from pathlib import Path

import pandas as pd

from src.config import PipelineConfig

ARMS = [
    ("fwd_opcf_rev_63d_feature_enabled", "fwd_opcf_rev_63d"),
    ("fwd_opcf_rev_126d_feature_enabled", "fwd_opcf_rev_126d"),
    ("fwd_opcf_rev_252d_feature_enabled", "fwd_opcf_rev_252d"),
]
NEW_KEYS = {key for _, key in ARMS}

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VARIANT_YAMLS = {
    "fwd_opcf_rev_63d_feature_enabled":
        _REPO_ROOT / "variants" / "arm_s13_5a_fwd_opcf_rev63.yaml",
    "fwd_opcf_rev_126d_feature_enabled":
        _REPO_ROOT / "variants" / "arm_s13_5b_fwd_opcf_rev126.yaml",
    "fwd_opcf_rev_252d_feature_enabled":
        _REPO_ROOT / "variants" / "arm_s13_5c_fwd_opcf_rev252.yaml",
}

_WL_MEMBERS = ["beta_63d", "momentum_252d", "eps_rev"]


def _tiny_df():
    return pd.DataFrame({"x": [0.0, 1.0]})


def _synthetic_panel():
    names = _WL_MEMBERS + sorted(NEW_KEYS)
    features = {n: _tiny_df() for n in names}
    feature_groups = {
        "Price": ["beta_63d", "momentum_252d"],
        "Sellside": ["eps_rev"] + sorted(NEW_KEYS),
    }
    return features, feature_groups


def test_config_flags_default_off():
    cfg = PipelineConfig()
    for flag, _ in ARMS:
        assert hasattr(cfg, flag), f"PipelineConfig missing S13.5 flag {flag!r}"
        assert getattr(cfg, flag) is False


def test_off_parity_prunes_rev_keys():
    from src.features.assembly import apply_core_filter, CORE_FEATURE_WHITELIST

    for key in NEW_KEYS:
        assert key not in CORE_FEATURE_WHITELIST
    features, feature_groups = _synthetic_panel()
    original = set(features.keys())

    apply_core_filter(features, feature_groups)  # OFF path

    assert set(features.keys()) == (original & set(CORE_FEATURE_WHITELIST))
    for key in NEW_KEYS:
        assert key not in features


def test_on_adds_exactly_one_key_each():
    from src.features.assembly import apply_core_filter, CORE_FEATURE_WHITELIST

    for _, key in ARMS:
        features, feature_groups = _synthetic_panel()
        off_survivors = set(features.keys()) & set(CORE_FEATURE_WHITELIST)

        apply_core_filter(features, feature_groups, extra_whitelist={key})

        assert set(features.keys()) == off_survivors | {key}
        for o in NEW_KEYS - {key}:
            assert o not in features


def test_assembly_flag_wiring():
    import src.features.assembly as assembly

    src_text = inspect.getsource(assembly)
    for flag, key in ARMS:
        assert flag in src_text, f"assembly wiring missing for {flag}"
        assert key in src_text, f"assembly wiring missing key {key}"


def test_flags_not_cache_safe():
    import run_variant

    src_text = inspect.getsource(run_variant.run)
    marker = "SAFE_FOR_CACHE_REUSE = frozenset({"
    start = src_text.index(marker)
    region = src_text[start:src_text.index("})", start)]
    assert "cov_lookback" in region  # positive control
    for flag, _ in ARMS:
        assert flag not in region


def test_variant_yamls_load_and_pin_single_flag():
    import run_variant

    all_flags = {flag for flag, _ in ARMS}
    # S13.4 플래그와도 상호 배타 — 어떤 arm도 타 arm 플래그를 켜지 않는다.
    s13_4_flags = {
        "eps_surprise_feature_enabled",
        "sales_surprise_feature_enabled",
        "fwd_opcf_feature_enabled",
    }
    for flag, path in _VARIANT_YAMLS.items():
        assert path.exists(), f"variant manifest missing: {path}"
        manifest = run_variant.load_manifest(path)
        overrides = manifest.get("overrides") or {}

        valid_fields = run_variant._valid_config_fields()
        assert set(overrides.keys()) <= valid_fields
        assert overrides.get(flag) is True, f"{path.name} must set {flag}: true"
        for other in (all_flags | s13_4_flags) - {flag}:
            assert other not in overrides, (
                f"{path.name} must pin exactly ONE arm flag (found {other})"
            )
        assert overrides.get("expected_universe_size") == 200
