# -*- coding: utf-8 -*-
"""§S16.8 — 단위 수정 후 OCF 재도전 arm 2종 (S13.4 청사진 미러).

  A: fwd_opcf_level_z = cs_z(rolling_tsz(Factset_Fwd_OpCashflow, 756, 252))
     — §S16.1 P2 관용구(통화 스케일 자기정규화 후 횡단면 z).
  B: fwd_opcf_invest_divergence = cs_z(pct_chg(OpCF,252)) − cs_z(pct_chg(FCF,252))
     — 투자 주도 FCF 압축(OCF 성장 지속 + CAPEX 흡수) vs 영업 악화 구분.
무조건 빌드(S8 관용구), 플래그는 whitelist 승인만 제어. default-OFF.
"""
import inspect
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PipelineConfig

ARMS = [
    ("fwd_opcf_level_feature_enabled", "fwd_opcf_level_z"),
    ("fwd_opcf_invest_divergence_feature_enabled", "fwd_opcf_invest_divergence"),
]
NEW_KEYS = {key for _, key in ARMS}

_REPO_ROOT = Path(__file__).resolve().parents[1]
_VARIANT_YAMLS = {
    "fwd_opcf_level_feature_enabled":
        _REPO_ROOT / "variants" / "s16_8a_opcf_level.yaml",
    "fwd_opcf_invest_divergence_feature_enabled":
        _REPO_ROOT / "variants" / "s16_8b_invest_divergence.yaml",
}


class _FakeData:
    def __init__(self, sheets, local_prices):
        self._sheets = dict(sheets)
        self.local_prices = local_prices
        self.prices = local_prices

    def get_sheet(self, name):
        if name not in self._sheets:
            raise KeyError(name)
        return self._sheets[name]


def _long_panel(seed):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    return pd.DataFrame(
        {"AAA": 100 + rng.normal(0, 5, 300).cumsum(),
         "BBB": 5000 + rng.normal(0, 100, 300).cumsum()},
        index=idx,
    )


def test_config_flags_default_off():
    cfg = PipelineConfig()
    for flag, _ in ARMS:
        assert getattr(cfg, flag) is False


def test_builder_level_z_matches_p2_idiom():
    from src.features.sellside import build_sellside_features
    from src.features.utils import cross_sectional_zscore, rolling_tsz

    opcf = _long_panel(1)
    data = _FakeData({"Factset_Fwd_OpCashflow": opcf}, local_prices=_long_panel(2).abs())
    feats = build_sellside_features(data, config=PipelineConfig())

    expected = cross_sectional_zscore(rolling_tsz(opcf, window=756, min_periods=252))
    pd.testing.assert_frame_equal(feats["fwd_opcf_level_z"], expected)
    # FCF 시트 부재 → divergence 만 스킵, level 은 존재
    assert "fwd_opcf_invest_divergence" not in feats


def test_builder_invest_divergence_matches_definition():
    from src.features.sellside import build_sellside_features
    from src.features.utils import cross_sectional_zscore, safe_pct_change

    opcf, fcf = _long_panel(3), _long_panel(4)
    data = _FakeData(
        {"Factset_Fwd_OpCashflow": opcf, "BEST_CALCULATED_FCF": fcf},
        local_prices=_long_panel(5).abs(),
    )
    feats = build_sellside_features(data, config=PipelineConfig())

    expected = (
        cross_sectional_zscore(safe_pct_change(opcf, 252))
        - cross_sectional_zscore(safe_pct_change(fcf, 252))
    )
    pd.testing.assert_frame_equal(feats["fwd_opcf_invest_divergence"], expected)


def test_whitelist_admission_gated_per_flag():
    from src.features.assembly import CORE_FEATURE_WHITELIST, apply_core_filter

    for key in NEW_KEYS:
        assert key not in CORE_FEATURE_WHITELIST
    for _, key in ARMS:
        features = {"beta_63d": pd.DataFrame({"x": [0.0]}),
                    **{k: pd.DataFrame({"x": [0.0]}) for k in NEW_KEYS}}
        groups = {"Price": ["beta_63d"], "Sellside": sorted(NEW_KEYS)}
        apply_core_filter(features, groups, extra_whitelist={key})
        assert key in features
        for other in NEW_KEYS - {key}:
            assert other not in features


def test_assembly_flag_wiring():
    import src.features.assembly as assembly

    src_text = inspect.getsource(assembly)
    for flag, key in ARMS:
        assert flag in src_text
        assert key in src_text


def test_flags_not_cache_safe():
    import run_variant

    src_text = inspect.getsource(run_variant.run)
    start = src_text.index("SAFE_FOR_CACHE_REUSE = frozenset({")
    region = src_text[start:src_text.index("})", start)]
    for flag, _ in ARMS:
        assert flag not in region


def test_variant_yamls_pin_single_flag_on_new_baseline():
    import run_variant

    all_flags = {flag for flag, _ in ARMS}
    for flag, path in _VARIANT_YAMLS.items():
        assert path.exists(), f"variant manifest missing: {path}"
        manifest = run_variant.load_manifest(path)
        overrides = manifest.get("overrides") or {}
        assert set(overrides.keys()) <= run_variant._valid_config_fields()
        assert overrides.get(flag) is True
        for other in all_flags - {flag}:
            assert other not in overrides
        # 비교 기준 = 새 S0′(S16.7 flip) config 여야 한다.
        assert overrides.get("name_risk_share_cap_enabled") is True
        assert overrides.get("expected_universe_size") == 250
