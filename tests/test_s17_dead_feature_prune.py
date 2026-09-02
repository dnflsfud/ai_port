# -*- coding: utf-8 -*-
"""§S17 M4 — 모델이 구조적으로 소비하지 못하는 코어 피처 9개의 배제
(s17_dead_feature_prune_enabled).

결함(결정 로그 §S17 P4): 코어 65 중 (a) 날짜별 상수(bcast) 7개 — cal_is_Q1·
regime_mkt_ret_21d·fac_*(5) — 는 날짜 그룹 rank_xendcg 에서 쿼리 내 상수라 33/33 모델
split 0(§S13.18 기전); (b) fin_roe_level_z ≡ best_roe_level_z, fin_pb_level_z ≡
best_px_bps_ratio_level_z 는 코드 동일식(820,750셀 max|diff| 0.0). EWMA 드롭 예산
(n_drop=3)이 이 죽은 피처에만 소진되어 선택층이 inert 하다.

수정 계약: ON 이면 core filter 에서 이 9개를 제외한다(best_* 사본 유지 —
interactions.py 부모 참조). OFF(기본)는 `exclude=None` 으로 레거시와 바이트 동일.
부수 기록: 56개가 되면 n_drop=2 < ewma_min_features(60) 여유라 드롭 0 — 선택층은
여전히 inert(별도 사전등록 대상, 이 arm 의 범위 아님).
"""

import inspect

import numpy as np
import pandas as pd

from src.config import PipelineConfig

DEAD_EXPECTED = {
    "cal_is_Q1", "regime_mkt_ret_21d",
    "fac_yield_slope", "fac_F_Quality_mom_63d", "fac_F_Growth_mom_63d",
    "fac_F_Value_mom_63d", "fac_value_growth_63d",
    "fin_roe_level_z", "fin_pb_level_z",
}
LIVE = ["beta_63d", "momentum_252d", "eps_rev", "best_roe_level_z"]
DUMMIES = ["dummy_junk_a", "dummy_junk_b"]


def _synthetic_panel():
    tiny = pd.DataFrame({"x": [0.0, 1.0]})
    names = LIVE + sorted(DEAD_EXPECTED) + DUMMIES
    features = {n: tiny.copy() for n in names}
    groups = {
        "Price": ["beta_63d", "momentum_252d"],
        "Sellside": ["eps_rev"],
        "Accounting": ["best_roe_level_z", "fin_roe_level_z", "fin_pb_level_z"],
        "Factor": [n for n in sorted(DEAD_EXPECTED) if n.startswith("fac_")],
        "Regime": ["regime_mkt_ret_21d", "cal_is_Q1"],
        "Junk": list(DUMMIES),
    }
    return features, groups


def test_s17_dead_feature_prune_flag_default_off():
    assert PipelineConfig().s17_dead_feature_prune_enabled is False


def test_dead_feature_set_is_the_nine_measured_members_of_the_whitelist():
    from src.features.assembly import CORE_FEATURE_WHITELIST, S17_DEAD_FEATURES
    assert set(S17_DEAD_FEATURES) == DEAD_EXPECTED
    assert set(S17_DEAD_FEATURES) <= set(CORE_FEATURE_WHITELIST)


def test_dead_feature_exclusions_none_when_off_and_set_when_on():
    from src.features.assembly import S17_DEAD_FEATURES, dead_feature_exclusions
    assert dead_feature_exclusions(PipelineConfig()) is None
    assert dead_feature_exclusions(None) is None
    assert dead_feature_exclusions(
        PipelineConfig(s17_dead_feature_prune_enabled=True)) == S17_DEAD_FEATURES


def test_apply_core_filter_exclude_none_is_legacy_behavior():
    from src.features.assembly import CORE_FEATURE_WHITELIST, apply_core_filter
    f_legacy, g_legacy = _synthetic_panel()
    f_none, g_none = _synthetic_panel()
    apply_core_filter(f_legacy, g_legacy)
    apply_core_filter(f_none, g_none, exclude=None)
    assert set(f_none) == set(f_legacy) == (set(LIVE) | DEAD_EXPECTED) & set(CORE_FEATURE_WHITELIST)
    assert g_none == g_legacy


def test_apply_core_filter_exclude_prunes_exactly_the_dead_features():
    from src.features.assembly import S17_DEAD_FEATURES, apply_core_filter
    f_legacy, _ = _synthetic_panel()
    apply_core_filter(f_legacy, _synthetic_panel()[1])
    features, groups = _synthetic_panel()
    apply_core_filter(features, groups, exclude=S17_DEAD_FEATURES)
    assert set(features) == set(f_legacy) - DEAD_EXPECTED
    assert "best_roe_level_z" in features                 # best_* 사본은 유지
    assert groups.get("Accounting") == ["best_roe_level_z"]
    assert "Factor" not in groups and "Regime" not in groups   # 빈 그룹 제거
    assert groups.get("Price") == ["beta_63d", "momentum_252d"]


def test_build_all_features_wires_exclusions_from_config():
    from src.features import assembly
    src = inspect.getsource(assembly.build_all_features)
    assert "exclude=dead_feature_exclusions(config)" in src


def test_ewma_drop_budget_arithmetic_pinned():
    """결정 로그 기록용 특성화: 65 → 3 드롭(죽은 피처만), 56 → 0 드롭(선택층 inert)."""
    from src.model_trainer import EWMAFeatureTracker
    cfg = PipelineConfig(ewma_enabled=True, ewma_min_retrains=2,
                         ewma_drop_pct=0.05, ewma_min_features=60)
    names65 = [f"live_{i}" for i in range(56)] + sorted(DEAD_EXPECTED)
    t = EWMAFeatureTracker(cfg)
    t.init_full_features(names65)
    imp = np.ones(65)
    imp[56:] = 1e-6                     # 죽은 9개 최하위
    t.ewma_importance = imp / imp.sum()
    t.n_updates = 2
    active = t.get_active_features(names65)
    assert len(active) == 62
    assert set(names65) - set(active) <= DEAD_EXPECTED
    names56 = names65[:56]
    t56 = EWMAFeatureTracker(cfg)
    t56.init_full_features(names56)
    t56.ewma_importance = np.linspace(1.0, 2.0, 56) / np.linspace(1.0, 2.0, 56).sum()
    t56.n_updates = 2
    assert t56.get_active_features(names56) == names56
