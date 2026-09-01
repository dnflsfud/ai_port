# -*- coding: utf-8 -*-
"""§S16.8-A 판정 헬퍼 단위테스트 (경량)."""
from scripts.eval_s16_8a_arm import evaluate_e2, feature_status


def test_evaluate_e2():
    e2 = evaluate_e2(arm_te=0.037, base_as=0.20, arm_as=0.19,
                     turnover_ratio=1.1, arm_fallback_rate=0.0)
    assert all(e2.values())
    e2b = evaluate_e2(arm_te=0.046, base_as=0.20, arm_as=0.16,
                      turnover_ratio=1.3, arm_fallback_rate=0.01)
    assert not any(e2b.values())


def test_feature_status_detects_noop():
    # gain 0 또는 미생존 → arm은 no-op (§S13.21 전례)
    s = feature_status(active=True, gain_pct=1.2)
    assert s["is_noop"] is False
    assert feature_status(active=False, gain_pct=0.0)["is_noop"] is True
    assert feature_status(active=True, gain_pct=0.0)["is_noop"] is True
