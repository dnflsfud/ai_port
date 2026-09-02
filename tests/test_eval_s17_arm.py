# -*- coding: utf-8 -*-
"""§S17.1 arm 판정 스크립트 헬퍼 단위테스트 (경량 — pkl 로드 없음)."""
from scripts.eval_s13_46_arm import E1_DELTA_IR
from scripts.eval_s17_arm import FRAMES, evaluate_e2, evaluate_no_harm


def test_frames_cover_the_four_preregistered_arms():
    assert set(FRAMES) == {"s17_1_coverage_gap_fix", "s17_2_cov_corr_overlap",
                           "s17_3_beta_overlap", "s17_4_dead_feature_prune"}
    assert FRAMES["s17_1_coverage_gap_fix"]["frame"] == "correctness"
    assert FRAMES["s17_3_beta_overlap"]["frame"] == "correctness"
    assert FRAMES["s17_2_cov_corr_overlap"]["frame"] == "risk_discipline"
    assert FRAMES["s17_2_cov_corr_overlap"]["alpha_identical"] is True   # 옵티마이저 전용
    assert FRAMES["s17_4_dead_feature_prune"]["frame"] == "hygiene"


def test_no_harm_frame_is_bar_and_not_all_splits_negative():
    assert E1_DELTA_IR == 0.36
    assert evaluate_no_harm(+0.04, [+0.06, +0.03, +0.02]) is True
    assert evaluate_no_harm(-0.10, [-0.2, +0.1, -0.05]) is True      # 노이즈 밴드·부호 혼재
    assert evaluate_no_harm(-0.10, [-0.2, -0.1, -0.05]) is False     # 3분할 전부 음
    assert evaluate_no_harm(-0.40, [+0.1, +0.1, -0.9]) is False      # 바 초과 손실


def test_evaluate_e2_do_no_harm_bounds():
    ok = evaluate_e2(arm_te=0.037, base_as=0.200, arm_as=0.21, turnover_ratio=1.1,
                     arm_fallback_rate=0.0, turnover_max=1.25)
    assert all(ok.values())
    assert evaluate_e2(0.046, 0.2, 0.21, 1.1, 0.0, 1.25)["te_ok"] is False
    assert evaluate_e2(0.037, 0.2, 0.24, 1.1, 0.0, 1.25)["active_share_ok"] is False
    assert evaluate_e2(0.037, 0.2, 0.21, 1.21, 0.0, 1.20)["turnover_ok"] is False
    assert evaluate_e2(0.037, 0.2, 0.21, 1.1, 0.01, 1.25)["fallback_zero"] is False
