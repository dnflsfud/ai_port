# -*- coding: utf-8 -*-
"""§S16.5-P 사전점검 헬퍼 단위테스트 (경량 — 데이터 로드·재해 없음)."""
import numpy as np

from scripts.precheck_s16_5 import (
    evaluate_p1,
    evaluate_p3,
    lambda_star_turnover_rule,
)


def test_evaluate_p1_gate_on_gap2():
    assert evaluate_p1(gap2_median_pct=6.0)["p1_pass"] is True
    assert evaluate_p1(gap2_median_pct=4.9)["p1_pass"] is False
    assert evaluate_p1(gap2_median_pct=float("nan"))["p1_pass"] is False


def test_evaluate_p3_gates():
    p3 = evaluate_p3(as_drift_max_pp=0.02, te_ratio_median=0.8, all_optimal=True)
    assert p3["p3_pass"] is True
    # active share 붕괴
    assert evaluate_p3(0.05, 0.8, True)["p3_pass"] is False
    # TE 무음 붕괴 (bm 방향)
    assert evaluate_p3(0.02, 0.4, True)["p3_pass"] is False
    # 해 실패
    assert evaluate_p3(0.02, 0.8, False)["p3_pass"] is False


def test_lambda_star_turnover_rule():
    # λ* = median( tp*L1 / risk ) — 리스크 항이 turnover 항과 같아지는 값
    tp = 0.03
    turnover_l1 = [0.10, 0.20, 0.30]
    risk = [1e-6, 2e-6, 3e-6]           # 일간 분산 스케일
    lam = lambda_star_turnover_rule(tp, turnover_l1, risk)
    assert np.isclose(lam, np.median([0.03 * t / r for t, r in zip(turnover_l1, risk)]))
    assert lam > 0
