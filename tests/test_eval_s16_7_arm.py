"""§S16.7 판정 헬퍼 단위테스트 (경량 — 데이터 로드 없음)."""
import numpy as np

from scripts.eval_s16_7_arm import (
    CAP,
    TOL,
    euler_shares,
    evaluate_g0,
    evaluate_g1,
    evaluate_g2,
)


def test_euler_shares_sum_to_one():
    bm = np.array([0.25, 0.25, 0.25, 0.25])
    w = np.array([0.40, 0.20, 0.20, 0.20])
    cov = np.diag([0.02, 0.01, 0.01, 0.01])
    s = euler_shares(w, bm, cov)
    assert abs(float(np.sum(s)) - 1.0) < 1e-12
    assert s[0] > CAP  # 단일 집중 북 — 지배 종목 몫이 캡 초과


def test_evaluate_g0():
    base = {"data_vintage": {"a": 1}, "metrics": {"avg_ic": 0.018181}}
    arm_ok = {"data_vintage": {"a": 1}, "metrics": {"avg_ic": 0.018181}}
    arm_bad = {"data_vintage": {"a": 2}, "metrics": {"avg_ic": 0.018182}}
    g0 = evaluate_g0(base, arm_ok)
    assert g0["vintage_equal"] and g0["avg_ic_bit_identical"] and g0["g0_pass"]
    g0b = evaluate_g0(base, arm_bad)
    assert not g0b["g0_pass"]


def test_evaluate_g1():
    ok_shares = {f"d{i}": 0.30 for i in range(10)}
    g1 = evaluate_g1(ok_shares)
    assert g1["g1_pass"] and g1["compliance_rate"] == 1.0
    # 초반 2/10 breach (준수율 0.8 < 0.9) → FAIL
    mixed = dict(ok_shares)
    mixed["d0"] = 0.50
    mixed["d1"] = 0.45
    assert not evaluate_g1(mixed)["g1_pass"]
    # 라이브 북(마지막) breach → 준수율 0.9라도 FAIL
    last_bad = {f"d{i}": 0.30 for i in range(9)}
    last_bad["d9"] = CAP + TOL + 0.05
    assert not evaluate_g1(last_bad)["g1_pass"]


def test_evaluate_g2():
    g2 = evaluate_g2(arm_te=0.037, base_as=0.20, arm_as=0.19,
                     turnover_ratio=1.05, arm_fallback_rate=0.0,
                     degenerate_equal=True)
    assert all(g2.values())
    g2b = evaluate_g2(arm_te=0.046, base_as=0.20, arm_as=0.15,
                      turnover_ratio=1.5, arm_fallback_rate=0.01,
                      degenerate_equal=False)
    assert not any(g2b.values())
