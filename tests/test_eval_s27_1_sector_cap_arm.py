# -*- coding: utf-8 -*-
"""§S27.1 섹터 리스크 몫 상한 arm 판정 스크립트의 순수 함수 테스트 (결정 로그 §S27.1 사전등록 규칙 핀)."""

import numpy as np

from scripts.eval_s27_1_sector_cap_arm import (
    ARM,
    BASE,
    FLAG,
    G1_COMPLIANCE_MIN,
    NAME_GATE,
    REFERENCE_BASE_IR,
    SECTOR_CAP,
    TURNOVER_MAX,
    book_risk_shares,
    evaluate_g1,
)


def test_arm_pins():
    assert ARM == "s27_1_sector_risk_cap" and BASE == "s27_om_pit_restamp" and FLAG == "sector_risk_share_cap_enabled"
    assert REFERENCE_BASE_IR == 1.708662580837287
    assert SECTOR_CAP == 0.85 and NAME_GATE == 0.36 and G1_COMPLIANCE_MIN == 0.90 and TURNOVER_MAX == 1.25


def test_book_risk_shares_matches_the_gate_definition():
    cov = np.diag([4.0, 1.0, 1.0, 1.0])
    bm = np.full(4, 0.25)
    w = np.array([0.45, 0.25, 0.15, 0.15])                # a = [+0.2, 0, -0.1, -0.1]
    top_sector, sector, max_name = book_risk_shares(w, bm, cov, ["A", "B", "C", "D"], {"A": "Tech", "B": "Tech", "C": "Fin"})
    # var = 4*0.04 + 0.01 + 0.01 = 0.18; Tech share = 0.16/0.18; D unmapped -> "Unknown" like the gate export
    assert sector == "Tech" and abs(top_sector - 0.16 / 0.18) < 1e-12 and abs(max_name - 0.16 / 0.18) < 1e-12


def test_g1_needs_compliance_rate_live_book_and_name_gate_not_worse():
    base = [("d1", 0.80, 0.30), ("d2", 0.90, 0.30), ("d3", 0.92, 0.33)]
    ok = evaluate_g1([("d1", 0.80, 0.30), ("d2", 0.84, 0.31), ("d3", 0.85, 0.34)], base)
    assert ok["g1_pass"] is True and ok["sector_compliance_rate"] == 1.0 and ok["base_sector_compliance_rate"] == round(1 / 3, 4)
    live_breach = evaluate_g1([("d1", 0.80, 0.30), ("d2", 0.84, 0.31), ("d3", 0.86, 0.34)], base)
    assert live_breach["sector_live_ok"] is False and live_breach["g1_pass"] is False
    name_worse = evaluate_g1([("d1", 0.80, 0.30), ("d2", 0.84, 0.31), ("d3", 0.85, 0.37)], base)
    assert name_worse["name_live_ok"] is False and name_worse["g1_pass"] is False
    low_rate = evaluate_g1([("d%d" % i, 0.9 if i < 2 else 0.8, 0.3) for i in range(10)] + [("dz", 0.8, 0.3)],
                           [("d%d" % i, 0.8, 0.3) for i in range(11)])
    assert low_rate["sector_compliance_rate"] < G1_COMPLIANCE_MIN and low_rate["g1_pass"] is False
