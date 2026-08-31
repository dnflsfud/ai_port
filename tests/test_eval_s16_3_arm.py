# -*- coding: utf-8 -*-
"""§S16.3 E1/E2 판정 스크립트의 순수 헬퍼 테스트."""
from scripts.eval_s16_3_arm import count_fresh_fires, evaluate_e2


def test_e2_all_pass():
    e2 = evaluate_e2(arm_te=0.037, base_as=0.200, arm_as=0.201,
                     turnover_ratio=1.05, arm_fallback_rate=0)
    assert e2 == {"te_ok": True, "active_share_ok": True,
                  "turnover_ok": True, "fallback_zero": True}


def test_e2_individual_guards():
    assert evaluate_e2(0.046, 0.200, 0.201, 1.0, 0)["te_ok"] is False
    assert evaluate_e2(0.037, 0.200, 0.235, 1.0, 0)["active_share_ok"] is False
    assert evaluate_e2(0.037, 0.200, 0.201, 1.26, 0)["turnover_ok"] is False
    assert evaluate_e2(0.037, 0.200, 0.201, 1.0, 0.01)["fallback_zero"] is False


def test_count_fresh_fires_filters_by_fallback_tag():
    events = [
        {"date": "2020-08-03", "fallback": "reuse_prev"},
        {"date": "2021-04-23", "fallback": "fresh_fixed"},
        {"date": "2021-07-21", "fallback": "fresh_fixed"},
        {"date": "2021-10-18"},
    ]
    n, dates = count_fresh_fires(events)
    assert n == 2
    assert dates == ["2021-04-23", "2021-07-21"]


def test_count_fresh_fires_empty():
    assert count_fresh_fires([]) == (0, [])
