# -*- coding: utf-8 -*-
"""§S13.19a 전수 재검정 스크립트 순수 함수 단위테스트."""
from __future__ import annotations

import pandas as pd

from scripts.preflight_s13_19a_all_spreads import judge_row


def test_judge_row_matches_s13_19_gate_semantics():
    row = pd.Series({"gap_delta": 0.0226, "spec_delta": 0.0001,
                     "gap_delta_H1": 0.0093, "gap_delta_H2": 0.0344})
    g = judge_row(row)
    assert g["G1"] and g["G2"] and g["G3"] and g["verdict"] == "PROCEED"
    # 부호 불일치 → G2 FAIL
    row2 = row.copy()
    row2["gap_delta_H1"] = -0.001
    assert not judge_row(row2)["G2"]
    # 국지화 실패 → G3 FAIL
    row3 = row.copy()
    row3["spec_delta"] = -0.05
    assert not judge_row(row3)["G3"]
    # 문턱 미달 → G1 FAIL
    row4 = row.copy()
    row4["gap_delta"] = 0.01
    assert not judge_row(row4)["G1"]
