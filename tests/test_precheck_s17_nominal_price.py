# -*- coding: utf-8 -*-
"""§S17.3 G1-01b 사전점검 판정 헬퍼 — 데이터 게이트 A~D 의 판정 논리(순수 함수)."""

import numpy as np
import pandas as pd

from scripts.precheck_s17_nominal_price import (
    MO_NOMINAL, N_HIGH_DIV, NO_DIVIDEND, TG_PX_ON_BAND, Z_GAP_ON_MAX,
    _group_median_by_year, evaluate_gates,
)


def test_preregistered_constants():
    assert NO_DIVIDEND == ["TSLA", "AMZN", "ADBE", "NFLX", "ISRG"]
    assert MO_NOMINAL == 41.94
    assert TG_PX_ON_BAND == (1.00, 1.25)
    assert Z_GAP_ON_MAX == 0.5
    assert N_HIGH_DIV == 12


def test_evaluate_gates_all_pass_only_when_every_gate_holds():
    ok = evaluate_gates(True, True, 1.10, 0.20)
    assert ok == {
        "A_no_dividend_identical": True,
        "B_mo_2014_06_30_nominal": True,
        "C_high_div_tg_px_2014_on_in_band": True,
        "D_high_div_z_gap_on_below_half": True,
    }
    assert evaluate_gates(False, True, 1.10, 0.20)["A_no_dividend_identical"] is False
    assert evaluate_gates(True, False, 1.10, 0.20)["B_mo_2014_06_30_nominal"] is False
    # C: 밴드 경계 포함, 밖이면 FAIL, NaN 이면 FAIL
    assert evaluate_gates(True, True, 1.25, 0.2)["C_high_div_tg_px_2014_on_in_band"] is True
    assert evaluate_gates(True, True, 1.26, 0.2)["C_high_div_tg_px_2014_on_in_band"] is False
    assert evaluate_gates(True, True, 2.02, 0.2)["C_high_div_tg_px_2014_on_in_band"] is False
    assert evaluate_gates(True, True, float("nan"), 0.2)["C_high_div_tg_px_2014_on_in_band"] is False
    # D: 절대값 기준, 0.5 미만
    assert evaluate_gates(True, True, 1.1, -0.49)["D_high_div_z_gap_on_below_half"] is True
    assert evaluate_gates(True, True, 1.1, 0.5)["D_high_div_z_gap_on_below_half"] is False
    assert evaluate_gates(True, True, 1.1, 1.56)["D_high_div_z_gap_on_below_half"] is False


def test_group_median_by_year_is_median_of_daily_group_medians():
    dates = pd.bdate_range("2014-01-01", periods=40).append(pd.bdate_range("2015-01-01", periods=40))
    panel = pd.DataFrame(
        {"MO": np.r_[np.full(40, 2.0), np.full(40, 1.0)],
         "T": np.r_[np.full(40, 4.0), np.full(40, 1.0)],
         "TSLA": np.r_[np.full(40, 1.1), np.full(40, 1.1)]},
        index=dates,
    )
    assert _group_median_by_year(panel, ["MO", "T"], (2014, 2014)) == 3.0
    assert _group_median_by_year(panel, ["MO", "T"], (2015, 2015)) == 1.0
    assert _group_median_by_year(panel, ["MO", "T"], (2014, 2015)) == 2.0   # 일별 중앙값(3.0×40, 1.0×40)의 중앙값
    # 패널에 없는 종목은 무시, 아무것도 없으면 NaN
    assert _group_median_by_year(panel, ["MO", "ZZZ"], (2014, 2014)) == 2.0
    assert np.isnan(_group_median_by_year(panel, ["ZZZ"], (2014, 2014)))
    assert np.isnan(_group_median_by_year(panel, ["MO"], (2020, 2020)))
