# -*- coding: utf-8 -*-
"""§S26 정확성 arm 판정 스크립트의 순수 함수 테스트 (결정 로그 §S26 사전등록 규칙 핀)."""

from scripts.eval_s26_stale_fund_arm import (
    ARM,
    BASE,
    FLAG,
    FUND_MAX_RUN,
    FUND_NAMED_CASES,
    FUND_SHEETS,
    REFERENCE_BASE_IR,
    _FUND_TOKENS,
    mechanism_fund_mask,
)


def test_arm_pins():
    assert ARM == "s26_stale_fund_mask" and FLAG == "stale_run_fund_mask_enabled" and BASE == "s26_s0_base"
    assert REFERENCE_BASE_IR == 1.7737993717667202
    assert FUND_SHEETS == ("BEST_GROSS_MARGIN", "BEST_CALCULATED_FCF", "BEST_CAPEX", "BEST_ROE") and FUND_MAX_RUN == 126
    assert FUND_NAMED_CASES == (("BEST_GROSS_MARGIN", "APO"), ("BEST_CALCULATED_FCF", "C"), ("BEST_CAPEX", "APO"))


def _dq(**over):
    rec = {
        "BEST_GROSS_MARGIN": {"masked_cells": 25000, "tickers": 20, "max_run": 126, "top": {"APO": 2962, "GS": 1080}},
        "BEST_CALCULATED_FCF": {"masked_cells": 15000, "tickers": 12, "max_run": 126, "top": {"APO": 2459, "C": 1523}},
        "BEST_CAPEX": {"masked_cells": 10000, "tickers": 10, "max_run": 126, "top": {"APO": 2969, "PGR": 587}},
        "BEST_ROE": {"masked_cells": 5000, "tickers": 8, "max_run": 126, "top": {"X": 900}},
        "BEST_PE_RATIO": {"masked_cells": 5070, "tickers": 19, "max_run": 21, "top": {"RBLX": 1212}},
    }
    rec.update(over)
    return {"stale_run_mask": rec}


def test_mechanism_requires_every_fund_sheet_window_and_named_cases():
    ok = mechanism_fund_mask(_dq())
    assert ok["pass"] is True and ok["total_masked_cells"] == 55000
    assert ok["missing_sheets"] == [] and ok["wrong_window"] == [] and ok["named_cases_missing"] == []
    assert "BEST_PE_RATIO" in ok["ratio_sheets_reported"] and "top" not in ok["ratio_sheets_reported"]["BEST_PE_RATIO"]
    missing = mechanism_fund_mask({"stale_run_mask": {k: v for k, v in _dq()["stale_run_mask"].items() if k != "BEST_ROE"}})
    assert missing["missing_sheets"] == ["BEST_ROE"] and missing["pass"] is False
    wrong = mechanism_fund_mask(_dq(BEST_CAPEX={"masked_cells": 10, "tickers": 1, "max_run": 21, "top": {"APO": 10}}))
    assert wrong["wrong_window"] == ["BEST_CAPEX"] and wrong["pass"] is False
    named = mechanism_fund_mask(_dq(BEST_CALCULATED_FCF={"masked_cells": 10, "tickers": 1, "max_run": 126, "top": {"Z": 10}}))
    assert named["named_cases_missing"] == ["BEST_CALCULATED_FCF/C"] and named["pass"] is False
    assert mechanism_fund_mask({})["pass"] is False


def test_fund_tokens_cover_every_whitelist_consumer_and_exclude_unrelated_features():
    consumers = ["best_gross_margin_chg_63d", "best_gross_margin_chg_252d", "op_leverage_63d",
                 "best_calculated_fcf_level_z", "cash_conversion_z", "best_capex_level_z", "capex_intensity_z",
                 "best_roe_level_z", "fin_roe_level_z", "fin_roe_chg_63d", "fin_roe_chg_252d", "fin_roe_pe_gap", "fin_roe_pb_gap"]
    assert all(any(tok in f for tok in _FUND_TOKENS) for f in consumers)
    unrelated = ["momentum_252d", "eps_rev", "fin_pe_level_z", "fin_pb_chg_63d", "oper_margin_chg_63d", "best_peg_ratio_level_z",
                 "earnings_quality_252d", "best_sales_chg_252d", "tg_upside", "analyst_rec_level"]
    assert not any(any(tok in f for tok in _FUND_TOKENS) for f in unrelated)
