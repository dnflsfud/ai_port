# -*- coding: utf-8 -*-
"""§S27 룩어헤드 클래스 arm 판정 스크립트의 순수 함수 테스트 (결정 로그 §S27 사전등록 규칙 핀)."""

import numpy as np
import pandas as pd

from scripts.eval_s27_om_pit_arm import (
    ARM,
    BASE,
    FLAG,
    MATCH_RATE_MIN,
    NW_LAGS,
    QE_STAMP_MIN,
    QE_VISIBLE_MAX_X_BASE,
    REFERENCE_BASE_IR,
    TURNOVER_MAX,
    _OM_TOKENS,
    mechanism_restamp,
    paired_nw_t,
)


def test_arm_pins():
    assert ARM == "s27_om_pit_restamp" and FLAG == "oper_margin_pit_restamp_enabled" and BASE == "s26_s0_base"
    assert REFERENCE_BASE_IR == 1.7737993717667202 and TURNOVER_MAX == 1.25
    assert MATCH_RATE_MIN == 0.90 and QE_STAMP_MIN == 0.50 and QE_VISIBLE_MAX_X_BASE == 2.0 and NW_LAGS == 21


def _dq(**over):
    rec = {"changes": 11362, "matched": 11170, "match_rate": 0.9831, "fallback": 192, "changed_cells": 249944,
           "quarter_end_3d_share_changes": 0.8469, "quarter_end_3d_share_visible": 0.0056,
           "quarter_end_3d_share_sessions": 0.0772}
    rec.update(over)
    return {"oper_margin_pit_restamp": rec}


def test_mechanism_requires_match_rate_and_the_stamp_signature_to_move():
    assert mechanism_restamp(_dq())["pass"] is True
    assert mechanism_restamp(_dq(match_rate=0.85))["checks"]["match_rate_ok"] is False
    assert mechanism_restamp(_dq(quarter_end_3d_share_changes=0.3))["checks"]["stamp_signature_before"] is False
    assert mechanism_restamp(_dq(quarter_end_3d_share_visible=0.2))["checks"]["stamp_gone_after"] is False
    assert mechanism_restamp(_dq(changed_cells=0))["pass"] is False
    assert mechanism_restamp({})["pass"] is False


def test_om_tokens_cover_every_whitelist_consumer_and_exclude_unrelated_features():
    consumers = ["oper_margin_chg_63d", "oper_margin_chg_252d", "oper_margin_accel", "op_leverage_63d"]
    assert all(any(tok in f for tok in _OM_TOKENS) for f in consumers)
    unrelated = ["best_gross_margin_chg_63d", "momentum_252d", "fin_pe_level_z", "earnings_quality_252d",
                 "best_sales_chg_252d", "cash_conversion_z", "best_roe_level_z", "idio_vol_63d"]
    assert not any(any(tok in f for tok in _OM_TOKENS) for f in unrelated)


def test_paired_nw_t_sign_and_iid_limit():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(0.001, 0.01, 4000))
    t = paired_nw_t(x)
    assert t > 3                                             # mean 0.1 sd -> t ~ 6 over 4,000 days
    assert abs(paired_nw_t(x, lags=0) - x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))) < 1e-9
    assert paired_nw_t(-x) == -t
    assert np.isnan(paired_nw_t(x.iloc[:10]))
