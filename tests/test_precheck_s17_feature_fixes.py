# -*- coding: utf-8 -*-
"""§S17.1 피처 채널 정확성 사전점검 헬퍼 단위테스트 (경량 — 데이터 로드 없음)."""
import numpy as np

from scripts.precheck_s17_feature_fixes import (
    BETA_ASIA_RATIO_MIN,
    BETA_US_BAND,
    T01_CS_STD_MIN,
    evaluate_beta,
    evaluate_t01,
    region_of,
)


def test_region_of_maps_bloomberg_suffixes():
    assert region_of("KS") == "ASIA" and region_of("JP") == "ASIA"
    assert region_of("LN") == "EU" and region_of("SM") == "EU"
    assert region_of("US") == "US"


def test_evaluate_t01_requires_defect_reproduced_gap_nan_and_std_restored():
    assert T01_CS_STD_MIN == 0.90
    ok = evaluate_t01(off_plus5_rows=409, on_nan_rows=409, window_rows=409, cs_std_on=0.99)
    assert ok["t01_pass"] is True
    # 결함이 재현되지 않으면(OFF 에 +5 없음) 검정 자체가 무효
    assert evaluate_t01(0, 409, 409, 0.99)["t01_pass"] is False
    # 갭창 일부만 NaN → 수정 미완
    assert evaluate_t01(409, 400, 409, 0.99)["t01_pass"] is False
    # 압축이 안 풀리면 FAIL
    assert evaluate_t01(409, 409, 409, 0.5)["t01_pass"] is False
    assert evaluate_t01(409, 409, 409, float("nan"))["t01_pass"] is False


def test_evaluate_beta_gates():
    assert BETA_ASIA_RATIO_MIN == 2.0 and BETA_US_BAND == 0.15
    assert evaluate_beta(ratio_asia=2.8, ratio_us=1.05)["beta_pass"] is True
    assert evaluate_beta(1.5, 1.05)["beta_pass"] is False       # 아시아 회복 부족
    assert evaluate_beta(2.8, 1.30)["beta_pass"] is False       # 미국 베타 왜곡
    assert evaluate_beta(float("nan"), 1.0)["beta_pass"] is False
