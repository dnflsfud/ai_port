# -*- coding: utf-8 -*-
"""§S13.25: Fwd Sales 기간구조 slope 피처 (4개).

``Fwd_Sales_Slope_1FY2FY`` 시트(BEST_SALES_1FY/2FY에서 유도한 FY1→FY2 내재
매출 성장률 ``(2FY−1FY)/1FY``)를 소비한다. 선형 2(레벨·Δ63) + 비선형 2 —
트리 랭커는 단일 피처의 단조 변환에 불변이므로, 실질 비선형은 S13.15
soft-AND(min-confirm) 2변수 결합으로만 만든다.

S8 idiom: 무조건 빌드, 채택은 core-whitelist에서
``config.fwd_sales_slope_features_enabled``로 게이트 — OFF면 패널 바이트 동일.
상장 전 구간은 로더 이중 마스킹(§S11.4)이 NaN으로 고정하고, 원천의 백필
접두는 생성 단계(_mask_backfilled_prefix)에서 이미 NaN이다(결정 로그 §S13.25).
"""

from typing import Dict

import pandas as pd

from src.features.nonlinear_confirmation import _soft_and
from src.features.utils import cross_sectional_zscore

FWD_SALES_SLOPE_SHEET = "Fwd_Sales_Slope_1FY2FY"

FWD_SALES_SLOPE_FEATURES = (
    "fwd_sales_slope_level",
    "fwd_sales_slope_chg_63d",
    "nl_fslope_rev_confirm",
    "nl_fslope_growth_confirm",
)

# 비선형 confirm의 파트너 — 이미 core-whitelist에 승인된 피처만 사용한다.
NL_CONFIRM_PARENTS = {
    "nl_fslope_rev_confirm": "sales_rev_ma_63d",
    "nl_fslope_growth_confirm": "best_sales_chg_252d",
}


def admitted_fwd_sales_slope_features(config) -> set:
    """core-whitelist에 승인할 피처 셋 — flag OFF면 빈 셋(parity 경로)."""
    if not getattr(config, "fwd_sales_slope_features_enabled", False):
        return set()
    return set(FWD_SALES_SLOPE_FEATURES)


def build_fwd_sales_slope_features(
    all_features: Dict[str, pd.DataFrame],
    data,
) -> Dict[str, pd.DataFrame]:
    """slope 시트에서 선형 2 + min-confirm 비선형 2 피처를 만든다.

    파트너 피처가 아직 없으면 해당 confirm만 건너뛴다(§9: 추정으로 메우지
    않음). 레벨·Δ63은 assembly의 공통 CS z-score를 그대로 따른다.
    """
    try:
        raw = data.get_sheet(FWD_SALES_SLOPE_SHEET)
    except KeyError:
        print(f"[FwdSalesSlope] {FWD_SALES_SLOPE_SHEET} sheet not found — skipping")
        return {}

    tickers = list(data.tickers)
    slope = raw.reindex(index=data.dates, columns=tickers)

    out: Dict[str, pd.DataFrame] = {
        "fwd_sales_slope_level": slope,
        "fwd_sales_slope_chg_63d": slope.diff(63),
    }

    slope_z = cross_sectional_zscore(slope)
    for name, parent in NL_CONFIRM_PARENTS.items():
        if parent not in all_features:
            print(f"[FwdSalesSlope] parent {parent} missing — skipping {name}")
            continue
        parent_z = cross_sectional_zscore(
            all_features[parent].reindex(index=slope.index, columns=tickers)
        )
        out[name] = _soft_and(slope_z, parent_z)
    return out
