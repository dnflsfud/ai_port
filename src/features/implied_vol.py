# -*- coding: utf-8 -*-
"""§S13.34: 종목 레벨 implied-vol 서피스 피처 (3개, 단일 arm).

옵션시장 정보 3축 — 워크북의 종목 레벨 IV 시트 5종(§S13.34 품질점검:
200종 전부 최근 252영업일 커버리지 ≥80%, 실현 vol과 동일한 연율 %p 단위)
에서 사전등록한 3개 피처를 만든다:

    iv_skew_level = (IV90 − IV110) / IV100    # 풋스큐 — crash-risk 프라이싱
    iv_vrp_level  = IV30 − RV30               # implied−realized 프리미엄
    iv_term_level = (IV3M − IV30) / IV30      # 종목별 IV 기간구조

레벨(IV30)은 실현 vol과 CS corr +0.88이라 기존 vol 축과 중복 — 스프레드만
채택한다(skew −0.49 / term −0.03 / vrp −0.39, 결정 로그 §S13.34 사전점검).

S8 idiom: 무조건 빌드, 채택은 core-whitelist에서
``config.implied_vol_features_enabled``로 게이트 — OFF면 모델 입력 바이트
동일. 상장 전 구간은 로더 이중 마스킹(§S11.4)이 NaN으로 고정한다.
IV100 ≤ 0 셀은 가드로 NaN(inf 유출 금지).
"""

from typing import Dict

import pandas as pd

IV_30D_100_SHEET = "30DAY_IMPVOL_100.0%MNY_DF"
IV_30D_90_SHEET = "30DAY_IMPVOL_90.0%MNY_DF"
IV_30D_110_SHEET = "30DAY_IMPVOL_110.0%MNY_DF"
IV_3M_100_SHEET = "3MTH_IMPVOL_100.0%MNY_DF"
REALIZED_VOL_SHEET = "VOLATILITY_30D"

IMPLIED_VOL_FEATURES = (
    "iv_skew_level",
    "iv_vrp_level",
    "iv_term_level",
)


def admitted_implied_vol_features(config) -> set:
    """core-whitelist에 승인할 피처 셋 — flag OFF면 빈 셋(parity 경로)."""
    if not getattr(config, "implied_vol_features_enabled", False):
        return set()
    return set(IMPLIED_VOL_FEATURES)


def build_implied_vol_features(data) -> Dict[str, pd.DataFrame]:
    """IV 서피스 스프레드 3피처를 만든다.

    입력 시트가 없으면 해당 피처만 건너뛴다(§9: 추정으로 메우지 않음).
    정규화는 assembly의 공통 CS z-score를 그대로 따른다.
    """
    tickers = list(data.tickers)

    def _sheet(name):
        try:
            raw = data.get_sheet(name)
        except KeyError:
            print(f"[ImpliedVol] {name} sheet not found — skipping dependents")
            return None
        return raw.reindex(index=data.dates, columns=tickers)

    iv100 = _sheet(IV_30D_100_SHEET)
    if iv100 is None:
        return {}
    denom = iv100.where(iv100 > 0)

    out: Dict[str, pd.DataFrame] = {}

    iv90 = _sheet(IV_30D_90_SHEET)
    iv110 = _sheet(IV_30D_110_SHEET)
    if iv90 is not None and iv110 is not None:
        out["iv_skew_level"] = (iv90 - iv110) / denom

    rv30 = _sheet(REALIZED_VOL_SHEET)
    if rv30 is not None:
        out["iv_vrp_level"] = iv100 - rv30

    iv3m = _sheet(IV_3M_100_SHEET)
    if iv3m is not None:
        out["iv_term_level"] = (iv3m - iv100) / denom

    return out
