# -*- coding: utf-8 -*-
"""§S13.38: 옵션 리스크 표준화 피처 (5개, 단일 arm — 사용자 지시).

§S13.37 데이터 계층(create_ai_signal_data.py)이 워크북에 만든 파생 시트를
(dates×tickers)로 정렬해 그대로 싣는 **패스스루**다. 252D TS z-score·Δ5 등
파생 정의는 데이터 계층이 유일 정본 — 여기서 재계산하지 않는다(단일 정의).

사전등록 5피처(사용자 지시 "Skew_Z, ΔSkew_5D, IV_Z, Term_Z 네 개부터" +
term의 earnings 효과 분리 장치):

    iv30_z               = Z252(30D ATM IV)         # 자기 역사 대비 불확실성
    downside_skew_z      = Z252(25Δ 풋 − 콜 IV)     # 평소 대비 downside fear
    downside_skew_chg_5d = Δ5(25Δ 풋 − 콜 IV)       # skew 급등 조기경보
    iv_term_structure_z  = Z252(30D − 3M IV)        # 단기 이벤트 프라이싱
    days_to_earnings     = 다음 실적발표까지 달력일수 # term의 earnings 분리용
                           (트리 분기 조건화 — 휴면 S13.9 earn_days_to_next와
                           의미 중첩을 사전 기록)

vol_risk_premium_z는 사전등록에서 제외(사용자 지시 — 방향 불확실).
TS z-score라 종목 간 구조적 IV 수준 차이가 이미 제거됨 — 사전점검(2026-08-12)
vol 축 CS ρ ≤ 0.08 (§S13.34 레벨형 +0.88과 대조). IC 부호는 가설과 반대인
양(+)이었음을 사전 기록(트리가 방향을 학습하므로 설계 불변).

S8 idiom: 무조건 빌드, 채택은 core-whitelist에서
``config.option_risk_features_enabled``로 게이트 — OFF면 모델 입력 바이트
동일. 상장 전 구간은 로더 이중 마스킹(§S11.4)이 NaN으로 고정한다.
"""

from typing import Dict

import pandas as pd

# 피처명 = 워크북 시트명 (§S13.37 데이터 계층 정본)
OPTION_RISK_FEATURES = (
    "iv30_z",
    "downside_skew_z",
    "downside_skew_chg_5d",
    "iv_term_structure_z",
    "days_to_earnings",
)


def admitted_option_risk_features(config) -> set:
    """core-whitelist에 승인할 피처 셋 — flag OFF면 빈 셋(parity 경로)."""
    if not getattr(config, "option_risk_features_enabled", False):
        return set()
    return set(OPTION_RISK_FEATURES)


def build_option_risk_features(data) -> Dict[str, pd.DataFrame]:
    """워크북 파생 시트를 유니버스 격자에 정렬해 싣는다.

    입력 시트가 없으면 해당 피처만 건너뛴다(§9: 추정으로 메우지 않음).
    SPX Index 등 비유니버스 열은 reindex가 버린다. 정규화는 assembly의
    공통 CS z-score를 그대로 따른다.
    """
    tickers = list(data.tickers)

    out: Dict[str, pd.DataFrame] = {}
    for name in OPTION_RISK_FEATURES:
        try:
            raw = data.get_sheet(name)
        except KeyError:
            print(f"[OptionRisk] {name} sheet not found — skipping")
            continue
        out[name] = raw.reindex(index=data.dates, columns=tickers)
    return out
