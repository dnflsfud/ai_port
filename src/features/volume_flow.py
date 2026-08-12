# -*- coding: utf-8 -*-
"""§S13.35: 거래량·옵션 포지셔닝 피처 (arm A: volume 3, arm B: putcall 2).

2026-08-11 수집 개시한 종목 레벨 시트 2종(PX_VOLUME 200/200,
PUT_CALL_OPEN_INTEREST_RATIO 152/200 — 비US 48종은 열 자체가 없어 정직한
NaN → 기본 경로의 per-date median 채움이 중립 처리)에서 사전등록 5피처를
만든다. 통화 단위가 식에서 소거되도록 구성한다(다통화 유니버스 안전):

  arm A (``volume_features_enabled``):
    vol_abnormal_21_126 = log(ADV21 / ADV126)          # 이상 거래량 (비율)
    share_turnover_63d  = mean63(V·P) / MKT_CAP        # ≈ V/발행주식수
    amihud_illiq_63d    = log(mean63(|r|·MKT_CAP/(V·P)))    # 정규화 Amihud

CUR_MKT_CAP은 백만 단위라 Amihud 원값이 ~1e-6 스케일 — log1p는 항등에
수렴해 꼬리 압축이 무효화되므로 순수 log를 쓴다(≤0은 NaN 가드). 스케일
자체는 전 종목 동일 배율이라 CS z-score에 inert(사전점검 2026-08-11).

  arm B (``putcall_features_enabled``):
    pcr_oi_level    = 풋/콜 미결제약정 비율 레벨
    pcr_oi_chg_21d  = 레벨 − 21일 전 레벨 (포지셔닝 변화)

롤링 min_periods는 window//2 (기존 252/126 관례와 동일). 가드: MKT_CAP≤0·
V·P≤0·ADV126≤0 셀은 NaN(inf 유출 금지). 상장 전 구간은 로더 이중
마스킹(§S11.4)이 NaN으로 고정한다.

S8 idiom: 무조건 빌드, 채택은 core-whitelist에서 두 플래그로 각각 게이트 —
OFF면 모델 입력 바이트 동일.
"""

from typing import Dict

import numpy as np
import pandas as pd

VOLUME_SHEET = "PX_VOLUME"
PUTCALL_SHEET = "PUT_CALL_OPEN_INTEREST_RATIO"

VOLUME_FEATURES = (
    "vol_abnormal_21_126",
    "share_turnover_63d",
    "amihud_illiq_63d",
)
PUTCALL_FEATURES = (
    "pcr_oi_level",
    "pcr_oi_chg_21d",
)


def admitted_volume_features(config) -> set:
    """core-whitelist에 승인할 피처 셋 — flag OFF면 빈 셋(parity 경로)."""
    if not getattr(config, "volume_features_enabled", False):
        return set()
    return set(VOLUME_FEATURES)


def admitted_putcall_features(config) -> set:
    """core-whitelist에 승인할 피처 셋 — flag OFF면 빈 셋(parity 경로)."""
    if not getattr(config, "putcall_features_enabled", False):
        return set()
    return set(PUTCALL_FEATURES)


def build_volume_flow_features(data) -> Dict[str, pd.DataFrame]:
    """volume 3 + putcall 2 피처를 만든다.

    입력 시트가 없으면 해당 축만 건너뛴다(§9: 추정으로 메우지 않음).
    정규화는 assembly의 공통 CS z-score를 그대로 따른다.
    """
    tickers = list(data.tickers)

    def _sheet(name):
        try:
            raw = data.get_sheet(name)
        except KeyError:
            print(f"[VolumeFlow] {name} sheet not found — skipping dependents")
            return None
        return raw.reindex(index=data.dates, columns=tickers)

    out: Dict[str, pd.DataFrame] = {}

    vol = _sheet(VOLUME_SHEET)
    if vol is not None:
        adv21 = vol.rolling(21, min_periods=10).mean()
        adv126 = vol.rolling(126, min_periods=63).mean()
        out["vol_abnormal_21_126"] = np.log(adv21 / adv126.where(adv126 > 0))

        px = _sheet("PX_LAST")
        mc = _sheet("CUR_MKT_CAP")
        ret = _sheet("Daily_Returns")
        if px is not None and mc is not None:
            mc = mc.where(mc > 0)
            dollar = (vol * px).where(vol * px > 0)
            out["share_turnover_63d"] = (
                dollar.rolling(63, min_periods=31).mean() / mc
            )
            if ret is not None:
                amihud = (
                    (ret.abs() * mc / dollar).rolling(63, min_periods=31).mean()
                )
                out["amihud_illiq_63d"] = np.log(amihud.where(amihud > 0))

    pcr = _sheet(PUTCALL_SHEET)
    if pcr is not None:
        out["pcr_oi_level"] = pcr
        out["pcr_oi_chg_21d"] = pcr - pcr.shift(21)

    return out
