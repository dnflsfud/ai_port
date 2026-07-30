# -*- coding: utf-8 -*-
"""S13.23: 국가-매핑 지수 리비전 레벨 피처 (1개).

bcast 형태는 §S13.18/21에서 gain 정확히 0으로 폐쇄 — 종목→소속시장 지수
리비전 매핑으로 날짜내 횡단면 변동(5그룹)을 만들어 쿼리내 상수 함정을
벗어난다. 매핑 테이블은 결정 로그 §S13.23 사전약정(스윕 금지)이며
scripts/preflight_s13_23_index_rev_mapping.py와 테스트로 동기화된다.

index_eps와 동일 idiom: 무조건 빌드(S8), Factor 그룹 합류(CS z-score 스킵),
채택은 core-whitelist에서 config.index_revision_feature_enabled로 게이트 —
OFF면 필터 후 패널 바이트 동일.
"""

from typing import Dict

import pandas as pd

from src.data_loader import UniverseData

INDEX_REVISION_FEATURES = ("fac_idx_rev",)

EXCHANGE_TO_REV = {
    "US": "SPX_REV",
    "FP": "CAC_REV",
    "GR": "DAX_REV",
    "JP": "JPN_REV",
    "NA": "SX5E_REV",
    "SM": "SX5E_REV",
    "LN": "SX5E_REV",  # 유럽 프록시(비유로존 근사, 선언된 부정확)
    "SW": "SX5E_REV",
    "DC": "SX5E_REV",
    "KS": "SPX_REV",   # 글로벌 앵커 fallback(선언된 부정확)
}


def admitted_index_revision_features(config) -> set:
    """core-whitelist에 승인할 피처 셋 — flag OFF면 빈 셋(parity 경로)."""
    if not getattr(config, "index_revision_feature_enabled", False):
        return set()
    return set(INDEX_REVISION_FEATURES)


def build_index_revision_features(data: UniverseData) -> Dict[str, pd.DataFrame]:
    factor_px = data.factor_prices
    needed = sorted(set(EXCHANGE_TO_REV.values()))
    if factor_px is None or any(c not in factor_px.columns for c in needed):
        print("[IndexRev] revision columns not found in factor_prices — skipping")
        return {}
    tickers = list(data.tickers)
    exchange = data.meta.loc[tickers, "exchange_code"]
    unknown = sorted(set(exchange.dropna()) - set(EXCHANGE_TO_REV))
    if unknown:
        # 무조건 빌드 경로이므로 crash 대신 명시적 skip (§9: 추정으로 메우지 않음)
        print(f"[IndexRev] unmapped exchange codes {unknown} — skipping")
        return {}
    common_dates = data.dates.intersection(factor_px.index)
    cols = {
        t: factor_px[EXCHANGE_TO_REV[exchange.loc[t]]].reindex(common_dates)
        for t in tickers
    }
    return {"fac_idx_rev": pd.DataFrame(cols, index=common_dates)}
