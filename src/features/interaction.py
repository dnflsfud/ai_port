"""Category 6: Sector Interaction Features (Adaptive Feature Weighting)."""

import pandas as pd
import numpy as np
from typing import Dict
from collections import Counter

from src.data_loader import UniverseData, TICKERS
from src.features.utils import cs_rank


# 핵심 피처 x 섹터 interaction -> GBT가 섹터별 패턴 학습 가능
# 예: "반도체 섹터에서 EPS revision 급등 -> 높은 specific return"
INTERACTION_KEY_FEATURES = [
    # Sellside
    "eps_rev", "eps_rev_diff_21d", "sales_rev", "tg_upside",
    # Price
    "momentum_63d", "reversal_21d", "realized_vol_21d",
    # Sentiment
    "news_trend",
    # Accounting
    "best_eps_chg_63d", "best_roe_chg_63d",
]


def build_sector_interaction_features(
    data: UniverseData,
    base_features: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """Sector x Key Feature interaction 피처 생성.

    목적:
      - LightGBM이 각 섹터에서 어떤 피처가 중요한지 명시적으로 학습
      - 반도체 섹터의 EPS revision과 Technology 섹터의 EPS revision을 구분
      - 섹터별 고유한 알파 드라이버 패턴 포착

    구현:
      - 주요 섹터(4개 이상 종목)에 대해서만 interaction 생성 (과적합 방지)
      - 핵심 피처 10개 x 주요 섹터 N개 = 약 30~50개 피처 추가
    """
    features: Dict[str, pd.DataFrame] = {}
    # data.tickers = intersection across all loaded sheets (authoritative).
    tickers = list(data.tickers)

    # 섹터 매핑
    meta = data.meta
    if isinstance(meta, pd.DataFrame) and "sector" in meta.columns:
        sector_map = meta["sector"]
    elif isinstance(meta, pd.DataFrame) and len(meta.columns) > 0:
        sector_map = meta.iloc[:, 0]
    else:
        return features

    # 섹터별 종목 수 계산 -> 2개 이상인 섹터만 interaction 생성
    sector_counts = Counter()
    for t in tickers:
        sec = sector_map.get(t, "Unknown")
        if str(sec) not in ("nan", "Unknown"):
            sector_counts[sec] += 1

    valid_sectors = [sec for sec, cnt in sector_counts.items() if cnt >= 2]

    if not valid_sectors:
        return features

    for feat_name in INTERACTION_KEY_FEATURES:
        if feat_name not in base_features:
            continue
        feat_df = base_features[feat_name]

        for sector in valid_sectors:
            # 섹터 마스크: 해당 섹터 종목만 원래 값, 나머지는 0
            mask_df = pd.DataFrame(0.0, index=feat_df.index, columns=feat_df.columns)
            for t in tickers:
                if t in feat_df.columns and sector_map.get(t, "") == sector:
                    mask_df[t] = 1.0

            interaction = feat_df * mask_df
            # 이름에서 공백 제거
            sec_clean = sector.replace(" ", "_").replace(".", "")
            features[f"ix_{feat_name}_{sec_clean}"] = interaction

    # 추가: Peer-relative features (종목의 동일 섹터 내 상대 강도)
    for feat_name in ["eps_rev", "momentum_63d", "tg_upside"]:
        if feat_name not in base_features:
            continue
        feat_df = base_features[feat_name]

        # 섹터별 평균 대비 상대값
        peer_rel = pd.DataFrame(0.0, index=feat_df.index, columns=feat_df.columns)
        for sector in valid_sectors:
            sec_tickers = [t for t in tickers if t in feat_df.columns and sector_map.get(t, "") == sector]
            if len(sec_tickers) < 2:
                continue
            sector_mean = feat_df[sec_tickers].mean(axis=1)
            for t in sec_tickers:
                peer_rel[t] = feat_df[t] - sector_mean

        features[f"peer_rel_{feat_name}"] = peer_rel

    print(f"[SectorInteraction] {len(features)}개 interaction 피처 생성 "
          f"(sectors={len(valid_sectors)}, key_features={len(INTERACTION_KEY_FEATURES)})")

    return features
