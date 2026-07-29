# -*- coding: utf-8 -*-
"""S13.18: Index forward-EPS growth/spread block (4 bcast features).

지수 레벨 BEST_EPS(SPX/NDX/MXWD)의 로그 성장률을 nested level/spread로 분해한
새 데이터 축(톱다운 어닝스 기대) — new_ai_port regime_v2 피처 설계 승계:
  fac_eps_g63        : 세계(MXWD) 레벨 63d 로그 성장
  fac_eps_us_lead63  : US-세계 스프레드 63d (빠른 축, 반감기 ~43BD)
  fac_eps_us_lead252 : US-세계 스프레드 252d (연간 리비전 사이클, ~192BD)
  fac_eps_tech_lead63: 테크(NDX)-US 스프레드 63d (~181BD)
원시 성장률 상관 0.80-0.97 → 분해 후 최대 |corr| 0.48 실측(2026-07-28).

다른 fac_* 브로드캐스트 피처처럼 per-date 상수이므로 assembly에서 Factor
그룹에 합류해 CS z-score를 건너뛴다. 빌드는 무조건(S8 idiom), 채택은
core-whitelist에서 config.index_eps_features_enabled로 게이트 — OFF면
필터 후 패널 바이트 동일.
"""

from typing import Dict

import numpy as np
import pandas as pd

from src.data_loader import UniverseData

INDEX_EPS_FEATURES = ("fac_eps_g63", "fac_eps_us_lead63",
                      "fac_eps_us_lead252", "fac_eps_tech_lead63")

_REQUIRED = ("SPX_FWD_EPS", "NDX_FWD_EPS", "MXWD_FWD_EPS")


def admitted_index_eps_features(config) -> set:
    """core-whitelist에 승인할 index-EPS 피처 셋 (S13.21 subset 지원).

    flag OFF → 빈 셋(기존 parity 경로 그대로). ON이면
    config.index_eps_feature_names(None/부재 = 전체 4종)의 subset만 승인.
    미지의 이름은 무음 inert 대신 ValueError.
    """
    if not getattr(config, "index_eps_features_enabled", False):
        return set()
    subset = getattr(config, "index_eps_feature_names", None) or INDEX_EPS_FEATURES
    unknown = set(subset) - set(INDEX_EPS_FEATURES)
    if unknown:
        raise ValueError(
            f"unknown index_eps_feature_names: {sorted(unknown)}")
    return set(subset)


def build_index_eps_features(data: UniverseData) -> Dict[str, pd.DataFrame]:
    factor_px = data.factor_prices
    if factor_px is None or any(c not in factor_px.columns for c in _REQUIRED):
        print("[IndexEPS] FWD_EPS columns not found in factor_prices — skipping")
        return {}
    tickers = list(data.tickers)
    n = len(tickers)
    common_dates = data.dates.intersection(factor_px.index)

    def bcast(series: pd.Series) -> pd.DataFrame:
        vals = series.reindex(common_dates).values.reshape(-1, 1)
        return pd.DataFrame(np.tile(vals, (1, n)), index=common_dates,
                            columns=tickers)

    eps = {k: np.log(factor_px[f"{k}_FWD_EPS"]) for k in ("SPX", "NDX", "MXWD")}
    return {
        "fac_eps_g63": bcast(eps["MXWD"].diff(63)),
        "fac_eps_us_lead63": bcast(eps["SPX"].diff(63) - eps["MXWD"].diff(63)),
        "fac_eps_us_lead252": bcast(eps["SPX"].diff(252) - eps["MXWD"].diff(252)),
        "fac_eps_tech_lead63": bcast(eps["NDX"].diff(63) - eps["SPX"].diff(63)),
    }
