# -*- coding: utf-8 -*-
"""S13.22: 캐리 TE-캡 조건화 (포트폴리오 구성 계층).

§S13.19 사전점검(PROCEED)의 후속 — S0 캐리(gap) 성분이 지수 EPS 사이클과
공변(Δgap top−bottom +2.26%/yr, 캐리 국지화)하므로, 리밸런싱 시점의 EPS
상태로 ex-ante TE 캡을 스케일한다:

    state = 조건 피처(들)의 PIT 확장 백분위 평균(composite)
    top(>2/3) -> max_te_annual x (1+kappa)
    mid       -> x 1.0
    bottom(<=1/3) -> x (1-kappa)

랭커·예측·피처 패널은 일절 건드리지 않는다(§S13.18/21: 랭커는 bcast를 소비
불가). default-OFF: flag OFF면 build가 None을 반환하고 backtest의 optimize
호출 경로가 기존과 바이트 동일하다.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

DEFAULT_FEATURES = ("fac_eps_g63",)

_FORMULAS = {
    "fac_eps_g63": lambda e: e["MXWD"].diff(63),
    "fac_eps_us_lead63": lambda e: e["SPX"].diff(63) - e["MXWD"].diff(63),
    "fac_eps_us_lead252": lambda e: e["SPX"].diff(252) - e["MXWD"].diff(252),
    "fac_eps_tech_lead63": lambda e: e["NDX"].diff(63) - e["SPX"].diff(63),
}
_REQUIRED = ("SPX_FWD_EPS", "NDX_FWD_EPS", "MXWD_FWD_EPS")


def index_eps_series(factor_px: pd.DataFrame, name: str) -> pd.Series:
    """지수 EPS 스프레드 시리즈 (src/features/index_eps.py와 동일 수식)."""
    if name not in _FORMULAS:
        raise ValueError(f"unknown carry_te_conditioning feature: {name}")
    eps = {k: np.log(factor_px[f"{k}_FWD_EPS"]) for k in ("SPX", "NDX", "MXWD")}
    return _FORMULAS[name](eps)


def expanding_percentile(s: pd.Series) -> pd.Series:
    """PIT 확장 백분위: t 값의 과거(자기 포함) 분포 내 (<=) 비율."""
    return s.expanding().apply(lambda a: float((a <= a[-1]).mean()), raw=True)


def te_multiplier_from_composite(comp: pd.Series, kappa: float) -> pd.Series:
    """composite 백분위 -> 승수 (NaN은 중립 1.0)."""
    mult = pd.Series(1.0, index=comp.index)
    mult[comp <= 1.0 / 3.0] = 1.0 - kappa
    mult[comp > 2.0 / 3.0] = 1.0 + kappa
    return mult


def build_te_cap_multipliers(factor_px: Optional[pd.DataFrame],
                             config) -> Optional[pd.Series]:
    """일별 TE-캡 승수 시리즈. OFF/데이터 부재 -> None (기존 경로 그대로)."""
    if not getattr(config, "carry_te_conditioning_enabled", False):
        return None
    if factor_px is None or any(c not in factor_px.columns for c in _REQUIRED):
        print("[CarryTE] FWD_EPS columns unavailable — conditioning disabled")
        return None
    features: List[str] = list(
        getattr(config, "carry_te_conditioning_features", None)
        or DEFAULT_FEATURES)
    kappa = float(getattr(config, "carry_te_conditioning_kappa", 0.25))
    min_history = int(getattr(config, "carry_te_conditioning_min_history", 504))

    pcts = [expanding_percentile(index_eps_series(factor_px, n).dropna())
            for n in features]
    comp = pd.concat(pcts, axis=1).dropna().mean(axis=1)
    mult = te_multiplier_from_composite(comp, kappa)
    # 확장 분포가 안정되기 전(min_history 미만 관측)은 중립
    mult.iloc[:min_history] = 1.0
    return mult.reindex(factor_px.index).fillna(1.0)
