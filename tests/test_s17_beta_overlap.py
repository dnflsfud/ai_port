# -*- coding: utf-8 -*-
"""§S17 M2 (피처 채널) — beta_63d·idio_vol_63d 를 K일 겹침 수익률로 추정
(s17_beta_overlap_enabled, K = s17_beta_overlap_days = 5, 사전등록 상수).

결함(결정 로그 §S17 P3): 동시 rolling cov/var 베타는 미국 종가 이후 거래되는
아시아 종목에서 3× 과소(0.324 vs Dimson 0.997). 수정은 종목·EW 시장 수익률을
K일 겹침 합으로 바꿔 같은 rolling 식을 적용하고, idio_vol은 √K 로 일별 환산한다.
OFF(기본)는 기존 식과 바이트 동일(config 인자 없이 호출해도 동일).
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.features.price import build_price_features

N_DATES = 160
US = ["U1", "U2", "U3", "U4", "U5"]
ASIA = "AS"


def _stub(seed=3):
    """미국 5종 = 충격_t + 잡음, 아시아 1종 = 0.8·충격_{t-1} + 잡음."""
    rng = np.random.default_rng(seed)
    shock = rng.normal(0.0, 0.010, N_DATES + 1)
    cols = {u: shock[1:] + rng.normal(0.0, 0.006, N_DATES) for u in US}
    cols[ASIA] = 0.8 * shock[:-1] + rng.normal(0.0, 0.006, N_DATES)
    dates = pd.bdate_range("2022-01-03", periods=N_DATES)
    ret = pd.DataFrame(cols, index=dates)
    prices = (1 + ret).cumprod() * 100.0
    return SimpleNamespace(returns=ret, returns_masked=ret, prices=prices,
                           market_cap=prices * 10.0)


def _inline_reference_beta(returns, w=63):
    mkt = returns.mean(axis=1)
    e_xy = returns.mul(mkt, axis=0).rolling(w, min_periods=w).mean()
    e_x = returns.rolling(w, min_periods=w).mean()
    e_y = mkt.rolling(w, min_periods=w).mean()
    var_y = mkt.rolling(w, min_periods=w).var().replace(0, np.nan)
    return (e_xy - e_x.mul(e_y, axis=0)).div(var_y, axis=0)


def test_s17_beta_overlap_flag_defaults():
    cfg = PipelineConfig()
    assert cfg.s17_beta_overlap_enabled is False
    assert cfg.s17_beta_overlap_days == 5


def test_off_parity_with_and_without_config():
    data = _stub()
    legacy = build_price_features(data)
    off = build_price_features(data, config=PipelineConfig())
    pd.testing.assert_frame_equal(off["beta_63d"], legacy["beta_63d"])
    pd.testing.assert_frame_equal(off["idio_vol_63d"], legacy["idio_vol_63d"])
    pd.testing.assert_frame_equal(off["beta_63d"], _inline_reference_beta(data.returns))


def test_on_recovers_lagged_beta_and_keeps_us_beta():
    data = _stub()
    off = build_price_features(data, config=PipelineConfig())
    on = build_price_features(data, config=PipelineConfig(s17_beta_overlap_enabled=True))
    b_off, b_on = off["beta_63d"], on["beta_63d"]
    valid = b_on.dropna().index
    asia_off = b_off.loc[valid, ASIA].median()
    asia_on = b_on.loc[valid, ASIA].median()
    assert asia_off < 0.35                       # 일별: 시차 베타 소실
    assert asia_on > 3.0 * asia_off              # 겹침: 회복 (≥3×)
    us_off = b_off.loc[valid, US].median().median()
    us_on = b_on.loc[valid, US].median().median()
    assert abs(us_on / us_off - 1.0) < 0.30      # 동시 거래 종목은 대체로 보존
    # 겹침 워밍업(K-1일)만큼만 유효 구간이 늦어진다.
    assert len(valid) == len(b_off.dropna().index) - 4


def test_on_idio_vol_is_daily_equivalent_scale():
    data = _stub()
    off = build_price_features(data, config=PipelineConfig())
    on = build_price_features(data, config=PipelineConfig(s17_beta_overlap_enabled=True))
    valid = on["idio_vol_63d"].dropna().index
    ratio = on["idio_vol_63d"].loc[valid, US].median() / off["idio_vol_63d"].loc[valid, US].median()
    assert ((ratio > 0.6) & (ratio < 1.6)).all()   # √K 환산으로 같은 스케일
    assert on["idio_vol_63d"].loc[valid].notna().all().all()
