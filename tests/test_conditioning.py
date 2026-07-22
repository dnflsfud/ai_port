# -*- coding: utf-8 -*-
"""Breadth 분모 계약 테스트 (§S11.4 point-in-time universe).

regime_*_breadth류의 분모는 고정 유니버스 폭(shape[1])이 아니라 날짜별 유효
(상장) 종목 수여야 한다. 상장 전 셀은 listing mask가 NaN으로 만들므로
notna 수가 곧 유효 종목 수다.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import PipelineConfig
from src.data_loader import UniverseData
from src.features.conditioning import build_conditioning_features


ESSENTIAL_SHEETS = {
    "PX_LAST",
    "Daily_Returns",
    "CUR_MKT_CAP",
    "BEST_EPS",
    "BEST_SALES",
    "BEST_PE_RATIO",
    "OPER_MARGIN",
    "BEST_ROE",
    "NEWS_SENTIMENT_DAILY_AVG",
    "EQY_REC_CONS",
    "Factset_EPS_Revision",
    "Factset_Sales_Revision",
    "Factset_TG_Price",
}


def _fixture(n_dates=12, listing_pos=6):
    """AAA/BBB 상시 상장 + NEW는 listing_pos부터 상장(그 전은 유령 상수)."""
    dates = pd.bdate_range("2021-01-04", periods=n_dates)
    listing_date = dates[listing_pos]
    aaa = pd.Series(
        np.linspace(100.0, 100.0 + 2 * (n_dates - 1), n_dates), index=dates
    )
    prices = pd.DataFrame({"AAA": aaa, "BBB": aaa * 2.0, "NEW": 50.0}, index=dates)
    prices.loc[listing_date:, "NEW"] = np.linspace(
        55.0, 55.0 + n_dates - listing_pos - 1, n_dates - listing_pos
    )
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    meta = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB", "NEW"],
            "Name": ["Alpha", "Beta", "Newco"],
            "Sector": ["Test", "Test", "Test"],
            "Status": ["Active", "Active", "Active"],
        },
        index=["AAA US Equity", "BBB US Equity", "NEW US Equity"],
    )
    raw = {
        "Universe_Meta": meta,
        "PX_LAST": prices,
        "Daily_Returns": returns,
    }
    for sheet in ESSENTIAL_SHEETS - {"PX_LAST", "Daily_Returns"}:
        raw[sheet] = pd.DataFrame(1.0, index=dates, columns=["AAA", "BBB", "NEW"])
    # 부호가 갈리는 시트: AAA(+), BBB(-), NEW(+, 상장 전은 유령)
    for sheet in ("Factset_EPS_Revision", "Factset_Sales_Revision",
                  "NEWS_SENTIMENT_DAILY_AVG"):
        raw[sheet] = pd.DataFrame(
            {"AAA": 1.0, "BBB": -1.0, "NEW": 1.0}, index=dates
        )
    return raw, dates, listing_date


def _build(monkeypatch, raw, listing_date):
    monkeypatch.setattr(
        "src.data_loader.load_all_sheets",
        lambda _path: {name: frame.copy() for name, frame in raw.items()},
    )
    data = UniverseData(
        "unused.xlsx",
        config=PipelineConfig(
            fx_source_path="missing.xlsx",
            listing_dates={"NEW": str(listing_date.date())},
        ),
    )
    return build_conditioning_features(data)


def test_rev_breadth_denominator_is_per_date_valid_count(monkeypatch):
    raw, dates, listing_date = _fixture()
    features = _build(monkeypatch, raw, listing_date)
    # 상장 전 날짜: 유효 2종(AAA +, BBB -) -> 1/2. 고정 분모라면 1/3.
    assert features["regime_rev_breadth_eps"].loc[dates[1]].iloc[0] == pytest.approx(0.5)
    assert features["regime_rev_breadth_sales"].loc[dates[1]].iloc[0] == pytest.approx(0.5)


def test_sent_breadth_denominator_is_per_date_valid_count(monkeypatch):
    raw, dates, listing_date = _fixture()
    features = _build(monkeypatch, raw, listing_date)
    assert features["regime_sent_breadth"].loc[dates[1]].iloc[0] == pytest.approx(0.5)


def test_breadth_50d_denominator_is_per_date_valid_count(monkeypatch):
    raw, dates, listing_date = _fixture(n_dates=60, listing_pos=55)
    features = _build(monkeypatch, raw, listing_date)
    # 마지막 날: AAA/BBB는 상승 추세라 50d MA 위, NEW는 50d MA 미형성(NaN)
    # -> 유효 분모 2, breadth == 1.0 (고정 분모라면 2/3).
    assert features["regime_breadth_50d"].iloc[-1, 0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# §S11.7: regime 시장 통계(ew_ret 등)는 dense returns가 아니라 상장 전 NaN 뷰
# (returns_masked)에서 계산돼야 한다. 상시 상장 3종 + 유령 1종 픽스처 —
# 상장 2종이면 median==mean 이라 dense/masked가 우연히 일치하므로 3종 이상 필요.
# ---------------------------------------------------------------------------
def _fixture4(n_dates=30, listing_pos=25):
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2021-01-04", periods=n_dates)
    listing_date = dates[listing_pos]
    tickers = ["AAA", "BBB", "CCC", "NEW"]
    prices = pd.DataFrame(
        100.0 * np.cumprod(1 + rng.normal(0.001, 0.02, (n_dates, 4)), axis=0),
        index=dates, columns=tickers,
    )
    prices.loc[dates < listing_date, "NEW"] = 50.0  # 유령 상수 백필
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    meta = pd.DataFrame(
        {
            "Ticker": tickers,
            "Name": tickers,
            "Sector": ["Test"] * 4,
            "Status": ["Active"] * 4,
        },
        index=[f"{t} US Equity" for t in tickers],
    )
    raw = {"Universe_Meta": meta, "PX_LAST": prices, "Daily_Returns": returns}
    for sheet in ESSENTIAL_SHEETS - {"PX_LAST", "Daily_Returns"}:
        raw[sheet] = pd.DataFrame(1.0, index=dates, columns=tickers)
    return raw, dates, listing_date


def test_regime_market_return_excludes_ghost(monkeypatch):
    raw, dates, listing_date = _fixture4()
    monkeypatch.setattr(
        "src.data_loader.load_all_sheets",
        lambda _path: {name: frame.copy() for name, frame in raw.items()},
    )
    data = UniverseData(
        "unused.xlsx",
        config=PipelineConfig(
            fx_source_path="missing.xlsx",
            listing_dates={"NEW": str(listing_date.date())},
        ),
    )
    features = build_conditioning_features(data)
    d = dates[24]  # 상장 전 마지막 날 (21d 윈도우 형성됨)
    expected = (
        data.returns_masked.mean(axis=1).rolling(21, min_periods=21).sum().loc[d]
    )
    assert features["regime_mkt_ret_21d"].loc[d].iloc[0] == pytest.approx(
        expected, abs=1e-12
    )
