# -*- coding: utf-8 -*-
"""§S16.1 unit fix-pack — 단일 default-OFF 플래그(s16_unit_fixpack_enabled) 뒤
4건 정확성 수정의 OFF 파리티·ON 동작 테스트.

P2(FCF/CAPEX level_z)·P3(cash_conversion_z)는 accounting 모듈분이라 TDD 가드
파일명 규약에 따라 tests/test_accounting.py 에 있다(§S15의 tests/test_factor.py
선례). 여기서는 플래그 기본값 · P1(가격 단위) · P4(매크로 교차항)를 다룬다.
"""
import logging

import numpy as np
import pandas as pd
import pytest

from src.config import PipelineConfig
from src.data_loader import PRICE_UNIT_SCALE, UniverseData
from src.features.macro_cross import (_rolling_zscore,
                                      build_macro_cross_features)

ESSENTIAL_SHEETS = [
    "CUR_MKT_CAP", "BEST_EPS", "BEST_SALES", "BEST_PE_RATIO", "OPER_MARGIN",
    "BEST_ROE", "NEWS_SENTIMENT_DAILY_AVG", "EQY_REC_CONS",
    "Factset_EPS_Revision", "Factset_Sales_Revision", "Factset_TG_Price",
]


def test_s16_unit_fixpack_flag_default_off():
    assert PipelineConfig().s16_unit_fixpack_enabled is False


# ---------------------------------------------------------------------------
# P1. LSE(LN) 상장의 GBp 시세 -> GBP 목표주가 단위 정합
# ---------------------------------------------------------------------------

def _ln_universe(monkeypatch, target_prices=(330.0, 159.0)):
    """AAPL(US, USD) + AZN(LN, GBp) 2종 합성 워크북."""
    dates = pd.bdate_range("2026-01-01", periods=30)
    n = len(dates)
    drift = np.linspace(1.0, 1.05, n)
    prices = pd.DataFrame(
        {"AAPL": 300.0 * drift, "AZN": 12000.0 * drift}, index=dates
    )
    meta = pd.DataFrame(
        {"Ticker": ["AAPL", "AZN"], "Name": ["Apple", "AstraZeneca"],
         "Sector": ["Test", "Test"], "Status": ["Active", "Active"]},
        index=["AAPL US Equity", "AZN LN Equity"],
    )
    raw = {
        "Universe_Meta": meta,
        "PX_LAST": prices,
        "Daily_Returns": prices.pct_change(fill_method=None).fillna(0.0),
        "Factor_PX_LAST": pd.DataFrame({"GBPUSD": np.full(n, 1.30)}, index=dates),
    }
    for sheet in ESSENTIAL_SHEETS:
        raw[sheet] = pd.DataFrame(1.0, index=dates, columns=["AAPL", "AZN"])
    raw["Factset_TG_Price"] = pd.DataFrame(
        {"AAPL": np.full(n, target_prices[0]), "AZN": np.full(n, target_prices[1])},
        index=dates,
    )
    monkeypatch.setattr(
        "src.data_loader.load_all_sheets",
        lambda _path: {name: frame.copy() for name, frame in raw.items()},
    )
    return prices


def _load(flag):
    return UniverseData(
        "unused.xlsx",
        config=PipelineConfig(
            fx_source_path="missing.xlsx",
            listing_auto_infer_enabled=False,
            s16_unit_fixpack_enabled=flag,
        ),
    )


def test_price_unit_scale_off_parity_leaves_px_last_untouched(monkeypatch):
    prices = _ln_universe(monkeypatch)
    data = _load(False)
    pd.testing.assert_frame_equal(data.local_prices, prices)
    assert "price_unit_scaled" not in data.data_quality["currency"]


def test_price_unit_scale_on_rescales_only_ln_lines(monkeypatch):
    prices = _ln_universe(monkeypatch)
    data = _load(True)
    assert PRICE_UNIT_SCALE == {"LN": 0.01}
    # 로컬 패널: LN만 x0.01, 미국 라인은 불변.
    assert np.allclose(data.local_prices["AZN"], prices["AZN"] * 0.01)
    assert np.allclose(data.local_prices["AAPL"], prices["AAPL"])
    # USD 패널도 같은 스케일을 그대로 물려받는다 (변환 전에 적용했으므로).
    assert np.allclose(data.prices["AZN"], prices["AZN"] * 0.01 * 1.30)
    assert np.allclose(data.prices["AAPL"], prices["AAPL"])
    assert data.data_quality["currency"]["price_unit_scaled"] == {"AZN": 0.01}


def test_target_price_ratio_guard_runs_regardless_of_flag(monkeypatch, caplog):
    prices = _ln_universe(monkeypatch)
    med = prices.median()
    with caplog.at_level(logging.WARNING, logger="src.data_loader"):
        off = _load(False)
    ratios_off = off.data_quality["currency"]["tg_px_ratio_median"]
    assert ratios_off["AZN"] == pytest.approx(159.0 / med["AZN"], rel=1e-6)
    assert ratios_off["AAPL"] == pytest.approx(330.0 / med["AAPL"], rel=1e-6)
    mismatches = [rec for rec in caplog.records
                  if "quote-unit mismatch" in rec.message]
    assert len(mismatches) == 1
    assert "AZN" in str(mismatches[0].args[-1])

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="src.data_loader"):
        on = _load(True)
    ratios_on = on.data_quality["currency"]["tg_px_ratio_median"]
    assert ratios_on["AZN"] == pytest.approx(159.0 / (med["AZN"] * 0.01), rel=1e-6)
    assert not [rec for rec in caplog.records if "quote-unit mismatch" in rec.message]


# ---------------------------------------------------------------------------
# P4. 매크로 교차항 이중 z-score 제거 + slope 스칼라 z화
# ---------------------------------------------------------------------------

class _FakeData:
    """UniverseData 중 build_macro_cross_features가 쓰는 속성만."""

    def __init__(self, sheets, dates, tickers, factor_prices, returns_masked):
        self._sheets = sheets
        self.dates = dates
        self.tickers = tickers
        self.factor_prices = factor_prices
        self.returns_masked = returns_masked
        self.earnings_timeline = None

    def get_sheet(self, name):
        if name in self._sheets:
            return self._sheets[name]
        raise KeyError(name)

    def has_factor_data(self):
        return True


# 상수 revision 패널 [1.0, 2.0] -> cross_sectional_zscore 는 매 날짜 ±1/sqrt(2).
_EPS_REV_CS_B = 1.0 / np.sqrt(2.0)


def _macro_stub(n=200):
    rng = np.random.default_rng(16)
    dates = pd.bdate_range("2020-01-01", periods=n)
    tickers = ["A", "B"]
    factor_px = pd.DataFrame(
        {
            "UST_10Y": 2.0 + rng.normal(0, 0.05, n).cumsum(),
            "UST_2Y": 1.0 + rng.normal(0, 0.05, n).cumsum(),
        },
        index=dates,
    )
    sheets = {
        "Factset_EPS_Revision": pd.DataFrame(
            {"A": np.full(n, 1.0), "B": np.full(n, 2.0)}, index=dates
        )
    }
    returns = pd.DataFrame(rng.normal(0, 0.01, (n, 2)), index=dates,
                           columns=tickers)
    return _FakeData(sheets, dates, tickers, factor_px, returns), factor_px


def _slope_scalar_from(features):
    """mc_slope_x_eps_rev 의 B열에서 broadcast 된 slope 스칼라를 복원."""
    return features["mc_slope_x_eps_rev"]["B"] / _EPS_REV_CS_B


def test_macro_cross_slope_off_parity_uses_raw_spread():
    data, factor_px = _macro_stub()
    features = build_macro_cross_features(data, config=PipelineConfig())
    raw_slope = factor_px["UST_10Y"] - factor_px["UST_2Y"]
    assert np.allclose(_slope_scalar_from(features), raw_slope)


def test_macro_cross_slope_on_is_63d_rolling_zscore():
    data, factor_px = _macro_stub()
    features = build_macro_cross_features(
        data, config=PipelineConfig(s16_unit_fixpack_enabled=True)
    )
    raw_slope = factor_px["UST_10Y"] - factor_px["UST_2Y"]
    expected = _rolling_zscore(raw_slope, window=63)
    got = _slope_scalar_from(features)
    assert np.allclose(got.fillna(0), expected.fillna(0), atol=1e-6)
    assert not np.allclose(got.fillna(0), raw_slope.fillna(0), atol=1e-6)


def test_macro_cross_clip_diagnostic_printed_only_when_on(capsys):
    data, _ = _macro_stub()
    build_macro_cross_features(data, config=PipelineConfig())
    assert "[MacroCross] skip-CS-z ON" not in capsys.readouterr().out
    build_macro_cross_features(
        data, config=PipelineConfig(s16_unit_fixpack_enabled=True)
    )
    out = capsys.readouterr().out
    assert "[MacroCross] skip-CS-z ON: pre-clip |z|>5 cells =" in out


# --- assembly 배선: skip_zscore 에 MacroCross 포함 여부 -----------------------
# 두 번째 CS z-score의 지문은 "날짜별 횡단면 표준편차 == 1.0 고정"이다.
# 통과하면 매크로 스칼라의 크기가 전부 소거되고 부호만 남는다(리뷰 실측
# |corr(mc_rate_x_eps_rev, eps_rev)| = 1.0000). 스킵하면 크기가 살아난다.

def _assembly_panel(monkeypatch, flag, n=400):
    rng = np.random.default_rng(16)
    dates = pd.bdate_range("2020-01-01", periods=n)
    tickers = ["AAA", "BBB", "CCC"]
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, 3)), axis=0)),
        index=dates, columns=tickers,
    )
    meta = pd.DataFrame(
        {"Ticker": tickers, "Name": tickers, "Sector": ["Test"] * 3,
         "Status": ["Active"] * 3},
        index=[f"{t} US Equity" for t in tickers],
    )
    factor_px = pd.DataFrame(
        {
            "UST_10Y": 2.0 + rng.normal(0, 0.05, n).cumsum(),
            "UST_2Y": 1.0 + rng.normal(0, 0.05, n).cumsum(),
            "VIX": 18.0 + rng.normal(0, 0.5, n).cumsum(),
            "DXY": 100.0 + rng.normal(0, 0.5, n).cumsum(),
        },
        index=dates,
    )
    raw = {
        "Universe_Meta": meta,
        "PX_LAST": prices,
        "Daily_Returns": prices.pct_change(fill_method=None).fillna(0.0),
        "Factor_PX_LAST": factor_px,
        "Factor_Returns": factor_px.pct_change(fill_method=None).fillna(0.0),
    }
    for sheet in ESSENTIAL_SHEETS:
        raw[sheet] = pd.DataFrame(
            rng.normal(10.0, 1.0, (n, 3)), index=dates, columns=tickers
        )
    monkeypatch.setattr(
        "src.data_loader.load_all_sheets",
        lambda _path: {name: frame.copy() for name, frame in raw.items()},
    )
    from src.features.assembly import build_all_features
    cfg = PipelineConfig(
        fx_source_path="missing.xlsx",
        listing_auto_infer_enabled=False,
        s16_unit_fixpack_enabled=flag,
    )
    data = UniverseData("unused.xlsx", config=cfg)
    panel, _names, groups = build_all_features(data, config=cfg)
    return panel, groups


_MACRO_SCALED = ["mc_rate_x_eps_rev", "mc_slope_x_eps_rev", "mc_vix_x_mom252"]


def test_assembly_off_parity_second_zscore_erases_macro_amplitude(monkeypatch):
    panel, groups = _assembly_panel(monkeypatch, flag=False)
    assert set(_MACRO_SCALED).issubset(groups["MacroCross"])
    for col in _MACRO_SCALED:
        per_date_std = panel[col].groupby(level="date").std().iloc[-100:]
        assert np.allclose(per_date_std, 1.0, atol=1e-5)


def test_assembly_on_keeps_macro_amplitude(monkeypatch):
    panel, groups = _assembly_panel(monkeypatch, flag=True)
    assert set(_MACRO_SCALED).issubset(groups["MacroCross"])
    for col in _MACRO_SCALED:
        per_date_std = panel[col].groupby(level="date").std().iloc[-100:]
        assert per_date_std.std() > 0.1
        assert not np.allclose(per_date_std, 1.0, atol=1e-5)
