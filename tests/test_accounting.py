# -*- coding: utf-8 -*-
"""§S16.1 P2·P3 — accounting.py 단위 수정의 OFF 파리티·ON 동작.

fix-pack 전체(플래그 기본값·P1·P4)는 tests/test_s16_unit_fixpack.py 에 있고,
accounting 모듈분만 TDD 가드 파일명 규약에 따라 여기로 분리한다(§S15의
tests/test_factor.py 선례).
"""
import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.features.accounting import build_accounting_features
from src.features.utils import cross_sectional_zscore, rolling_tsz


class _FakeData:
    """UniverseData 중 build_accounting_features가 쓰는 속성만."""

    def __init__(self, sheets, market_cap=None, prices=None):
        self._sheets = sheets
        self.market_cap = market_cap
        self.prices = prices

    def get_sheet(self, name):
        if name in self._sheets:
            return self._sheets[name]
        raise KeyError(name)


def _close(a, b):
    return np.allclose(a.fillna(0), b.fillna(0), atol=1e-6)


def _synthetic(n=800):
    rng = np.random.default_rng(16)
    dates = pd.bdate_range("2018-01-01", periods=n)
    cols = ["A", "B", "C"]

    def panel(base, scale):
        vals = base + rng.normal(0, 0.5, size=(n, len(cols)))
        return pd.DataFrame(vals * scale, index=dates, columns=cols)

    sheets = {
        "BEST_EPS": panel(10.0, [1.0, 2.0, 0.5]),
        "BEST_SALES": panel(200.0, [1.0, 3.0, 0.8]),
        "BEST_CALCULATED_FCF": panel(500.0, [1.0, 1500.0, 130.0]),
        "BEST_CAPEX": panel(120.0, [1.0, 1500.0, 130.0]),
        "BEST_ROE": panel(15.0, [1.0, 1.1, 0.9]),
    }
    market_cap = pd.DataFrame(
        np.tile([2.0e11, 9.0e10, 5.0e11], (n, 1)), index=dates, columns=cols
    )
    prices = pd.DataFrame(
        np.tile([150.0, 30.0, 90.0], (n, 1)), index=dates, columns=cols
    )
    return _FakeData(sheets, market_cap=market_cap, prices=prices)


# ---------------------------------------------------------------------------
# P2. FCF / CAPEX level_z: 현지통화 절대금액 횡단면 비교 제거
# ---------------------------------------------------------------------------

def test_level_z_off_parity_compares_raw_levels():
    data = _synthetic()
    implicit = build_accounting_features(data)
    explicit = build_accounting_features(data, PipelineConfig())
    for feats in (implicit, explicit):
        for sheet, key in (("BEST_CALCULATED_FCF", "best_calculated_fcf_level_z"),
                           ("BEST_CAPEX", "best_capex_level_z"),
                           ("BEST_ROE", "best_roe_level_z")):
            assert _close(feats[key],
                          cross_sectional_zscore(data.get_sheet(sheet)))


def test_level_z_on_self_normalizes_only_currency_level_sheets():
    data = _synthetic()
    off = build_accounting_features(data, PipelineConfig())
    on = build_accounting_features(
        data, PipelineConfig(s16_unit_fixpack_enabled=True)
    )
    for sheet, key in (("BEST_CALCULATED_FCF", "best_calculated_fcf_level_z"),
                       ("BEST_CAPEX", "best_capex_level_z")):
        raw = data.get_sheet(sheet)
        assert _close(on[key], cross_sectional_zscore(
            rolling_tsz(raw, window=756, min_periods=252)
        ))
        assert not _close(on[key], off[key])
    # 퍼센트 시트와 whitelist 밖 파생물은 불변
    assert _close(on["best_roe_level_z"], off["best_roe_level_z"])
    for key in ("best_calculated_fcf_rank", "best_calculated_fcf_vs_median",
                "best_capex_rank", "best_capex_vs_median"):
        assert _close(on[key], off[key])


def test_level_z_on_removes_currency_magnitude_dominance():
    """1000배 단위 차이가 횡단면 순서를 지배하던 것이 ON에서 해소된다."""
    n = 800
    dates = pd.bdate_range("2018-01-01", periods=n)
    wave = np.sin(np.arange(n) / 50.0)
    fcf = pd.DataFrame(
        {"A": 100.0 + 10.0 * wave, "B": 1000.0 * (100.0 - 10.0 * wave)},
        index=dates,
    )
    eps = pd.DataFrame({"A": 10.0, "B": 10.0}, index=dates)
    data = _FakeData(
        {"BEST_CALCULATED_FCF": fcf, "BEST_EPS": eps},
        market_cap=pd.DataFrame({"A": 1.0e11, "B": 1.0e11}, index=dates),
        prices=pd.DataFrame({"A": 100.0, "B": 100.0}, index=dates),
    )
    off = build_accounting_features(data, PipelineConfig())
    on = build_accounting_features(
        data, PipelineConfig(s16_unit_fixpack_enabled=True)
    )
    peak = int(np.argmax(wave[600:])) + 600     # A는 자기 노름 위, B는 아래
    off_row = off["best_calculated_fcf_level_z"].iloc[peak]
    on_row = on["best_calculated_fcf_level_z"].iloc[peak]
    # OFF: 크기(=단위)만 보므로 B가 늘 위 — 전 구간 부호가 고정된다.
    off_frame = off["best_calculated_fcf_level_z"].iloc[252:]
    assert (off_frame["B"] > off_frame["A"]).all()
    assert off_row["B"] > off_row["A"]
    # ON: 각자 자기 히스토리 대비 위치 -> 이 날짜에서는 A가 위.
    assert on_row["A"] > on_row["B"]


# ---------------------------------------------------------------------------
# P3. cash_conversion_z: 총액 / 주당 -> 총액 / 순이익
# ---------------------------------------------------------------------------

def test_cash_conversion_off_parity_total_over_per_share():
    data = _synthetic()
    feats = build_accounting_features(data, PipelineConfig())
    fcf = data.get_sheet("BEST_CALCULATED_FCF")
    eps = data.get_sheet("BEST_EPS")
    ref = cross_sectional_zscore(fcf / eps.replace(0, np.nan).abs())
    assert _close(feats["cash_conversion_z"], ref)
    assert _close(build_accounting_features(data)["cash_conversion_z"], ref)


def test_cash_conversion_on_divides_by_net_income():
    data = _synthetic()
    feats = build_accounting_features(
        data, PipelineConfig(s16_unit_fixpack_enabled=True)
    )
    fcf = data.get_sheet("BEST_CALCULATED_FCF")
    eps = data.get_sheet("BEST_EPS")
    shares = data.market_cap / data.prices.replace(0, np.nan)
    net_income = eps * shares
    ref = cross_sectional_zscore(fcf / net_income.replace(0, np.nan).abs())
    assert _close(feats["cash_conversion_z"], ref)
    off = build_accounting_features(data, PipelineConfig())
    assert not _close(feats["cash_conversion_z"], off["cash_conversion_z"])
