# -*- coding: utf-8 -*-
"""§S15 fix-pack — 단일 default-OFF 플래그(s15_fixpack_enabled) 뒤 8건
정확성 수정의 OFF 파리티·ON 동작 테스트."""
import numpy as np
import pandas as pd
import pytest

from src.backtest import (_matured_trailing_ic_mean, apply_growth_tilt,
                          apply_pead_boost)
from src.config import PipelineConfig
from src.features.assembly import build_lean_momentum_composites
from src.features.macro_cross import build_macro_cross_features
from src.features.sellside import clean_revision_spikes
from src.portfolio_optimizer import _pairwise_covariance


class _FakeData:
    """UniverseData의 필요 속성만 갖는 테스트 대역."""

    def __init__(self, sheets=None, dates=None, tickers=None,
                 factor_prices=None, factor_returns=None,
                 returns_masked=None, prices=None, earnings_timeline=None):
        self._sheets = sheets or {}
        self.dates = dates
        self.tickers = tickers or []
        self.factor_prices = factor_prices
        self.factor_returns = factor_returns
        self.returns_masked = returns_masked
        self.prices = prices
        self.earnings_timeline = earnings_timeline
        self.local_prices = None

    def get_sheet(self, name):
        if name in self._sheets:
            return self._sheets[name]
        raise KeyError(name)

    def has_factor_data(self):
        return self.factor_returns is not None or self.factor_prices is not None


def test_fixpack_flag_default_off():
    assert PipelineConfig().s15_fixpack_enabled is False


# ---------------------------------------------------------------------------
# 1. growth tilt: NaN boost가 유효 예측을 파괴하지 않도록 (ON)
# ---------------------------------------------------------------------------

def _growth_setup(fixpack):
    dates = pd.bdate_range("2020-01-01", periods=300)
    cols = ["A", "B"]
    predictions = pd.DataFrame(1.0, index=dates, columns=cols)
    eps = pd.DataFrame(10.0, index=dates, columns=cols)
    sales = pd.DataFrame(20.0, index=dates, columns=cols)
    data = _FakeData(sheets={"BEST_EPS": eps, "BEST_SALES": sales})
    cfg = PipelineConfig(
        growth_tilt_enabled=True, growth_tilt_weight=0.25,
        s15_fixpack_enabled=fixpack,
    )
    return apply_growth_tilt(predictions, data, cfg)


def test_growth_tilt_off_parity_nan_destruction_preserved():
    # OFF: shift(252) 워밍업 구간의 tilt NaN이 유효 예측을 NaN으로 만든다
    # (기존 동작 그대로 = 파리티).
    out = _growth_setup(fixpack=False)
    assert out.iloc[0].isna().all()
    assert out.iloc[100].isna().all()


def test_growth_tilt_on_preserves_valid_predictions():
    out = _growth_setup(fixpack=True)
    # ON: tilt NaN 셀은 boost 0으로 처리 — 유효 예측 보존.
    assert np.allclose(out.iloc[0], 1.0)
    assert np.allclose(out.iloc[100], 1.0)
    # 성숙 구간(워밍업 이후)은 OFF와 동일해야 한다.
    off = _growth_setup(fixpack=False)
    tail_off = off.iloc[280]
    tail_on = out.iloc[280]
    assert np.allclose(tail_off, tail_on)


# ---------------------------------------------------------------------------
# 2. PEAD: 달력일 -> 거래일 거리 (ON)
# ---------------------------------------------------------------------------

def _pead_setup(fixpack):
    dates = pd.bdate_range("2020-03-02", periods=60)   # 월~금 그리드
    cols = ["A"]
    predictions = pd.DataFrame(0.0, index=dates, columns=cols)
    event_date = dates[39]
    assert event_date.dayofweek == 4  # 금요일이어야 주말 교차 검증이 됨
    tl = pd.DataFrame(0, index=dates, columns=cols)
    tl.loc[event_date, "A"] = 1
    rev = pd.DataFrame(50.0, index=dates, columns=cols)
    data = _FakeData(sheets={"Factset_EPS_Revision": rev}, earnings_timeline=tl)
    cfg = PipelineConfig(
        pead_boost_enabled=True, pead_boost_weight=0.30,
        pead_decay_days=7.0, pead_max_days=21,
        s15_fixpack_enabled=fixpack,
    )
    return apply_pead_boost(predictions, data, cfg), dates, event_date


def test_pead_off_parity_calendar_days():
    out, dates, event = _pead_setup(fixpack=False)
    monday = dates[40]
    assert (monday - event).days == 3
    expected = 0.30 * np.exp(-3.0 / 7.0) * 0.5
    assert out.loc[monday, "A"] == pytest.approx(expected, rel=1e-9)


def test_pead_on_trading_days():
    out, dates, event = _pead_setup(fixpack=True)
    monday = dates[40]
    expected = 0.30 * np.exp(-1.0 / 7.0) * 0.5   # 거래일 1일
    assert out.loc[monday, "A"] == pytest.approx(expected, rel=1e-9)
    # 이벤트 당일은 양쪽 동일 (거리 0)
    assert out.loc[event, "A"] == pytest.approx(0.30 * 0.5, rel=1e-9)


# ---------------------------------------------------------------------------
# 3. factor 캘린더 테스트는 tests/test_factor.py 로 분리 (TDD 가드 파일명 규약)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 4. macro_cross: 유니버스 캘린더 (ON)
# ---------------------------------------------------------------------------

def _macro_setup(fixpack):
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2020-01-01", periods=150)
    factor_cal = dates.delete(100)
    fp = pd.DataFrame({"VIX": 15 + rng.standard_normal(149).cumsum()},
                      index=factor_cal)
    rets = pd.DataFrame(rng.normal(0, 0.01, size=(150, 2)),
                        index=dates, columns=["A", "B"])
    data = _FakeData(dates=dates, tickers=["A", "B"], factor_prices=fp,
                     returns_masked=rets)
    cfg = PipelineConfig(s15_fixpack_enabled=fixpack)
    return build_macro_cross_features(data, config=cfg), dates


def test_macro_cross_off_parity_restricted_calendar():
    feats, dates = _macro_setup(fixpack=False)
    f = feats["mc_vol_x_mom63"]
    assert dates[100] not in f.index


def test_macro_cross_on_full_calendar():
    feats, dates = _macro_setup(fixpack=True)
    f = feats["mc_vol_x_mom63"]
    assert f.index.equals(dates)
    assert np.isfinite(f.loc[dates[100]]).all()


# ---------------------------------------------------------------------------
# 5. mom_accel_63_252: 부분창 금지 (ON)
# ---------------------------------------------------------------------------

def _mom_setup(fixpack, short_history=True):
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2019-01-01", periods=320)
    rets = pd.DataFrame(rng.normal(0, 0.01, size=(320, 3)),
                        index=dates, columns=["A", "B", "C"])
    if short_history:
        rets.iloc[:150, 2] = np.nan   # C = 신규 상장 (마스크)
    prices = (1.0 + rets.fillna(0.0)).cumprod() * 100.0
    data = _FakeData(dates=dates, tickers=["A", "B", "C"],
                     returns_masked=rets, prices=prices)
    cfg = PipelineConfig(s15_fixpack_enabled=fixpack)
    return build_lean_momentum_composites(data, ["A", "B", "C"], config=cfg)


def test_mom_accel_off_parity_partial_window():
    feats = _mom_setup(fixpack=False)
    # OFF: C는 170obs로 min_periods=126을 충족 -> 부분합 기반 값 존재
    assert np.isfinite(feats["mom_accel_63_252"].iloc[-1]["C"])


def test_mom_accel_on_requires_full_window():
    feats = _mom_setup(fixpack=True)
    # ON: C는 252obs 미만 -> NaN (median-fill 중립으로 하류 처리)
    assert np.isnan(feats["mom_accel_63_252"].iloc[-1]["C"])
    # 전 종목 252obs 이후(글로벌 워밍업 램프 밖)에서는 ON == OFF
    full_off = _mom_setup(fixpack=False, short_history=False)
    full_on = _mom_setup(fixpack=True, short_history=False)
    pd.testing.assert_frame_equal(full_off["mom_accel_63_252"].iloc[252:],
                                  full_on["mom_accel_63_252"].iloc[252:])


# ---------------------------------------------------------------------------
# 6. revision cleaner: 지속형 롤오버 전방 연장 (ON)
# ---------------------------------------------------------------------------

def _rollover_series():
    dates = pd.bdate_range("2020-03-02", periods=30)   # 3월 = 캘린더 fallback 비활성 월
    vals = np.concatenate([np.full(10, 80.0), np.full(10, 5.0), np.full(10, 60.0)])
    return pd.DataFrame({"A": vals}, index=dates)


def test_cleaner_off_parity_one_day_delay():
    rev = _rollover_series()
    out = clean_revision_spikes(rev, threshold=30, mode="reversion_gated",
                                extreme_threshold=50.0, reversion_ratio=0.5)
    # 기존 동작: 전환일만 마스킹 -> t+1에 스텝 전량 유입
    assert out["A"].iloc[10] == pytest.approx(80.0)
    assert out["A"].iloc[11] == pytest.approx(5.0)


def test_cleaner_on_extends_through_persistent_collapse():
    rev = _rollover_series()
    out = clean_revision_spikes(rev, threshold=30, mode="reversion_gated",
                                extreme_threshold=50.0, reversion_ratio=0.5,
                                persistent_rollover_extension=True)
    # 붕괴 지속 구간(11..19) 전체 마스킹 -> 직전 정상값 80 유지
    assert np.allclose(out["A"].iloc[10:20], 80.0)
    # 회복(60: |60-80|=20 < 30) 시 해제
    assert out["A"].iloc[20] == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# 7. pairwise 공분산 대각 최소 관측수 (ON)
# ---------------------------------------------------------------------------

def _pairwise_setup():
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2020-01-01", periods=126)
    df = pd.DataFrame({
        "A": rng.normal(0, 0.01, 126),
        "B": rng.normal(0, 0.012, 126),
        "C": np.nan,
    }, index=idx)
    df.iloc[-10:, 2] = rng.normal(0, 0.05, 10)   # C: 10obs만
    return df


def test_pairwise_off_parity_small_sample_diag():
    recent = _pairwise_setup()
    cov = _pairwise_covariance(recent)
    var_c_small = recent["C"].var()
    assert cov[2, 2] == pytest.approx(var_c_small, rel=1e-6)


def test_pairwise_on_enforces_diag_min_obs():
    recent = _pairwise_setup()
    cov = _pairwise_covariance(recent, enforce_diag_min_obs=True)
    fallback = np.median([recent["A"].var(), recent["B"].var()])
    assert cov[2, 2] == pytest.approx(fallback, rel=1e-6)
    # 관측 충분한 이름의 대각은 불변
    cov_off = _pairwise_covariance(recent)
    assert cov[0, 0] == pytest.approx(cov_off[0, 0], rel=1e-9)
    assert cov[1, 1] == pytest.approx(cov_off[1, 1], rel=1e-9)


# ---------------------------------------------------------------------------
# 8. trailing-IC 성숙 필터 (ON 경로 helper)
# ---------------------------------------------------------------------------

def test_matured_trailing_ic_mean_filters_unmatured():
    ic_values = [(pd.Timestamp("2020-01-15"), 0.1),
                 (pd.Timestamp("2020-02-14"), 0.2),
                 (pd.Timestamp("2020-03-16"), 0.3)]
    append_idxs = [10, 31, 52]
    # t_idx=62, horizon=20: 10+20<=62, 31+20<=62 성숙; 52+20=72 미성숙
    got = _matured_trailing_ic_mean(ic_values, append_idxs, t_idx=62,
                                    horizon=20, window=6)
    assert got == pytest.approx(0.15)


def test_matured_trailing_ic_mean_needs_two_entries():
    ic_values = [(pd.Timestamp("2020-01-15"), 0.1),
                 (pd.Timestamp("2020-02-14"), 0.2)]
    append_idxs = [10, 31]
    # 성숙 1개뿐 -> 기존 관례대로 0.0
    assert _matured_trailing_ic_mean(ic_values, append_idxs, t_idx=32,
                                     horizon=20, window=6) == 0.0
    # 프로덕션 스페이싱(21BD >= horizon 20)에서는 전 엔트리 성숙 = 기존 동일
    got = _matured_trailing_ic_mean(ic_values, append_idxs, t_idx=52,
                                    horizon=20, window=6)
    assert got == pytest.approx(np.nanmean([0.1, 0.2]))
