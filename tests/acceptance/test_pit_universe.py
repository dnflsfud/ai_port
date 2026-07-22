"""Acceptance tests — point-in-time universe (§S11.4).

Written from the spec BEFORE implementation. The spec extends the pre-listing
mask from PX_LAST/CUR_MKT_CAP to EVERY per-ticker model-input sheet with a
double-mask pipeline (mask before impute so ghosts cannot seed the
cross-sectional median, re-mask after align so pre-listing cells end NaN),
keeps Daily_Returns dense for the simulation P&L path (first-pass mask +
median refill, no re-mask; §S11.7 이후 PCA/피처는 returns_masked 뷰를 소비하고
공분산은 마스킹된 raw_returns를 사용), converts fixed-N breadth denominators
to per-date valid counts, and adds an ``expected_universe_size`` guard
(default None).

======================================================================
합격기준(spec) ↔ 테스트 매핑표
----------------------------------------------------------------------
P-A  expected_universe_size: default None, <=0 거부                    -> test_expected_universe_size_defaults
P-B  재무 시트 상장 전 셀 NaN 확정, 상장일 당일 보존                     -> test_fundamental_sheet_masked_pre_listing
P-C  1차 마스킹: 유령 상수가 임퓨트 median에 불포함                      -> test_impute_median_excludes_ghost
P-D  OFF 파리티: mask OFF면 유령 보존 + 레거시 median(유령 포함)         -> test_off_parity_keeps_ghost
P-E  Daily_Returns dense 유지(유령 0.0 -> 상장종목 median; 시뮬 P&L용),
     raw_returns(공분산 경로)는 상장 전 NaN                              -> test_daily_returns_dense_and_raw_masked
P-F  expected_universe_size 가드: 불일치 raise, 일치/None 통과           -> test_expected_universe_size_guard
P-G  regime_breadth_50d 분모 = 날짜별 유효 종목 수                       -> test_breadth_denominator_is_per_date_valid_count
======================================================================
"""

import numpy as np
import pandas as pd
import pytest

from src.config import PipelineConfig
from src.data_loader import UniverseData


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

GHOST_EPS = 7.0  # NEW's constant backfill in BEST_EPS (pre- AND post-listing)


def _pit_fixture(n_dates=12, listing_pos=6):
    """3 US tickers; NEW carries constant ghost backfill before listing.

    AAA and BBB share identical percentage moves (BBB = 2 x AAA) so the
    cross-sectional median return of listed names equals AAA's return.
    AAA's BEST_EPS starts NaN to exercise the cross-sectional median impute.
    """
    dates = pd.bdate_range("2021-01-04", periods=n_dates)
    listing_date = dates[listing_pos]
    aaa = pd.Series(np.linspace(100.0, 100.0 + 2 * (n_dates - 1), n_dates), index=dates)
    prices = pd.DataFrame({"AAA": aaa, "BBB": aaa * 2.0, "NEW": 50.0}, index=dates)
    # NEW trades for real only from the listing date on.
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
    best_eps = pd.DataFrame(
        {
            "AAA": [np.nan] + [2.0] * (n_dates - 1),  # leading NaN -> median path
            "BBB": 1.0,
            "NEW": GHOST_EPS,  # constant backfill ghost
        },
        index=dates,
    )
    raw = {
        "Universe_Meta": meta,
        "PX_LAST": prices,
        "Daily_Returns": returns,
        "BEST_EPS": best_eps,
    }
    for sheet in ESSENTIAL_SHEETS - {"PX_LAST", "Daily_Returns", "BEST_EPS"}:
        raw[sheet] = pd.DataFrame(1.0, index=dates, columns=["AAA", "BBB", "NEW"])
    return raw, dates, listing_date


def _patch_raw(monkeypatch, raw):
    monkeypatch.setattr(
        "src.data_loader.load_all_sheets",
        lambda _path: {name: frame.copy() for name, frame in raw.items()},
    )


def _pit_config(**kwargs):
    kwargs.setdefault("fx_source_path", "missing.xlsx")
    return PipelineConfig(**kwargs)


# ---------------------------------------------------------------------------
# P-A: config defaults
# ---------------------------------------------------------------------------
def test_expected_universe_size_defaults():
    assert PipelineConfig().expected_universe_size is None
    with pytest.raises(ValueError):
        PipelineConfig(expected_universe_size=0)


# ---------------------------------------------------------------------------
# P-B: fundamental sheets are masked pre-listing (listing day preserved)
# ---------------------------------------------------------------------------
def test_fundamental_sheet_masked_pre_listing(monkeypatch):
    raw, dates, listing_date = _pit_fixture()
    _patch_raw(monkeypatch, raw)
    data = UniverseData(
        "unused.xlsx",
        config=_pit_config(listing_dates={"NEW": str(listing_date.date())}),
    )

    eps = data.get_sheet("BEST_EPS")["NEW"]
    assert eps.loc[: listing_date - pd.Timedelta(days=1)].isna().all()
    # listing day itself is the first real observation (inclusive=False)
    assert eps.loc[listing_date] == GHOST_EPS


# ---------------------------------------------------------------------------
# P-C: first-pass mask keeps ghosts out of the impute median
# ---------------------------------------------------------------------------
def test_impute_median_excludes_ghost(monkeypatch):
    raw, dates, listing_date = _pit_fixture()
    _patch_raw(monkeypatch, raw)
    data = UniverseData(
        "unused.xlsx",
        config=_pit_config(listing_dates={"NEW": str(listing_date.date())}),
    )

    # AAA's leading NaN is median-imputed on date[0]. With NEW masked first,
    # the cross-sectional median at date[0] sees only BBB (=1.0).
    assert data.get_sheet("BEST_EPS")["AAA"].iloc[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# P-D: OFF parity — legacy behaviour keeps the ghost and its median footprint
# ---------------------------------------------------------------------------
def test_off_parity_keeps_ghost(monkeypatch):
    raw, dates, listing_date = _pit_fixture()
    _patch_raw(monkeypatch, raw)
    data = UniverseData(
        "unused.xlsx",
        config=_pit_config(
            listing_mask_enabled=False,
            listing_dates={"NEW": str(listing_date.date())},
        ),
    )

    eps = data.get_sheet("BEST_EPS")
    # ghost preserved, and AAA's leading NaN imputed with the ghost included:
    # median(BBB=1.0, NEW=7.0) = 4.0
    assert (eps["NEW"] == GHOST_EPS).all()
    assert eps["AAA"].iloc[0] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# P-E: Daily_Returns stays dense (simulation P&L path) with ghost 0.0 replaced
#      by the median return of LISTED names; raw_returns (covariance path)
#      keeps NaN pre-listing. (§S11.7: PCA/피처는 returns_masked 뷰 소비)
# ---------------------------------------------------------------------------
def test_daily_returns_dense_and_raw_masked(monkeypatch):
    raw, dates, listing_date = _pit_fixture()
    _patch_raw(monkeypatch, raw)
    data = UniverseData(
        "unused.xlsx",
        config=_pit_config(listing_dates={"NEW": str(listing_date.date())}),
    )

    dr = data.get_sheet("Daily_Returns")
    assert not dr.isna().any().any()  # dense for the simulation P&L path
    # date[1] is pre-listing: NEW's ghost 0.0 must be replaced by the
    # cross-sectional median of listed names == AAA's return that day.
    expected = raw["Daily_Returns"].loc[dates[1], "AAA"]
    assert expected != 0.0  # fixture sanity: discriminating value
    assert dr.loc[dates[1], "NEW"] == pytest.approx(expected)

    # cov-path raw returns keep the mask (inclusive=True: listing day too)
    assert data.raw_returns["NEW"].loc[:listing_date].isna().all()
    post = data.raw_returns["NEW"].loc[listing_date + pd.Timedelta(days=1):]
    assert post.notna().all()


# ---------------------------------------------------------------------------
# P-F: expected_universe_size guard
# ---------------------------------------------------------------------------
def test_expected_universe_size_guard(monkeypatch):
    raw, dates, listing_date = _pit_fixture()

    _patch_raw(monkeypatch, raw)
    with pytest.raises(ValueError):
        UniverseData(
            "unused.xlsx",
            config=_pit_config(expected_universe_size=150),
        )

    _patch_raw(monkeypatch, raw)
    data = UniverseData(
        "unused.xlsx",
        config=_pit_config(expected_universe_size=3),
    )
    assert len(data.tickers) == 3


# ---------------------------------------------------------------------------
# P-G: breadth denominator adapts to the per-date valid (listed) count
# ---------------------------------------------------------------------------
def test_breadth_denominator_is_per_date_valid_count(monkeypatch):
    from src.features.conditioning import build_conditioning_features

    raw, dates, listing_date = _pit_fixture(n_dates=60, listing_pos=55)
    _patch_raw(monkeypatch, raw)
    data = UniverseData(
        "unused.xlsx",
        config=_pit_config(listing_dates={"NEW": str(listing_date.date())}),
    )

    features = build_conditioning_features(data)
    breadth = features["regime_breadth_50d"]
    # On the last date AAA/BBB sit above their rising 50d MA while NEW (listed
    # 5 days ago) has no 50d MA yet: valid denominator is 2, so breadth == 1.0.
    # The legacy fixed-N denominator would report 2/3.
    assert breadth.iloc[-1, 0] == pytest.approx(1.0)
