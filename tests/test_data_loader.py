"""TDD-guard stem test for src/data_loader.mask_pre_listing.

Authoritative coverage lives in tests/acceptance/test_listing_mask.py. This
stem-named file exists so the pytest-tdd PreToolUse guard permits editing
src/data_loader.py.
"""

import numpy as np
import pandas as pd
import pytest


def test_mask_pre_listing_inclusive_masks_listing_day():
    from src.data_loader import mask_pre_listing

    idx = pd.to_datetime(["2020-09-29", "2020-09-30", "2020-10-01"])
    df = pd.DataFrame({"PLTR": [1.0, 2.0, 3.0]}, index=idx)
    out = mask_pre_listing(df, {"PLTR": "2020-09-30"}, inclusive=True)
    assert bool(np.isnan(out["PLTR"].iloc[0]))
    assert bool(np.isnan(out["PLTR"].iloc[1]))
    assert out["PLTR"].iloc[2] == 3.0
    # input frame is not mutated
    assert df["PLTR"].iloc[0] == 1.0


def test_align_dates_removes_weekends_from_intersection_and_tail():
    from src.config import PipelineConfig
    from src.data_loader import align_dates

    px_dates = pd.to_datetime([
        "2026-06-04", "2026-06-05", "2026-06-06", "2026-06-07", "2026-06-08",
    ])
    slow_dates = pd.to_datetime(["2026-06-04", "2026-06-05", "2026-06-06"])
    processed = {
        "PX_LAST": pd.DataFrame({"AAA": range(len(px_dates))}, index=px_dates),
        "SLOW": pd.DataFrame({"AAA": range(len(slow_dates))}, index=slow_dates),
    }
    diagnostics = {}
    aligned = align_dates(
        processed,
        config=PipelineConfig(max_tail_ffill_days=10),
        diagnostics=diagnostics,
    )

    assert list(aligned["PX_LAST"].index.strftime("%Y-%m-%d")) == [
        "2026-06-04", "2026-06-05", "2026-06-08",
    ]
    assert diagnostics["weekend_dates_removed"] == 2
    assert diagnostics["calendar_type"] == "weekday_index"
    assert diagnostics["tail_ffill_days"] == 1


def test_align_dates_can_fail_fast_on_stale_weekday_tail():
    from src.config import PipelineConfig
    from src.data_loader import align_dates

    px_dates = pd.bdate_range("2026-06-01", periods=5)
    processed = {
        "PX_LAST": pd.DataFrame({"AAA": range(5)}, index=px_dates),
        "SLOW": pd.DataFrame({"AAA": range(2)}, index=px_dates[:2]),
    }
    with pytest.raises(ValueError, match="Tail ffill length 3 exceeds"):
        align_dates(
            processed,
            config=PipelineConfig(
                max_tail_ffill_days=2,
                fail_on_stale_tail_ffill=True,
            ),
        )


# ---------------------------------------------------------------------------
# §S11.7: returns_masked — 상장 전 NaN 뷰 (피처·PCA 소비용). dense returns는
# 시뮬레이션 P&L용으로 불변(공분산 경로는 마스킹된 raw_returns 사용).
# ---------------------------------------------------------------------------
def _masked_view_data(config):
    from src.data_loader import UniverseData

    idx = pd.bdate_range("2020-09-28", periods=5)
    dense = pd.DataFrame({"AAA": [0.01] * 5, "NEW": [0.02] * 5}, index=idx)
    data = UniverseData.__new__(UniverseData)
    data.sheets = {"Daily_Returns": dense}
    data.config = config
    return data


def test_returns_masked_masks_through_listing_day():
    from src.config import PipelineConfig

    data = _masked_view_data(PipelineConfig(listing_dates={"NEW": "2020-09-30"}))
    out = data.returns_masked
    # inclusive=True: 상장일 수익률도 가짜(백필 기준가 대비)이므로 NaN
    assert out["NEW"].loc[:"2020-09-30"].isna().all()
    assert out["NEW"].loc["2020-10-01":].notna().all()
    assert out["AAA"].notna().all()
    # dense 원본은 불변
    assert data.returns["NEW"].notna().all()


def test_returns_masked_is_identity_when_mask_disabled():
    from src.config import PipelineConfig

    data = _masked_view_data(
        PipelineConfig(listing_mask_enabled=False, listing_dates={"NEW": "2020-09-30"})
    )
    assert data.returns_masked is data.returns


# ---------------------------------------------------------------------------
# resolve_listing_dates: listing_meta_columns 우선순위 (PIT 계약서 §1 —
# Eligibility_Start_Date > Listing_Date > IPO_Date > First_Trade_Date)
# ---------------------------------------------------------------------------
def test_resolve_listing_dates_first_meta_column_wins():
    from src.config import PipelineConfig
    from src.data_loader import resolve_listing_dates

    meta = pd.DataFrame(
        {
            "Eligibility_Start_Date": ["2021-03-01", None],
            "IPO_Date": ["2019-06-15", "2020-01-02"],
        },
        index=pd.Index(["AAA", "BBB"], name="ticker"),
    )
    resolved, sources = resolve_listing_dates(meta, raw={}, config=PipelineConfig())
    # AAA carries both columns -> first-listed (Eligibility) must win
    assert resolved["AAA"] == "2021-03-01"
    assert sources["AAA"] == "meta:Eligibility_Start_Date"
    # BBB only has the lower-priority column -> it still fills the gap
    assert resolved["BBB"] == "2020-01-02"
    assert sources["BBB"] == "meta:IPO_Date"


def test_resolve_listing_dates_config_override_beats_meta():
    from src.config import PipelineConfig
    from src.data_loader import resolve_listing_dates

    meta = pd.DataFrame(
        {"Eligibility_Start_Date": ["2021-03-01"]},
        index=pd.Index(["AAA"], name="ticker"),
    )
    cfg = PipelineConfig(listing_dates={"AAA": "2022-01-05"})
    resolved, sources = resolve_listing_dates(meta, raw={}, config=cfg)
    assert resolved["AAA"] == "2022-01-05"
    assert sources["AAA"] == "config_override"


# ---------------------------------------------------------------------------
# UniverseData currency 해석: 접미사도 fallback 사전 항목도 없는 티커가
# 조용히 USD로 기장되면 안 된다 (Universe_Meta가 있을 때만 강제).
# ---------------------------------------------------------------------------
def test_universedata_rejects_silent_usd_for_unknown_bare_ticker(monkeypatch):
    import src.data_loader as dl

    meta = pd.DataFrame(
        {"Ticker": ["005930", "ZZZQ"], "Name": ["Samsung", "Mystery"]}
    )
    monkeypatch.setattr(dl, "load_all_sheets", lambda path: {"Universe_Meta": meta})
    # 005930 is bare but in FALLBACK_TICKER_CURRENCY -> passes; ZZZQ is bare
    # with no fallback entry -> must fail loudly, naming the ticker.
    with pytest.raises(ValueError, match="ZZZQ"):
        dl.UniverseData("unused.xlsx")


# ---------------------------------------------------------------------------
# 캘린더 소스 시트 제한 (구조 리뷰 2026-08-27, default-OFF).
#
# align_dates가 모든 시트를 교집합에 넣기 때문에, 플래그-OFF 연구 시트 하나가
# production 백테스트 캘린더를 자른다(실측: 교집합 3,282행 vs 최장 시트 4,984행).
# ON이면 CALENDAR_EXEMPT_SHEETS만 교집합에서 빠지고, reindex 대상에는 남는다.
# ---------------------------------------------------------------------------
def test_calendar_exempt_flag_default_off():
    from src.config import PipelineConfig

    assert PipelineConfig().calendar_exempt_sheets_enabled is False


def test_live_sheets_are_not_calendar_exempt():
    """production variant가 실제로 소비하는 시트는 절대 면제되면 안 된다.
    Fwd_Sales_Slope_1FY2FY = §S13.25 (fwd_sales_slope_features_enabled: true),
    iv30_z = §S13.41 공분산 대각 (option_vol_covariance_enabled: true)."""
    from src.data_loader import CALENDAR_EXEMPT_SHEETS

    assert "Fwd_Sales_Slope_1FY2FY" not in CALENDAR_EXEMPT_SHEETS
    assert "iv30_z" not in CALENDAR_EXEMPT_SHEETS
    # essential 시트(유니버스 멤버십 정의)도 면제 대상이 아니다.
    for essential in ("PX_LAST", "Daily_Returns", "CUR_MKT_CAP", "BEST_EPS"):
        assert essential not in CALENDAR_EXEMPT_SHEETS


def _calendar_processed():
    """core 시트는 10영업일, arm 전용 시트(PX_VOLUME)는 뒤쪽 4일만 보유."""
    core_dates = pd.bdate_range("2026-06-01", periods=10)
    arm_dates = core_dates[-4:]
    return {
        "PX_LAST": pd.DataFrame({"AAA": range(10)}, index=core_dates),
        "Daily_Returns": pd.DataFrame({"AAA": range(10)}, index=core_dates),
        "PX_VOLUME": pd.DataFrame({"AAA": range(4)}, index=arm_dates),
    }, core_dates


def test_calendar_exempt_off_keeps_all_sheet_intersection():
    """OFF 파리티: 짧은 arm 시트가 여전히 교집합을 자른다(기존 동작)."""
    from src.config import PipelineConfig
    from src.data_loader import align_dates

    processed, _ = _calendar_processed()
    diagnostics = {}
    aligned = align_dates(
        processed, config=PipelineConfig(), diagnostics=diagnostics
    )
    assert len(aligned["PX_LAST"].index) == 4
    assert diagnostics["intersection_dates"] == 4
    # OFF 경로는 진단 키를 추가하지 않는다 (metrics.json 스키마 파리티)
    assert "calendar_exempt_sheets_applied" not in diagnostics
    assert "calendar_source_sheet_count" not in diagnostics


def test_calendar_exempt_on_excludes_arm_sheet_from_intersection():
    """ON: PX_VOLUME이 교집합에서 빠져 캘린더가 10일 전부 살아난다.
    면제 시트 자체는 새 캘린더로 reindex되어 계속 존재한다."""
    from src.config import PipelineConfig
    from src.data_loader import align_dates

    processed, core_dates = _calendar_processed()
    diagnostics = {}
    aligned = align_dates(
        processed,
        config=PipelineConfig(calendar_exempt_sheets_enabled=True),
        diagnostics=diagnostics,
    )
    assert list(aligned["PX_LAST"].index) == list(core_dates)
    assert diagnostics["intersection_dates"] == 10
    assert diagnostics["calendar_exempt_sheets_applied"] == ["PX_VOLUME"]
    assert diagnostics["calendar_source_sheet_count"] == 2
    # 면제 시트는 사라지지 않고 같은 캘린더로 정렬된다
    assert list(aligned["PX_VOLUME"].index) == list(core_dates)


def test_calendar_exempt_on_falls_back_when_every_sheet_is_exempt():
    """전 시트가 면제인 합성 워크북에서는 기존(전 시트 교집합) 동작 유지."""
    from src.config import PipelineConfig
    from src.data_loader import align_dates

    dates_a = pd.bdate_range("2026-06-01", periods=6)
    processed = {
        "PX_VOLUME": pd.DataFrame({"AAA": range(6)}, index=dates_a),
        "SHORT_INT_RATIO": pd.DataFrame({"AAA": range(4)}, index=dates_a[:4]),
    }
    diagnostics = {}
    aligned = align_dates(
        processed,
        config=PipelineConfig(calendar_exempt_sheets_enabled=True),
        diagnostics=diagnostics,
    )
    assert diagnostics["intersection_dates"] == 4
    assert "calendar_exempt_sheets_applied" not in diagnostics
    assert len(aligned["PX_VOLUME"].index) >= 4


# ---------------------------------------------------------------------------
# raw_sheet_observed_mask — 로더 임퓨트 이전 관측 마스크 (구조 리뷰 2026-08-27).
#
# preprocess_sheets의 _fill_missing(ffill -> 날짜별 횡단면 median)과 align_dates의
# 2차 fill이 결측을 전부 메우므로, 소비자는 관측 셀과 임퓨트 셀을 구별할 수 없다.
# "결측 입력 -> inert" 계약을 가진 소비자(§S13.41 공분산 스케일)가 조용히 무력화된다.
# ---------------------------------------------------------------------------
_OBS_ESSENTIAL = {
    "PX_LAST", "Daily_Returns", "CUR_MKT_CAP", "BEST_EPS", "BEST_SALES",
    "BEST_PE_RATIO", "OPER_MARGIN", "BEST_ROE", "NEWS_SENTIMENT_DAILY_AVG",
    "EQY_REC_CONS", "Factset_EPS_Revision", "Factset_Sales_Revision",
    "Factset_TG_Price",
}


def _observed_mask_workbook(n_dates=12):
    """AAA/BBB 2종목. iv30_z는 Bloomberg 열 이름 + BBB 앞 절반이 미관측."""
    dates = pd.bdate_range("2021-01-04", periods=n_dates)
    prices = pd.DataFrame(
        {"AAA": np.linspace(100.0, 111.0, n_dates),
         "BBB": np.linspace(50.0, 61.0, n_dates)},
        index=dates,
    )
    meta = pd.DataFrame(
        {"Ticker": ["AAA", "BBB"], "Name": ["Alpha", "Beta"],
         "Sector": ["Test", "Test"]},
        index=["AAA US Equity", "BBB US Equity"],
    )
    iv30_z = pd.DataFrame(
        {"AAA US Equity": np.linspace(0.5, 1.6, n_dates),
         "BBB US Equity": [np.nan] * (n_dates // 2)
                          + list(np.linspace(-0.4, 0.4, n_dates - n_dates // 2))},
        index=dates,
    )
    raw = {
        "Universe_Meta": meta,
        "PX_LAST": prices,
        "Daily_Returns": prices.pct_change(fill_method=None).fillna(0.0),
        "iv30_z": iv30_z,
    }
    for sheet in _OBS_ESSENTIAL - {"PX_LAST", "Daily_Returns"}:
        raw[sheet] = pd.DataFrame(1.0, index=dates, columns=["AAA", "BBB"])
    return raw, dates, n_dates // 2


def _observed_mask_data(monkeypatch):
    from src.config import PipelineConfig
    from src.data_loader import UniverseData

    raw, dates, split = _observed_mask_workbook()
    monkeypatch.setattr(
        "src.data_loader.load_all_sheets",
        lambda _path: {k: v.copy() for k, v in raw.items()},
    )
    cfg = PipelineConfig(fx_source_path="missing.xlsx")
    return UniverseData("unused.xlsx", config=cfg), dates, split


def test_loader_fill_hides_missing_cells_from_consumers(monkeypatch):
    """회귀 고정: get_sheet만으로는 미관측을 알 수 없다 (가드 무력화의 원인)."""
    data, _dates, split = _observed_mask_data(monkeypatch)
    sheet = data.get_sheet("iv30_z")
    assert not sheet["BBB"].isna().any()          # 결측이 메워져 사라졌다
    assert sheet["BBB"].iloc[:split].notna().all()


def test_raw_sheet_observed_mask_marks_unobserved_cells(monkeypatch):
    data, dates, split = _observed_mask_data(monkeypatch)
    mask = data.raw_sheet_observed_mask("iv30_z")

    assert mask is not None
    assert list(mask.columns) == list(data.tickers)   # Bloomberg 열 이름 정규화됨
    assert list(mask.index) == list(data.dates)
    assert mask["AAA"].all()                          # 전 구간 관측
    assert not mask["BBB"].iloc[:split].any()         # 앞 절반 미관측
    assert mask["BBB"].iloc[split:].all()             # 뒷 절반 관측


def test_raw_sheet_observed_mask_absent_sheet_is_none(monkeypatch):
    data, _dates, _split = _observed_mask_data(monkeypatch)
    assert data.raw_sheet_observed_mask("NO_SUCH_SHEET") is None


# ---------------------------------------------------------------------------
# 필수 시트 부재 fail-fast (구조 리뷰 2026-08-31, O8).
#
# 유니버스 교집합은 "워크북에 존재하는" 필수 시트로만 계산되므로, 시트 하나가
# 통째로 빠지면 티커 수는 그대로라 가드가 통과하고 손실은 피처층의 "whitelist
# misses" 한 줄로만 드러난다 (예: Factset_TG_Price 누락 -> tg_mom_63d/tg_upside
# 없이 학습된 결과가 실패 없이 metrics.json으로 나감).
# ---------------------------------------------------------------------------
def _data_without_sheet(monkeypatch, sheet, expected_universe_size):
    from src.config import PipelineConfig
    from src.data_loader import UniverseData

    raw, _dates, _split = _observed_mask_workbook()
    raw.pop(sheet)
    monkeypatch.setattr(
        "src.data_loader.load_all_sheets",
        lambda _path: {k: v.copy() for k, v in raw.items()},
    )
    cfg = PipelineConfig(
        fx_source_path="missing.xlsx",
        expected_universe_size=expected_universe_size,
    )
    return UniverseData("unused.xlsx", config=cfg)


def test_missing_essential_sheet_fails_fast_for_sized_universe(monkeypatch):
    with pytest.raises(ValueError, match="Factset_TG_Price"):
        _data_without_sheet(monkeypatch, "Factset_TG_Price", expected_universe_size=2)


def test_missing_essential_sheet_is_diagnostic_only_without_expected_size(monkeypatch):
    """합성 픽스처(expected_universe_size=None)는 계속 통과하되 진단은 남는다."""
    data = _data_without_sheet(
        monkeypatch, "Factset_TG_Price", expected_universe_size=None
    )
    assert data.data_quality["universe"]["missing_essential_sheets"] == [
        "Factset_TG_Price"
    ]
    assert list(data.tickers) == ["AAA", "BBB"]


# ---------------------------------------------------------------------------
# 영업일 캘린더 (감사 2026-09-09 Medium-2, default-OFF).
#
# price_v4 가 모든 시트를 일력(주말+휴일)으로 reindex+ffill 하고 align_dates 는
# 주말만 제거하므로, 미 휴일 119행이 production 캘린더에 남는다(US 종목 수익률
# 0 · 비US 는 실제 이동). ON 이면 raw 단계에서 BusinessDays 시트의 거래일만
# 남기고 Daily_Returns 를 남은 PX_LAST 행에서 재계산한다(휴일을 가로지르는
# 비US 이동은 다음 거래일에 복리로 합산 -- 유실 0).
# ---------------------------------------------------------------------------

def _holiday_workbook():
    # Thu 07-02, Fri 07-03 (US holiday: AAA ffilled, BBB moves), Sat, Sun, Mon 07-06
    dates = pd.to_datetime(
        ["2026-07-02", "2026-07-03", "2026-07-04", "2026-07-05", "2026-07-06"]
    )
    px = pd.DataFrame(
        {"AAA": [100.0, 100.0, 100.0, 100.0, 102.0],
         "BBB": [100.0, 101.0, 101.0, 101.0, 103.0]},
        index=dates,
    )
    raw = {
        "Universe_Meta": pd.DataFrame(
            {"Ticker": ["AAA", "BBB"], "Name": ["Alpha", "Beta"]},
            index=["AAA US Equity", "BBB US Equity"],
        ),
        "BusinessDays": pd.DataFrame(
            index=pd.Index(pd.to_datetime(["2026-07-02", "2026-07-06"]), name="BusinessDay")
        ),
        "PX_LAST": px,
        "Daily_Returns": px.pct_change(fill_method=None),
        "BEST_EPS": pd.DataFrame(1.0, index=dates, columns=["AAA", "BBB"]),
    }
    return raw


def test_business_day_calendar_flag_default_off():
    from src.config import PipelineConfig
    assert PipelineConfig().business_day_calendar_enabled is False


def test_restrict_to_business_days_drops_holidays_and_compounds_returns():
    from src.data_loader import restrict_to_business_days

    raw = _holiday_workbook()
    out, diag = restrict_to_business_days(raw)

    kept = list(out["PX_LAST"].index.strftime("%Y-%m-%d"))
    assert kept == ["2026-07-02", "2026-07-06"]
    assert list(out["BEST_EPS"].index) == list(out["PX_LAST"].index)
    ret = out["Daily_Returns"]
    assert bool(np.isnan(ret["AAA"].iloc[0]))
    assert ret["AAA"].iloc[1] == pytest.approx(0.02)
    assert ret["BBB"].iloc[1] == pytest.approx(0.03)   # 휴일 이동이 복리로 합산
    assert out["Universe_Meta"].equals(raw["Universe_Meta"])
    assert diag["business_days"] == 2
    assert diag["rows_dropped_by_sheet"]["PX_LAST"] == 3
    # 입력은 변경되지 않는다
    assert len(raw["PX_LAST"]) == 5


def test_restrict_to_business_days_requires_business_days_sheet():
    from src.data_loader import restrict_to_business_days

    raw = _holiday_workbook()
    del raw["BusinessDays"]
    with pytest.raises(ValueError, match="BusinessDays"):
        restrict_to_business_days(raw)


def test_universedata_business_day_calendar_off_is_parity_and_on_restricts(monkeypatch):
    from src.config import PipelineConfig
    from src.data_loader import UniverseData

    raw = _holiday_workbook()
    for sheet in _OBS_ESSENTIAL - {"PX_LAST", "Daily_Returns", "BEST_EPS"}:
        raw[sheet] = pd.DataFrame(1.0, index=raw["PX_LAST"].index, columns=["AAA", "BBB"])
    monkeypatch.setattr(
        "src.data_loader.load_all_sheets",
        lambda _path: {k: v.copy() for k, v in raw.items()},
    )
    off = UniverseData("unused.xlsx", config=PipelineConfig(fx_source_path="missing.xlsx"))
    on = UniverseData(
        "unused.xlsx",
        config=PipelineConfig(fx_source_path="missing.xlsx", business_day_calendar_enabled=True),
    )
    assert list(off.dates.strftime("%Y-%m-%d")) == ["2026-07-02", "2026-07-03", "2026-07-06"]
    assert "business_day_calendar" not in off.data_quality
    assert list(on.dates.strftime("%Y-%m-%d")) == ["2026-07-02", "2026-07-06"]
    assert on.data_quality["business_day_calendar"]["business_days"] == 2
    assert on.get_sheet("Daily_Returns")["BBB"].iloc[1] == pytest.approx(0.03)
