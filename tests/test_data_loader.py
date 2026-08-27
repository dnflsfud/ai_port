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
