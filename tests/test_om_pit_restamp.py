# -*- coding: utf-8 -*-
"""§S27 OPER_MARGIN point-in-time restamp (decision log §S27 preregistration). The workbook stamps each
OPER_MARGIN change at the fiscal period end: 98.5% of the 11,113 changes in the 10-21 workbook precede the next
Earnings_Date (median 29 calendar days). ON moves every change to the first session after its matched
announcement (unmatched: frozen per-currency p90 delay); OFF is byte-identical."""

import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.data_loader import (
    OM_PIT_FALLBACK_DAYS,
    OM_PIT_FALLBACK_DAYS_POOLED,
    OM_PIT_MAX_MATCH_SESSIONS,
    UniverseData,
    restamp_to_announcements,
)

DATES = pd.bdate_range("2021-01-04", periods=120)


def _events(**rows):
    ev = pd.DataFrame(0, index=DATES, columns=list(rows))
    for ticker, positions in rows.items():
        for p in positions:
            ev.iloc[p, ev.columns.get_loc(ticker)] = 1
    return ev


def _fallback_row(change_row, days=38):
    return int(DATES.searchsorted(DATES[change_row] + pd.Timedelta(days=days), side="right"))


def test_frozen_rule_constants_and_default_off():
    assert OM_PIT_MAX_MATCH_SESSIONS == 90
    assert OM_PIT_FALLBACK_DAYS == {"USD": 38, "EUR": 51, "JPY": 43, "GBP": 54, "CHF": 45}
    assert OM_PIT_FALLBACK_DAYS_POOLED == 40
    assert PipelineConfig().oper_margin_pit_restamp_enabled is False


def test_production_variant_pins_s27_flip_state():
    """§8/§S27: 사용자 승인 flip(2026-11-03) 이후 production 상태 핀 — production variant 는
    oper_margin_pit_restamp_enabled=True(새 S0′ 'OM-PIT' 1.7087, 1.7738 은퇴), 기본값은 False 유지."""
    import yaml

    with open("variants/codex_causal_rank_65.yaml", encoding="utf-8") as fh:
        overrides = yaml.safe_load(fh).get("overrides") or {}
    assert overrides.get("oper_margin_pit_restamp_enabled") is True
    assert PipelineConfig().oper_margin_pit_restamp_enabled is False


def test_matched_change_becomes_visible_the_session_after_the_announcement():
    v = pd.DataFrame({"A": [10.0] * 20 + [12.0] * 100}, index=DATES)    # period-end stamp at row 20
    out, vis, diag = restamp_to_announcements(v, _events(A=[35]), {"A": 38}, 90)
    assert (out["A"].iloc[:36] == 10.0).all() and (out["A"].iloc[36:] == 12.0).all()
    assert vis["A"].iloc[20] == 36 and vis["A"].iloc[0] == 0          # row 0 = data start, not restamped
    assert diag["changes"] == 1 and diag["matched"] == 1 and diag["lag_sessions_median"] == 16.0


def test_unmatched_or_out_of_window_change_uses_the_calendar_day_fallback():
    v = pd.DataFrame({"B": [5.0] * 10 + [6.0] * 110}, index=DATES)
    first = _fallback_row(10)
    for events in (None, _events(B=[110])):                            # none / 100 sessions later (> 90)
        out, vis, diag = restamp_to_announcements(v, events, {"B": 38}, 90)
        assert vis["B"].iloc[10] == first and diag["matched"] == 0 and diag["fallback"] == 1
        assert (out["B"].iloc[:first] == 5.0).all() and (out["B"].iloc[first:] == 6.0).all()


def test_one_announcement_publishes_only_the_latest_change():
    v = pd.DataFrame({"C": [1.0] * 10 + [2.0] * 10 + [3.0] * 100}, index=DATES)
    out, _, _ = restamp_to_announcements(v, _events(C=[25]), {"C": 38}, 90)
    assert (out["C"].iloc[:26] == 1.0).all() and (out["C"].iloc[26:] == 3.0).all()   # 2.0 is never public


def test_an_older_change_visible_later_never_overwrites_a_newer_one():
    v = pd.DataFrame({"C": [1.0] * 10 + [2.0] * 10 + [3.0] * 100}, index=DATES)
    out, vis, _ = restamp_to_announcements(v, _events(C=[25]), {"C": 38}, 5)   # row 10: 15 sessions > 5 -> fallback
    assert vis["C"].iloc[20] == 26 and vis["C"].iloc[10] == _fallback_row(10) > 26
    assert (out["C"].iloc[:26] == 1.0).all() and (out["C"].iloc[26:] == 3.0).all()


def test_first_value_after_data_start_is_nan_until_public():
    v = pd.DataFrame({"D": [np.nan] * 5 + [7.0] * 115}, index=DATES)
    out, _, _ = restamp_to_announcements(v, _events(D=[12]), {"D": 38}, 90)
    assert out["D"].iloc[:13].isna().all() and (out["D"].iloc[13:] == 7.0).all()


def test_quarter_end_shares_in_diagnostics():
    qe, ann = DATES.get_loc(pd.Timestamp("2021-03-31")), DATES.get_loc(pd.Timestamp("2021-04-27"))
    v = pd.DataFrame({"E": [1.0] * qe + [2.0] * (len(DATES) - qe)}, index=DATES)
    _, _, diag = restamp_to_announcements(v, _events(E=[ann]), {"E": 38}, 90)
    assert diag["quarter_end_3d_share_changes"] == 1.0 and diag["quarter_end_3d_share_visible"] == 0.0


# ---------------------------------------------------------------------------
# UniverseData.get_sheet hook
# ---------------------------------------------------------------------------
_ESSENTIAL = {
    "PX_LAST", "Daily_Returns", "CUR_MKT_CAP", "BEST_EPS", "BEST_SALES",
    "BEST_PE_RATIO", "OPER_MARGIN", "BEST_ROE", "NEWS_SENTIMENT_DAILY_AVG",
    "EQY_REC_CONS", "Factset_EPS_Revision", "Factset_Sales_Revision",
    "Factset_TG_Price",
}


def _data(monkeypatch, extra=None, **cfg_kwargs):
    n = len(DATES)
    prices = pd.DataFrame({"AAA": np.linspace(100.0, 219.0, n), "BBB": np.linspace(50.0, 169.0, n)}, index=DATES)
    meta = pd.DataFrame({"Ticker": ["AAA", "BBB"], "Name": ["Alpha", "Beta"], "Sector": ["Test", "Test"]},
                        index=["AAA US Equity", "BBB US Equity"])
    om = pd.DataFrame({"AAA": [10.0] * 20 + [12.0] * (n - 20), "BBB": [5.0] * 10 + [6.0] * (n - 10)}, index=DATES)
    raw = {"Universe_Meta": meta, "PX_LAST": prices,
           "Daily_Returns": prices.pct_change(fill_method=None).fillna(0.0),
           "OPER_MARGIN": om, "Earnings_Date": _events(AAA=[35], BBB=[])}
    for sheet in _ESSENTIAL - set(raw):
        raw[sheet] = pd.DataFrame(1.0, index=DATES, columns=["AAA", "BBB"])
    raw.update(extra or {})
    monkeypatch.setattr("src.data_loader.load_all_sheets", lambda _path: {k: v.copy() for k, v in raw.items()})
    return UniverseData("unused.xlsx", config=PipelineConfig(fx_source_path="missing.xlsx", **cfg_kwargs))


def test_off_get_sheet_returns_the_untouched_sheet_object(monkeypatch):
    data = _data(monkeypatch)
    assert data.get_sheet("OPER_MARGIN") is data.sheets["OPER_MARGIN"]
    assert "oper_margin_pit_restamp" not in data.data_quality


def test_on_restamps_only_oper_margin_through_get_sheet(monkeypatch):
    data = _data(monkeypatch, oper_margin_pit_restamp_enabled=True)
    assert data._full_currency_map["BBB"] == "USD"
    stored = data.sheets["OPER_MARGIN"].copy()
    om = data.get_sheet("OPER_MARGIN")
    assert list(om.index) == list(DATES)
    assert (om["AAA"].iloc[:36] == 10.0).all() and (om["AAA"].iloc[36:] == 12.0).all()
    first = _fallback_row(10, OM_PIT_FALLBACK_DAYS["USD"])
    assert (om["BBB"].iloc[:first] == 5.0).all() and (om["BBB"].iloc[first:] == 6.0).all()
    assert data.sheets["OPER_MARGIN"].equals(stored)                       # loader's sheet not mutated
    assert data.get_sheet("BEST_PE_RATIO") is data.sheets["BEST_PE_RATIO"]
    assert data.get_sheet("OPER_MARGIN") is om                             # cached
    dq = data.data_quality["oper_margin_pit_restamp"]
    assert dq["changes"] == 2 and dq["matched"] == 1 and dq["fallback"] == 1
    assert dq["changed_cells"] == (36 - 20) + (first - 10) and dq["changed_to_nan_cells"] == 0


def test_source_restamped_workbook_is_not_restamped_again(monkeypatch):
    """§S27.3: from the 11-04+ vintages the generator (re_study/create_universe_data.py) restamps OPER_MARGIN at the
    source and records it in a Data_Provenance sheet (index Sheet, column Basis). Restamping again would match each
    change (already at announcement + 1) to the NEXT quarter's announcement — one quarter late — so the loader keeps
    the sheet as is. The marker sheet is metadata, never a data sheet (also through the business-day calendar)."""
    marker = pd.DataFrame({"Basis": ["announcement_pit_v1"]}, index=pd.Index(["OPER_MARGIN"], name="Sheet"))
    extra = {"Data_Provenance": marker, "BusinessDays": pd.DataFrame({"BusinessDay": DATES})}
    for calendar in (False, True):
        data = _data(monkeypatch, extra=extra, oper_margin_pit_restamp_enabled=True,
                     business_day_calendar_enabled=calendar)
        assert "Data_Provenance" not in data.sheets
        assert data.get_sheet("OPER_MARGIN") is data.sheets["OPER_MARGIN"]
        assert (data.sheets["OPER_MARGIN"]["AAA"].iloc[20:] == 12.0).all()           # untouched stamps
        assert data.data_quality["oper_margin_pit_restamp"] == {"source_restamped": "announcement_pit_v1"}
