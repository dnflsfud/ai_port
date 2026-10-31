# -*- coding: utf-8 -*-
"""§S25.4 stale-run mask (decision log §S25 F1 follow-up): price-dependent ratio sheets whose vendor
value stopped updating are carried forward by price_v4 / the loader ffill for months or years
(RBLX BEST_PE_RATIO 14,847 since 2022-02, VRSN BEST_PX_BPS_RATIO since 2014-07). ON re-NaNs every cell
whose value has been unchanged for more than ``stale_run_max_days`` consecutive rows; OFF is byte-identical."""

import numpy as np
import pandas as pd
import pytest

from src.config import PipelineConfig
from src.data_loader import UniverseData, mask_stale_runs

_ESSENTIAL = {
    "PX_LAST", "Daily_Returns", "CUR_MKT_CAP", "BEST_EPS", "BEST_SALES",
    "BEST_PE_RATIO", "OPER_MARGIN", "BEST_ROE", "NEWS_SENTIMENT_DAILY_AVG",
    "EQY_REC_CONS", "Factset_EPS_Revision", "Factset_Sales_Revision",
    "Factset_TG_Price",
}


def test_mask_stale_runs_keeps_first_max_run_rows_and_nans_the_rest():
    s = pd.Series([1.0] * 30 + [2.0, 3.0] + [4.0] * 5 + [np.nan] + [4.0] * 5)
    out = mask_stale_runs(pd.DataFrame({"x": s}), max_run=21)["x"]
    assert out.iloc[:21].notna().all()            # a 21-row run is kept in full
    assert out.iloc[21:30].isna().all()           # rows 22..30 of the 30-row run are stale
    assert out.iloc[30:32].tolist() == [2.0, 3.0]  # changing values untouched
    assert out.iloc[32:37].notna().all()          # 5-row run < max_run kept
    assert np.isnan(out.iloc[37])                 # the NaN itself stays NaN
    assert out.iloc[38:].notna().all()            # NaN breaks the run: the second 4.0 run restarts at 1


def test_mask_stale_runs_is_per_column_and_exact_equality():
    df = pd.DataFrame({"a": [5.0] * 25, "b": np.linspace(1, 2, 25), "c": [5.0] * 10 + [5.0000001] * 15})
    out = mask_stale_runs(df, max_run=21)
    assert out["a"].iloc[:21].notna().all() and out["a"].iloc[21:].isna().all()
    assert out["b"].notna().all()
    assert out["c"].notna().all()                 # a tiny change restarts the run (exact equality rule)


def test_config_defaults_off_and_validates_window():
    cfg = PipelineConfig()
    assert cfg.stale_run_mask_enabled is False
    assert cfg.stale_run_max_days == 21
    assert cfg.stale_run_mask_sheets == ("BEST_PE_RATIO", "BEST_PX_BPS_RATIO", "BEST_PEG_RATIO", "BEST_EV_TO_BEST_EBITDA")
    with pytest.raises(ValueError):
        PipelineConfig(stale_run_max_days=0)


def _workbook(n_dates=60):
    dates = pd.bdate_range("2021-01-04", periods=n_dates)
    prices = pd.DataFrame({"AAA": np.linspace(100.0, 159.0, n_dates), "BBB": np.linspace(50.0, 109.0, n_dates)}, index=dates)
    meta = pd.DataFrame({"Ticker": ["AAA", "BBB"], "Name": ["Alpha", "Beta"], "Sector": ["Test", "Test"]},
                        index=["AAA US Equity", "BBB US Equity"])
    pe = pd.DataFrame({"AAA": np.linspace(10.0, 15.9, n_dates),
                       "BBB": list(np.linspace(20.0, 21.0, 20)) + [21.0] * (n_dates - 20)}, index=dates)
    raw = {"Universe_Meta": meta, "PX_LAST": prices,
           "Daily_Returns": prices.pct_change(fill_method=None).fillna(0.0), "BEST_PE_RATIO": pe}
    for sheet in _ESSENTIAL - set(raw):
        raw[sheet] = pd.DataFrame(1.0, index=dates, columns=["AAA", "BBB"])
    return raw


def _data(monkeypatch, **cfg_kwargs):
    raw = _workbook()
    monkeypatch.setattr("src.data_loader.load_all_sheets", lambda _path: {k: v.copy() for k, v in raw.items()})
    return UniverseData("unused.xlsx", config=PipelineConfig(fx_source_path="missing.xlsx", **cfg_kwargs))


def test_off_get_sheet_returns_the_untouched_sheet_object(monkeypatch):
    data = _data(monkeypatch)
    sheet = data.get_sheet("BEST_PE_RATIO")
    assert sheet is data.sheets["BEST_PE_RATIO"]
    assert not sheet.isna().any().any()


def test_on_masks_only_configured_sheets_after_the_run_window(monkeypatch):
    data = _data(monkeypatch, stale_run_mask_enabled=True)
    pe = data.get_sheet("BEST_PE_RATIO")
    # BBB: 21.0 from row 19 onward (41 identical rows): first 21 kept, the remaining 20 are NaN
    bbb = pe["BBB"]
    assert bbb.iloc[:40].notna().all()
    assert bbb.iloc[40:].isna().all()
    assert pe["AAA"].notna().all()
    # the loader's stored sheet is not mutated and other sheets are untouched
    assert not data.sheets["BEST_PE_RATIO"].isna().any().any()
    assert data.get_sheet("OPER_MARGIN") is data.sheets["OPER_MARGIN"]
    # repeat calls reuse the masked frame
    assert data.get_sheet("BEST_PE_RATIO") is pe
    assert data.data_quality["stale_run_mask"]["BEST_PE_RATIO"]["masked_cells"] == 20


# ---------------------------------------------------------------------------
# §S26 (2026-10-31): second stale-run group — consensus fundamental sheets
# (BEST_GROSS_MARGIN / BEST_CALCULATED_FCF / BEST_CAPEX / BEST_ROE) at a 126-row
# window. Reference distribution (10-21 workbook): BEST_EPS / BEST_SALES, the
# densest consensus fields, have <= 1 run >= 63 rows over 250 names x 12 years,
# while GM/FCF/CAPEX carry whole-history frozen values (APO 3,088 rows, GS GM
# 0.7 for 1,206 rows, C FCF -16,851 for 1,649 rows). 126 rows (one half-year,
# the longest legitimate reporting cadence in the universe) is the single
# preregistered window. OFF is byte-identical; the ratio group keeps its own
# 21-row window.
# ---------------------------------------------------------------------------
_FUND_SHEETS = ("BEST_GROSS_MARGIN", "BEST_CALCULATED_FCF", "BEST_CAPEX", "BEST_ROE")


def _workbook_long(n_dates=200):
    dates = pd.bdate_range("2021-01-04", periods=n_dates)
    prices = pd.DataFrame({"AAA": np.linspace(100.0, 199.0, n_dates), "BBB": np.linspace(50.0, 149.0, n_dates)}, index=dates)
    meta = pd.DataFrame({"Ticker": ["AAA", "BBB"], "Name": ["Alpha", "Beta"], "Sector": ["Test", "Test"]},
                        index=["AAA US Equity", "BBB US Equity"])
    # BBB gross margin: 30 changing rows, then frozen at 42.0 for 170 rows
    gm = pd.DataFrame({"AAA": np.linspace(30.0, 31.99, n_dates),
                       "BBB": list(np.linspace(40.0, 41.0, 30)) + [42.0] * (n_dates - 30)}, index=dates)
    # BBB PE: 20 changing rows, then frozen at 21.0 for 180 rows (ratio group, 21-row window)
    pe = pd.DataFrame({"AAA": np.linspace(10.0, 11.99, n_dates),
                       "BBB": list(np.linspace(20.0, 21.0, 20)) + [21.0] * (n_dates - 20)}, index=dates)
    raw = {"Universe_Meta": meta, "PX_LAST": prices,
           "Daily_Returns": prices.pct_change(fill_method=None).fillna(0.0),
           "BEST_PE_RATIO": pe, "BEST_GROSS_MARGIN": gm}
    for sheet in (_ESSENTIAL | set(_FUND_SHEETS)) - set(raw):
        raw[sheet] = pd.DataFrame(1.0, index=dates, columns=["AAA", "BBB"])
    return raw


def _data_long(monkeypatch, **cfg_kwargs):
    raw = _workbook_long()
    monkeypatch.setattr("src.data_loader.load_all_sheets", lambda _path: {k: v.copy() for k, v in raw.items()})
    return UniverseData("unused.xlsx", config=PipelineConfig(fx_source_path="missing.xlsx", **cfg_kwargs))


def test_fund_group_config_defaults_off_and_validates():
    cfg = PipelineConfig()
    assert cfg.stale_run_fund_mask_enabled is False
    assert cfg.stale_run_fund_mask_sheets == _FUND_SHEETS
    assert cfg.stale_run_fund_max_days == 126
    with pytest.raises(ValueError):
        PipelineConfig(stale_run_fund_max_days=0)
    with pytest.raises(ValueError):          # the two groups must stay disjoint (one window per sheet)
        PipelineConfig(stale_run_fund_mask_sheets=("BEST_PE_RATIO",))


def test_fund_group_off_leaves_fund_sheets_untouched_even_with_ratio_group_on(monkeypatch):
    data = _data_long(monkeypatch, stale_run_mask_enabled=True)
    gm = data.get_sheet("BEST_GROSS_MARGIN")
    assert gm is data.sheets["BEST_GROSS_MARGIN"]
    assert not gm.isna().any().any()
    assert "BEST_GROSS_MARGIN" not in data.data_quality.get("stale_run_mask", {})


def test_fund_group_on_masks_at_126_rows_and_ratio_group_keeps_its_21_row_window(monkeypatch):
    data = _data_long(monkeypatch, stale_run_mask_enabled=True, stale_run_fund_mask_enabled=True)
    gm = data.get_sheet("BEST_GROSS_MARGIN")["BBB"]
    assert gm.iloc[:30 + 126].notna().all()          # changing rows + the first 126 frozen rows are kept
    assert gm.iloc[30 + 126:].isna().all()           # the remaining 44 frozen rows are stale
    assert data.get_sheet("BEST_GROSS_MARGIN")["AAA"].notna().all()
    pe = data.get_sheet("BEST_PE_RATIO")["BBB"]
    assert pe.iloc[:19 + 21].notna().all() and pe.iloc[19 + 21:].isna().all()   # ratio group unchanged: 21-row window
    assert data.get_sheet("OPER_MARGIN") is data.sheets["OPER_MARGIN"]           # constant 1.0 sheet outside both groups
    dq = data.data_quality["stale_run_mask"]
    assert dq["BEST_GROSS_MARGIN"]["masked_cells"] == 44 and dq["BEST_GROSS_MARGIN"]["max_run"] == 126
    assert dq["BEST_PE_RATIO"]["max_run"] == 21


def test_fund_group_on_alone_does_not_touch_ratio_sheets(monkeypatch):
    data = _data_long(monkeypatch, stale_run_fund_mask_enabled=True)
    assert data.get_sheet("BEST_PE_RATIO") is data.sheets["BEST_PE_RATIO"]
    fcf = data.get_sheet("BEST_CALCULATED_FCF")      # constant 1.0 for 200 rows: first 126 kept, 74 masked
    assert fcf["AAA"].iloc[:126].notna().all() and fcf["AAA"].iloc[126:].isna().all()
