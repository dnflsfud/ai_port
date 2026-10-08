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
