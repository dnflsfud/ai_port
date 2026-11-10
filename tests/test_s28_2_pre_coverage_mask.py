# -*- coding: utf-8 -*-
"""§S28.2 pre-coverage mask for consensus sheets (decision log §S28 H4 / §S28.2 preregistration).

Defect: after listing but before analyst coverage starts, the loader's ``_fill_missing`` (ffill -> per-date
cross-sectional MEDIAN of the level) fills a ticker's consensus cells with OTHER names' levels. Derived
features are then wrong: ``safe_pct_change(raw, w)`` is a spurious "median -> actual" jump for the first w
rows after coverage starts, level / rank features sit on a fake level. §S17 T-01 fixed this for two sheets
inside sellside.py; ON generalises it through ``UniverseData.get_sheet`` for the preregistered consensus
sheets: cells before a ticker's first observed raw value are re-NaNed (the panel builder then imputes the
FEATURE with its per-date median). Cells from the first observation on keep the loader's fill (interior
gaps stay forward-filled). OFF (default) returns the untouched sheet object."""

import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.data_loader import UniverseData
from src.features.utils import safe_pct_change

_SHEETS = (
    "BEST_EPS", "BEST_SALES", "BEST_CALCULATED_FCF", "BEST_GROSS_MARGIN", "OPER_MARGIN", "BEST_CAPEX",
    "BEST_ROE", "BEST_PE_RATIO", "BEST_PEG_RATIO", "BEST_PX_BPS_RATIO", "BEST_EV_TO_BEST_EBITDA",
)
_ESSENTIAL = {
    "PX_LAST", "Daily_Returns", "CUR_MKT_CAP", "BEST_EPS", "BEST_SALES",
    "BEST_PE_RATIO", "OPER_MARGIN", "BEST_ROE", "NEWS_SENTIMENT_DAILY_AVG",
    "EQY_REC_CONS", "Factset_EPS_Revision", "Factset_Sales_Revision",
    "Factset_TG_Price",
}
_TICKERS = ["AAA", "BBB", "CCC", "DDD"]
N = 60
K = 20       # BBB: first observed row (analyst coverage start)
GAP = 30     # BBB: interior gap after coverage start (the loader's ffill must survive)
LIST = 10    # DDD: listing row (PX_LAST starts here)
DDD_K = 15   # DDD: coverage starts 5 rows after listing
W = 5        # pct-change window of the feature-consequence test


def _workbook():
    dates = pd.bdate_range("2021-01-04", periods=N)
    prices = pd.DataFrame({"AAA": np.linspace(100.0, 159.0, N), "BBB": np.linspace(50.0, 109.0, N),
                           "CCC": np.linspace(70.0, 129.0, N), "DDD": np.linspace(30.0, 89.0, N)}, index=dates)
    prices.iloc[:LIST, prices.columns.get_loc("DDD")] = np.nan          # DDD lists at row LIST
    meta = pd.DataFrame({"Ticker": _TICKERS, "Name": ["Alpha", "Beta", "Gamma", "Delta"], "Sector": ["Test"] * 4},
                        index=[f"{t} US Equity" for t in _TICKERS])
    # BEST_EPS: AAA fully covered; BBB covered from row K with one interior gap; CCC never covered;
    # DDD covered from DDD_K (5 rows after its listing).
    eps = pd.DataFrame({"AAA": np.linspace(10.0, 15.9, N),
                        "BBB": [np.nan] * K + list(np.linspace(2.0, 3.0, N - K)),
                        "CCC": np.nan,
                        "DDD": [np.nan] * DDD_K + list(np.linspace(40.0, 45.0, N - DDD_K))}, index=dates)
    eps.iloc[GAP, eps.columns.get_loc("BBB")] = np.nan
    # BEST_PE_RATIO (also a stale-run sheet): BBB covered from row K, 10 changing rows ending at 21.0, then
    # frozen at 21.0 -> a 31-row run from row K + 9: the first 21 rows are kept, rows K + 30.. are stale.
    pe = pd.DataFrame({"AAA": np.linspace(10.0, 15.9, N),
                       "BBB": [np.nan] * K + list(np.linspace(20.0, 21.0, 10)) + [21.0] * (N - K - 10),
                       "CCC": np.linspace(30.0, 35.9, N), "DDD": np.linspace(40.0, 45.9, N)}, index=dates)
    raw = {"Universe_Meta": meta, "PX_LAST": prices,
           "Daily_Returns": prices.pct_change(fill_method=None).fillna(0.0),
           "BEST_EPS": eps, "BEST_PE_RATIO": pe}
    for sheet in _ESSENTIAL - set(raw):
        raw[sheet] = pd.DataFrame(1.0, index=dates, columns=_TICKERS)
    return raw


def _data(monkeypatch, **cfg_kwargs):
    raw = _workbook()
    monkeypatch.setattr("src.data_loader.load_all_sheets", lambda _path: {k: v.copy() for k, v in raw.items()})
    return UniverseData("unused.xlsx", config=PipelineConfig(fx_source_path="missing.xlsx", **cfg_kwargs))


def test_config_defaults_off_with_the_preregistered_sheet_set():
    from src.features.accounting import ACCOUNTING_BASE, VALUATION_SHEETS

    cfg = PipelineConfig()
    assert cfg.pre_coverage_mask_enabled is False
    assert cfg.pre_coverage_mask_sheets == _SHEETS
    assert set(cfg.pre_coverage_mask_sheets) == set(ACCOUNTING_BASE) | set(VALUATION_SHEETS)


def test_off_get_sheet_returns_the_untouched_sheet_object(monkeypatch):
    data = _data(monkeypatch)
    for name in _SHEETS:
        if name in data.sheets:
            assert data.get_sheet(name) is data.sheets[name]
    assert data.sheets["BEST_EPS"]["BBB"].iloc[:K].notna().all()       # regression pin: the loader fills the gap
    assert "pre_coverage_mask" not in data.data_quality


def test_on_masks_cells_before_the_first_observation_only(monkeypatch):
    data = _data(monkeypatch, pre_coverage_mask_enabled=True)
    stored = data.sheets["BEST_EPS"].copy()
    eps = data.get_sheet("BEST_EPS")
    # BBB: NaN before coverage, the loader's values from the first observation on (interior gap stays ffilled)
    assert eps["BBB"].iloc[:K].isna().all()
    pd.testing.assert_series_equal(eps["BBB"].iloc[K:], stored["BBB"].iloc[K:])
    assert eps["BBB"].iloc[K:].notna().all()
    assert eps["BBB"].iloc[GAP] == eps["BBB"].iloc[GAP - 1]
    # CCC has no raw observation at all; AAA is fully covered
    assert eps["CCC"].isna().all()
    pd.testing.assert_series_equal(eps["AAA"], stored["AAA"])
    # the loader's stored sheet is not mutated, repeat calls reuse the masked frame
    assert data.sheets["BEST_EPS"].equals(stored)
    assert data.get_sheet("BEST_EPS") is eps
    # a sheet outside the preregistered tuple is returned untouched
    assert data.get_sheet("Factset_TG_Price") is data.sheets["Factset_TG_Price"]
    dq = data.data_quality["pre_coverage_mask"]
    assert "Factset_TG_Price" not in dq
    assert dq["BEST_EPS"] == {
        "masked_cells": K + N + (DDD_K - LIST),
        "tickers": 3,
        "never_observed_tickers": 1,
        "top": {"CCC": N, "BBB": K, "DDD": DDD_K - LIST},
    }


def test_on_changes_only_post_listing_cells(monkeypatch):
    """Pre-listing cells are already NaN in the loader's sheet (listing re-mask, §S11.4): the pre-coverage mask
    adds the listing -> first-observation stretch only, and only those cells are counted."""
    data = _data(monkeypatch, pre_coverage_mask_enabled=True)
    stored = data.sheets["BEST_EPS"]["DDD"]
    assert stored.iloc[:LIST].isna().all() and stored.iloc[LIST:].notna().all()
    ddd = data.get_sheet("BEST_EPS")["DDD"]
    assert ddd.iloc[:DDD_K].isna().all()
    pd.testing.assert_series_equal(ddd.iloc[DDD_K:], stored.iloc[DDD_K:])
    assert data.data_quality["pre_coverage_mask"]["BEST_EPS"]["top"]["DDD"] == DDD_K - LIST


def test_on_composes_with_the_stale_run_mask(monkeypatch):
    data = _data(monkeypatch, pre_coverage_mask_enabled=True, stale_run_mask_enabled=True)
    pe = data.get_sheet("BEST_PE_RATIO")
    bbb = pe["BBB"]
    assert bbb.iloc[:K].isna().all()                                   # pre-coverage
    assert bbb.iloc[K:K + 30].notna().all()                            # 9 changing rows + the 21 kept run rows
    assert bbb.iloc[K + 30:].isna().all()                              # stale-run
    assert pe[["AAA", "CCC"]].notna().all().all()
    assert data.data_quality["stale_run_mask"]["BEST_PE_RATIO"]["masked_cells"] == N - K - 30
    assert data.data_quality["pre_coverage_mask"]["BEST_PE_RATIO"]["masked_cells"] == K
    assert data.data_quality["pre_coverage_mask"]["BEST_PE_RATIO"]["never_observed_tickers"] == 0


def test_on_without_a_raw_counterpart_returns_the_sheet_untouched():
    shell = UniverseData.__new__(UniverseData)                         # no raw dict -> no observed mask
    shell.config = PipelineConfig(pre_coverage_mask_enabled=True)
    shell.sheets = {"BEST_EPS": pd.DataFrame({"AAA": [1.0, 2.0]})}
    assert shell.get_sheet("BEST_EPS") is shell.sheets["BEST_EPS"]


def test_pct_change_feature_loses_the_spurious_coverage_start_jump(monkeypatch):
    off = safe_pct_change(_data(monkeypatch).get_sheet("BEST_EPS"), W)["BBB"]
    # OFF: the denominator at the coverage-start rows is another name's level (median fill): 2.0 vs ~10
    assert off.iloc[K:K + W].notna().all()
    assert (off.iloc[K:K + W].abs() > 0.5).all()
    on = safe_pct_change(_data(monkeypatch, pre_coverage_mask_enabled=True).get_sheet("BEST_EPS"), W)["BBB"]
    assert on.iloc[:K + W].isna().all()                                # no own history yet -> NaN, not a jump
    pd.testing.assert_series_equal(on.iloc[K + W:], off.iloc[K + W:])  # own-history changes are untouched
    assert on.iloc[K + W:].abs().max() < 0.1
