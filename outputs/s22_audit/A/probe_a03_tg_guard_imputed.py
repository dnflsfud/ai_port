"""A-03: the TG/price guard medians include loader-imputed pre-coverage TG cells
(other names' target-price LEVELS from _fill_missing's cross-sectional median),
the exact cells the production s17_coverage_gap_fix re-NaNs for tg_upside.

Real functions: preprocess_sheets, align_dates, UniverseData._check_target_price_unit_ratio
(stub self), annotate_basis_guard.
Run from the repo dir:  PYTHONPATH=. <PY> <this file>
"""
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.data_loader import UniverseData, align_dates, preprocess_sheets
from src.tg_basis_guard import annotate_basis_guard

idx = pd.bdate_range(end="2026-09-14", periods=300)
px = pd.DataFrame({"AAA": 100.0, "BBB": 500.0, "NEW": 20.0}, index=idx)
tg = pd.DataFrame({"AAA": 110.0, "BBB": 550.0, "NEW": np.nan}, index=idx)
tg.iloc[-100:, tg.columns.get_loc("NEW")] = 22.0  # FactSet coverage starts 100 rows before the end

raw = {"PX_LAST": px, "Factset_TG_Price": tg}
sheets = preprocess_sheets(raw, tickers=["AAA", "BBB", "NEW"])
sheets = align_dates(sheets, config=PipelineConfig(), diagnostics={})
print("imputed NEW TG cell (first row) =", float(sheets["Factset_TG_Price"]["NEW"].iloc[0]))

stub = SimpleNamespace(sheets=sheets, local_prices=sheets["PX_LAST"],
                       data_quality={"currency": {}})
UniverseData._check_target_price_unit_ratio(stub)
cur = stub.data_quality["currency"]
observed_only = float((tg["NEW"] / px["NEW"]).tail(252).median())
print(f"guard median NEW = {cur['tg_px_ratio_median']['NEW']:.2f} | observed-cells median = {observed_only:.2f}")
print("suspect =", cur["tg_px_ratio_suspect"])
with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as d:
    pending = annotate_basis_guard(d, stub.data_quality, 0.25)
    print("pending =", pending, "| tg_basis_guard_ok =", cur["tg_basis_guard_ok"])
