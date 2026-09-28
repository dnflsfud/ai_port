"""A-05: restrict_to_business_days never checks that the BusinessDays sheet
covers the price panel's last date; a short BusinessDays sheet silently drops
the newest rows of EVERY sheet (no warning, no raise, tail guard sees 0).

Real functions: restrict_to_business_days, preprocess_sheets, align_dates.
Run from the repo dir:  PYTHONPATH=. <PY> <this file>
"""
import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.data_loader import align_dates, preprocess_sheets, restrict_to_business_days

px_days = pd.bdate_range("2026-08-03", "2026-09-14")
bdays = px_days[px_days <= "2026-09-10"]            # BusinessDays sheet 2 rows short
px = pd.DataFrame({"AAA": np.linspace(100, 110, len(px_days))}, index=px_days)
raw = {
    "PX_LAST": px,
    "Daily_Returns": px.pct_change(),
    # load_all_sheets reads with index_col=0, so the BusinessDay column is the index
    "BusinessDays": pd.DataFrame(index=pd.Index(bdays, name="BusinessDay")),
}
out, diag = restrict_to_business_days(raw)
sheets = align_dates(preprocess_sheets(out, tickers=["AAA"]), config=PipelineConfig(),
                     diagnostics=(dq := {}))
print("raw PX_LAST last date     =", px.index.max().date())
print("model calendar last date  =", sheets["PX_LAST"].index.max().date())
print("rows dropped (PX_LAST)    =", diag["rows_dropped_by_sheet"]["PX_LAST"],
      "| tail_ffill_days =", dq["tail_ffill_days"])
