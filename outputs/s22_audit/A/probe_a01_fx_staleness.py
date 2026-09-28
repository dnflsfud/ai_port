"""A-01: FX staleness guard is inert because Factor_PX_LAST is pre-ffilled upstream.

Uses the real src.data_loader.build_fx_rates_usd_per_local on synthetic quotes.
The "workbook" Factor_PX_LAST is built with the exact upstream transform
(re_study/create_ai_signal_data.py:467):
    df_factor_px = df_idx.reindex(bdays, method="ffill").ffill()
Run from the repo dir:  PYTHONPATH=. <PY> <this file>
"""
import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.data_loader import build_fx_rates_usd_per_local

cfg = PipelineConfig()  # production defaults: fail_on_missing_fx=True, max_fx_staleness_days=7
bdays = pd.bdate_range("2026-08-03", "2026-09-14")

# Scenario 1: vendor stops updating EURUSD after 2026-08-21 (Index.xlsx cells NaN).
true_quotes = pd.DataFrame({"EURUSD": np.linspace(1.10, 1.20, len(bdays))}, index=bdays)
index_xlsx = true_quotes.copy()
index_xlsx.loc["2026-08-22":, "EURUSD"] = np.nan
workbook_factor_px = index_xlsx.reindex(bdays, method="ffill").ffill()  # upstream transform

try:
    build_fx_rates_usd_per_local(bdays, ["USD", "EUR"], cfg,
                                 factor_prices=index_xlsx, external_quotes=index_xlsx)
    print("S1 raw observations  -> no error (unexpected)")
except ValueError as exc:
    print("S1 raw observations  -> ValueError:", exc)

rates, diag = build_fx_rates_usd_per_local(bdays, ["USD", "EUR"], cfg,
                                           factor_prices=workbook_factor_px,
                                           external_quotes=index_xlsx)
print("S1 workbook factor   -> stale_currencies =", diag["stale_currencies"],
      "| max_staleness_days =", diag["max_staleness_days_by_currency"]["EUR"],
      "| latest_source_date =", diag["latest_source_date_by_currency"]["EUR"])

# Scenario 2: Index.xlsx lagged 2 business days when the workbook was built;
# at run time the external Index.xlsx has the fresh (moved) quotes.
fresh = pd.DataFrame({"EURUSD": 1.10}, index=bdays)
fresh.loc["2026-09-11":, "EURUSD"] = 1.13  # +2.7% EUR move on the last 2 rows
index_at_build = fresh.loc[:"2026-09-10"]
workbook_factor_px2 = index_at_build.reindex(bdays, method="ffill").ffill()
rates2, diag2 = build_fx_rates_usd_per_local(bdays, ["USD", "EUR"], cfg,
                                             factor_prices=workbook_factor_px2,
                                             external_quotes=fresh)
print("S2 EUR rate used on 2026-09-14 =", float(rates2.loc["2026-09-14", "EUR"]),
      "| fresh external quote =", float(fresh.loc["2026-09-14", "EURUSD"]),
      "| stale_currencies =", diag2["stale_currencies"])
