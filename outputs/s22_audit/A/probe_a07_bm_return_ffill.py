"""A-07: compute_performance_metrics forward-fills BENCHMARK RETURNS onto
portfolio dates, so a missing benchmark day is booked as a repeat of the
previous day's benchmark return (a fabricated return, not an as-of level).

Real function: src.utils.compute_performance_metrics.
Run from the repo dir:  PYTHONPATH=. <PY> <this file>
"""
import numpy as np
import pandas as pd

from src.utils import compute_performance_metrics

idx = pd.bdate_range("2026-01-05", periods=4)
port = pd.Series([0.01, 0.00, 0.00, 0.00], index=idx)
bm_full = pd.Series([0.01, -0.05, 0.00, 0.00], index=idx)     # true benchmark
bm_missing = bm_full.drop(idx[1])                              # day 2 absent
a = compute_performance_metrics(port, bm_full, 252)["active_return"]
b = compute_performance_metrics(port, bm_missing, 252)["active_return"]
bm_used = bm_missing.reindex(port.index).ffill().fillna(0)
print("benchmark return booked on missing day =", float(bm_used.iloc[1]),
      "(true", float(bm_full.iloc[1]), ")")
print(f"active_return true={a:.4f} | with missing day={b:.4f}")
