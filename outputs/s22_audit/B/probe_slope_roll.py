"""B-04 probe: fwd_sales_slope_chg_63d = slope.diff(63) across a 1FY/2FY roll.

The source sheet is (2FY - 1FY)/1FY built from BEST_SALES with
BEST_FPERIOD_OVERRIDE=1FY/2FY (re_study/create_ai_signal_data.py:245-250),
i.e. relative fiscal periods that roll when the company reports its fiscal
year.  A roll swaps the pair (FY_n, FY_n+1) -> (FY_n+1, FY_n+2) with NO
estimate revision, yet diff(63) carries the full step for 63 rows.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:/Users/westl/PycharmProjects/pythonProject/venv_vf_new/machine/re_study/c2/ai_port")
from src.features.fwd_sales_slope import build_fwd_sales_slope_features  # noqa: E402

dates = pd.bdate_range("2024-01-01", periods=200)
roll = dates[80]
# ROLLER: consensus never changes. FY24=100, FY25=130, FY26=150. Reports FY24 at `roll`.
fy1 = pd.Series(np.where(dates < roll, 100.0, 130.0), index=dates)
fy2 = pd.Series(np.where(dates < roll, 130.0, 150.0), index=dates)
# REVISER: no roll in window, FY2 estimate revised up 1% steadily (a genuine steepening).
r1 = pd.Series(100.0, index=dates)
r2 = pd.Series(np.linspace(110.0, 111.1, len(dates)), index=dates)
slope = pd.DataFrame({"ROLLER": (fy2 - fy1) / fy1, "REVISER": (r2 - r1) / r1})

class Stub:
    tickers = ["ROLLER", "REVISER"]
    def __init__(self):
        self.dates = dates
    def get_sheet(self, name):
        return slope

out = build_fwd_sales_slope_features({}, Stub())
chg = out["fwd_sales_slope_chg_63d"]
held = chg["ROLLER"].loc[roll:].abs() > 0.1
print(f"ROLLER: estimates unchanged; slope {slope['ROLLER'].iloc[0]:.3f} -> {slope['ROLLER'].iloc[-1]:.3f} at roll {roll.date()}")
print(f"ROLLER chg_63d on roll day {chg['ROLLER'].loc[roll]:+.3f}; rows with |chg| > 0.10 after roll: {int(held.sum())}")
print(f"REVISER (genuine FY2 up-revision) chg_63d max |value|: {chg['REVISER'].abs().max():.4f}")
