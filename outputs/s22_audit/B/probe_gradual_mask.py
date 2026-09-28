"""B-02 probe: Pattern-2 'gradual' mask in get_cleaned_revision (production path).

With earnings_timeline=None (every production caller), Pattern 2 masks any
3-day cumulative drop < -threshold with >= 2 daily drops < -3 in 8 of 12
calendar months, down-side only, and with no duration cap (the S16.2 21-BD cap
applies to the Pattern-1 extension only).  A genuine, steady downgrade is held
at its pre-decline level for its whole duration; the mirror-image upgrade flows
immediately; the same downgrade in a non-earnings month also flows.
"""
import copy
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:/Users/westl/PycharmProjects/pythonProject/venv_vf_new/machine/re_study/c2/ai_port")
from src.config import DEFAULT_CONFIG  # noqa: E402
from src.features.sellside import get_cleaned_revision  # noqa: E402

dates = pd.bdate_range("2024-01-01", "2024-06-28")
n = len(dates)

def ramp(start_date, lvl0, lvl1, days):
    s = pd.Series(float(lvl0), index=dates)
    i0 = dates.get_indexer([pd.Timestamp(start_date)])[0]
    step = (lvl1 - lvl0) / days
    for k in range(1, days + 1):
        s.iloc[i0 + k - 1] = lvl0 + step * k
    s.iloc[i0 + days:] = float(lvl1)
    return s

rev = pd.DataFrame({
    "DOWN_FEB": ramp("2024-02-05", 40, -40, 10),   # earnings month, downgrade
    "UP_FEB":   ramp("2024-02-05", -40, 40, 10),   # earnings month, upgrade
    "DOWN_MAR": ramp("2024-03-04", 40, -40, 10),   # non-earnings month
})

class Stub:
    def get_sheet(self, name):
        return rev

cfg = copy.copy(DEFAULT_CONFIG)
cfg.s15_fixpack_enabled = True           # production
cfg.revision_extension_max_days = 21     # production
print("mode:", cfg.revision_clean_mode, cfg.revision_clean_threshold,
      cfg.revision_clean_extreme_threshold, cfg.revision_clean_reversion_ratio)
clean = get_cleaned_revision(Stub(), "Factset_EPS_Revision", config=cfg)

for col in rev:
    raw_hit = rev[col].index[(rev[col] - rev[col].iloc[-1]).abs() < 1e-9][0]
    cl_hit = clean[col].index[(clean[col] - rev[col].iloc[-1]).abs() < 1e-9][0]
    frozen = int(((clean[col] - rev[col]).abs() > 1e-9).sum())
    lag = dates.get_loc(cl_hit) - dates.get_loc(raw_hit)
    print(f"{col:9s} raw reaches final {raw_hit.date()} | cleaned reaches final "
          f"{cl_hit.date()} | lag {lag} BD | cells differing from raw {frozen}")

seg = slice("2024-02-05", "2024-02-20")
print(pd.DataFrame({"raw_DOWN_FEB": rev.loc[seg, "DOWN_FEB"],
                    "clean_DOWN_FEB": clean.loc[seg, "DOWN_FEB"]}).T.round(1).to_string())
