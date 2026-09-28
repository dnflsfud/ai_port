"""B-03 probe: CS z-score BEFORE any winsorisation (assembly.py:837-853).

safe_pct_change divides by |base|; one name whose base is near zero (EPS
0.02 -> 1.0) produces a huge raw value, owns most of the cross-sectional
variance, gets clipped to +5 AFTER the z-score, and leaves the other 249
names compressed into a narrow band.  The compression factor varies date by
date with the outlier's size, so a pooled tree threshold means a different
quantile on different dates.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:/Users/westl/PycharmProjects/pythonProject/venv_vf_new/machine/re_study/c2/ai_port")
from src.features.utils import safe_pct_change, cross_sectional_zscore, clip_outliers  # noqa: E402

rng = np.random.default_rng(0)
N = 250
dates = pd.bdate_range("2023-01-02", periods=260)
base = rng.lognormal(np.log(5.0), 0.6, N)
growth = rng.normal(0.08, 0.20, N)                       # realistic 1y EPS growth
eps = pd.DataFrame(np.outer(np.ones(len(dates)), base), index=dates,
                   columns=[f"T{i}" for i in range(N)])
eps.iloc[-1] = base * (1 + growth)
for name, b0, b1 in [("T0", 0.02, 1.0), ("T1", 0.10, 1.0), ("T2", 0.50, 1.0)]:
    eps.loc[:, name] = b0
    eps.loc[dates[-1], name] = b1

def post_clip_stats(frame):
    z = clip_outliers(cross_sectional_zscore(frame)).iloc[-1]
    rest = z.drop(["T0", "T1", "T2"], errors="ignore")
    return z.std(), rest.std(), rest.quantile(0.9) - rest.quantile(0.1)

chg = safe_pct_change(eps, 252)
for label, cols_to_keep in [("no near-zero base", [c for c in eps if c not in ("T0", "T1", "T2")]),
                            ("base 0.50 -> 1.0", [c for c in eps if c not in ("T0", "T1")]),
                            ("base 0.10 -> 1.0", [c for c in eps if c not in ("T0", "T2")]),
                            ("base 0.02 -> 1.0", [c for c in eps if c not in ("T1", "T2")])]:
    s_all, s_rest, p90_10 = post_clip_stats(chg[cols_to_keep])
    print(f"{label:18s}: post-clip CS std {s_all:.3f} | other names std {s_rest:.3f} "
          f"| other names p90-p10 {p90_10:.3f}")
