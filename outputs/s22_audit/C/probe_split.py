"""Probe: production causal split (train 1260 / val 126 / H 20 / retrain 63)
on a 3,176-row business-day calendar, plus the target realization contract of
target_engine.compute_forward_returns (label at t realises at t+H)."""
import numpy as np, pandas as pd
from src.model_trainer import build_walk_forward_split
from src.target_engine import compute_forward_returns

dates = pd.bdate_range("2014-01-24", periods=3176)
bad = []
for t in range(1260, len(dates), 63):
    s = build_walk_forward_split(dates, t, 1260, 126, 20)
    a = s["audit"]
    tr_last = dates.get_loc(s["train_dates"][-1]); va_first = dates.get_loc(s["val_dates"][0])
    va_last = dates.get_loc(s["val_dates"][-1])
    if not (a["causal_validation_ok"] and va_last + 20 <= t and tr_last + 20 < va_first):
        bad.append(t)
print("retrains:", len(range(1260, len(dates), 63)), "causal violations:", bad)
s = build_walk_forward_split(dates, 1260, 1260, 126, 20)["audit"]
print("first split:", s["train_start"], s["train_end"], s["validation_start"], s["validation_end"],
      "embargo", s["embargo_days"])
# label realisation: fwd at t uses rows t+1..t+H
r = pd.DataFrame({"A": np.arange(1, 31) / 1000.0}, index=pd.bdate_range("2020-01-01", periods=30))
f = compute_forward_returns(r, 3)
exp = (1 + r["A"].iloc[1:4]).prod() - 1
print("fwd[0] == prod(r[1..3])-1:", bool(np.isclose(f["A"].iloc[0], exp)))
