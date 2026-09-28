"""A-02: the TG/price basis guard divides by dividend-ADJUSTED PX_LAST while
production tg_upside divides by the nominal PX_LAST_UNADJ panel, so a
distribution-type spin-off (RTX/T class, decision log S18 P1 class 3) is
invisible to both the suspect band and the S19 jump detector.

Real functions: UniverseData._check_target_price_unit_ratio (called on a stub
carrying the attributes it reads) and src.tg_basis_guard.annotate_basis_guard.
Run from the repo dir:  PYTHONPATH=. <PY> <this file>
"""
import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.data_loader import UniverseData
from src.tg_basis_guard import annotate_basis_guard

idx = pd.bdate_range(end="2026-09-14", periods=300)
event = idx[-60]  # spin-off 60 rows before the last date (inside the 252 tail)
f = 0.6           # post-spin value retained (spin worth 40%)

# XYZ: nominal (UNADJ) price 100 before the spin, 60 after (not back-adjusted).
unadj = pd.Series(np.where(idx < event, 100.0, 60.0), index=idx)
# Bloomberg DPDF PX_LAST back-adjusts the pre-event history by f.
adj = pd.Series(60.0, index=idx)
# FactSet restates the TG history onto the post-event basis (115 * f = 69).
tg = pd.Series(69.0, index=idx)
# A clean control name.
ctl = pd.Series(100.0, index=idx)

stub = SimpleNamespace(
    sheets={"Factset_TG_Price": pd.DataFrame({"XYZ": tg, "CTL": ctl * 1.1})},
    local_prices=pd.DataFrame({"XYZ": adj, "CTL": ctl}),               # what the guard reads
    local_prices_nominal=pd.DataFrame({"XYZ": unadj, "CTL": ctl}),     # what tg_upside reads
    data_quality={"currency": {}},
)
UniverseData._check_target_price_unit_ratio(stub)
cur = stub.data_quality["currency"]
guard_median = cur["tg_px_ratio_median"]["XYZ"]
consumer_median = float((tg / unadj).tail(252).median())
print(f"guard median TG/PX_LAST (adjusted) = {guard_median:.4f}")
print(f"consumer median TG/UNADJ (nominal)  = {consumer_median:.4f}")
print("suspect =", cur["tg_px_ratio_suspect"])
print(f"pre-event tg_upside as consumed = {69/100 - 1:+.2f} (true basis {69/60 - 1:+.2f})")

with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as d:
    # State from the last clean run before the event: XYZ baseline 1.15.
    Path(d, "tg_basis_state.json").write_text(json.dumps(
        {"schema_version": 1, "baseline": {"XYZ": 1.15, "CTL": 1.1}, "pending": {}}))
    pending = annotate_basis_guard(d, stub.data_quality, 0.25)
    print("jump pending =", pending, "| tg_basis_guard_ok =", cur["tg_basis_guard_ok"])
print(f"|log(consumer/baseline)| = {abs(math.log(consumer_median / 1.15)):.3f} (> 0.25 would be flagged)")
