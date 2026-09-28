"""Clean-check: S16.2 cap semantics (production params) on a persistent 80 -> 10 collapse in a non-earnings month."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/westl/PycharmProjects/pythonProject/venv_vf_new/machine/re_study/c2/ai_port")
from src.features.sellside import clean_revision_spikes
d = pd.bdate_range("2024-03-01", periods=80)
s = pd.Series(80.0, index=d); s.iloc[10:] = 10.0
out = clean_revision_spikes(s.to_frame("X"), threshold=15.0, mode="reversion_gated", extreme_threshold=50.0,
                            reversion_ratio=0.5, persistent_rollover_extension=True, extension_max_days=21)["X"]
held = (out != s)
print("event row", d[10].date(), "| cells held at pre-event value:", int(held.sum()),
      "| first released row:", out.index[(~held) & (out.index > d[10])][0].date(),
      "| BD after event:", int(np.flatnonzero((~held.values) & (np.arange(80) > 10))[0] - 10))
