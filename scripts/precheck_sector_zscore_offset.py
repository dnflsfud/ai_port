# -*- coding: utf-8 -*-
"""S13.7 pre-check: does a Financials / non-Financials split actually bind?

CLAUDE.md 4.3 precedent: before concluding anything about a neutralization
scheme, prove the exposure it would remove actually exists.

Group-wise z-scoring only changes the panel to the extent each feature carries
a systematic Financials-vs-rest level offset. Measure that offset directly on
the production (S0(200)) feature panel:

  delta_f  = mean_z(Financials) - mean_z(rest), time-averaged
  eta2_f   = between-group variance share of the cross-sectional z

eta2 ~ 0 -> group-wise z-scoring is inert and the arm is not worth running.
"""
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, ".")

from src.config import PipelineConfig, NAN_TOLERANT_FEATURES
from src.data_loader import UniverseData
from src.features.assembly import build_all_features

VARIANT = "variants/codex_causal_rank_65.yaml"
MASKED = {"8306", "AXP", "BAC", "BN", "BRK/B", "C", "CB", "COF", "GS",
          "HSBA", "JPM", "MS", "MUV2", "PGR", "TSM", "WFC", "ZURN"}

overrides = yaml.safe_load(open(VARIANT, encoding="utf-8"))["overrides"]
cfg = PipelineConfig(**overrides)

print("[1/3] loading universe ...", flush=True)
data = UniverseData(cfg.data_path, config=cfg)

meta = data.meta
sector = meta["sector"].astype(str)
print("\n=== sector composition (n=%d) ===" % len(sector))
print(sector.value_counts().to_string())

fin = set(sector[sector.str.contains("Financ", case=False, na=False)].index)
print("\nFinancials n=%d" % len(fin))
print("  masked & financial   :", len(fin & MASKED))
print("  financial, unmasked  :", len(fin - MASKED))
print("  masked, non-financial:", sorted(MASKED - fin))

print("\n[2/3] building core feature panel ...", flush=True)
panel, names, groups = build_all_features(data, config=cfg)
print("panel", panel.shape, "features", len(names), flush=True)

tickers = panel.index.get_level_values("ticker")
is_fin = pd.Series(tickers.isin(fin), index=panel.index)

# Pre-listing rows are median-filled (~0 in z space) and would dilute the
# offset toward zero in BOTH groups. Drop them using the same listing dates
# the production mask uses, then restrict to 2020+.
dates = panel.index.get_level_values("date")
ld = data.listing_dates
listed_from = pd.Series(tickers.map(lambda t: ld.get(t, pd.Timestamp.min)),
                        index=panel.index)
keep = (dates >= pd.Timestamp("2020-01-01")) & (dates >= listed_from.values)
print(f"listed rows kept: {int(keep.sum()):,} / {len(panel):,} "
      f"({keep.mean():.1%}); pre-listing rows dropped from the estimate")
sub = panel[keep]
sub_fin = is_fin[keep]

print("\n[3/3] per-feature Financials offset (2020-01-01 onward, "
      "%d dates) ...\n" % sub.index.get_level_values("date").nunique(),
      flush=True)

d_idx = sub.index.get_level_values("date")
g = sub.groupby([d_idx, sub_fin.values])
means, sizes = g.mean(), g.size()
fin_mu, oth_mu = means.xs(True, level=1), means.xs(False, level=1)
n_f, n_o = sizes.xs(True, level=1), sizes.xs(False, level=1)
delta = (fin_mu - oth_mu).mean()

# between-group variance share (eta^2) per feature, pooled over dates,
# with the ACTUAL per-date listed counts in each group.
overall = sub.groupby(d_idx).mean()
between = (fin_mu - overall).pow(2).mul(n_f, axis=0) \
        + (oth_mu - overall).pow(2).mul(n_o, axis=0)
total = sub.groupby(d_idx).apply(lambda d: ((d - d.mean()) ** 2).sum())
eta2 = (between.sum() / total.sum().replace(0, np.nan)).astype(float)

rep = pd.DataFrame({"delta_fin_minus_rest": delta, "eta2": eta2})
rep["masked_feature"] = rep.index.isin(NAN_TOLERANT_FEATURES)
rep = rep.sort_values("eta2", ascending=False)

pd.set_option("display.width", 140)
print("--- top 20 by between-group variance share ---")
print(rep.head(20).to_string(float_format=lambda v: f"{v:+.4f}"))

print("\n--- the 8 structurally-absent features ---")
print(rep[rep.masked_feature].to_string(float_format=lambda v: f"{v:+.4f}"))

print("\n--- summary ---")
print(f"features                      : {len(rep)}")
print(f"mean eta2                     : {rep.eta2.mean():.4f}")
print(f"median eta2                   : {rep.eta2.median():.4f}")
print(f"features with eta2 > 0.05     : {(rep.eta2 > 0.05).sum()}")
print(f"features with |delta| > 0.25  : {(rep.delta_fin_minus_rest.abs() > 0.25).sum()}")
print(f"features with |delta| > 0.50  : {(rep.delta_fin_minus_rest.abs() > 0.50).sum()}")

rep.to_csv("outputs/s13_7_sector_offset_precheck.csv")
print("\nwrote outputs/s13_7_sector_offset_precheck.csv")
