# -*- coding: utf-8 -*-
"""§S27.3 parity: current workbook through (old) loader restamp vs (new) generator restamp + Data_Provenance
marker + loader skip. Same raw workbook, production config. Prints cell-category counts."""
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

AI = Path(r"C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port")
RS = AI.parents[1]
SCR = Path(__file__).resolve().parent
sys.path.insert(0, str(AI))
import src.data_loader as dl  # noqa: E402
from run_variant import compose_config  # noqa: E402
from src.harness import inject_config  # noqa: E402

sys.path.insert(0, str(RS))
import create_universe_data as cud  # noqa: E402
from universe_config import simple_ticker, ticker_currency  # noqa: E402

t0 = time.time()
cfg = compose_config(yaml.safe_load((AI / "variants" / "codex_causal_rank_65.yaml").read_text(encoding="utf-8")))
inject_config(cfg)
assert cfg.oper_margin_pit_restamp_enabled
wb = Path(cfg.data_path)
print("workbook", wb, pd.Timestamp(wb.stat().st_mtime, unit="s"), flush=True)
raw = dl.load_all_sheets(str(wb))
print("loaded", len(raw), "sheets", round(time.time() - t0), "s", flush=True)
real_load = dl.load_all_sheets

# ---------------- old path ----------------
dl.load_all_sheets = lambda _p: {k: v.copy() for k, v in raw.items()}
d1 = dl.UniverseData(str(wb), config=cfg)
s_old = d1.get_sheet("OPER_MARGIN").copy()
diag_old = dict(d1.data_quality["oper_margin_pit_restamp"])
sheet1 = d1.sheets["OPER_MARGIN"]
raw1 = d1._raw_sheet_standardized("OPER_MARGIN")
known_old = raw1.reindex(raw1.index.union(sheet1.index)).ffill().reindex(index=sheet1.index, columns=sheet1.columns)
fb = {t: dl.OM_PIT_FALLBACK_DAYS.get(str(d1._full_currency_map.get(t, "")).upper(), dl.OM_PIT_FALLBACK_DAYS_POOLED)
      for t in sheet1.columns}
p_old, _, _ = dl.restamp_to_announcements(known_old, d1.earnings_timeline, fb, dl.OM_PIT_MAX_MATCH_SESSIONS)
dates1, px1 = d1.dates, d1.sheets["PX_LAST"].copy()
cur_loader = dict(d1._full_currency_map)
print("old path done", round(time.time() - t0), "s", diag_old, flush=True)
del d1, sheet1, raw1
gc.collect()

# ---------------- generator restamp ----------------
om_wb = raw["OPER_MARGIN"]
om_cal = om_wb.copy()
om_cal.index = pd.to_datetime(om_cal.index)
earn = raw["Earnings_Date"].copy()
earn.index = pd.to_datetime(earn.index)
bd = pd.to_datetime(raw["BusinessDays"].index if "BusinessDay" not in raw["BusinessDays"] else raw["BusinessDays"]["BusinessDay"])
meta = raw["Universe_Meta"]
bbg = [str(x) for x in (meta["Ticker"] if "Ticker" in meta.columns else meta.index)]
currency_of = {simple_ticker(t): ticker_currency(t) for t in bbg}
cur_mismatch = {t: (currency_of.get(t), c) for t, c in cur_loader.items() if currency_of.get(t) != c}
gen_om, gen_diag = cud.restamp_oper_margin(om_cal, earn, bd, currency_of)
gen_om.index = om_wb.index
print("generator", gen_diag, "currency mismatches", cur_mismatch, round(time.time() - t0), "s", flush=True)

# ---------------- new path ----------------
raw_new = dict(raw)
raw_new["OPER_MARGIN"] = gen_om
raw_new["Data_Provenance"] = pd.DataFrame({"Basis": [cud.OM_PIT_BASIS]}, index=pd.Index(["OPER_MARGIN"], name="Sheet"))
dl.load_all_sheets = lambda _p: {k: v.copy() for k, v in raw_new.items()}
d2 = dl.UniverseData(str(wb), config=cfg)
s_new = d2.get_sheet("OPER_MARGIN")
diag_new = d2.data_quality["oper_margin_pit_restamp"]
sheet2 = d2.sheets["OPER_MARGIN"]
raw2 = d2._raw_sheet_standardized("OPER_MARGIN")
p_new = raw2.reindex(raw2.index.union(sheet2.index)).ffill().reindex(index=sheet2.index, columns=sheet2.columns)
# counterfactual: what a second (loader) restamp would do to the generator output
fb2 = {t: dl.OM_PIT_FALLBACK_DAYS.get(str(d2._full_currency_map.get(t, "")).upper(), dl.OM_PIT_FALLBACK_DAYS_POOLED)
       for t in sheet2.columns}
p_double, _, diag_double = dl.restamp_to_announcements(p_new, d2.earnings_timeline, fb2, dl.OM_PIT_MAX_MATCH_SESSIONS)
same_dates = d2.dates.equals(dates1)
same_px = d2.sheets["PX_LAST"].equals(px1)
print("new path done", round(time.time() - t0), "s", diag_new, flush=True)


def cats(a, b):
    a, b = a.reindex_like(b), b
    an, bn = a.notna(), b.notna()
    both = an & bn
    eq = both & np.isclose(a.fillna(0).to_numpy(), b.fillna(0).to_numpy(), rtol=0, atol=1e-12)
    return {
        "cells": int(a.size), "both_nan": int((~an & ~bn).to_numpy().sum()),
        "equal": int(eq.to_numpy().sum()), "both_value_differ": int((both & ~eq).to_numpy().sum()),
        "old_nan_new_value": int((~an & bn).to_numpy().sum()), "old_value_new_nan": int((an & ~bn).to_numpy().sum()),
    }, (both & ~eq) | (an ^ bn)


pre, pre_mask = cats(p_old, p_new)
fin, fin_mask = cats(s_old, s_new)
dbl, _ = cats(p_new, p_double)


def where(mask, top=12):
    per_t = mask.sum().sort_values(ascending=False)
    per_t = per_t[per_t > 0]
    rows = mask.any(axis=1)
    by_year = pd.Series(mask.sum(axis=1).to_numpy(), index=mask.index).groupby(mask.index.year).sum()
    return {"tickers": int(len(per_t)), "top": {str(k): int(v) for k, v in per_t.head(top).items()},
            "first_date": str(mask.index[rows.to_numpy()].min().date()) if rows.any() else None,
            "by_year": {int(k): int(v) for k, v in by_year[by_year > 0].items()}}


out = {"workbook_mtime": str(pd.Timestamp(wb.stat().st_mtime, unit="s")), "same_dates": bool(same_dates),
       "same_px_last": bool(same_px), "loader_diag_old": diag_old, "loader_diag_new": diag_new,
       "generator_diag": gen_diag, "currency_mismatch": {k: list(v) for k, v in cur_mismatch.items()},
       "pre_impute": pre, "pre_impute_where": where(pre_mask),
       "get_sheet": fin, "get_sheet_where": where(fin_mask),
       "double_restamp_vs_single": dbl, "double_restamp_diag": {k: diag_double[k] for k in ("changes", "matched", "lag_sessions_median")},
       "loader_dates": [str(dates1[0].date()), str(dates1[-1].date()), int(len(dates1))],
       "business_days": [str(bd.min().date()), str(bd.max().date()), int(len(bd))]}
(SCR / "om_source_parity.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
pre_mask.to_pickle(SCR / "om_parity_pre_mask.pkl")
fin_mask.to_pickle(SCR / "om_parity_fin_mask.pkl")
print(json.dumps(out, indent=1, ensure_ascii=False))
print("total", round(time.time() - t0), "s")
