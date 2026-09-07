# -*- coding: utf-8 -*-
"""Probe C: leading CONSTANT (backfilled) prefixes in per-ticker sheets, post-listing.

The loader masks pre-listing rows and the FY-override sheets mask their flat prefix at
generation time, but the other Bloomberg BEST_* / FactSet sheets are consumed as-is.
A vendor backfill shows up as a leading constant run (value != NaN), invisible to the
NaN-based coverage audits. Count, per sheet, the post-listing leading-flat run length
(weekday rows) with the run measured on the raw calendar-day sheet, then converted.
"""
import json, pickle, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
WB = Path(r"C:/Users/westl/PycharmProjects/pythonProject/venv_vf_new/machine/re_study/ai_signal_data.xlsx")
SCRATCH = Path(__file__).resolve().parent
c = pickle.load(open(SCRATCH / "probe_a_raw.pkl", "rb"))
listing = {k: pd.Timestamp(v) for k, v in c["listing"].items()}
SHEETS = ["BEST_EPS", "BEST_SALES", "BEST_ROE", "BEST_PE_RATIO", "BEST_PX_BPS_RATIO", "BEST_PEG_RATIO",
          "BEST_EV_TO_BEST_EBITDA", "BEST_CALCULATED_FCF", "BEST_CAPEX", "BEST_GROSS_MARGIN", "OPER_MARGIN",
          "EQY_REC_CONS", "Factset_EPS_Revision", "Factset_Sales_Revision", "Factset_TG_Price",
          "NEWS_SENTIMENT_DAILY_AVG", "CUR_MKT_CAP"]
raw = pd.read_excel(WB, sheet_name=SHEETS, index_col=0, engine="openpyxl")
def std(s):
    s = s.copy()
    s.columns = [str(x).strip().split()[0] if str(x).strip().endswith("Equity") else str(x).strip() for x in s.columns]
    s.index = pd.to_datetime(s.index, errors="coerce"); s = s[~s.index.isna()].sort_index()
    s = s[~s.index.duplicated(keep="first")]
    s = s[[t for t in listing if t in s.columns]].apply(pd.to_numeric, errors="coerce")
    return s[s.index.dayofweek < 5]
out = {}
for name, df in raw.items():
    df = std(df)
    rows = {}
    for t in df.columns:
        s = df[t]
        s = s[s.index >= listing[t]].dropna()
        if len(s) < 30:
            continue
        flat = int((s == s.iloc[0]).cummin().sum())
        rows[t] = (flat, str(s.index[0].date()), str(s.index[min(flat, len(s) - 1)].date()), float(s.iloc[0]))
    fl = pd.Series({t: v[0] for t, v in rows.items()})
    # only flag runs longer than 3 months (~65 weekday rows) as backfill-like
    long = fl[fl > 65].sort_values(ascending=False)
    out[name] = {
        "n_tickers": int(len(fl)), "median_flat_rows": float(fl.median()), "p90_flat_rows": float(fl.quantile(0.9)),
        "n_flat_gt_65": int((fl > 65).sum()), "n_flat_gt_252": int((fl > 252).sum()), "n_flat_gt_756": int((fl > 756).sum()),
        "total_flat_rows_gt65": int(long.sum()),
        "top": {t: {"flat_rows": int(rows[t][0]), "from": rows[t][1], "to": rows[t][2], "value": rows[t][3]} for t in long.index[:12]},
    }
    print(f"{name:26s} flat>65: {out[name]['n_flat_gt_65']:3d} names, >252: {out[name]['n_flat_gt_252']:3d}, >756: {out[name]['n_flat_gt_756']:3d}, "
          f"median {out[name]['median_flat_rows']:.0f} p90 {out[name]['p90_flat_rows']:.0f}  top: {list(long.index[:5])} {list(long.values[:5])}")
json.dump(out, open(SCRATCH / "probe_c_results.json", "w", encoding="utf-8"), indent=1, default=str)
