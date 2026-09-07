# -*- coding: utf-8 -*-
"""Structural review r5 — raw workbook probes (read-only, no backtest).

A1 coverage-gap imputation audit on ALL per-ticker sheets (W3 critic #2)
A2 spin-off / capital-change audit on the nominal price + TG ratio (W3 critic #3)
A3 stale (zero) return share by currency
Saves a compact pickle for probe_b_pkl.py.
"""
import json, sys, pickle, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
AI_PORT = Path(r"C:/Users/westl/PycharmProjects/pythonProject/venv_vf_new/machine/re_study/c2/ai_port")
WB = Path(r"C:/Users/westl/PycharmProjects/pythonProject/venv_vf_new/machine/re_study/ai_signal_data.xlsx")
SCRATCH = Path(__file__).resolve().parent
OUT_JSON = SCRATCH / "probe_a_results.json"
OUT_PKL = SCRATCH / "probe_a_raw.pkl"

LEVEL_SHEETS = [
    "PX_LAST", "PX_LAST_UNADJ", "CUR_MKT_CAP", "Factset_TG_Price",
    "BEST_EPS", "BEST_SALES", "BEST_ROE", "BEST_PE_RATIO", "BEST_PX_BPS_RATIO",
    "BEST_PEG_RATIO", "BEST_EV_TO_BEST_EBITDA", "BEST_CALCULATED_FCF", "BEST_CAPEX",
    "BEST_GROSS_MARGIN", "OPER_MARGIN", "EQY_REC_CONS", "Factset_EPS_Revision",
    "Factset_Sales_Revision", "NEWS_SENTIMENT_DAILY_AVG", "Factset_Fwd_OpCashflow",
    "Fwd_Sales_Slope_1FY2FY",
]
SHEETS = LEVEL_SHEETS + ["Daily_Returns", "Universe_Meta"]

metrics = json.load(open(AI_PORT / "outputs/s17_5_nominal_price/metrics.json", encoding="utf-8"))
listing = {k: pd.Timestamp(v) for k, v in metrics["data_quality"]["listing_mask"]["dates"].items()}
tickers = list(listing)

print("[A] reading workbook sheets ...", flush=True)
raw = pd.read_excel(WB, sheet_name=SHEETS, index_col=0, engine="openpyxl")
print("[A] read done", flush=True)

meta = raw.pop("Universe_Meta")
def _split(v):
    parts = str(v).strip().rsplit(" ", 2)
    return (parts[0], parts[1]) if len(parts) == 3 else (str(v), None)
mkt = {}
for idx, row in meta.iterrows():
    t, ex = _split(row["Ticker"] if "Ticker" in meta.columns else idx)
    mkt[t] = ex
M2C = {"US": "USD", "KS": "KRW", "JP": "JPY", "FP": "EUR", "GR": "EUR", "NA": "EUR", "SW": "CHF",
       "LN": "GBP", "DC": "DKK", "SM": "EUR", "IM": "EUR"}
ccy = {t: M2C.get(mkt.get(t), "USD") for t in tickers}

def _std(df):
    df = df.copy()
    df.columns = [str(c).strip().split()[0] if str(c).strip().endswith("Equity") else str(c).strip() for c in df.columns]
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df[[t for t in tickers if t in df.columns]]
    return df.apply(pd.to_numeric, errors="coerce")

sheets = {k: _std(v) for k, v in raw.items()}
del raw
dates_ref = sheets["PX_LAST"].index

def mask_pre(df, inclusive):
    out = df.copy()
    for t, d in listing.items():
        if t in out.columns:
            out.loc[out.index <= d if inclusive else out.index < d, t] = np.nan
    return out

results = {}

# ---------------------------------------------------------------- A1 coverage-gap audit
print("[A1] coverage-gap audit", flush=True)
a1 = {}
for name in LEVEL_SHEETS:
    if name not in sheets:
        continue
    df = mask_pre(sheets[name], inclusive=False)
    ff = df.ffill()
    row_med = ff.median(axis=1)
    lead = ff.isna()  # cells still NaN after ffill => leading (post-listing coverage gap) => CS median fill
    # restrict to rows within each ticker's listed period and to the common date range
    listed = pd.DataFrame({t: (df.index >= listing[t]) for t in df.columns}, index=df.index)
    lead = lead & listed
    n_cells = int(lead.sum().sum())
    per_t = lead.sum()
    per_t = per_t[per_t > 0].sort_values(ascending=False)
    # magnitude: fill value (row median) vs first observed value of the ticker
    mags = {}
    for t in per_t.index[:40]:
        first = df[t].first_valid_index()
        if first is None:
            mags[t] = None; continue
        fv = float(df.loc[first, t])
        fills = row_med[lead[t]]
        if fv == 0 or not np.isfinite(fv) or fills.empty:
            mags[t] = {"rows": int(per_t[t]), "first_obs": fv, "fill_median": float(fills.median())}
        else:
            ratio = (fills / fv)
            mags[t] = {"rows": int(per_t[t]), "first_obs": round(fv, 4), "fill_median": round(float(fills.median()), 4),
                       "median_fill_over_first": round(float(ratio.median()), 3),
                       "gap_start": str(fills.index.min().date()), "gap_end": str(fills.index.max().date()),
                       "ccy": ccy.get(t)}
    interior = int((df.isna() & ff.notna() & listed).sum().sum())
    a1[name] = {"leading_gap_cells": n_cells, "tickers_with_gap": int((per_t > 0).sum()),
                "interior_ffilled_cells": interior, "top": mags}
results["A1_coverage_gap"] = a1

# CUR_MKT_CAP gaps -> benchmark weight impact (share of cap-weight that is imputed, per date)
mc = mask_pre(sheets["CUR_MKT_CAP"], inclusive=False)
mcff = mc.ffill()
lead_mc = mcff.isna() & pd.DataFrame({t: (mc.index >= listing[t]) for t in mc.columns}, index=mc.index)
filled = mcff.copy()
rm = mcff.median(axis=1)
for t in filled.columns:
    m = lead_mc[t]
    if m.any():
        filled.loc[m, t] = rm[m]
imputed_share = (filled.where(lead_mc, 0.0).sum(axis=1) / filled.sum(axis=1))
results["A1_mktcap_bm_impact"] = {
    "dates_with_imputed_bm_weight": int((imputed_share > 0).sum()),
    "max_imputed_bm_share": float(imputed_share.max()),
    "p95_imputed_bm_share_on_affected_dates": float(imputed_share[imputed_share > 0].quantile(0.95)) if (imputed_share > 0).any() else 0.0,
    "worst_dates": {str(d.date()): round(float(v), 4) for d, v in imputed_share.sort_values(ascending=False).head(8).items()},
}

# ---------------------------------------------------------------- A2 spin-off / capital-change audit
print("[A2] spin-off audit", flush=True)
px = sheets["PX_LAST"]; un = sheets["PX_LAST_UNADJ"].reindex_like(px); mcap = sheets["CUR_MKT_CAP"].reindex_like(px)
tg = sheets["Factset_TG_Price"].reindex_like(px); ret = sheets["Daily_Returns"].reindex_like(px)
scale = {t: (0.01 if mkt.get(t) == "LN" else 1.0) for t in px.columns}
un_s = un * pd.Series(scale)
px_s = px * pd.Series(scale)
log_sh_un = np.log(mcap / un_s.replace(0, np.nan))   # implied shares from nominal price
log_sh_adj = np.log(mcap / px_s.replace(0, np.nan))
adj_ratio = np.log(un_s / px_s.replace(0, np.nan))    # cumulative dividend(+abnormal) adjustment

SPINS = {
    "WDC": "2025-02-24", "GE": "2024-04-02", "DHR": "2023-10-02", "IBM": "2021-11-04", "DELL": "2021-11-01",
    "T": "2022-04-11", "PFE": "2020-11-16", "MRK": "2021-06-03", "SIE": "2020-09-28", "NOVN": "2023-10-04",
    "6758": "2025-10-01", "HON": "2025-10-30", "CMCSA": "2026-01-02", "GE_GEHC": "2023-01-04",
}
def _step(series, d, w=5):
    s = series.dropna()
    before = s[(s.index < d) & (s.index >= d - pd.Timedelta(days=w * 2 + 4))]
    after = s[(s.index >= d) & (s.index <= d + pd.Timedelta(days=w * 2 + 4))]
    if before.empty or after.empty:
        return None
    return float(after.median() - before.median())
def _ratio_med(num, den, d, side, days=180):
    r = (num / den.replace(0, np.nan)).dropna()
    if side == "before":
        r = r[(r.index < d) & (r.index >= d - pd.Timedelta(days=days))]
    else:
        r = r[(r.index >= d) & (r.index <= d + pd.Timedelta(days=days))]
    return float(r.median()) if len(r) else None
a2 = {}
for key, ds in SPINS.items():
    t = key.split("_")[0]
    d = pd.Timestamp(ds)
    if t not in px.columns:
        a2[key] = {"present": False}; continue
    win = ret[t][(ret.index >= d - pd.Timedelta(days=7)) & (ret.index <= d + pd.Timedelta(days=7))]
    a2[key] = {
        "ticker": t, "date": ds,
        "min_daily_return_pm7d": round(float(win.min()), 4) if len(win) else None,
        "step_log_shares_nominal": None if _step(log_sh_un[t], d) is None else round(_step(log_sh_un[t], d), 4),
        "step_log_shares_adjusted": None if _step(log_sh_adj[t], d) is None else round(_step(log_sh_adj[t], d), 4),
        "step_log_unadj_over_adj": None if _step(adj_ratio[t], d) is None else round(_step(adj_ratio[t], d), 4),
        "tg_over_nominal_px_before180d": None if _ratio_med(tg[t], un_s[t], d, "before") is None else round(_ratio_med(tg[t], un_s[t], d, "before"), 3),
        "tg_over_nominal_px_after180d": None if _ratio_med(tg[t], un_s[t], d, "after") is None else round(_ratio_med(tg[t], un_s[t], d, "after"), 3),
    }
results["A2_known_spinoffs"] = a2

# generic scan: implied-share jumps (>10% in one day) on the nominal price panel, excluding listing start
jumps = []
d_log = log_sh_un.diff()
for t in px.columns:
    s = d_log[t].dropna()
    s = s[s.index > listing[t] + pd.Timedelta(days=10)]
    big = s[s.abs() > 0.10]
    for d, v in big.items():
        jumps.append({"ticker": t, "date": str(d.date()), "d_log_shares_nominal": round(float(v), 3),
                      "ret_local": round(float(ret.loc[d, t]), 4) if d in ret.index and np.isfinite(ret.loc[d, t]) else None,
                      "d_log_unadj_over_adj": round(float(adj_ratio[t].diff().loc[d]), 4) if d in adj_ratio.index and np.isfinite(adj_ratio[t].diff().loc[d]) else None})
jumps = sorted(jumps, key=lambda r: -abs(r["d_log_shares_nominal"]))
results["A2_share_jumps_gt10pct"] = {"count": len(jumps), "top": jumps[:60]}

# per ticker-year median TG/nominal px > 1.5 (inflated upside — spin-off or unit artefact)
tgr = (tg / un_s.replace(0, np.nan))
ty = tgr.groupby(tgr.index.year).median()
flag = ty.stack()
flag = flag[flag > 1.5].sort_values(ascending=False)
results["A2_tg_over_nominal_gt1p5_ticker_years"] = {"count": int(len(flag)),
    "top": [{"year": int(y), "ticker": str(t), "median_tg_over_px": round(float(v), 3)} for (y, t), v in flag.head(40).items()]}

# ---------------------------------------------------------------- A3 stale-return share
print("[A3] stale returns", flush=True)
r_m = mask_pre(ret, inclusive=True)
zero_share = (r_m == 0).sum() / r_m.notna().sum()
by_ccy = {}
for c in sorted(set(ccy.values())):
    cols = [t for t in r_m.columns if ccy[t] == c]
    if cols:
        by_ccy[c] = {"n": len(cols), "median_zero_share": round(float(zero_share[cols].median()), 4),
                     "max": round(float(zero_share[cols].max()), 4), "max_ticker": str(zero_share[cols].idxmax())}
results["A3_zero_return_share_by_ccy"] = by_ccy

json.dump(results, open(OUT_JSON, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
compact = {k: sheets[k].astype("float32") for k in ["PX_LAST", "PX_LAST_UNADJ", "CUR_MKT_CAP", "Factset_TG_Price",
                                                    "BEST_ROE", "BEST_EPS", "BEST_PX_BPS_RATIO", "Daily_Returns"]}
compact["listing"] = {k: str(v.date()) for k, v in listing.items()}
compact["ccy"] = ccy
compact["mkt"] = {t: mkt.get(t) for t in tickers}
pickle.dump(compact, open(OUT_PKL, "wb"))
print("[A] done ->", OUT_JSON, flush=True)
