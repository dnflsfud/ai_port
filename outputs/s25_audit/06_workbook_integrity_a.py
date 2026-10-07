"""Read-only integrity audit of the current ai_signal_data workbook (cached raw sheets)."""
import sys, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port")
sys.path.insert(0, r"C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study")
from src import data_loader as dl
from src.trading_calendar import exchange_sessions, business_days_from_raw
from universe_config import mask_backfilled_prefix

raw = pickle.load(open(sys.argv[1], "rb"))
meta = dl.load_universe_meta(raw)
tickers = list(meta.index)
print("sheets:", len(raw), "| Universe_Meta rows:", len(meta),
      "| status counts:", raw["Universe_Meta"]["Status"].value_counts().to_dict() if "Status" in raw["Universe_Meta"].columns else "n/a")


def prep(name, drop_weekend=True):
    df = dl._standardize_columns(raw[name].copy())
    if name in dl.BLOOMBERG_EQUITY_SHEETS:
        df = dl._rename_bloomberg_equity_columns(df)
    df = dl._standardize_index(df)
    df = dl._filter_tickers(df, tickers=tickers).apply(pd.to_numeric, errors="coerce")
    return df[df.index.dayofweek < 5] if drop_weekend else df


# ---- A. calendar + structure
bd = business_days_from_raw(raw)
xnys = exchange_sessions(bd[0].strftime("%Y-%m-%d"), bd[-1].strftime("%Y-%m-%d"))
print(f"BusinessDays: {len(bd)} {bd[0].date()}..{bd[-1].date()} | XNYS sessions: {len(xnys)} | "
      f"missing from BD: {len(xnys.difference(bd))} {list(xnys.difference(bd)[:5].date)} | "
      f"extra in BD: {len(bd.difference(xnys))} {list(bd.difference(xnys)[:5].date)}")
rows = []
for name, df0 in raw.items():
    if name in ("Universe_Meta", "BusinessDays", "Summary_Stats", "Factor_Meta"):
        continue
    idx = pd.DatetimeIndex(pd.to_datetime(df0.index, errors="coerce"))
    n_bad = int(idx.isna().sum()); dup = int(idx.duplicated().sum()); mono = bool(idx.dropna().is_monotonic_increasing)
    df = df0.apply(pd.to_numeric, errors="coerce")
    nonnum = int((df0.notna() & df.isna()).sum().sum())
    ninf = int(np.isinf(df.to_numpy(float)).sum()) if df.size else 0
    lv = pd.to_datetime(df.apply(lambda s: s.last_valid_index()).dropna(), errors="coerce")
    rows.append((name, df0.shape[0], df0.shape[1], str(idx.min().date()), str(idx.max().date()), n_bad, dup, mono, nonnum, ninf,
                 str(lv.min().date()) if len(lv) else None, int((lv < lv.max()).sum()) if len(lv) else None,
                 int(df.isna().all().sum())))
tab = pd.DataFrame(rows, columns=["sheet", "rows", "cols", "first", "last", "bad_idx", "dup_idx", "monotonic",
                                  "nonnumeric", "inf", "earliest_last_valid", "cols_ending_early", "all_nan_cols"])
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
print(tab.to_string(index=False))

# ---- B. tail / last-row consistency (High-1 regression)
px = prep("PX_LAST"); un = prep("PX_LAST_UNADJ"); mc = prep("CUR_MKT_CAP")
print("\nPX_LAST last:", px.index[-1].date(), "| BusinessDays last:", bd[-1].date(), "| PX rows after BD last:", int((px.index > bd[-1]).sum()))
dev = ((px / un) - 1).abs().iloc[-3:]
print("PX_LAST/UNADJ |ratio-1| max per row (last 3):", dev.max(axis=1).round(5).tolist(),
      "| last-row names >1e-3:", dev.iloc[-1][dev.iloc[-1] > 1e-3].round(4).to_dict())
shares = mc / px.replace(0, np.nan)
sh_chg = shares.pct_change().iloc[-5:].abs()
print("implied shares last-5-row |pct chg|>3%:",
      {d.strftime("%m-%d"): sh_chg.loc[d][sh_chg.loc[d] > 0.03].round(3).to_dict() for d in sh_chg.index if (sh_chg.loc[d] > 0.03).any()})
print("PX_LAST <=0 cells:", int((px <= 0).sum().sum()),
      "| NaN cells after first valid:", int(sum(px[t].loc[px[t].first_valid_index():].isna().sum() for t in px.columns)))

# ---- D. splits / corporate actions
step_ua = np.log(un / px).diff().abs().stack()
print("UNADJ/ADJ single-day steps >30% (unadjusted split in nominal series):", step_ua[step_ua > 0.3].round(3).to_dict())
bp = np.log(px).diff().abs().stack(); bp = bp[bp > 0.45]
print("PX_LAST single-day |log move|>0.45:", len(bp), {f"{t}@{d.date()}": round(v, 2) for (d, t), v in bp.items()})
tg = prep("Factset_TG_Price"); tg = tg.where(tg.notna().cummax())
btg = np.log(tg).diff().abs().stack(); btg = btg[btg > 0.5]
print("Factset_TG_Price single-day |log move|>0.5:", len(btg), {f"{t}@{d.date()}": round(v, 2) for (d, t), v in btg.items()})

# ---- E. reproductions
dr = prep("Daily_Returns", drop_weekend=False); px_all = prep("PX_LAST", drop_weekend=False)
rep = px_all.pct_change(fill_method=None)
common = dr.index.intersection(rep.index)
diff = (dr.loc[common] - rep.loc[common]).abs()
print("\nDaily_Returns vs PX_LAST.pct_change: max|diff|", float(np.nanmax(diff.to_numpy())),
      "| NaN-pattern mismatch cells:", int((dr.loc[common].isna() != rep.loc[common].isna()).sum().sum()))
fpx = raw["Factor_PX_LAST"].apply(pd.to_numeric, errors="coerce"); fpx.index = pd.to_datetime(fpx.index)
fr = raw["Factor_Returns"].apply(pd.to_numeric, errors="coerce"); fr.index = pd.to_datetime(fr.index)
rep_f = fpx.pct_change(fill_method=None)
d_f = (fr - rep_f).abs()
lvl = [c for c in fpx.columns if float(np.nanmax(d_f[c].to_numpy())) > 1e-9]
d_lvl = (fr[lvl] - fpx[lvl].diff()).abs() if lvl else None
print("Factor_Returns: cols not pct_change-reproduced:", lvl,
      "| those as level diff max|diff|:", float(np.nanmax(d_lvl.to_numpy())) if lvl else None,
      "| other cols max|diff|:", float(np.nanmax(d_f.drop(columns=lvl).to_numpy())))
s1 = prep("BEST_SALES", drop_weekend=False); s2 = prep("BEST_SALES_2BF", drop_weekend=False); sl = prep("Fwd_Sales_Slope_1BF2BF", drop_weekend=False)
m1 = mask_backfilled_prefix(s1); m2 = mask_backfilled_prefix(s2)
ci = m1.index.intersection(m2.index).intersection(sl.index); cc = [c for c in m1.columns if c in m2.columns and c in sl.columns]
rep_sl = (m2.loc[ci, cc] - m1.loc[ci, cc]) / m1.loc[ci, cc].replace(0, np.nan)
dsl = (sl.loc[ci, cc] - rep_sl).abs()
print("Fwd_Sales_Slope_1BF2BF vs (2BF-1BF)/1BF(prefix-masked): max|diff|", float(np.nanmax(dsl.to_numpy())),
      "| NaN-pattern mismatch cells:", int((sl.loc[ci, cc].isna() != rep_sl.isna()).sum().sum()),
      "| last-row coverage:", int(sl.iloc[-1].notna().sum()), "/", len(cc))
iv = prep("30DAY_IMPVOL_100.0%MNY_DF"); iv30 = prep("iv30"); ivz = prep("iv30_z")
print("iv30 == 30DAY_IMPVOL: max|diff|", float(np.nanmax((iv30 - iv.reindex_like(iv30)).abs().to_numpy())),
      "| iv30_z range:", round(float(np.nanmin(ivz.to_numpy())), 2), round(float(np.nanmax(ivz.to_numpy())), 2))

# ---- F. value ranges
rec = prep("EQY_REC_CONS"); ns = prep("NEWS_SENTIMENT_DAILY_AVG"); om = prep("OPER_MARGIN"); roe = prep("BEST_ROE")
print("\nEQY_REC_CONS outside [1,5]:", int(((rec < 1) | (rec > 5)).sum().sum()),
      "| NEWS_SENTIMENT outside [-1,1]:", int(((ns < -1) | (ns > 1)).sum().sum()),
      "| OPER_MARGIN |x|>100:", int((om.abs() > 100).sum().sum()),
      "| BEST_ROE |x|>500 names:", (roe.abs() > 500).any()[lambda s: s].index.tolist())

# ---- G. unit consistency: PE*EPS vs nominal price (last 252 rows)
pe = prep("BEST_PE_RATIO"); eps = prep("BEST_EPS"); pb = prep("BEST_PX_BPS_RATIO")
k = (pe * eps / un.replace(0, np.nan)).tail(252).median()
off = k[(k < 0.7) | (k > 1.4)].round(3)
print("PE*EPS/UNADJ 252d median outside [0.7,1.4]:", len(off), off.to_dict())

# ---- H. earnings sheets
et = raw["Earnings_Timeline"]; ed = raw["Earnings_Date"]; d2e = prep("days_to_earnings")
same = et.shape == ed.shape and bool((et.fillna(-1).to_numpy() == ed.fillna(-1).to_numpy()).all())
print("Earnings_Timeline shape", et.shape, "| Earnings_Date shape", ed.shape, "| identical:", same)
print("days_to_earnings: min", float(np.nanmin(d2e.to_numpy())), "max", float(np.nanmax(d2e.to_numpy())),
      "| last-row NaN:", int(d2e.iloc[-1].isna().sum()))

# ---- I. known F1 live tails (count only)
print("\n(known F1) frozen live tails >=252 rows: PB", int(sum(pb[t].iloc[-252:].nunique() <= 1 for t in pb.columns)),
      "PE", int(sum(pe[t].iloc[-252:].nunique() <= 1 for t in pe.columns)))

# ---- J. sentiment / factset last dates
for name in ("Sent_Trend_Momentum_Timeseries", "Sent_Trend_21d_Timeseries", "Factset_EPS_Revision", "Factset_TG_Price",
             "Factset_Sales_Revision_FY2", "Factset_EPS_Est_StdDev", "Factset_Fwd_OpCashflow"):
    df = raw[name].apply(pd.to_numeric, errors="coerce"); idx = pd.to_datetime(df.index, errors="coerce")
    print(f"{name}: last {idx.max().date()} | cols {df.shape[1]} | last-row non-NaN {int(df.iloc[-1].notna().sum())} | all-NaN cols {int(df.isna().all().sum())}")
