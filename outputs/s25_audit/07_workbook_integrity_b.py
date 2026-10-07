import sys, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port")
from src import data_loader as dl
raw = pickle.load(open(sys.argv[1], "rb"))
meta = dl.load_universe_meta(raw); tickers = list(meta.index)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)


def prep(name, drop_weekend=True):
    df = dl._rename_bloomberg_equity_columns(dl._standardize_columns(raw[name].copy()))
    df = dl._standardize_index(df)
    df = dl._filter_tickers(df, tickers=tickers).apply(pd.to_numeric, errors="coerce")
    return df[df.index.dayofweek < 5] if drop_weekend else df


px = prep("PX_LAST"); un = prep("PX_LAST_UNADJ"); tg = prep("Factset_TG_Price")
# TTE tail
r = (px["TTE"] / un["TTE"])
print("TTE PX_LAST/UNADJ ratio, last 12 rows:\n", pd.DataFrame({"PX_LAST": px["TTE"], "UNADJ": un["TTE"], "ratio": r.round(5)}).tail(12).to_string())
print("rows where ratio != 1 within last 60:", int((r.tail(60).sub(1).abs() > 1e-6).sum()), "| ratio unique values (last 60):", sorted(r.tail(60).round(5).unique().tolist()))
# any other name whose ratio != 1 anywhere in the last 20 rows
dev = ((px / un) - 1).abs().tail(20)
print("names with |ratio-1|>1e-4 in last 20 rows:", {t: int((dev[t] > 1e-4).sum()) for t in dev.columns if (dev[t] > 1e-4).any()})
# DELL / HPE TG jumps
for t, d in (("DELL", "2026-05-29"), ("HPE", "2026-06-02")):
    pos = px.index.get_loc(pd.Timestamp(d))
    print(f"\n{t} around {d}:\n", pd.DataFrame({"TG": tg[t], "UNADJ": un[t], "TG/px": (tg[t] / un[t]).round(3)}).iloc[pos - 3:pos + 4].to_string())
# iv30 reproduction
iv = prep("30DAY_IMPVOL_100.0%MNY_DF"); iv30 = prep("iv30"); ivz = prep("iv30_z")
print("\niv30 == 30DAY_IMPVOL: max|diff|", float(np.nanmax((iv30 - iv.reindex_like(iv30)).abs().to_numpy())),
      "| iv30_z range:", round(float(np.nanmin(ivz.to_numpy())), 2), round(float(np.nanmax(ivz.to_numpy())), 2),
      "| iv30_z last-row non-NaN:", int(ivz.iloc[-1].notna().sum()))
# FactSet columns ending early / all-NaN
for name in ("Factset_Fwd_OpCashflow", "Factset_EPS_Surprise", "Factset_Sales_Surprise", "Factset_EPS_Revision", "Factset_TG_Price", "Factset_Sales_Revision"):
    df = prep(name)
    lv = df.apply(lambda s: s.last_valid_index())
    allnan = lv[lv.isna()].index.tolist()
    early = {t: str(pd.Timestamp(v).date()) for t, v in lv.dropna().items() if pd.Timestamp(v) < df.index[-1] - pd.Timedelta(days=45)}
    fv = df.apply(lambda s: s.first_valid_index()).dropna()
    late = {t: str(pd.Timestamp(v).date()) for t, v in fv.items() if pd.Timestamp(v) > pd.Timestamp("2024-01-01")}
    print(f"\n{name}: all-NaN {len(allnan)} {allnan} | ended >45d before sheet end {len(early)} {early} | coverage starting after 2024 {len(late)} {late}")
# Earnings_Timeline per-ticker last event date
et = raw["Earnings_Timeline"].copy(); et.columns = [str(c).strip() for c in et.columns]
et.index = pd.to_datetime(et.index, errors="coerce"); et = et.apply(pd.to_numeric, errors="coerce")
ev = (et > 0)
last_ev = ev.apply(lambda s: s[s].index.max() if s.any() else pd.NaT)
first_ev = ev.apply(lambda s: s[s].index.min() if s.any() else pd.NaT)
cnt = ev.sum()
print("\nEarnings_Timeline: events/ticker median", float(cnt.median()), "| tickers with 0 events:", cnt[cnt == 0].index.tolist(),
      "\n  last event before 2026-06-01:", {t: str(d.date()) for t, d in last_ev.dropna().items() if d < pd.Timestamp("2026-06-01")},
      "\n  events in last 120 days (should be ~1 per name):", ev.loc[ev.index >= ev.index[-1] - pd.Timedelta(days=120)].sum().value_counts().sort_index().to_dict())
d2e = prep("days_to_earnings")
print("days_to_earnings last row: non-NaN", int(d2e.iloc[-1].notna().sum()), "| range", float(np.nanmin(d2e.to_numpy())), float(np.nanmax(d2e.to_numpy())))
# EQY/NEWS/OM/ROE ranges
rec = prep("EQY_REC_CONS"); ns = prep("NEWS_SENTIMENT_DAILY_AVG"); om = prep("OPER_MARGIN"); roe = prep("BEST_ROE")
print("\nEQY_REC_CONS outside [1,5]:", int(((rec < 1) | (rec > 5)).sum().sum()),
      "| NEWS_SENTIMENT outside [-1,1]:", int(((ns < -1) | (ns > 1)).sum().sum()), "NEWS range", round(float(np.nanmin(ns.to_numpy())), 3), round(float(np.nanmax(ns.to_numpy())), 3),
      "| OPER_MARGIN |x|>100 names:", (om.abs() > 100).any()[lambda s: s].index.tolist(),
      "| BEST_ROE |x|>500 names:", (roe.abs() > 500).any()[lambda s: s].index.tolist())
# unit consistency PE*EPS/UNADJ
pe = prep("BEST_PE_RATIO"); eps = prep("BEST_EPS"); pb = prep("BEST_PX_BPS_RATIO"); mc = prep("CUR_MKT_CAP")
k = (pe * eps / un.replace(0, np.nan)).tail(252).median()
off = k[(k < 0.7) | (k > 1.4)].round(3)
print("\nPE*EPS/UNADJ 252d median outside [0.7,1.4]:", len(off), off.to_dict())
# Sentiment sheets: coverage
for name in ("Sent_Trend_Momentum_Timeseries", "Sent_Trend_21d_Timeseries"):
    df = raw[name].copy(); df.index = pd.to_datetime(df.index, errors="coerce"); df = df.apply(pd.to_numeric, errors="coerce")
    tail_nan = df.tail(5).isna().sum()
    print(f"{name}: last-row NaN cols {int(df.iloc[-1].isna().sum())} | cols all-NaN last 21 rows: {int(df.tail(21).isna().all().sum())} | range {round(float(np.nanmin(df.to_numpy())),3)}..{round(float(np.nanmax(df.to_numpy())),3)}")
# CUR_MKT_CAP sanity: USD mkt cap top/bottom and implied shares vs known (AAPL ~15bn)
sh = (mc / px.replace(0, np.nan)).iloc[-1]
print("\nimplied shares (bn) last row: AAPL", round(sh["AAPL"] / 1e9, 2), "005930", round(sh["005930"] / 1e9, 2), "7203", round(sh["7203"] / 1e9, 2), "HSBA", round(sh["HSBA"] / 1e9, 2), "AZN", round(sh["AZN"] / 1e9, 2),
      "| mktcap USD bn min/max:", round(float(mc.iloc[-1].min() / 1e9), 1), round(float(mc.iloc[-1].max() / 1e9), 1), "| min name", mc.iloc[-1].idxmin())
