import sys, pickle, json
import numpy as np, pandas as pd
sys.path.insert(0, r"C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port")
from src import data_loader as dl
raw = pickle.load(open(sys.argv[1], "rb"))
raw2, _ = dl.restrict_to_business_days(raw)
meta = dl.load_universe_meta(raw2); tickers = list(meta.index)
def prep(name):
    df = dl._standardize_columns(raw2[name].copy())
    if name in dl.BLOOMBERG_EQUITY_SHEETS: df = dl._rename_bloomberg_equity_columns(df)
    df = dl._filter_tickers(dl._standardize_index(df), tickers=tickers)
    return df.apply(pd.to_numeric, errors="coerce")
o = pd.read_csv(sys.argv[2])
o["suspect"] = (o["lead_run"] >= 40) & (o["lead_run"] > 3 * o["max_later_run"].clip(lower=1))
pd.set_option("display.width", 220)
# sample values around the first change
for sh, t in [("BEST_PE_RATIO","UBER"),("BEST_ROE","FICO"),("EQY_REC_CONS","AON"),("BEST_EV_TO_BEST_EBITDA","IBKR"),("BEST_CAPEX","BNP"),("BEST_EPS","LIN"),("BEST_PE_RATIO","NET")]:
    s = prep(sh)[t]; r = o[(o.sheet==sh)&(o.ticker==t)].iloc[0]
    fc = pd.Timestamp(r.first_change); pos = s.index.get_loc(fc)
    print(f"\n{sh} {t}: listing {r.listing} first_change {r.first_change} lead_run {r.lead_run}")
    print("  first 3:", s[s.index>=pd.Timestamp(r.listing)].head(3).round(4).tolist(), "| around change:", s.iloc[pos-3:pos+4].round(4).tolist())
# contamination by year for level-consumed sheets
rows = []
for sh in ["BEST_ROE","BEST_PE_RATIO","BEST_PX_BPS_RATIO","EQY_REC_CONS","BEST_CAPEX","BEST_CALCULATED_FCF","BEST_EV_TO_BEST_EBITDA","BEST_PEG_RATIO","BEST_GROSS_MARGIN","BEST_EPS","BEST_SALES","OPER_MARGIN","NEWS_SENTIMENT_DAILY_AVG"]:
    df = prep(sh); d = o[(o.sheet==sh)&o.suspect]
    cnt = pd.Series(0, index=df.index)
    for _, r in d.iterrows():
        end = pd.Timestamp(r.first_change) if isinstance(r.first_change, str) else df.index[-1] + pd.Timedelta(days=1)
        m = (df.index >= pd.Timestamp(r.listing)) & (df.index < end)
        cnt[m] += 1
    by = cnt.groupby(cnt.index.year).mean().round(1)
    rows.append(pd.Series(by, name=sh))
print("\nAverage # of names in back-filled prefix per day, by year (universe 250):")
print(pd.DataFrame(rows).fillna(0).to_string())
