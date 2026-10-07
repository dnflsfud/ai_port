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
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
daily = ["PX_LAST","PX_LAST_UNADJ","CUR_MKT_CAP","BEST_EPS","BEST_SALES","BEST_PE_RATIO","BEST_PX_BPS_RATIO","BEST_EV_TO_BEST_EBITDA","BEST_PEG_RATIO",
         "BEST_ROE","BEST_CALCULATED_FCF","BEST_CAPEX","BEST_GROSS_MARGIN","EQY_REC_CONS","NEWS_SENTIMENT_DAILY_AVG","OPER_MARGIN",
         "30DAY_IMPVOL_100.0%MNY_DF","iv30_z","Factset_TG_Price","Factset_EPS_Revision","Factset_Sales_Revision","PX_VOLUME"]
rows = []
for sh in daily:
    df = prep(sh)
    last = df.iloc[-1]
    v = df.to_numpy(float)
    # trailing constant run length per ticker
    eq_last = np.isclose(v, v[-1], rtol=1e-10, atol=1e-12, equal_nan=True)
    tr = np.array([ (np.argmax(~eq_last[::-1, j]) if (~eq_last[:, j]).any() else len(v)) for j in range(v.shape[1]) ])
    nan_last = int(np.isnan(v[-1]).sum())
    rows.append((sh, df.shape[1], str(df.index[-1].date()), nan_last, int(np.median(tr)), int(np.percentile(tr, 90)), int(tr.max()),
                 int((tr >= 21).sum()), int((tr >= 63).sum()),
                 ",".join(f"{t}:{n}" for t, n in sorted(zip(df.columns, tr), key=lambda x: -x[1])[:8])))
print(pd.DataFrame(rows, columns=["sheet","n","last_row","nan_last","med_trail","p90_trail","max_trail","n>=21","n>=63","top"]).to_string(index=False))
# Demonstration: cross-sectional PE rank of UBER on 2020-06-30
pe = prep("BEST_PE_RATIO"); d = pd.Timestamp("2020-06-30")
print("\nUBER PE 2020-06-30:", pe.loc[d, "UBER"], "rank pct:", round(float(pe.loc[d].rank(pct=True)["UBER"]), 4), "| cross-section median", round(float(pe.loc[d].median()), 2))
roe = prep("BEST_ROE"); d = pd.Timestamp("2017-06-30")
print("FICO ROE 2017-06-30:", roe.loc[d, "FICO"], "rank pct:", round(float(roe.loc[d].rank(pct=True)["FICO"]), 4), "| median", round(float(roe.loc[d].median()), 2))
