import sys, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port")
from src import data_loader as dl
raw = pickle.load(open(sys.argv[1], "rb"))
meta = dl.load_universe_meta(raw); tickers = list(meta.index)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)


def prep(name):
    df = dl._rename_bloomberg_equity_columns(dl._standardize_columns(raw[name].copy()))
    df = dl._standardize_index(df)
    df = dl._filter_tickers(df, tickers=tickers).apply(pd.to_numeric, errors="coerce")
    return df[df.index.dayofweek < 5]


px = prep("PX_LAST"); un = prep("PX_LAST_UNADJ"); tg = prep("Factset_TG_Price").reindex(px.index)
for t, d in (("DELL", "2026-05-29"), ("HPE", "2026-06-02"), ("TTD", "2026-08-07")):
    pos = px.index.get_loc(pd.Timestamp(d))
    print(f"{t} around {d}:\n", pd.DataFrame({"TG": tg[t], "UNADJ": un[t], "TG/px": (tg[t] / un[t]).round(3)}).iloc[pos - 3:pos + 4].to_string())

ns = prep("NEWS_SENTIMENT_DAILY_AVG")
bad = ns.stack(); bad = bad[(bad < -1) | (bad > 1)]
print("\nNEWS_SENTIMENT outside [-1,1]: cells", len(bad), "| names:", bad.groupby(level=1).size().to_dict())
print("  date range of bad cells:", bad.index.get_level_values(0).min().date(), "..", bad.index.get_level_values(0).max().date())
print("  sample:", {f"{t}@{d.date()}": v for (d, t), v in bad.sort_values().head(8).items()})
for t in bad.index.get_level_values(1).unique()[:3]:
    s = ns[t]; b = s[(s < -1) | (s > 1)]
    print(f"  {t}: bad rows {len(b)} {b.index.min().date()}..{b.index.max().date()} | unique bad values {sorted(b.round(1).unique().tolist())[:5]} | normal range {round(float(s[(s>=-1)&(s<=1)].min()),3)}..{round(float(s[(s>=-1)&(s<=1)].max()),3)}")

om = prep("OPER_MARGIN")
ext = pd.DataFrame({"min": om.min(), "max": om.max(), "last": om.iloc[-1]}).loc[(om.abs() > 100).any()[lambda s: s].index]
print("\nOPER_MARGIN names with |x|>100:\n", ext.round(1).to_string())
over = om.stack(); over = over[over > 100]
print("  cells >100:", len(over), "| names:", over.groupby(level=1).size().to_dict())

et = raw["Earnings_Timeline"].copy(); et.columns = [str(c).strip() for c in et.columns]
et.index = pd.to_datetime(et.index, errors="coerce"); et = et.apply(pd.to_numeric, errors="coerce")
b = et["BRK/B"]; evd = b[b > 0].index
print("\nBRK/B Earnings_Timeline events:", len(evd), "first", evd.min().date(), "last", evd.max().date(), "| per year:", evd.year.value_counts().sort_index().to_dict())
ev = (et > 0); two = ev.loc[ev.index >= ev.index[-1] - pd.Timedelta(days=120)].sum()
print("names with 2 events in last 120d:", {t: [str(d.date()) for d in ev.index[ev.index >= ev.index[-1] - pd.Timedelta(days=120)][ev.loc[ev.index >= ev.index[-1] - pd.Timedelta(days=120), t].to_numpy(bool)]] for t in two[two == 2].index})
print("Earnings_Timeline value set:", sorted(pd.unique(et.to_numpy().ravel()))[:6])

# production use of OpCashflow / Surprise sheets
import re, pathlib
root = pathlib.Path(r"C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port")
src = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in root.glob("src/**/*.py"))
print("\nsrc references: Factset_Fwd_OpCashflow", src.count("Factset_Fwd_OpCashflow"), "| EPS_Surprise", src.count("EPS_Surprise"), "| Sales_Surprise", src.count("Sales_Surprise"), "| days_to_earnings", src.count("days_to_earnings"))
asm = (root / "src/features/assembly.py").read_text(encoding="utf-8")
wl_start = asm.find("CORE_FEATURE_WHITELIST"); wl = asm[wl_start:wl_start + 6000]
print("whitelist mentions: opcf", len(re.findall(r"opcf", wl)), "| surprise", len(re.findall(r"surprise", wl)))
