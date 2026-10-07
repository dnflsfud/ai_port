import sys, pickle, json
import numpy as np, pandas as pd
sys.path.insert(0, r"C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port")
from src import data_loader as dl
from src.tg_basis_guard import event_consistency
raw = pickle.load(open(sys.argv[1], "rb"))
raw2, _ = dl.restrict_to_business_days(raw)
meta = dl.load_universe_meta(raw2); tickers = list(meta.index)
m = json.load(open(r"C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port\outputs\codex_causal_rank_65\metrics.json", encoding="utf-8"))
listing = {t: pd.Timestamp(d) for t, d in m["data_quality"]["listing_mask"]["dates"].items()}
def prep(name):
    df = dl._standardize_columns(raw2[name].copy())
    if name in dl.BLOOMBERG_EQUITY_SHEETS: df = dl._rename_bloomberg_equity_columns(df)
    df = dl._filter_tickers(dl._standardize_index(df), tickers=tickers)
    return df.apply(pd.to_numeric, errors="coerce")
un = prep("PX_LAST_UNADJ"); adj = prep("PX_LAST"); tg = prep("Factset_TG_Price")
idx = un.index.intersection(tg.index); un, adj, tg = un.loc[idx], adj.loc[idx], tg.loc[idx]
for t in un.columns:
    pre = idx < listing.get(t, idx[0]); un.loc[pre, t] = np.nan; adj.loc[pre, t] = np.nan; tg.loc[pre, t] = np.nan
tg = tg.where(tg.notna().cummax())
ratio = tg / un.replace(0, np.nan)
# (1) every single-day step in UNADJ/ADJ larger than 5% = corporate action left unadjusted in the nominal price
step = np.log(un / adj).diff()
ev = step.stack(); ev = ev[ev.abs() > 0.05]
print("UNADJ/ADJ single-day steps > 5%:", len(ev))
events = {}
for (d, t), v in ev.items():
    events.setdefault(t, {})[str(d.date())] = 1.0     # factor 1.0 => residual = log(raw TG/price step)
chk = event_consistency(ratio, events)
rows = [(e["ticker"], e["date"], round(float(ev.loc[(pd.Timestamp(e["date"]), e["ticker"])]), 3), e.get("raw_step"), e.get("log_residual"), e["status"]) for e in chk["events"]]
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
print(pd.DataFrame(rows, columns=["ticker","date","dlog(UNADJ/ADJ)","TG/px step","log_resid(no factor)","status"]).to_string(index=False))
# (2) the two registered events, as the loader does
print(json.dumps(event_consistency(ratio, {"DELL": {"2021-11-02": 0.506}, "DHR": {"2016-07-05": 0.758}})["events"], indent=0)[:900])
# (3) HPE 2026-06-02
for t, d in [("HPE", "2026-06-02"), ("TTD", "2026-08-07")]:
    pos = idx.get_loc(pd.Timestamp(d))
    print(t, d, "\n", pd.DataFrame({"TG": tg[t], "UNADJ": un[t], "ADJ": adj[t]}).iloc[pos-4:pos+5].round(3).to_string())
