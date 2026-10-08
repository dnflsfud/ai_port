"""§S25.5 read-only prechecks on the new S0' pkl (s25_4_combined_recert) + 2026-10-08 workbook cache.
(5) top-decile convexity: per rebalance date OLS fwd_ret ~ a + b*pct_rank(pred) + c*1[top decile]; Fama-MacBeth t(c) >= 2 ?
(4) sector gate: Euler Tech active-risk share vs net Tech active weight; counterfactual Tech-net-neutral book."""
import sys, pickle, warnings, json
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port")
from src import data_loader as dl
from src.portfolio_optimizer import estimate_covariance
from src.config import PipelineConfig

res = pickle.load(open(sys.argv[1], "rb")); raw = pickle.load(open(sys.argv[2], "rb"))
meta = dl.load_universe_meta(raw); tickers = list(meta.index); sector = meta["sector"]
currency = meta["currency"].fillna("USD").str.upper()
def prep(name):
    df = dl._rename_bloomberg_equity_columns(dl._standardize_columns(raw[name].copy()))
    return dl._filter_tickers(dl._standardize_index(df), tickers=tickers).apply(pd.to_numeric, errors="coerce")
px, mc = prep("PX_LAST"), prep("CUR_MKT_CAP")
fpx = raw["Factor_PX_LAST"].apply(pd.to_numeric, errors="coerce"); fpx.index = pd.to_datetime(fpx.index)
usd_per = pd.DataFrame(1.0, index=fpx.index, columns=["USD"])
for ccy, col, inv in (("KRW","USDKRW",True),("JPY","USDJPY",True),("EUR","EURUSD",False),("CHF","USDCHF",True),("GBP","GBPUSD",False),("DKK","USDDKK",True)):
    if col in fpx.columns: usd_per[ccy] = (1.0/fpx[col]) if inv else fpx[col]
dates = res.predictions.index
px = px.reindex(dates).ffill(); mc = mc.reindex(dates).ffill()
fx = pd.DataFrame({t: usd_per.reindex(dates).ffill()[currency[t] if currency[t] in usd_per.columns else "USD"] for t in px.columns})
ret = (px*fx).pct_change(fill_method=None)
pred = res.predictions.reindex(dates).copy()
for t, d in res.data_quality["listing_mask"]["dates"].items():
    if t in pred.columns:
        pred.loc[pred.index < pd.Timestamp(d), t] = np.nan; ret.loc[ret.index <= pd.Timestamp(d), t] = np.nan
cum = (1+ret.fillna(0)).cumprod()
fwd = (cum.shift(-22)/cum.shift(-1) - 1)          # t+2..t+22 (production label window after B-05)
reb = [d for d in sorted(res.portfolio_weights) if d >= pd.Timestamp("2019-01-29")]

print("=== (5) top-decile convexity precheck (Fama-MacBeth over rebalance dates) ===")
rows = []
for d in reb:
    p, y = pred.loc[d], fwd.loc[d]; m = p.notna() & y.notna()
    if m.sum() < 100: continue
    p, y = p[m], y[m]; r = p.rank(pct=True)
    X = {"top10": (r > 0.9).astype(float), "top5": (r > 0.95).astype(float), "top20n": (p.rank(ascending=False) <= 20).astype(float)}
    out = {"date": d}
    for k, dummy in X.items():
        A = np.column_stack([np.ones(len(r)), r.to_numpy(), dummy.to_numpy()])
        beta, *_ = np.linalg.lstsq(A, y.to_numpy(), rcond=None)
        out[f"b_{k}"], out[f"c_{k}"] = beta[1], beta[2]
    # linear-only slope and decile means for reference
    A1 = np.column_stack([np.ones(len(r)), r.to_numpy()]); b1, *_ = np.linalg.lstsq(A1, y.to_numpy(), rcond=None); out["b_linear"] = b1[1]
    q = pd.qcut(r, 10, labels=False); dm = y.groupby(q).mean(); out["d9_minus_d8"] = dm.get(9, np.nan) - dm.get(8, np.nan); out["d8_minus_d7"] = dm.get(8, np.nan) - dm.get(7, np.nan)
    rows.append(out)
df = pd.DataFrame(rows).set_index("date")
def fm(s):
    s = s.dropna(); return round(float(s.mean()*100),3), round(float(s.mean()/s.std()*np.sqrt(len(s))),2), int(len(s))
summary = {k: fm(df[f"c_{k}"]) for k in ("top10","top5","top20n")}
summary["b_linear"] = fm(df["b_linear"]); summary["d9_minus_d8"] = fm(df["d9_minus_d8"]); summary["d8_minus_d7"] = fm(df["d8_minus_d7"])
print("mean(%) / FM t / n:", summary)
for k in ("top10","top5","top20n"):
    print(f"  {k}: dummy coef mean {summary[k][0]}% t {summary[k][1]} | share of dates c>0: {round(float((df[f'c_{k}']>0).mean()),3)} | by year t:",
          {y: round(float(g[f'c_{k}'].mean()/g[f'c_{k}'].std()*np.sqrt(len(g))),2) if len(g)>2 else None for y, g in df.groupby(df.index.year)})
print("PASS (t>=2 on top10 dummy):", summary["top10"][1] >= 2)

print("\n=== (4) sector gate decomposition (Euler active-risk share, cov_lookback 126, LW) ===")
cfg = PipelineConfig()
bm_all = mc.shift(1); bm_all = bm_all.div(bm_all.sum(axis=1), axis=0).fillna(0.0)
out4 = []
for d in reb[-24:]:
    w = pd.Series(res.portfolio_weights[d]).reindex(px.columns).fillna(0.0)
    bm = bm_all.loc[d]; a = (w - bm)
    hist = ret.loc[ret.index < d].tail(126).fillna(0.0)
    cov = np.asarray(estimate_covariance(hist, bm_weights=bm.to_numpy(), config=cfg), dtype=float)
    def shares(av):
        av = av.to_numpy(); var = float(av @ cov @ av); contrib = av * (cov @ av)
        s = pd.Series(contrib, index=px.columns).groupby(sector).sum() / var
        return s
    s0 = shares(a); tech = s0.get("Technology", np.nan)
    net_tech = float(a[sector == "Technology"].sum())
    # counterfactual 1: remove the net Tech bet by scaling Tech actives to net zero (shift by constant over Tech names)
    a1 = a.copy(); idx = sector[sector == "Technology"].index.intersection(a1.index); a1[idx] = a1[idx] - a1[idx].mean()
    s1 = shares(a1); tech1 = s1.get("Technology", np.nan)
    # counterfactual 2: keep the net Tech bet, kill within-Tech dispersion (every Tech name gets the same active weight)
    a2 = a.copy(); a2[idx] = net_tech / len(idx)
    s2 = shares(a2); tech2 = s2.get("Technology", np.nan)
    out4.append({"date": d.date(), "tech_share": round(float(tech),3), "net_tech_active": round(net_tech,3), "tech_share_net_neutral": round(float(tech1),3),
                 "tech_share_no_dispersion": round(float(tech2),3), "top_sector": s0.idxmax(), "max_abs_net_sector": round(float(a.groupby(sector).sum().abs().max()),3)})
d4 = pd.DataFrame(out4).set_index("date")
print(d4.tail(8).to_string())
print("\nlast 24 rebalances: tech_share mean", round(float(d4.tech_share.mean()),3), "| > 0.85 on", int((d4.tech_share > 0.85).sum()), "/", len(d4),
      "| net Tech active mean", round(float(d4.net_tech_active.mean()),3), "max", round(float(d4.net_tech_active.max()),3),
      "| counterfactual net-neutral Tech share mean", round(float(d4.tech_share_net_neutral.mean()),3),
      "| no-dispersion Tech share mean", round(float(d4.tech_share_no_dispersion.mean()),3),
      "| |net sector active| > 0.10 on", int((d4.max_abs_net_sector > 0.10).sum()), "dates")
json.dump({"convexity": summary, "sector": d4.reset_index().astype(str).to_dict("records")}, open(sys.argv[3], "w"), indent=1)
