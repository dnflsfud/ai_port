"""M2 supplementary: (a) ASIA x US Sigma-block ratio over ALL 97 rebalance dates,
(b) per-name risk share (a_i (Sigma a)_i / a'Sigma a) of ASIA OW names under daily vs 5d-overlap Sigma,
(c) sub-period stability of the weekly-vs-daily correlation recovery."""
import os, sys, pickle, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())
from src.config import PipelineConfig
VD = r"C:\Users\westl\AppData\Local\Temp\claude\C--Users-westl-PycharmProjects-pythonProject-venv-vf-new-machine-re-study-c2-ai-port\0caf960f-9660-4a4b-a91e-809a2f4ab53d\scratchpad\w3\verify"
PKL = r"C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port\outputs\s16_7_name_risk_cap\backtest_result.pkl"
S = pickle.load(open(os.path.join(VD, "m2_sheets.pkl"), "rb"))
cfg = PipelineConfig()
for k in ("Daily_Returns", "PX_LAST", "CUR_MKT_CAP"):
    S[k] = S[k][S[k].index.dayofweek < 5]
ret = S["Daily_Returns"]; px = S["PX_LAST"]; mcap = S["CUR_MKT_CAP"]; idx = S["Index_PX_LAST_subset"]
meta = S["Universe_Meta"].set_index("ticker"); tickers = list(ret.columns)
cur = meta["currency"].reindex(tickers)
region = pd.Series(np.where(cur == "USD", "US", np.where(cur.isin(["JPY", "KRW"]), "ASIA", "EUROPE")), index=tickers)
US = [t for t in tickers if region[t] == "US"]; ASIA = [t for t in tickers if region[t] == "ASIA"]; EU = [t for t in tickers if region[t] == "EUROPE"]
first_px = px.apply(lambda s: s.first_valid_index())
mask = pd.DataFrame(True, index=ret.index, columns=tickers)
for t in tickers:
    d0 = first_px[t]
    if t in cfg.listing_dates:
        d0 = max(pd.Timestamp(cfg.listing_dates[t]), d0) if d0 is not None else pd.Timestamp(cfg.listing_dates[t])
    if d0 is not None:
        mask.loc[mask.index <= d0, t] = False
ret = ret.where(mask)
fxmap = {"JPY": ("USDJPY Curncy", "inv"), "KRW": ("USDKRW Curncy", "inv"), "EUR": ("EURUSD Curncy", "dir"), "CHF": ("USDCHF Curncy", "inv"), "GBP": ("GBPUSD Curncy", "dir"), "DKK": ("USDDKK Curncy", "inv")}
fx = pd.DataFrame(index=ret.index)
for c, (col, d) in fxmap.items():
    s = idx[col].where(idx[col] > 0); s = (1.0 / s) if d == "inv" else s
    fx[c] = s.reindex(ret.index.union(s.index)).ffill().reindex(ret.index)
fx["USD"] = 1.0
fxr = pd.DataFrame({t: fx[cur[t]].pct_change().fillna(0.0) for t in tickers})
ret_usd = ((1 + ret) * (1 + fxr) - 1).where(mask)
us_ew = ret[US].mean(axis=1)

res = pickle.load(open(PKL, "rb"))
rebal = sorted(res.portfolio_weights.keys())
mc_ff = mcap.reindex(columns=tickers).ffill()
def bm_at(dt):
    row = mc_ff.loc[:dt].iloc[-1].astype(float).copy()
    elig = mask.loc[:dt].iloc[-1]
    row[~elig.to_numpy()] = 0.0; row[~np.isfinite(row) | (row <= 0)] = 0.0
    return row / row.sum()
ia = np.array([t in ASIA for t in tickers])
rows = []; per_name = []
for dt in rebal:
    pos = ret_usd.index.get_loc(dt)
    win = ret_usd.iloc[max(0, pos - 126):pos]
    if len(win) < 60:
        continue
    C1 = win.corr(min_periods=30); ov = win.rolling(5).sum().dropna(how="all"); C5 = ov.corr(min_periods=30)
    cov1 = win.cov(min_periods=30).reindex(index=tickers, columns=tickers).fillna(0.0).to_numpy()
    cov5 = (ov.cov(min_periods=30) / 5.0).reindex(index=tickers, columns=tickers).fillna(0.0).to_numpy()
    a = (res.portfolio_weights[dt].reindex(tickers).fillna(0.0) - bm_at(dt)).to_numpy()
    v1 = a @ cov1 @ a; v5 = a @ cov5 @ a
    rs1 = a * (cov1 @ a) / v1; rs5 = a * (cov5 @ a) / v5
    rows.append({"date": dt, "corr_AxUS_daily": float(np.nanmean(C1.loc[ASIA, US].values)), "corr_AxUS_5dov": float(np.nanmean(C5.loc[ASIA, US].values)),
                 "corr_EUxUS_daily": float(np.nanmean(C1.loc[EU, US].values)), "corr_EUxUS_5dov": float(np.nanmean(C5.loc[EU, US].values)),
                 "corr_USxUS_daily": float(np.nanmean(C1.loc[US, US].values)), "corr_USxUS_5dov": float(np.nanmean(C5.loc[US, US].values)),
                 "te_daily": np.sqrt(v1 * 252), "te_5dov": np.sqrt(v5 * 252), "asia_rs_daily": rs1[ia].sum(), "asia_rs_5dov": rs5[ia].sum(),
                 "asia_net_active": a[ia].sum(), "max_asia_rs_daily": rs1[ia].max(), "max_asia_rs_5dov": rs5[ia].max(),
                 "max_asia_rs_name": tickers[int(np.where(ia)[0][np.argmax(rs1[ia])])]})
    for i in np.where(ia)[0]:
        if a[i] > 0.005:
            per_name.append({"date": dt, "ticker": tickers[i], "active": a[i], "rs_daily": rs1[i], "rs_5dov": rs5[i]})
df = pd.DataFrame(rows).set_index("date")
df["ratio_AxUS"] = df["corr_AxUS_5dov"] / df["corr_AxUS_daily"]
df["ratio_EUxUS"] = df["corr_EUxUS_5dov"] / df["corr_EUxUS_daily"]
df["ratio_USxUS"] = df["corr_USxUS_5dov"] / df["corr_USxUS_daily"]
df["diff_AxUS"] = df["corr_AxUS_5dov"] - df["corr_AxUS_daily"]
print("=== (a) 97 rebalance dates: block corr ===")
print(df[["corr_AxUS_daily", "corr_AxUS_5dov", "diff_AxUS", "ratio_AxUS", "corr_EUxUS_daily", "corr_EUxUS_5dov", "ratio_EUxUS", "corr_USxUS_daily", "corr_USxUS_5dov", "ratio_USxUS"]].describe().round(3).T.to_string())
print("share of dates with corr_AxUS_5dov > corr_AxUS_daily: %.3f" % (df["diff_AxUS"] > 0).mean())
print("yearly mean (daily, 5dov):"); print(df.groupby(df.index.year)[["corr_AxUS_daily", "corr_AxUS_5dov", "corr_USxUS_daily", "corr_USxUS_5dov"]].mean().round(3).to_string())
print("=== (b) risk shares ===")
print(df[["te_daily", "te_5dov", "asia_rs_daily", "asia_rs_5dov", "asia_net_active", "max_asia_rs_daily", "max_asia_rs_5dov"]].describe().round(4).T.to_string())
pn = pd.DataFrame(per_name)
print("ASIA OW names (>0.5%% active) risk share daily vs 5dov: n=%d" % len(pn))
print(pn.groupby("ticker")[["active", "rs_daily", "rs_5dov"]].mean().round(4).to_string())
print("last 5 dates per-name:"); print(pn[pn["date"] >= rebal[-5]].round(4).to_string())
print("share of (date,name) with rs_5dov > 0.35 while rs_daily <= 0.35: %d / %d" % (((pn.rs_5dov > 0.35) & (pn.rs_daily <= 0.35)).sum(), len(pn)))

print("=== (c) sub-period weekly/daily corr recovery (ASIA EW vs US EW) ===")
asia_ew = ret[ASIA].mean(axis=1)
for a, b in [("2014", "2017"), ("2018", "2021"), ("2022", "2026")]:
    d = pd.concat([asia_ew, us_ew], axis=1).loc[a:b]
    cd = d.iloc[:, 0].corr(d.iloc[:, 1]); cl = d.iloc[:, 0].corr(d.iloc[:, 1].shift(1))
    w = d.resample("W-FRI").sum(min_count=3); cw = w.iloc[:, 0].corr(w.iloc[:, 1])
    print(f"{a}-{b}: daily same {cd:.3f} | vs US_(t-1) {cl:.3f} | weekly {cw:.3f} | ratio {cw/cd:.2f}")
out = {"a_ratio_AxUS_median": float(df["ratio_AxUS"].median()), "a_corr_AxUS_daily_mean": float(df["corr_AxUS_daily"].mean()), "a_corr_AxUS_5dov_mean": float(df["corr_AxUS_5dov"].mean()),
       "a_ratio_USxUS_median": float(df["ratio_USxUS"].median()), "a_ratio_EUxUS_median": float(df["ratio_EUxUS"].median()),
       "b_asia_rs_daily_mean": float(df["asia_rs_daily"].mean()), "b_asia_rs_5dov_mean": float(df["asia_rs_5dov"].mean()),
       "b_te_ratio_mean": float((df["te_5dov"] / df["te_daily"]).mean())}
json.dump(out, open(os.path.join(VD, "M2_supp_numbers.json"), "w"), indent=1)
df.to_csv(os.path.join(VD, "M2_supp_sigma_all_rebal.csv"))
