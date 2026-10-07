"""Read-only alpha diagnostics on the production backtest pickle + cached workbook sheets."""
import sys, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port")
from src import data_loader as dl
from scipy.stats import spearmanr

res = pickle.load(open(sys.argv[1], "rb"))
raw = pickle.load(open(sys.argv[2], "rb"))
meta = dl.load_universe_meta(raw); tickers = list(meta.index)
sector = meta["sector"]
currency = meta["currency"].fillna("USD").str.upper()


def prep(name):
    df = dl._rename_bloomberg_equity_columns(dl._standardize_columns(raw[name].copy()))
    df = dl._filter_tickers(dl._standardize_index(df), tickers=tickers).apply(pd.to_numeric, errors="coerce")
    return df


px = prep("PX_LAST"); mc = prep("CUR_MKT_CAP")
fpx = raw["Factor_PX_LAST"].apply(pd.to_numeric, errors="coerce"); fpx.index = pd.to_datetime(fpx.index)
usd_per = pd.DataFrame(1.0, index=fpx.index, columns=["USD"])
for ccy, col, inv in (("KRW", "USDKRW", True), ("JPY", "USDJPY", True), ("EUR", "EURUSD", False), ("CHF", "USDCHF", True),
                      ("GBP", "GBPUSD", False), ("DKK", "USDDKK", True)):
    if col in fpx.columns:
        usd_per[ccy] = (1.0 / fpx[col]) if inv else fpx[col]
dates = res.predictions.index
px = px.reindex(dates).ffill(); mc = mc.reindex(dates).ffill()
fx = pd.DataFrame({t: usd_per.reindex(dates).ffill()[currency[t] if currency[t] in usd_per.columns else "USD"] for t in px.columns})
px_usd = px * fx
ret = px_usd.pct_change(fill_method=None)
pred = res.predictions.reindex(dates)
# mask pre-listing via loader-inferred listing dates
listing = res.data_quality["listing_mask"]["dates"]
for t, d in listing.items():
    if t in pred.columns:
        pred.loc[pred.index < pd.Timestamp(d), t] = np.nan
        ret.loc[ret.index <= pd.Timestamp(d), t] = np.nan
start = pd.Timestamp("2019-01-29")
reb_dates = sorted(res.portfolio_weights)
cum = (1 + ret.fillna(0)).cumprod()


def fwd(h):
    return cum.shift(-h) / cum - 1


def ic_on(dates_, p, f, min_n=30):
    out = []
    for d in dates_:
        a, b = p.loc[d], f.loc[d]
        m = a.notna() & b.notna()
        if m.sum() > min_n:
            out.append(spearmanr(a[m], b[m])[0])
        else:
            out.append(np.nan)
    return pd.Series(out, index=dates_)


samp_all = None
print("=== 1. IC decay (Spearman, rebalance dates, h-day forward USD return from t+1) ===")
samp = [d for d in reb_dates if d >= start]
row = {}
for h in (5, 10, 21, 42, 63, 126):
    s = ic_on(samp, pred, fwd(h).shift(-1))
    row[h] = (round(s.mean(), 4), round(s.mean() / s.std() * np.sqrt(len(s)), 2))
print("h: (mean IC, t-stat) ->", row)
print("IC vs training label (targets, PCA-residual 20d):", round(ic_on(samp, pred, res.targets.reindex(dates)).mean(), 4),
      "| stored ic_series mean", round(float(res.ic_series.mean()), 4))

print("\n=== 2. Within-sector vs between-sector IC (h=21) ===")
f21 = fwd(21).shift(-1)
def demean_by_sector(df):
    return df.sub(df.T.groupby(sector).transform("mean").T)
within = ic_on(samp, demean_by_sector(pred), demean_by_sector(f21))
sec_p = pred.T.groupby(sector).mean().T; sec_f = f21.T.groupby(sector).mean().T
between = ic_on(samp, sec_p, sec_f, min_n=6)
print("within-sector IC", round(within.mean(), 4), "t", round(within.mean() / within.std() * np.sqrt(len(within)), 2),
      "| sector-level IC (11 sectors)", round(between.mean(), 4), "t", round(between.mean() / between.std() * np.sqrt(len(between)), 2))

print("\n=== 3. Decile forward 21d USD return (cross-sectional, avg over rebalance dates) ===")
dec = []
for d in samp:
    a, b = pred.loc[d], f21.loc[d]; m = a.notna() & b.notna()
    if m.sum() < 50:
        continue
    q = pd.qcut(a[m].rank(method="first"), 10, labels=False)
    dec.append(b[m].groupby(q).mean())
dec = pd.DataFrame(dec)
print("mean by decile (0=low .. 9=high):", (dec.mean() * 100).round(2).tolist(), "| top-bottom spread", round((dec[9] - dec[0]).mean() * 100, 2), "%")
print("long-side excess (D9 - mean) ", round((dec[9] - dec.mean(axis=1)).mean() * 100, 2), "% | short-side excess (mean - D0)", round((dec.mean(axis=1) - dec[0]).mean() * 100, 2), "%")

print("\n=== 4. Active return by side and sector (daily weights vs cap-weighted bm from CUR_MKT_CAP) ===")
W = pd.DataFrame(res.daily_weights).T.sort_index().reindex(columns=px.columns).fillna(0.0)
bm = mc.shift(1).reindex(W.index); bm = bm.where(W.index.to_series().notna(), bm)
bm = bm.div(bm.sum(axis=1), axis=0).fillna(0.0)
act = W - bm
r = ret.reindex(W.index).fillna(0.0)
contrib = act * r
recon = contrib.sum(axis=1)
true_act = (res.portfolio_returns - res.benchmark_returns).reindex(W.index)
print("reconstruction check: corr(recon active, true active) =", round(float(np.corrcoef(recon.dropna(), true_act.loc[recon.dropna().index])[0, 1]), 3),
      "| ann active recon", round(float(recon.mean() * 252), 4), "true", round(float(true_act.mean() * 252), 4))
ow = (act.clip(lower=0) * r).sum(axis=1); uw = (act.clip(upper=0) * r).sum(axis=1)
print("ann contribution: OW side", round(float(ow.mean() * 252), 4), "| UW side", round(float(uw.mean() * 252), 4))
by_sec = contrib.T.groupby(sector).sum().T
print("ann contribution by sector:", (by_sec.mean() * 252).round(4).sort_values(ascending=False).to_dict())
yrs = contrib.groupby(contrib.index.year).sum()
print("by year (recon active):", yrs.sum(axis=1).round(3).to_dict())
tech_share = (act.T.groupby(sector).sum().T["Technology"]).describe()
print("Technology active weight: mean", round(float(tech_share["mean"]), 3), "max", round(float(tech_share["max"]), 3))

print("\n=== 5. Yearly IR / IC / turnover ===")
pr, br = res.portfolio_returns, res.benchmark_returns
for y in sorted(set(pr.index.year)):
    a = (pr - br)[pr.index.year == y]
    ics = [v for d, v in res.ic_series.items() if d.year == y]
    tos = [v for d, v in res.turnover.items() if d.year == y]
    print(y, "IR", round(float(a.mean() / a.std() * np.sqrt(252)), 2), "active", round(float(a.mean() * 252), 3), "TE", round(float(a.std() * np.sqrt(252)), 3),
          "| mean IC", round(float(np.mean(ics)), 4) if ics else None, "| two-way turnover/rebal", round(float(np.mean(tos)), 3) if tos else None)

print("\n=== 6. Signal persistence: rank autocorrelation of predictions ===")
for h in (5, 21, 42, 63):
    ac = ic_on(samp, pred, pred.shift(-h))
    print(f"lag {h}: {round(ac.mean(), 3)}", end=" | ")
print()

print("\n=== 7. Model staleness: IC when the live model was a reused (degenerate) fit ===")
events = {pd.Timestamp(e["date"]) for e in res.model_quality.get("events", [])}
audit = res.model_quality.get("split_audit", [])
fit_dates = sorted(res.models)
def model_for(d):
    prior = [f for f in fit_dates if f <= d]
    return prior[-1] if prior else None
ic21 = ic_on(samp, pred, f21)
flag = pd.Series([model_for(d) in events for d in samp], index=samp)
print("rebalances on degenerate-reuse models:", int(flag.sum()), "/", len(flag), "| mean IC reuse", round(float(ic21[flag].mean()), 4), "vs fresh", round(float(ic21[~flag].mean()), 4))
ages = pd.Series([(d - model_for(d)).days for d in samp], index=samp)
print("model age (days) at rebalance: median", float(ages.median()), "max", float(ages.max()))
