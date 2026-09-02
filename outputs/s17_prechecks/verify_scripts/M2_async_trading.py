"""M2 verify — asynchronous-trading lead (ASIA beta/idio_vol/Sigma distortion).

Refutation-first checks 1..4 from the task brief. Reads only:
  * scratchpad cache m2_sheets.pkl (built by M2_extract.py from the workbook + Index.xlsx)
  * outputs/s16_7_name_risk_cap/backtest_result.pkl (certified S0' run)
Writes only into the scratchpad verify dir.

Run from ai_port with PYTHONPATH=. (single foreground process).
"""
import os, sys, time, pickle, json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())
from src.config import PipelineConfig

VD = r"C:\Users\westl\AppData\Local\Temp\claude\C--Users-westl-PycharmProjects-pythonProject-venv-vf-new-machine-re-study-c2-ai-port\0caf960f-9660-4a4b-a91e-809a2f4ab53d\scratchpad\w3\verify"
PKL = r"C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port\outputs\s16_7_name_risk_cap\backtest_result.pkl"
OUT = {}
def log(*a):
    print(*a, flush=True)

# ----------------------------------------------------------------------------
# 0. data
# ----------------------------------------------------------------------------
with open(os.path.join(VD, "m2_sheets.pkl"), "rb") as f:
    S = pickle.load(f)
ret_local = S["Daily_Returns"].copy()
px_local = S["PX_LAST"].copy()
mcap = S["CUR_MKT_CAP"].copy()
meta = S["Universe_Meta"].set_index("ticker")
idx = S["Index_PX_LAST_subset"]
cfg = PipelineConfig()
# pipeline calendar = weekday rows only (data_loader.py:728-729 drops Sat/Sun; US holidays stay as rows)
for _k in ("Daily_Returns", "PX_LAST", "CUR_MKT_CAP"):
    S[_k] = S[_k][S[_k].index.dayofweek < 5]
ret_local = S["Daily_Returns"].copy(); px_local = S["PX_LAST"].copy(); mcap = S["CUR_MKT_CAP"].copy()
log("weekday rows:", len(ret_local), ret_local.index.min().date(), ret_local.index.max().date())

tickers = list(ret_local.columns)
cur = meta["currency"].reindex(tickers)
region = pd.Series(np.where(cur == "USD", "US", np.where(cur.isin(["JPY", "KRW"]), "ASIA", "EUROPE")), index=tickers)
US = [t for t in tickers if region[t] == "US"]
ASIA = [t for t in tickers if region[t] == "ASIA"]
EU = [t for t in tickers if region[t] == "EUROPE"]
JP = [t for t in tickers if cur[t] == "JPY"]
log("regions:", {k: len(v) for k, v in [("US", US), ("ASIA", ASIA), ("EUROPE", EU)]}, "JP", len(JP))

# listing mask (config registry + auto-infer from PX_LAST first valid)
first_px = px_local.apply(lambda s: s.first_valid_index())
mask = pd.DataFrame(True, index=ret_local.index, columns=tickers)
for t in tickers:
    d0 = first_px[t]
    if t in cfg.listing_dates:
        d0 = max(pd.Timestamp(cfg.listing_dates[t]), d0) if d0 is not None else pd.Timestamp(cfg.listing_dates[t])
    if d0 is not None:
        mask.loc[mask.index <= d0, t] = False  # inclusive, mirrors Daily_Returns listing-day exclusion
ret_local = ret_local.where(mask)
log("return magnitude check (median abs daily, AAPL):", float(ret_local["AAPL"].abs().median()))

# FX -> USD per local (Index.xlsx), ffill, fx returns
fxmap = {"JPY": ("USDJPY Curncy", "inv"), "KRW": ("USDKRW Curncy", "inv"), "EUR": ("EURUSD Curncy", "dir"),
         "CHF": ("USDCHF Curncy", "inv"), "GBP": ("GBPUSD Curncy", "dir"), "DKK": ("USDDKK Curncy", "inv")}
fx_lvl = pd.DataFrame(index=ret_local.index)
for c, (col, d) in fxmap.items():
    s = idx[col].where(idx[col] > 0)
    s = (1.0 / s) if d == "inv" else s
    fx_lvl[c] = s.reindex(ret_local.index.union(s.index)).ffill().reindex(ret_local.index)
fx_lvl["USD"] = 1.0
fx_ret_t = pd.DataFrame({t: fx_lvl[cur[t]].pct_change().fillna(0.0) for t in tickers})
ret_usd = (1.0 + ret_local) * (1.0 + fx_ret_t) - 1.0
ret_usd = ret_usd.where(mask)

# EW market proxies
us_ew = ret_local[US].mean(axis=1)             # US names: local == USD
all_ew_local = ret_local.mean(axis=1)
all_ew_usd = ret_usd.mean(axis=1)

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def corr_beta(y: pd.Series, x: pd.Series):
    d = pd.concat([y, x], axis=1).dropna()
    if len(d) < 50:
        return np.nan, np.nan, len(d)
    c = d.iloc[:, 0].corr(d.iloc[:, 1])
    b = d.iloc[:, 0].cov(d.iloc[:, 1]) / d.iloc[:, 1].var()
    return c, b, len(d)

def per_ticker_stats(R: pd.DataFrame, m: pd.Series, lags=(0, 1, -1)):
    rows = {}
    for t in R.columns:
        y = R[t]
        r = {}
        for k in lags:
            c, b, n = corr_beta(y, m.shift(k))
            r[f"corr_lag{k}"] = c
            r[f"beta_lag{k}"] = b
        r["n"] = n
        r["dimson_beta"] = sum(r.get(f"beta_lag{k}", 0.0) for k in (0, 1, -1))
        r["dimson_ratio"] = r["dimson_beta"] / r["beta_lag0"] if r["beta_lag0"] else np.nan
        rows[t] = r
    return pd.DataFrame(rows).T

def region_median(df, cols):
    return df.groupby(region.reindex(df.index))[cols].median()

# ----------------------------------------------------------------------------
# 1. reproduce daily async stats vs US EW (local returns; sample 2014-2026)
# ----------------------------------------------------------------------------
log("\n=== 1. daily stats vs US EW (191), local currency, listing-masked ===")
d1 = per_ticker_stats(ret_local, us_ew)
d1["region"] = region.reindex(d1.index)
rm1 = region_median(d1, ["corr_lag0", "corr_lag1", "corr_lag-1", "beta_lag0", "beta_lag1", "dimson_beta", "dimson_ratio"])
log(rm1.round(3).to_string())
OUT["step1_daily_region_median_local"] = rm1.round(4).to_dict()
# same in USD terms
d1u = per_ticker_stats(ret_usd, us_ew)
rm1u = region_median(d1u, ["corr_lag0", "corr_lag1", "beta_lag0", "beta_lag1", "dimson_ratio"])
log("USD-converted:\n" + rm1u.round(3).to_string())
OUT["step1_daily_region_median_usd"] = rm1u.round(4).to_dict()
# compare with pack
pack = pd.read_csv(os.path.join(VD, "..", "pack", "async_probe.csv")).set_index("ticker")
cmp = pd.concat([d1[["corr_lag0", "corr_lag1", "beta_lag0", "beta_lag1", "dimson_ratio"]],
                 pack[["corr_lag0_us", "corr_stock_t_vs_us_tm1", "beta0_us", "beta_lag1_us", "dimson_ratio"]].add_prefix("pack_")], axis=1)
log("pack vs mine (ASIA):\n" + cmp.loc[ASIA].round(3).to_string())
# region EW
for nm, grp in [("ASIA", ASIA), ("EUROPE", EU)]:
    g = ret_local[grp].mean(axis=1)
    c0, b0, _ = corr_beta(g, us_ew)
    c1, b1, _ = corr_beta(g, us_ew.shift(1))
    cm1, bm1, _ = corr_beta(g, us_ew.shift(-1))
    log(f"{nm} EW vs US EW: corr same-day {c0:.3f} | vs US_(t-1) {c1:.3f} | vs US_(t+1) {cm1:.3f}; beta {b0:.3f}/{b1:.3f}/{bm1:.3f}")
    OUT[f"step1_{nm}_EW_corr_same"] = round(c0, 4); OUT[f"step1_{nm}_EW_corr_us_lag1"] = round(c1, 4)

# ----------------------------------------------------------------------------
# 2. alternative hypothesis: genuinely low beta? weekly / monthly sums
# ----------------------------------------------------------------------------
log("\n=== 2. weekly (W-FRI non-overlapping 5d sums) and 21d sums ===")
def agg(R, rule):
    return R.resample(rule).apply(lambda s: s.sum(min_count=3) if s.notna().sum() >= 3 else np.nan)
def nonoverlap_sum(R, n):
    g = np.arange(len(R)) // n
    out = R.groupby(g).sum(min_count=n)
    out.index = R.index[(np.arange(len(out)) * n + n - 1).clip(max=len(R) - 1)]
    return out

wk_local = ret_local.resample("W-FRI").sum(min_count=3)
wk_us = us_ew.resample("W-FRI").sum(min_count=3)
mo_local = nonoverlap_sum(ret_local, 21)
mo_us = nonoverlap_sum(us_ew.to_frame("us"), 21)["us"]
wk = per_ticker_stats(wk_local, wk_us, lags=(0, 1))
mo = per_ticker_stats(mo_local, mo_us, lags=(0,))
tab = pd.DataFrame({
    "corr_daily": d1["corr_lag0"], "corr_daily_uslag1": d1["corr_lag1"],
    "corr_weekly": wk["corr_lag0"], "corr_weekly_uslag1": wk["corr_lag1"], "corr_21d": mo["corr_lag0"],
    "beta_daily": d1["beta_lag0"], "beta_dimson_daily": d1["dimson_beta"], "beta_weekly": wk["beta_lag0"], "beta_21d": mo["beta_lag0"],
})
tab["corr_ratio_wk_over_daily"] = tab["corr_weekly"] / tab["corr_daily"]
tab["corr_ratio_21d_over_daily"] = tab["corr_21d"] / tab["corr_daily"]
tab["beta_ratio_wk_over_daily"] = tab["beta_weekly"] / tab["beta_daily"]
tab["region"] = region.reindex(tab.index)
rm2 = tab.groupby("region").median(numeric_only=True)
log(rm2.round(3).T.to_string())
OUT["step2_region_median"] = rm2.round(4).to_dict()
log("ASIA per-name:\n" + tab.loc[ASIA].round(3).to_string())
tab.to_csv(os.path.join(VD, "M2_step2_weekly_vs_daily.csv"))
# US-vs-ASIA relative beta: is ASIA beta 'low' at weekly horizon relative to US names?
log("ASIA weekly beta / US weekly beta (medians): %.3f / %.3f ; daily: %.3f / %.3f" % (
    rm2.loc["ASIA", "beta_weekly"], rm2.loc["US", "beta_weekly"], rm2.loc["ASIA", "beta_daily"], rm2.loc["US", "beta_daily"]))

# ----------------------------------------------------------------------------
# 2b. W1 T-08 stamp discrimination: JP cross-correlogram, NKY reference
# ----------------------------------------------------------------------------
log("\n=== 2b. JP stamp test: corr(stock_t, US_EW_{t-k}), k=0..3 ===")
xc = {}
for t in JP + ["000660", "005930"]:
    xc[t] = {f"k{k}": corr_beta(ret_local[t], us_ew.shift(k))[0] for k in range(0, 4)}
xc = pd.DataFrame(xc).T
xc["peak_lag"] = xc.idxmax(axis=1)
log(xc.round(3).to_string())
OUT["step2b_JP_peak_lag_counts"] = xc.loc[JP, "peak_lag"].value_counts().to_dict()
OUT["step2b_JP_xcorr_median"] = xc.loc[JP, ["k0", "k1", "k2", "k3"]].median().round(4).to_dict()

nky = idx["NKY Index"].where(idx["NKY Index"] > 0)
nky_ret = nky.pct_change()
nky_ret = nky_ret[nky.diff() != 0]  # drop stale/holiday repeats
spx = idx["SPX Index"].where(idx["SPX Index"] > 0)
spx_ret = spx.pct_change(); spx_ret = spx_ret[spx.diff() != 0]
log("Index.xlsx NKY vs US EW (workbook): corr NKY_t vs US_{t-k}:",
    {k: round(corr_beta(nky_ret, us_ew.shift(k))[0], 3) for k in range(0, 3)})
log("Index.xlsx SPX vs US EW (workbook) same-day corr: %.3f (stamp alignment sanity of Index.xlsx vs workbook US rows)" % corr_beta(spx_ret, us_ew)[0])
jp_ew = ret_local[JP].mean(axis=1)
nk = {}
for t in ["7203", "8306", "6758", "8035", "6857", "4063"]:
    nk[t] = {f"NKY_lag{k}": round(corr_beta(ret_local[t], nky_ret.shift(k))[0], 3) for k in (-1, 0, 1, 2)}
nk["JP_EW"] = {f"NKY_lag{k}": round(corr_beta(jp_ew, nky_ret.shift(k))[0], 3) for k in (-1, 0, 1, 2)}
nk = pd.DataFrame(nk).T
log("corr(stock_t, NKY_{t-k}):\n" + nk.to_string())
OUT["step2b_7203_vs_NKY"] = nk.loc["7203"].to_dict()
OUT["step2b_JPEW_vs_NKY"] = nk.loc["JP_EW"].to_dict()

log("PX_LAST 7203 2013-12-27..2014-01-10:")
log(px_local.loc["2013-12-27":"2014-01-10", ["7203", "8306", "6758"]].to_string())
log("Index.xlsx NKY 2013-12-27..2014-01-10:")
log(nky.loc["2013-12-27":"2014-01-10"].to_string())
log("2024-08-02..08-07 returns: JP names vs NKY:")
w = pd.concat([S["Daily_Returns"].set_index(S["Daily_Returns"].index)[["7203", "8306", "6758", "8035", "005930"]].loc["2024-08-01":"2024-08-08"],
               nky_ret.rename("NKY").loc["2024-08-01":"2024-08-08"], us_ew.rename("US_EW").loc["2024-08-01":"2024-08-08"]], axis=1)
log(w.round(4).to_string())
OUT["step2b_20240805_7203"] = float(S["Daily_Returns"].loc["2024-08-05", "7203"])
OUT["step2b_20240806_7203"] = float(S["Daily_Returns"].loc["2024-08-06", "7203"])
OUT["step2b_20240805_NKY"] = float(nky_ret.get(pd.Timestamp("2024-08-05"), np.nan))
OUT["step2b_20240806_NKY"] = float(nky_ret.get(pd.Timestamp("2024-08-06"), np.nan))

# ----------------------------------------------------------------------------
# 3(i). price.py-style beta_63d / idio_vol_63d vs Dimson-corrected
# ----------------------------------------------------------------------------
log("\n=== 3(i). price.py replication (EW market of all 250, USD returns) vs Dimson ===")
R = ret_usd
m = all_ew_usd
W = 63
def price_py(R, m):
    xy = R.mul(m, axis=0)
    e_xy = xy.rolling(W, min_periods=W).mean()
    e_x = R.rolling(W, min_periods=W).mean()
    e_y = m.rolling(W, min_periods=W).mean()
    cov_xy = e_xy - e_x.mul(e_y, axis=0)
    var_y = m.rolling(W, min_periods=W).var().replace(0, np.nan)
    beta = cov_xy.div(var_y, axis=0)
    resid = R - beta.mul(m, axis=0)
    idio = resid.rolling(W, min_periods=W).std() * np.sqrt(252)
    return beta, idio

def rolling_ols_multi(R, X, W):
    """Rolling OLS y_t = a + sum_k b_k x_kt over trailing W, batched via normal equations.
    R: T x N, X: list of T-series. Returns list of beta DataFrames and residual (no-intercept) panel."""
    T, N = R.shape
    K = len(X)
    Xm = np.column_stack([x.to_numpy() for x in X])            # T x K
    valid = R.notna().to_numpy() & np.isfinite(Xm).all(axis=1)[:, None]
    Y = np.where(valid, R.to_numpy(), 0.0)
    P = K + 1
    # design with intercept
    D = np.concatenate([np.ones((T, 1)), Xm], axis=1)           # T x P
    D = np.where(np.isfinite(D), D, 0.0)
    def rs(a):  # rolling sum along time for 1d/2d arrays
        return pd.DataFrame(a).rolling(W, min_periods=W).sum().to_numpy()
    cnt = rs(valid.astype(float))                               # T x N
    XtX = np.empty((T, N, P, P))
    for i in range(P):
        for j in range(i, P):
            v = rs(valid * (D[:, i] * D[:, j])[:, None])
            XtX[:, :, i, j] = v; XtX[:, :, j, i] = v
    Xty = np.empty((T, N, P))
    for i in range(P):
        Xty[:, :, i] = rs(Y * D[:, i][:, None])
    ok = cnt >= W
    B = np.full((T, N, P), np.nan)
    sel = np.where(ok)
    try:
        B[sel] = np.linalg.solve(XtX[sel], Xty[sel][..., None])[..., 0]
    except np.linalg.LinAlgError:
        for a, b in zip(*sel):
            try:
                B[a, b] = np.linalg.solve(XtX[a, b], Xty[a, b])
            except np.linalg.LinAlgError:
                pass
    betas = [pd.DataFrame(B[:, :, k + 1], index=R.index, columns=R.columns) for k in range(K)]
    fitted = sum(b.mul(x, axis=0) for b, x in zip(betas, X))
    resid = R - fitted
    return betas, resid

t0 = time.time()
beta_s, idio_s = price_py(R, m)
b_lagonly, resid_lag = rolling_ols_multi(R, [m, m.shift(1)], W)
idio_lag = resid_lag.rolling(W, min_periods=W).std() * np.sqrt(252)
b_full, resid_full = rolling_ols_multi(R, [m, m.shift(1), m.shift(-1)], W)
idio_full = resid_full.rolling(W, min_periods=W).std() * np.sqrt(252)
log("rolling regressions done in %.0fs" % (time.time() - t0))
tot_vol = R.rolling(W, min_periods=W).std() * np.sqrt(252)

# load pkl
t0 = time.time()
with open(PKL, "rb") as f:
    res = pickle.load(f)
log("pkl loaded in %.0fs" % (time.time() - t0))
panel = res.panel
pred = res.predictions
# model gain share of beta_63d / idio_vol_63d across the 33 rankers
gs = []
for k, mdl in res.models.items():
    try:
        feats = list(getattr(mdl, "_active_features", None) or res.feature_names)
        g = np.asarray(mdl.booster_.feature_importance(importance_type="gain"), dtype=float)
        g = g / g.sum() if g.sum() > 0 else g
        d = dict(zip(feats, g))
        order = pd.Series(d).rank(ascending=False)
        gs.append({"model": str(k), "beta_63d_share": d.get("beta_63d", np.nan), "beta_63d_rank": order.get("beta_63d", np.nan),
                   "idio_vol_63d_share": d.get("idio_vol_63d", np.nan), "idio_vol_63d_rank": order.get("idio_vol_63d", np.nan)})
    except Exception as e:
        gs.append({"model": str(k), "err": str(e)})
gs = pd.DataFrame(gs)
log("model gain shares (33 models):" + chr(10) + gs.describe().round(4).T.to_string())
OUT["model_gain_share_median"] = {"beta_63d": float(gs["beta_63d_share"].median()), "idio_vol_63d": float(gs["idio_vol_63d_share"].median()),
                                  "beta_63d_rank_median": float(gs["beta_63d_rank"].median()), "idio_vol_63d_rank_median": float(gs["idio_vol_63d_rank"].median())}
dates_bt = pred.index
panel_idio = panel["idio_vol_63d"].unstack("ticker").reindex(columns=tickers)
panel_beta = panel["beta_63d"].unstack("ticker").reindex(columns=tickers)
panel_rvol = panel["realized_vol_63d"].unstack("ticker").reindex(columns=tickers) if "realized_vol_63d" in panel.columns else None

# replication check vs pkl panel (same dates)
common = panel_idio.index.intersection(idio_s.index)
rep = pd.DataFrame({
    "corr_idio": [idio_s.loc[common].corrwith(panel_idio.loc[common]).median()],
    "med_rel_absdiff_idio": [((idio_s.loc[common] - panel_idio.loc[common]).abs() / panel_idio.loc[common].abs()).stack().median()],
    "corr_beta": [beta_s.loc[common].corrwith(panel_beta.loc[common]).median()],
    "med_absdiff_beta": [(beta_s.loc[common] - panel_beta.loc[common]).abs().stack().median()],
})
log("replication vs pkl panel (my USD approx vs pipeline):\n" + rep.round(4).to_string())
OUT["step3i_replication"] = rep.round(4).iloc[0].to_dict()

sub = slice(dates_bt[0], dates_bt[-1])
def med_by_region(df):
    return df.loc[sub].median().groupby(region).median()
ratio_lag = (idio_s / idio_lag).loc[sub]
ratio_full = (idio_s / idio_full).loc[sub]
beta_ratio_lag = (beta_s / (b_lagonly[0] + b_lagonly[1])).loc[sub]
beta_ratio_full = (beta_s / (b_full[0] + b_full[1] + b_full[2])).loc[sub]
t3 = pd.DataFrame({
    "tot_vol": tot_vol.loc[sub].median(), "idio_simple": idio_s.loc[sub].median(),
    "idio_dimson_lag": idio_lag.loc[sub].median(), "idio_dimson_full": idio_full.loc[sub].median(),
    "idio_over_dimson_lag": ratio_lag.median(), "idio_over_dimson_full": ratio_full.median(),
    "beta_simple": beta_s.loc[sub].median(), "beta_dimson_lag": (b_lagonly[0] + b_lagonly[1]).loc[sub].median(),
    "beta_dimson_full": (b_full[0] + b_full[1] + b_full[2]).loc[sub].median(),
    "beta_simple_over_dimson_lag": beta_ratio_lag.median(), "beta_simple_over_dimson_full": beta_ratio_full.median(),
    "panel_idio": panel_idio.median(), "panel_beta": panel_beta.median(),
})
t3["region"] = region
log("3(i) region medians (backtest period):\n" + t3.groupby("region").median(numeric_only=True).round(3).T.to_string())
log("ASIA per name:\n" + t3.loc[ASIA].round(3).to_string())
OUT["step3i_region_median"] = t3.groupby("region").median(numeric_only=True).round(4).to_dict()
t3.to_csv(os.path.join(VD, "M2_step3i_idio_beta.csv"))
# fraction of ASIA names whose idio overstatement > 5% / 10%
OUT["step3i_ASIA_idio_overstate_gt5pct_frac"] = float((t3.loc[ASIA, "idio_over_dimson_full"] > 1.05).mean())
OUT["step3i_ASIA_idio_overstate_gt10pct_frac"] = float((t3.loc[ASIA, "idio_over_dimson_full"] > 1.10).mean())

# ----------------------------------------------------------------------------
# 3(ii). top idio-vol tercile membership frequency by region (pkl panel), and under corrected idio
# ----------------------------------------------------------------------------
log("\n=== 3(ii). top-tercile (idio_vol_63d) membership frequency among scored names ===")
def tercile_freq(vol_panel, pred):
    hits = {t: 0 for t in tickers}; n_dates = 0; asia_count = []
    for dt in pred.index:
        s = pred.loc[dt]; scored = s.index[s.notna()]
        if len(scored) < 30 or dt not in vol_panel.index:
            continue
        v = vol_panel.loc[dt].reindex(scored).dropna()
        if len(v) < 30:
            continue
        top = v.sort_values(ascending=False).head(len(v) // 3).index
        n_dates += 1
        for t in top:
            hits[t] += 1
        asia_count.append(sum(1 for t in top if t in ASIA))
    f = pd.Series(hits) / max(n_dates, 1)
    return f, n_dates, float(np.mean(asia_count)) if asia_count else np.nan
f_pk, nd, asia_top_mean = tercile_freq(panel_idio, pred)
f_s, _, asia_top_s = tercile_freq(idio_s, pred)
f_lag, _, asia_top_lag = tercile_freq(idio_lag, pred)
f_full, _, asia_top_full = tercile_freq(idio_full, pred)
f_tv, _, asia_top_tv = tercile_freq(tot_vol, pred)
f_rv, _, asia_top_rv = tercile_freq(panel_rvol, pred) if panel_rvol is not None else (None, None, np.nan)
tt = pd.DataFrame({"pkl_idio": f_pk, "my_idio_simple": f_s, "my_idio_dimson_lag": f_lag, "my_idio_dimson_full": f_full,
                   "my_total_vol": f_tv, "pkl_realized_vol_63d": f_rv})
tt["region"] = region
log(f"dates used {nd}; mean #ASIA names in top tercile: pkl {asia_top_mean:.2f} | simple {asia_top_s:.2f} | dimson_lag {asia_top_lag:.2f} | dimson_full {asia_top_full:.2f} | total_vol {asia_top_tv:.2f} | pkl realized_vol {asia_top_rv:.2f}")
log(tt.groupby("region").mean(numeric_only=True).round(3).to_string())
log("ASIA per name:\n" + tt.loc[ASIA].round(3).to_string())
OUT["step3ii_top_tercile_freq_region_mean"] = tt.groupby("region").mean(numeric_only=True).round(4).to_dict()
OUT["step3ii_mean_asia_in_top_tercile"] = {"pkl": asia_top_mean, "simple": asia_top_s, "dimson_lag": asia_top_lag, "dimson_full": asia_top_full, "total_vol": asia_top_tv, "pkl_realized_vol": asia_top_rv}
tt.to_csv(os.path.join(VD, "M2_step3ii_tercile.csv"))
# does the quality tilt bite on ASIA? predictions vs pre_overlay on ASIA names
pre = getattr(res, "pre_overlay_predictions", None)
if pre is not None:
    dlt = (pred - pre.reindex_like(pred))
    log("tilt delta (pred - pre_overlay) mean abs by region: " + str(dlt.abs().mean().groupby(region).mean().round(5).to_dict()))
    log("tilt delta sign mean by region: " + str(dlt.mean().groupby(region).mean().round(5).to_dict()))
    OUT["step3ii_tilt_delta_mean_by_region"] = dlt.mean().groupby(region).mean().round(6).to_dict()

# ----------------------------------------------------------------------------
# 3(iii). Sigma at recent rebalance dates: ASIA x US block correlation daily vs 5d-overlap
# ----------------------------------------------------------------------------
log("\n=== 3(iii). 126d Sigma blocks at the last 5 rebalance dates (USD approx of raw_returns) ===")
rebal_dates = sorted(res.portfolio_weights.keys())
last5 = rebal_dates[-5:]
def block_mean(C, A, B):
    return float(np.nanmean(C.loc[A, B].to_numpy()))
rows = []
for dt in last5:
    pos = ret_usd.index.get_loc(dt)
    win = ret_usd.iloc[max(0, pos - 126):pos]
    C1 = win.corr(min_periods=30)
    ov = win.rolling(5).sum().dropna(how="all")
    C5 = ov.corr(min_periods=30)
    win252 = ret_usd.iloc[max(0, pos - 252):pos]
    wk = win252.resample("W-FRI").sum(min_count=3)
    Cw = wk.corr(min_periods=20)
    cov1 = win.cov(min_periods=30); cov5 = ov.cov(min_periods=30) / 5.0
    rows.append({"date": dt.strftime("%Y-%m-%d"),
                 "corr_ASIAxUS_daily": block_mean(C1, ASIA, US), "corr_ASIAxUS_5dov": block_mean(C5, ASIA, US), "corr_ASIAxUS_weekly252": block_mean(Cw, ASIA, US),
                 "corr_EUxUS_daily": block_mean(C1, EU, US), "corr_EUxUS_5dov": block_mean(C5, EU, US),
                 "corr_USxUS_daily": block_mean(C1, US, US), "corr_USxUS_5dov": block_mean(C5, US, US),
                 "corr_ASIAxASIA_daily": block_mean(C1, ASIA, ASIA), "corr_ASIAxASIA_5dov": block_mean(C5, ASIA, ASIA),
                 "cov_ASIAxUS_daily": block_mean(cov1, ASIA, US), "cov_ASIAxUS_5dov_scaled": block_mean(cov5, ASIA, US),
                 "var_ASIA_daily": float(np.nanmean(np.diag(cov1.loc[ASIA, ASIA]))), "var_ASIA_5dov_scaled": float(np.nanmean(np.diag(cov5.loc[ASIA, ASIA]))),
                 "var_US_daily": float(np.nanmean(np.diag(cov1.loc[US, US]))), "var_US_5dov_scaled": float(np.nanmean(np.diag(cov5.loc[US, US])))})
s3 = pd.DataFrame(rows).set_index("date")
s3["ratio_ASIAxUS_5dov_over_daily"] = s3["corr_ASIAxUS_5dov"] / s3["corr_ASIAxUS_daily"]
s3["ratio_EUxUS"] = s3["corr_EUxUS_5dov"] / s3["corr_EUxUS_daily"]
s3["ratio_USxUS"] = s3["corr_USxUS_5dov"] / s3["corr_USxUS_daily"]
s3["cov_ratio_ASIAxUS"] = s3["cov_ASIAxUS_5dov_scaled"] / s3["cov_ASIAxUS_daily"]
log(s3.round(3).T.to_string())
OUT["step3iii_last5_mean"] = s3.mean().round(4).to_dict()
s3.to_csv(os.path.join(VD, "M2_step3iii_sigma_blocks.csv"))

# ----------------------------------------------------------------------------
# 3(iv). ASIA active weight time series (w - bm, bm = cap-weight approx)
# ----------------------------------------------------------------------------
log("\n=== 3(iv). ASIA active weights over 97 rebalance dates ===")
mc_ff = mcap.reindex(columns=tickers).ffill()
def bm_at(dt):
    row = mc_ff.loc[:dt].iloc[-1].astype(float).copy()
    elig = mask.loc[:dt].iloc[-1] if dt in mask.index else mask.loc[:dt].iloc[-1]
    row[~elig.to_numpy()] = 0.0
    row[~np.isfinite(row) | (row <= 0)] = 0.0
    return row / row.sum()
act_rows = []
A = {}
for dt in rebal_dates:
    w = res.portfolio_weights[dt].reindex(tickers).fillna(0.0)
    bm = bm_at(dt)
    a = w - bm
    A[dt] = a
    act_rows.append({"date": dt, "asia_w": w[ASIA].sum(), "asia_bm": bm[ASIA].sum(), "asia_active": a[ASIA].sum(),
                     "asia_gross_active": a[ASIA].abs().sum(), "asia_n_ow": int((a[ASIA] > 0.002).sum()), "asia_n_uw": int((a[ASIA] < -0.002).sum()),
                     "asia_max_ow": a[ASIA].max(), "asia_max_ow_name": a[ASIA].idxmax(),
                     "eu_active": a[EU].sum(), "us_active": a[US].sum(), "total_gross_active": a.abs().sum() / 2})
act = pd.DataFrame(act_rows).set_index("date")
log(act.describe().round(4).T.to_string())
log("sign of ASIA net active: pos %d / neg %d / |a|<0.1%% %d" % ((act["asia_active"] > 0.001).sum(), (act["asia_active"] < -0.001).sum(), (act["asia_active"].abs() <= 0.001).sum()))
log("yearly mean ASIA net active:\n" + act["asia_active"].groupby(act.index.year).mean().round(4).to_string())
log("last 8 rebalance dates:\n" + act.tail(8).round(4).to_string())
OUT["step3iv_asia_active_mean"] = float(act["asia_active"].mean()); OUT["step3iv_asia_active_median"] = float(act["asia_active"].median())
OUT["step3iv_asia_active_min"] = float(act["asia_active"].min()); OUT["step3iv_asia_active_max"] = float(act["asia_active"].max())
OUT["step3iv_asia_gross_active_mean"] = float(act["asia_gross_active"].mean())
OUT["step3iv_asia_bm_mean"] = float(act["asia_bm"].mean()); OUT["step3iv_asia_w_mean"] = float(act["asia_w"].mean())
OUT["step3iv_sign_counts"] = {"pos": int((act["asia_active"] > 0.001).sum()), "neg": int((act["asia_active"] < -0.001).sum())}
act.to_csv(os.path.join(VD, "M2_step3iv_asia_active.csv"))
# average active by ASIA name
per_name = pd.DataFrame(A).T[ASIA].mean().sort_values()
log("mean active weight per ASIA name:\n" + per_name.round(4).to_string())
OUT["step3iv_per_name_mean_active"] = per_name.round(5).to_dict()

# ----------------------------------------------------------------------------
# 4. TE contribution: ex-ante (daily Sigma vs 5d-overlap Sigma) vs realized (daily vs weekly)
# ----------------------------------------------------------------------------
log("\n=== 4. TE: ex-ante ASIA contribution under daily vs 5d-overlap Sigma; realized daily vs weekly ===")
rows = []
for dt in rebal_dates:
    pos = ret_usd.index.get_loc(dt)
    win = ret_usd.iloc[max(0, pos - 126):pos]
    if len(win) < 60:
        continue
    a = A[dt].to_numpy()
    cov1 = win.cov(min_periods=30).reindex(index=tickers, columns=tickers).fillna(0.0).to_numpy()
    ov = win.rolling(5).sum().dropna(how="all")
    cov5 = (ov.cov(min_periods=30) / 5.0).reindex(index=tickers, columns=tickers).fillna(0.0).to_numpy()
    ia = np.array([t in ASIA for t in tickers])
    def te_parts(cv):
        v = a @ cv @ a
        contrib = a * (cv @ a)
        return np.sqrt(max(v, 0) * 252), contrib[ia].sum() / v if v > 0 else np.nan
    te1, c1 = te_parts(cov1); te5, c5 = te_parts(cov5)
    rows.append({"date": dt, "te_daily": te1, "te_5dov": te5, "asia_share_daily": c1, "asia_share_5dov": c5})
te = pd.DataFrame(rows).set_index("date")
log(te.describe().round(4).T.to_string())
OUT["step4_exante_te_daily_mean"] = float(te["te_daily"].mean()); OUT["step4_exante_te_5dov_mean"] = float(te["te_5dov"].mean())
OUT["step4_exante_asia_share_daily_mean"] = float(te["asia_share_daily"].mean()); OUT["step4_exante_asia_share_5dov_mean"] = float(te["asia_share_5dov"].mean())
OUT["step4_exante_te_ratio_5dov_over_daily_mean"] = float((te["te_5dov"] / te["te_daily"]).mean())

# realized: active return series from pkl, daily vs weekly vs monthly variance
ra = (res.portfolio_returns - res.benchmark_returns).dropna()
te_d = ra.std() * np.sqrt(252)
wk_ra = ra.resample("W-FRI").sum(min_count=3).dropna()
te_w = wk_ra.std() * np.sqrt(52)
mo_ra = nonoverlap_sum(ra.to_frame("a"), 21)["a"].dropna()
te_m = mo_ra.std() * np.sqrt(252 / 21)
ac1 = ra.autocorr(1)
log(f"realized TE: daily {te_d:.4f} | weekly {te_w:.4f} | 21d {te_m:.4f}; active-return AC(1) {ac1:.3f}; VR(5)={(te_w/te_d)**2:.3f}")
OUT["step4_realized_te_daily"] = round(float(te_d), 5); OUT["step4_realized_te_weekly"] = round(float(te_w), 5); OUT["step4_realized_te_21d"] = round(float(te_m), 5)
OUT["step4_active_ac1"] = round(float(ac1), 4)

# realized ASIA sleeve contribution (weights held from rebalance to next rebalance, USD returns)
sleeve = []; tot = []
for i, dt in enumerate(rebal_dates):
    nxt = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else ret_usd.index[-1]
    seg = ret_usd.loc[dt:nxt].iloc[1:]  # returns after the rebalance date up to next rebalance
    a = A[dt]
    sleeve.append((seg[ASIA].fillna(0.0) * a[ASIA]).sum(axis=1))
    tot.append((seg.fillna(0.0) * a).sum(axis=1))
sleeve = pd.concat(sleeve); tot = pd.concat(tot)
sleeve = sleeve[~sleeve.index.duplicated()]; tot = tot[~tot.index.duplicated()]
def contrib(x, y):
    return float(x.cov(y) / y.var())
c_d = contrib(sleeve, tot)
sw = sleeve.resample("W-FRI").sum(min_count=3); tw = tot.resample("W-FRI").sum(min_count=3)
c_w = contrib(sw.dropna(), tw.reindex(sw.dropna().index))
log(f"static-weight active proxy: TE daily {tot.std()*np.sqrt(252):.4f} weekly {tw.dropna().std()*np.sqrt(52):.4f}; ASIA sleeve share of active variance: daily {c_d:.4f} | weekly {c_w:.4f}; ASIA sleeve stdev ann daily {sleeve.std()*np.sqrt(252):.4f} weekly {sw.dropna().std()*np.sqrt(52):.4f}")
log("corr(proxy active, pkl active) = %.3f" % tot.corr(ra.reindex(tot.index)))
OUT["step4_realized_asia_share_daily"] = round(c_d, 5); OUT["step4_realized_asia_share_weekly"] = round(c_w, 5)
OUT["step4_asia_sleeve_te_daily"] = round(float(sleeve.std() * np.sqrt(252)), 5); OUT["step4_asia_sleeve_te_weekly"] = round(float(sw.dropna().std() * np.sqrt(52)), 5)
OUT["step4_proxy_vs_pkl_active_corr"] = round(float(tot.corr(ra.reindex(tot.index))), 4)

with open(os.path.join(VD, "M2_numbers.json"), "w") as f:
    json.dump(OUT, f, indent=1, default=float)
log("\nsaved M2_numbers.json")
