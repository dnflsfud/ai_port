"""M1 lead verification: is PX_LAST dividend(total-return)-adjusted and does
tg_upside = TG / PX_LAST - 1 therefore embed future dividends?

Stages (run with argv[1]):
  load   - read the needed workbook sheets once, cache to scratch pickle
  s1     - reproduce: 2014-06-30 PX vs nominal, Daily_Returns == pct_change
  s2     - alternatives: TG split continuity, TG/nominal flatness, x-sec corr
  s3     - quantify: PE*EPS nominal proxy, corr, IC with pkl targets
  s4     - live-time shift of panel tg_upside z (needs pkl)
  s5     - cash_conversion_z / implied shares drift (needs pkl)
Nothing is written inside the repo; all outputs go to the verify dir.
"""
import sys, os, json, pickle, time
import numpy as np
import pandas as pd

REPO = r"C:/Users/westl/PycharmProjects/pythonProject/venv_vf_new/machine/re_study/c2/ai_port"
VER = (r"C:/Users/westl/AppData/Local/Temp/claude/C--Users-westl-PycharmProjects-pythonProject-venv-vf-new-"
       r"machine-re-study-c2-ai-port/0caf960f-9660-4a4b-a91e-809a2f4ab53d/scratchpad/w3/verify")
CACHE = os.path.join(VER, "M1_sheets.pkl")
PKL = os.path.join(REPO, "outputs/s16_7_name_risk_cap/backtest_result.pkl")
sys.path.insert(0, REPO)
os.chdir(REPO)

SHEETS = ["PX_LAST", "Daily_Returns", "Factset_TG_Price", "BEST_PE_RATIO", "BEST_EPS",
          "CUR_MKT_CAP", "Universe_Meta", "BEST_CALCULATED_FCF"]
HIGH_DIV = ["MO", "T", "VZ", "PFE", "KO", "XOM", "TTE", "BNP", "RIO", "SHEL", "ENEL", "IBM"]
NO_DIV = ["TSLA", "AMZN", "GOOGL", "META", "NFLX", "NVDA", "CRM", "ADBE", "PLTR", "ISRG"]
# Known split-adjusted nominal closes 2014-06-30 (from public price history; AAPL
# further /4 for the 2020 split, NVDA /4/10 for 2021+2024 splits)
NOMINAL_20140630 = {"MO": 41.94, "T": 35.36, "VZ": 48.93, "PFE": 29.68, "XOM": 100.68,
                    "KO": 42.36, "AAPL": 92.93 / 4, "JPM": 57.62, "PG": 78.59, "MSFT": 41.70}


def _date_index(df):
    from src.data_loader import _rename_bloomberg_equity_columns
    if "date" in df.columns:
        df = df.set_index(pd.to_datetime(df["date"])).drop(columns=["date"])
    else:
        first = df.columns[0]
        df = df.set_index(pd.to_datetime(df[first])).drop(columns=[first])
    df = df.sort_index()
    df = _rename_bloomberg_equity_columns(df)
    df.columns = [str(c).strip() for c in df.columns]
    return df.apply(pd.to_numeric, errors="coerce")


def stage_load():
    from src.config import DEFAULT_CONFIG
    from src.data_loader import load_universe_meta
    path = DEFAULT_CONFIG.data_path
    t0 = time.time()
    raw = pd.read_excel(path, sheet_name=SHEETS)
    print("read_excel %.0fs" % (time.time() - t0))
    out = {}
    for k, df in raw.items():
        if k == "Universe_Meta":
            meta = load_universe_meta({"Universe_Meta": df.set_index(df.columns[0])})
            out["meta"] = meta
            continue
        print(k, "raw cols[:3]", list(df.columns[:3]), "dtype0", df.dtypes.iloc[0])
        out[k] = _date_index(df)
        print(k, out[k].shape, out[k].index.min(), out[k].index.max())
    with open(CACHE, "wb") as f:
        pickle.dump(out, f)
    print("cached", CACHE)


def load_cache():
    with open(CACHE, "rb") as f:
        return pickle.load(f)


def local_px(d):
    """PX_LAST in the unit tg_upside uses in production (§S16.1 P1: LN x0.01)."""
    px = d["PX_LAST"].copy()
    meta = d["meta"]
    for t in px.columns:
        if t in meta.index and meta.loc[t, "exchange_code"] == "LN":
            px[t] = px[t] * 0.01
    return px


def stage_s1():
    d = load_cache()
    px = d["PX_LAST"]; ret = d["Daily_Returns"]
    res = {}
    # 1a: 2014-06-30 values vs known nominal
    day = pd.Timestamp("2014-06-30")
    row = px.loc[day] if day in px.index else px.loc[:day].iloc[-1]
    tab = {}
    for t, nom in NOMINAL_20140630.items():
        v = float(row[t]) if t in row.index else np.nan
        tab[t] = {"sheet": round(v, 3), "nominal": nom, "ratio": round(v / nom, 3)}
    res["px_20140630"] = tab
    # 1b: Daily_Returns == PX_LAST.pct_change()?
    chk = {}
    for t in ["MO", "KO", "T", "AAPL", "TSLA", "NVDA", "RIO", "AZN"]:
        p = px[t].dropna(); r = ret[t].reindex(p.index)
        pc = p.pct_change()
        diff = (pc - r).abs().dropna()
        chk[t] = {"n": int(len(diff)), "max_abs_diff": float(diff.max()),
                  "n_gt_1e-6": int((diff > 1e-6).sum()),
                  "ret_last": float(r.dropna().iloc[-1])}
    res["ret_vs_pctchange"] = chk
    # 1c: ex-div drop test on MO: in a dividend-adjusted series there is no
    # ~2% one-day drop on ex-dates; use PE*EPS nominal proxy day-over-day
    # ratio to detect steps: ratio_t = PX_LAST / nominal ; steps up on ex-dates.
    nom = d["BEST_PE_RATIO"] * d["BEST_EPS"]
    for t in ["MO", "T", "VZ", "KO", "TSLA", "NVDA"]:
        rr = (px[t] / nom[t]).replace([np.inf, -np.inf], np.nan).dropna()
        yr = rr.groupby(rr.index.year).median()
        res.setdefault("px_over_nominal_yearly", {})[t] = {int(k): round(float(v), 3) for k, v in yr.items()}
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(VER, "M1_s1.json"), "w"), indent=1)


def stage_s2():
    d = load_cache()
    px = local_px(d); tg = d["Factset_TG_Price"]
    nom = (d["BEST_PE_RATIO"] * d["BEST_EPS"]).replace([np.inf, -np.inf], np.nan)
    nom = nom.where((d["BEST_EPS"] > 0) & (d["BEST_PE_RATIO"] > 0) & (d["BEST_PE_RATIO"] < 500))
    res = {}
    # 2a: split continuity of TG/PX around split dates
    splits = {"AAPL": ["2014-06-09", "2020-08-31"], "NVDA": ["2021-07-20", "2024-06-10"],
              "TSLA": ["2020-08-31", "2022-08-25"]}
    for t, days in splits.items():
        for s in days:
            s = pd.Timestamp(s)
            r = (tg[t] / px[t]).dropna()
            before = r.loc[s - pd.Timedelta(days=30): s - pd.Timedelta(days=1)]
            after = r.loc[s: s + pd.Timedelta(days=30)]
            pb = px[t].loc[s - pd.Timedelta(days=10): s - pd.Timedelta(days=1)].dropna()
            pa = px[t].loc[s: s + pd.Timedelta(days=10)].dropna()
            tb = tg[t].loc[s - pd.Timedelta(days=10): s - pd.Timedelta(days=1)].dropna()
            ta = tg[t].loc[s: s + pd.Timedelta(days=10)].dropna()
            res.setdefault("split_continuity", {})[f"{t}@{s.date()}"] = {
                "tg_px_median_before": round(float(before.median()), 3),
                "tg_px_median_after": round(float(after.median()), 3),
                "px_before_last": round(float(pb.iloc[-1]), 3) if len(pb) else None,
                "px_after_first": round(float(pa.iloc[0]), 3) if len(pa) else None,
                "tg_before_last": round(float(tb.iloc[-1]), 3) if len(tb) else None,
                "tg_after_first": round(float(ta.iloc[0]), 3) if len(ta) else None,
            }
    # 2b: TG / nominal proxy yearly medians (flat => TG nominal, PX is adjusted one)
    for grp, names in [("HIGH_DIV", HIGH_DIV), ("NO_DIV", NO_DIV)]:
        names = [n for n in names if n in tg.columns and n in nom.columns]
        r_now = (tg[names] / px[names]).replace([np.inf, -np.inf], np.nan)
        r_nom = (tg[names] / nom[names]).replace([np.inf, -np.inf], np.nan)
        res.setdefault("tg_over_px_yearly_median", {})[grp] = {
            int(k): round(float(v), 3) for k, v in r_now.stack().groupby(level=0).median().groupby(lambda x: x.year).median().items()}
        res.setdefault("tg_over_nominal_yearly_median", {})[grp] = {
            int(k): round(float(v), 3) for k, v in r_nom.stack().groupby(level=0).median().groupby(lambda x: x.year).median().items()}
    for t in ["MO", "RIO", "T", "TSLA", "NVDA"]:
        r_nom = (tg[t] / nom[t]).replace([np.inf, -np.inf], np.nan).dropna()
        res.setdefault("tg_over_nominal_ticker", {})[t] = {int(k): round(float(v), 3) for k, v in r_nom.groupby(r_nom.index.year).median().items()}
    # 2c: cross-sectional: adjustment ratio proxy at 2014 (px/nominal) vs TG/PX 2014
    d14 = px.loc["2014-06":"2014-12"]
    adj14 = (px.loc["2014-06":"2014-12"] / nom.loc["2014-06":"2014-12"]).median()
    tgpx14 = (tg.loc["2014-06":"2014-12"] / px.loc["2014-06":"2014-12"]).median()
    tgnom14 = (tg.loc["2014-06":"2014-12"] / nom.loc["2014-06":"2014-12"]).median()
    both = pd.concat([adj14.rename("adj"), tgpx14.rename("tgpx"), tgnom14.rename("tgnom")], axis=1).dropna()
    both = both[(both.adj > 0.2) & (both.adj < 5)]
    res["xsec_2014"] = {
        "n": int(len(both)),
        "spearman_adj_vs_tgpx": round(float(both.adj.corr(both.tgpx, method="spearman")), 3),
        "spearman_adj_vs_tgnom": round(float(both.adj.corr(both.tgnom, method="spearman")), 3),
        "pearson_log_adj_vs_log_tgpx": round(float(np.log(both.adj).corr(np.log(both.tgpx))), 3),
        "adj_quantiles": {q: round(float(both.adj.quantile(q)), 3) for q in [0.05, 0.25, 0.5, 0.75, 0.95]},
    }
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(VER, "M1_s2.json"), "w"), indent=1)


def _spearman_rows(A, B, min_n=30):
    """Per-date Spearman between two aligned DataFrames (rows=dates)."""
    out = {}
    for dt in A.index:
        a = A.loc[dt]; b = B.loc[dt]
        m = a.notna() & b.notna()
        if m.sum() < min_n:
            continue
        out[dt] = a[m].rank().corr(b[m].rank())
    return pd.Series(out)


def _load_pkl():
    t0 = time.time()
    with open(PKL, "rb") as f:
        r = pickle.load(f)
    print("pkl loaded %.1fs" % (time.time() - t0))
    return r


def stage_s3():
    d = load_cache()
    px = local_px(d); tg = d["Factset_TG_Price"]
    pe = d["BEST_PE_RATIO"]; eps = d["BEST_EPS"]
    nom = (pe * eps).where((eps > 0) & (pe > 0) & (pe < 500)).replace([np.inf, -np.inf], np.nan)
    # LN names: PE*EPS is in GBP (major) — px already scaled to GBP. Check AZN ratio.
    res = {"nominal_proxy_nan_pct": round(float(nom.isna().mean().mean() * 100), 2)}
    adj = (px / nom).replace([np.inf, -np.inf], np.nan)
    adj = adj.where((adj > 0.2) & (adj < 5))
    # (i) per-date x-sec corr of log px vs log nominal, yearly median; and x-sec dispersion of adj
    lp = np.log(px.where(px > 0)); ln = np.log(nom.where(nom > 0))
    c = _spearman_rows(lp, ln)
    res["corr_logpx_lognom_yearly_median"] = {int(k): round(float(v), 4) for k, v in c.groupby(c.index.year).median().items()}
    disp = adj.std(axis=1)
    res["adj_xsec_std_yearly_median"] = {int(k): round(float(v), 4) for k, v in disp.groupby(disp.index.year).median().items()}
    res["adj_xsec_median_yearly"] = {int(k): round(float(v), 4) for k, v in adj.median(axis=1).groupby(adj.index.year).median().items()}
    # (ii) upside_now vs upside_nom per-date Spearman yearly
    up_now = (tg / px) - 1
    up_nom = (tg / nom) - 1
    up_now = up_now.where(up_nom.notna())  # same coverage
    c2 = _spearman_rows(up_now, up_nom)
    res["spearman_upnow_vs_upnom_yearly_median"] = {int(k): round(float(v), 4) for k, v in c2.groupby(c2.index.year).median().items()}
    res["spearman_upnow_vs_upnom_overall_median"] = round(float(c2.median()), 4)
    # (iii) forward IC with pkl targets
    r = _load_pkl()
    tgt = r.targets
    reb = sorted(r.portfolio_weights.keys())
    print("targets", tgt.shape, "reb dates", len(reb), reb[0], reb[-1])
    # every-21-row grid over all dates where targets valid, plus rebalance dates
    valid_dates = tgt.index[tgt.notna().sum(axis=1) >= 30]
    grid = valid_dates[::21]
    cols = [c for c in tgt.columns if c in up_now.columns]
    def ic_table(feat, dates, label):
        F = feat.reindex(index=dates, columns=cols)
        T = tgt.reindex(index=dates, columns=cols)
        s = _spearman_rows(F, T)
        out = {"label": label, "n": int(len(s)), "mean": round(float(s.mean()), 5),
               "t": round(float(s.mean() / s.std() * np.sqrt(len(s))), 3)}
        out["yearly"] = {int(k): round(float(v), 4) for k, v in s.groupby(s.index.year).mean().items()}
        n = len(s); thirds = [s.iloc[: n // 3], s.iloc[n // 3: 2 * n // 3], s.iloc[2 * n // 3:]]
        out["thirds"] = [round(float(x.mean()), 4) for x in thirds]
        return out, s
    res["ic_grid21"] = {}
    ic_now, s_now = ic_table(up_now, grid, "tg_upside_now (TG/PX_LAST-1)")
    ic_nom, s_nom = ic_table(up_nom, grid, "tg_upside_nom (TG/(PE*EPS)-1)")
    ic_adj, s_adj = ic_table(adj, grid, "adj proxy (PX_LAST/nominal)")
    ic_adj_inv, _ = ic_table(1.0 / adj, grid, "1/adj (dividend-yield-like dummy)")
    res["ic_grid21"]["now"] = ic_now; res["ic_grid21"]["nom"] = ic_nom
    res["ic_grid21"]["adj"] = ic_adj; res["ic_grid21"]["inv_adj"] = ic_adj_inv
    diff = (s_now - s_nom).dropna()
    res["ic_grid21"]["now_minus_nom"] = {"mean": round(float(diff.mean()), 5),
                                        "t": round(float(diff.mean() / diff.std() * np.sqrt(len(diff))), 3),
                                        "n": int(len(diff))}
    # residual IC: per date regress up_now on [1, up_nom, log(adj)] -> residual
    resid = {}
    for dt in grid:
        y = up_now.loc[dt, cols]; x1 = up_nom.loc[dt, cols]; x2 = np.log(adj.loc[dt, cols])
        m = y.notna() & x1.notna() & x2.notna()
        if m.sum() < 30:
            continue
        X = np.column_stack([np.ones(m.sum()), x1[m].values, x2[m].values])
        beta, *_ = np.linalg.lstsq(X, y[m].values, rcond=None)
        e = pd.Series(np.nan, index=cols); e[m] = y[m].values - X @ beta
        resid[dt] = e
    resid = pd.DataFrame(resid).T
    ic_res, s_res = ic_table(resid, grid, "resid(up_now | up_nom, log adj)")
    res["ic_grid21"]["resid"] = ic_res
    # also: partial - up_nom residualized on adj (what nominal upside contributes net of dividend dummy)
    # rebalance-date IC (for comparison with production 96-date avg_ic)
    res["ic_rebal"] = {}
    for lab, feat in [("now", up_now), ("nom", up_nom), ("adj", adj)]:
        o, _ = ic_table(feat, [x for x in reb if x in tgt.index], lab)
        res["ic_rebal"][lab] = o
    # variance decomposition: x-sec var(log(1+up_now)) explained by log adj per date
    share = {}
    for dt in grid:
        y = np.log1p(up_now.loc[dt, cols]); x = np.log(adj.loc[dt, cols])
        m = y.notna() & x.notna() & np.isfinite(y) & np.isfinite(x)
        if m.sum() < 30:
            continue
        share[dt] = float(np.corrcoef(y[m], x[m])[0, 1] ** 2)
    share = pd.Series(share)
    res["r2_logup_on_logadj_yearly"] = {int(k): round(float(v), 4) for k, v in share.groupby(share.index.year).median().items()}
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(VER, "M1_s3.json"), "w"), indent=1)


def stage_s4():
    r = _load_pkl()
    panel = r.panel
    res = {}
    for feat in ["tg_upside", "cash_conversion_z"]:
        if feat not in panel.columns:
            res[feat] = "MISSING"; continue
        s = panel[feat]
        for grp, names in [("HIGH_DIV", HIGH_DIV), ("NO_DIV", NO_DIV)]:
            sub = s[s.index.get_level_values(1).isin(names)]
            yr = sub.groupby(sub.index.get_level_values(0).year).median()
            res.setdefault(feat, {})[grp + "_yearly_median"] = {int(k): round(float(v), 3) for k, v in yr.items()}
        for t in ["MO", "RIO", "T", "TSLA", "NVDA", "9432", "KO"]:
            sub = s.xs(t, level=1) if t in s.index.get_level_values(1) else None
            if sub is None or len(sub) == 0:
                continue
            yr = sub.groupby(sub.index.year).median()
            res[feat][t] = {int(k): round(float(v), 3) for k, v in yr.items()}
        # train-period (<=2019) vs live (2026) gap for high-div group
        sub = s[s.index.get_level_values(1).isin(HIGH_DIV)]
        yrs = sub.index.get_level_values(0).year
        res[feat]["HIGH_DIV_train_2014_2019_median"] = round(float(sub[yrs <= 2019].median()), 3)
        res[feat]["HIGH_DIV_2025_2026_median"] = round(float(sub[yrs >= 2025].median()), 3)
        # trend slope of yearly medians vs year (high-div)
        yr = sub.groupby(yrs).median()
        res[feat]["HIGH_DIV_slope_per_year"] = round(float(np.polyfit(yr.index.values.astype(float), yr.values, 1)[0]), 4)
        # within-year cross-sectional rank of HIGH_DIV names' tg_upside (pct rank)
        if feat == "tg_upside":
            rk = s.groupby(level=0).rank(pct=True)
            sub = rk[rk.index.get_level_values(1).isin(HIGH_DIV)]
            yr = sub.groupby(sub.index.get_level_values(0).year).median()
            res[feat]["HIGH_DIV_pct_rank_yearly_median"] = {int(k): round(float(v), 3) for k, v in yr.items()}
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(VER, "M1_s4.json"), "w"), indent=1)


def stage_s5():
    d = load_cache()
    px = d["PX_LAST"]; mc = d["CUR_MKT_CAP"]
    nom = (d["BEST_PE_RATIO"] * d["BEST_EPS"]).where((d["BEST_EPS"] > 0) & (d["BEST_PE_RATIO"] > 0) & (d["BEST_PE_RATIO"] < 500))
    res = {}
    # implied shares = mktcap / px (production) vs mktcap / nominal ; ratio = nominal/px = 1/adj
    for t in ["MO", "T", "VZ", "XOM", "KO", "TSLA", "NVDA", "AAPL"]:
        sh_adj = (mc[t] / px[t]); sh_nom = (mc[t] / nom[t])
        yr_adj = sh_adj.groupby(sh_adj.index.year).median()
        yr_nom = sh_nom.groupby(sh_nom.index.year).median()
        res.setdefault("implied_shares_yearly", {})[t] = {
            int(k): {"mc_over_px": round(float(yr_adj[k]), 1), "mc_over_nom": round(float(yr_nom.get(k, np.nan)), 1),
                     "overstate_x": round(float(yr_adj[k] / yr_nom.get(k, np.nan)), 3)} for k in yr_adj.index}
    res["mktcap_2014_06_30_MO"] = float(mc.loc["2014-06-30", "MO"]) if pd.Timestamp("2014-06-30") in mc.index else None
    res["mktcap_last_MO"] = float(mc["MO"].dropna().iloc[-1]); res["px_last_MO"] = float(px["MO"].dropna().iloc[-1])
    res["mktcap_last_date"] = str(mc["MO"].dropna().index[-1].date())
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(VER, "M1_s5.json"), "w"), indent=1)


NOMINAL_NODIV_20140630 = {"TSLA": 240.06 / 15, "NVDA": 18.56 / 40, "AMZN": 324.78 / 20,
                          "META": 67.29, "GOOGL": 584.67 / 20, "NFLX": 440.60 / 7,
                          "ADBE": 72.34, "CRM": 58.07, "ISRG": 411.83 / 3}
# approx known values (public sources, memory): fwd PE mid-2014, shares (bn) mid-2014, mktcap ($bn) 2014-06-30
KNOWN_2014 = {"MO": {"fwd_pe": 16.0, "shares_bn": 1.98, "mktcap_bn": 83},
              "T": {"fwd_pe": 13.5, "shares_bn": 5.19, "mktcap_bn": 184},
              "KO": {"fwd_pe": 19.5, "shares_bn": 4.40, "mktcap_bn": 186},
              "XOM": {"fwd_pe": 13.0, "shares_bn": 4.29, "mktcap_bn": 432},
              "VZ": {"fwd_pe": 13.5, "shares_bn": 4.15, "mktcap_bn": 203},
              "AAPL": {"fwd_pe": 14.0, "shares_bn": 6.03 * 4, "mktcap_bn": 560}}


def stage_s6():
    """Independent anchors that do not rely on PE*EPS (which is == PX_LAST)."""
    d = load_cache()
    px = d["PX_LAST"]; mc = d["CUR_MKT_CAP"]; pe = d["BEST_PE_RATIO"]; eps = d["BEST_EPS"]
    res = {}
    day = pd.Timestamp("2014-06-30")
    row = px.loc[day]
    res["nodiv_px_20140630"] = {t: {"sheet": round(float(row[t]), 3), "nominal_splitadj": round(v, 3),
                                    "ratio": round(float(row[t]) / v, 3)}
                                for t, v in NOMINAL_NODIV_20140630.items() if t in row.index and pd.notna(row[t])}
    # PE * EPS vs PX exactness
    r = (px / (pe * eps)).replace([np.inf, -np.inf], np.nan)
    sub = r.loc["2014":"2016", ["MO", "T", "KO", "XOM", "AAPL", "TSLA"]].stack()
    res["px_over_pe_x_eps_2014_2016"] = {"median": float(sub.median()), "p05": float(sub.quantile(.05)),
                                         "p95": float(sub.quantile(.95)), "frac_within_1pct": float(((sub - 1).abs() < 0.01).mean())}
    # PE levels 2014H2 vs known forward PE; MC vs known; implied shares vs known
    tab = {}
    for t, k in KNOWN_2014.items():
        pe14 = float(pe.loc["2014-06":"2014-12", t].median())
        mc14 = float(mc.loc[day, t]); px14 = float(px.loc[day, t])
        tab[t] = {"sheet_fwd_pe_2014H2": round(pe14, 2), "known_fwd_pe": k["fwd_pe"], "pe_ratio": round(pe14 / k["fwd_pe"], 3),
                  "sheet_mktcap_20140630": round(mc14, 1), "known_mktcap_bn": k["mktcap_bn"],
                  "implied_shares_bn(mc/px)": round(mc14 / px14 / 1e3, 3), "known_shares_bn": k["shares_bn"],
                  "shares_overstate_x": round(mc14 / px14 / 1e3 / k["shares_bn"], 3)}
    res["anchors_2014"] = tab
    res["mktcap_unit_hint"] = {"MO_last_mc": float(mc["MO"].dropna().iloc[-1]), "MO_last_px": float(px["MO"].dropna().iloc[-1]),
                               "MO_last_implied_shares": float(mc["MO"].dropna().iloc[-1] / px["MO"].dropna().iloc[-1])}
    # implied-share step signature: daily dlog(mc/px). nominal px -> only share-count updates (both signs);
    # adjusted px -> additional quarterly negative steps of ~div yield on ex-dates.
    sig = {}
    for t in ["MO", "T", "KO", "XOM", "VZ", "PFE", "TSLA", "NVDA", "AMZN"]:
        s = np.log((mc[t] / px[t]).replace([np.inf, -np.inf], np.nan)).dropna()
        dl = s.diff().dropna()
        big = dl[dl.abs() > 0.003]
        neg = big[big < 0]
        yrs = (s.index[-1] - s.index[0]).days / 365.25
        sig[t] = {"n_days": int(len(dl)), "n_steps_gt_0.3pct": int(len(big)), "n_neg": int(len(neg)),
                  "n_pos": int((big > 0).sum()), "steps_per_year": round(len(big) / yrs, 2),
                  "sum_neg_steps": round(float(neg.sum()), 3), "sum_pos_steps": round(float(big[big > 0].sum()), 3),
                  "total_dlog_2014_to_end": round(float(s.iloc[-1] - s.iloc[0]), 3),
                  "frac_days_abs_lt_1e-4": round(float((dl.abs() < 1e-4).mean()), 3),
                  "median_neg_step": round(float(neg.median()), 4) if len(neg) else None,
                  "neg_step_month_hist": {int(m): int(c) for m, c in neg.groupby(neg.index.month).size().items()}}
    res["implied_share_step_signature"] = sig
    # MO neg-step dates first 12 (to compare with ex-dividend calendar: mid-Mar/Jun/Sep, ~Dec 20-24)
    s = np.log((mc["MO"] / px["MO"])).dropna(); dl = s.diff().dropna(); neg = dl[dl < -0.003]
    res["MO_first_neg_steps"] = {str(k.date()): round(float(v), 4) for k, v in neg.head(16).items()}
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(VER, "M1_s6.json"), "w"), indent=1)


def dy_proxy(d):
    """Per-ticker adjustment-drift proxy from implied shares: only NEGATIVE
    steps of log(mc/px) (ex-div steps + buybacks; issuance excluded).
    Returns cumulative negative-step series C(t) (>=0, decreasing to 0 at T)
    and per-ticker annualised rate."""
    px = d["PX_LAST"]; mc = d["CUR_MKT_CAP"]
    s = np.log((mc / px).replace([np.inf, -np.inf], np.nan))
    dl = s.diff()
    neg = dl.where(dl < -0.003, 0.0).where(dl.notna(), 0.0)
    # cumulative future negative steps: C(t) = -sum_{s>t} neg_s  (>= 0)
    C = (-neg[::-1].cumsum()[::-1]).shift(-1).fillna(0.0)
    return C


def stage_s34():
    d = load_cache()
    px = local_px(d); tg = d["Factset_TG_Price"]
    C = dy_proxy(d)  # proxy of -log A(t): cumulative future ex-div (+buyback) steps
    res = {}
    # sanity: C at 2014-06-30 vs known adj ratio  (-log ratio)
    day = pd.Timestamp("2014-06-30")
    res["C_vs_known_neglog_adj_20140630"] = {t: {"C": round(float(C.loc[day, t]), 3), "neglog_known": round(float(-np.log(float(px.loc[day, t]) / v)), 3)}
                                             for t, v in NOMINAL_20140630.items()}
    tg = tg.reindex(index=px.index, columns=px.columns)
    C = C.reindex(index=px.index, columns=px.columns)
    up_now = (tg / px) - 1
    nom_hat = px * np.exp(C)            # de-adjusted price proxy
    up_hat = (tg / nom_hat) - 1
    up_now = up_now.where(up_hat.notna())
    c2 = _spearman_rows(up_now, up_hat)
    res["spearman_upnow_vs_uphat_yearly_median"] = {int(k): round(float(v), 4) for k, v in c2.groupby(c2.index.year).median().items()}
    # x-sec R2 of log(1+up_now) on C per year
    share = {}
    for dt in up_now.index[::21]:
        y = np.log1p(up_now.loc[dt]); x = C.loc[dt]
        m = y.notna() & x.notna() & np.isfinite(y) & np.isfinite(x) & (y > -3)
        if m.sum() < 30:
            continue
        share[dt] = float(np.corrcoef(y[m], x[m])[0, 1] ** 2)
    share = pd.Series(share)
    res["r2_logup_on_C_yearly_median"] = {int(k): round(float(v), 4) for k, v in share.groupby(share.index.year).median().items()}
    res["C_xsec_std_yearly"] = {int(k): round(float(v), 4) for k, v in C.std(axis=1).groupby(C.index.year).median().items()}
    res["C_HIGH_DIV_yearly_median"] = {int(k): round(float(v), 3) for k, v in C[[t for t in HIGH_DIV if t in C.columns]].median(axis=1).groupby(C.index.year).median().items()}
    res["C_NO_DIV_yearly_median"] = {int(k): round(float(v), 3) for k, v in C[[t for t in NO_DIV if t in C.columns]].median(axis=1).groupby(C.index.year).median().items()}
    # ---- pkl ----
    r = _load_pkl()
    tgt = r.targets; panel = r.panel
    reb = sorted(r.portfolio_weights.keys())
    cols = [c for c in tgt.columns if c in up_now.columns]
    valid_dates = tgt.index[tgt.notna().sum(axis=1) >= 30]
    grid = valid_dates[::21]

    def ic_table(feat, dates, label):
        F = feat.reindex(index=dates, columns=cols); T = tgt.reindex(index=dates, columns=cols)
        s = _spearman_rows(F, T)
        out = {"label": label, "n": int(len(s)), "mean": round(float(s.mean()), 5),
               "t": round(float(s.mean() / s.std() * np.sqrt(len(s))), 3),
               "yearly": {int(k): round(float(v), 4) for k, v in s.groupby(s.index.year).mean().items()}}
        n = len(s); out["thirds"] = [round(float(x.mean()), 4) for x in (s.iloc[:n // 3], s.iloc[n // 3:2 * n // 3], s.iloc[2 * n // 3:])]
        return out, s
    # sanity: panel tg_upside (z) vs my up_now per date
    ptg = panel["tg_upside"].unstack()
    c0 = _spearman_rows(ptg.reindex(index=grid, columns=cols), up_now.reindex(index=grid, columns=cols))
    res["panel_tg_upside_vs_upnow_spearman_median"] = round(float(c0.median()), 4)
    res["panel_tg_upside_vs_upnow_spearman_min"] = round(float(c0.min()), 4)
    ic = {}
    ic["now"], s_now = ic_table(up_now, grid, "TG/PX_LAST-1")
    ic["hat"], s_hat = ic_table(up_hat, grid, "TG/(PX*exp(C))-1  (de-adjusted proxy)")
    ic["C"], s_C = ic_table(C, grid, "C = cumulative future ex-div steps (artifact term)")
    ic["panel_tg_upside"], _ = ic_table(ptg, grid, "panel tg_upside (z)")
    diff = (s_now - s_hat).dropna()
    ic["now_minus_hat"] = {"mean": round(float(diff.mean()), 5), "t": round(float(diff.mean() / diff.std() * np.sqrt(len(diff))), 3), "n": int(len(diff))}
    # residual of up_now on [1, up_hat, C]
    resid = {}
    for dt in grid:
        y = up_now.loc[dt, cols]; x1 = up_hat.loc[dt, cols]; x2 = C.loc[dt, cols]
        m = y.notna() & x1.notna() & x2.notna()
        if m.sum() < 30:
            continue
        X = np.column_stack([np.ones(m.sum()), x1[m].values, x2[m].values])
        beta, *_ = np.linalg.lstsq(X, y[m].values, rcond=None)
        e = pd.Series(np.nan, index=cols); e[m] = y[m].values - X @ beta
        resid[dt] = e
    ic["resid_now_on_hat_C"], _ = ic_table(pd.DataFrame(resid).T, grid, "resid(up_now|up_hat,C)")
    # rank-based: IC of rank(up_now) after partialling out C by per-date OLS on ranks
    res["ic_grid21"] = ic
    icr = {}
    rdates = [x for x in reb if x in tgt.index]
    for lab, f in [("now", up_now), ("hat", up_hat), ("C", C)]:
        icr[lab], _ = ic_table(f, rdates, lab)
    res["ic_rebal"] = icr
    # ---- s4: live-time shift of panel tg_upside z ----
    s4 = {}
    for feat in ["tg_upside", "cash_conversion_z"]:
        s = panel[feat]; tick = s.index.get_level_values(1); yrs = s.index.get_level_values(0).year
        s4[feat] = {}
        for grp, names in [("HIGH_DIV", HIGH_DIV), ("NO_DIV", NO_DIV)]:
            sub = s[tick.isin(names)]
            s4[feat][grp + "_yearly_median"] = {int(k): round(float(v), 3) for k, v in sub.groupby(sub.index.get_level_values(0).year).median().items()}
        for t in ["MO", "RIO", "T", "TSLA", "NVDA", "9432", "KO", "AAPL"]:
            sub = s.xs(t, level=1)
            s4[feat][t] = {int(k): round(float(v), 3) for k, v in sub.groupby(sub.index.year).median().items()}
        sub = s[tick.isin(HIGH_DIV)]; y2 = sub.index.get_level_values(0).year
        s4[feat]["HIGH_DIV_2014_2019_median"] = round(float(sub[y2 <= 2019].median()), 3)
        s4[feat]["HIGH_DIV_2025_2026_median"] = round(float(sub[y2 >= 2025].median()), 3)
        yr = sub.groupby(y2).median()
        s4[feat]["HIGH_DIV_slope_per_year"] = round(float(np.polyfit(yr.index.values.astype(float), yr.values, 1)[0]), 4)
        if feat == "tg_upside":
            rk = s.groupby(level=0).rank(pct=True)
            sub = rk[tick.isin(HIGH_DIV)]
            s4[feat]["HIGH_DIV_pct_rank_yearly_median"] = {int(k): round(float(v), 3) for k, v in sub.groupby(sub.index.get_level_values(0).year).median().items()}
            # last panel date: high-div z vs train-period z
            last = s.index.get_level_values(0).max()
            s4[feat]["last_date"] = str(last.date())
            s4[feat]["HIGH_DIV_last_date_z"] = {t: round(float(s.loc[(last, t)]), 3) for t in HIGH_DIV if (last, t) in s.index}
    res["s4"] = s4
    # ---- s5-lite: cash_conversion shares proxy ----
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(VER, "M1_s34.json"), "w"), indent=1)


def stage_s7():
    """Which sheets share PX_LAST's dividend adjustment? On ex-div dates (negative
    steps of log(MC/PX)), dlog(X/PX) ~ +D/P if X is nominal, ~0 if X is adjusted
    like PX. Ratio = dlog(X/PX) / (-dlog(MC/PX)) -> 1 nominal, 0 adjusted."""
    from src.config import DEFAULT_CONFIG
    d = load_cache()
    extra = ["BEST_PX_BPS_RATIO", "BEST_PEG_RATIO", "BEST_EV_TO_BEST_EBITDA", "PX_VOLUME"]
    t0 = time.time()
    raw = pd.read_excel(DEFAULT_CONFIG.data_path, sheet_name=extra)
    print("read extra %.0fs" % (time.time() - t0))
    for k, df in raw.items():
        d[k] = _date_index(df)
    px = d["PX_LAST"]; mc = d["CUR_MKT_CAP"]
    s_impl = np.log((mc / px).replace([np.inf, -np.inf], np.nan))
    ds = s_impl.diff()
    names = ["KO", "XOM", "T", "PFE", "VZ", "JPM", "PG", "MSFT", "MO", "IBM", "CVX", "HD"]
    names = [n for n in names if n in px.columns]
    res = {}
    for X in ["BEST_PE_RATIO", "BEST_PX_BPS_RATIO", "BEST_PEG_RATIO", "BEST_EV_TO_BEST_EBITDA", "Factset_TG_Price", "BEST_EPS"]:
        xs = d[X].reindex(index=px.index, columns=px.columns)
        if X in ("Factset_TG_Price", "BEST_EPS"):
            xs = xs.ffill()
        dx = np.log((xs / px).where(xs > 0).replace([np.inf, -np.inf], np.nan)).diff()
        ratios = []; per = {}
        for t in names:
            ex = ds[t][(ds[t] < -0.005)]  # ex-div-sized steps only
            r = (dx.loc[ex.index, t] / (-ex)).dropna()
            r = r[(r > -2) & (r < 3)]
            per[t] = {"n": int(len(r)), "median_ratio": round(float(r.median()), 3) if len(r) else None}
            ratios.append(r)
        allr = pd.concat(ratios)
        res[X] = {"pooled_n": int(len(allr)), "pooled_median_ratio": round(float(allr.median()), 3),
                  "pooled_frac_near0(|r|<0.25)": round(float((allr.abs() < 0.25).mean()), 3),
                  "pooled_frac_near1(|r-1|<0.25)": round(float(((allr - 1).abs() < 0.25).mean()), 3),
                  "per_ticker": per}
    # PX_VOLUME * PX = dollar volume: adjusted -> understated in past for high-div names (informational)
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(VER, "M1_s7.json"), "w"), indent=1)


def stage_s8():
    """Scope extension: valuation features built from dividend-adjusted PE/PB."""
    from src.config import DEFAULT_CONFIG
    from src.features.utils import rolling_tsz, cross_sectional_zscore
    d = load_cache()
    raw = pd.read_excel(DEFAULT_CONFIG.data_path, sheet_name=["BEST_PX_BPS_RATIO"])
    d["BEST_PX_BPS_RATIO"] = _date_index(raw["BEST_PX_BPS_RATIO"])
    px = d["PX_LAST"]
    C = dy_proxy(d).reindex(index=px.index, columns=px.columns)
    pe = d["BEST_PE_RATIO"].reindex(index=px.index, columns=px.columns)
    pb = d["BEST_PX_BPS_RATIO"].reindex(index=px.index, columns=px.columns)
    res = {}
    hd = [t for t in HIGH_DIV if t in px.columns]; nd = [t for t in NO_DIV if t in px.columns]
    # (b) tsz(PE) as production (window 756) vs tsz(PE * exp(C)) de-adjusted proxy
    for nm, X in [("pe", pe), ("pb", pb)]:
        Xp = X.where(X > 0)
        z_now = cross_sectional_zscore(rolling_tsz(Xp, window=756, min_periods=252))
        z_hat = cross_sectional_zscore(rolling_tsz(Xp * np.exp(C), window=756, min_periods=252))
        if nm == "pe":
            z_now = z_now.clip(upper=1.5); z_hat = z_hat.clip(upper=1.5)
        dz = z_now - z_hat
        res[f"fin_{nm}_level_z_shift(now-hat)_HIGH_DIV_yearly_median"] = {int(k): round(float(v), 3) for k, v in dz[hd].median(axis=1).groupby(dz.index.year).median().items()}
        res[f"fin_{nm}_level_z_shift(now-hat)_NO_DIV_yearly_median"] = {int(k): round(float(v), 3) for k, v in dz[nd].median(axis=1).groupby(dz.index.year).median().items()}
        # pure cross-sectional level z (accounting.py best_px_bps_ratio_level_z style)
        cz_now = cross_sectional_zscore(Xp); cz_hat = cross_sectional_zscore(Xp * np.exp(C))
        dcz = cz_now - cz_hat
        res[f"cs_level_z_{nm}_shift(now-hat)_HIGH_DIV_yearly_median"] = {int(k): round(float(v), 3) for k, v in dcz[hd].median(axis=1).groupby(dcz.index.year).median().items()}
        # (c) per-date R2 of log X on C
        share = {}
        for dt in px.index[::21]:
            y = np.log(Xp.loc[dt]); x = C.loc[dt]
            m = y.notna() & x.notna() & np.isfinite(y)
            if m.sum() < 30:
                continue
            share[dt] = float(np.corrcoef(y[m], x[m])[0, 1] ** 2)
        share = pd.Series(share)
        res[f"r2_log{nm}_on_C_yearly_median"] = {int(k): round(float(v), 4) for k, v in share.groupby(share.index.year).median().items()}
    # (a) panel yearly medians by group
    r = _load_pkl()
    panel = r.panel; tgt = r.targets
    feats = ["fin_pe_level_z", "fin_pb_level_z", "best_px_bps_ratio_level_z", "best_peg_ratio_level_z",
             "fin_roe_pb_gap", "fin_roe_pe_gap", "fin_pe_chg_63d", "fin_pb_chg_63d", "tg_upside", "cash_conversion_z"]
    tick = panel.index.get_level_values(1); yrs = panel.index.get_level_values(0).year
    pan = {}
    for f in feats:
        if f not in panel.columns:
            pan[f] = "MISSING"; continue
        s = panel[f]
        pan[f] = {"HIGH_DIV": {int(k): round(float(v), 3) for k, v in s[tick.isin(hd)].groupby(yrs[tick.isin(hd)]).median().items()},
                  "NO_DIV": {int(k): round(float(v), 3) for k, v in s[tick.isin(nd)].groupby(yrs[tick.isin(nd)]).median().items()}}
        a = s[tick.isin(hd)]; ya = yrs[tick.isin(hd)]
        pan[f]["HIGH_DIV_2014_2019_vs_2025_2026"] = [round(float(a[ya <= 2019].median()), 3), round(float(a[ya >= 2025].median()), 3)]
    res["panel_group_medians"] = pan
    # (d) IC of PB / PE cross-sectional z, now vs hat, 21d grid
    valid_dates = tgt.index[tgt.notna().sum(axis=1) >= 30]; grid = valid_dates[::21]
    cols = [c for c in tgt.columns if c in px.columns]
    def ic(F, dates):
        s = _spearman_rows(F.reindex(index=dates, columns=cols), tgt.reindex(index=dates, columns=cols))
        n = len(s)
        return {"n": int(n), "mean": round(float(s.mean()), 5), "t": round(float(s.mean() / s.std() * np.sqrt(n)), 3),
                "thirds": [round(float(x.mean()), 4) for x in (s.iloc[:n // 3], s.iloc[n // 3:2 * n // 3], s.iloc[2 * n // 3:])]}, s
    icd = {}
    for nm, X in [("pe", pe), ("pb", pb)]:
        Xp = X.where(X > 0)
        icd[f"{nm}_level_now"], s1 = ic(Xp, grid)
        icd[f"{nm}_level_hat"], s2 = ic(Xp * np.exp(C), grid)
        dd = (s1 - s2).dropna(); icd[f"{nm}_now_minus_hat"] = {"mean": round(float(dd.mean()), 5), "t": round(float(dd.mean() / dd.std() * np.sqrt(len(dd))), 3)}
        icd[f"{nm}_tsz_now"], s1 = ic(rolling_tsz(Xp, window=756, min_periods=252), grid)
        icd[f"{nm}_tsz_hat"], s2 = ic(rolling_tsz(Xp * np.exp(C), window=756, min_periods=252), grid)
        dd = (s1 - s2).dropna(); icd[f"{nm}_tsz_now_minus_hat"] = {"mean": round(float(dd.mean()), 5), "t": round(float(dd.mean() / dd.std() * np.sqrt(len(dd))), 3)}
    res["ic_grid21_valuation"] = icd
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(VER, "M1_s8.json"), "w"), indent=1)


if __name__ == "__main__":
    {"load": stage_load, "s1": stage_s1, "s2": stage_s2, "s3": stage_s3, "s4": stage_s4, "s5": stage_s5,
     "s6": stage_s6, "s34": stage_s34, "s7": stage_s7, "s8": stage_s8}[sys.argv[1]]()
