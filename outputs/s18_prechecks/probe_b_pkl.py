# -*- coding: utf-8 -*-
"""Probe B: certified S0' pkl (outputs/s17_5_nominal_price) — read-only diagnostics.

B1 vol-quality tilt x negative-equity ROE (W3 critic #1)
B2 execution confidence reconstruction: spread leg saturation, eta_t range, IC autocorrelation
B3 turnover per rebalance vs 0.15 cap; active share
B4 covariance branch (pairwise vs Ledoit-Wolf) by rebalance date; ex-ante TE vs realized fwd TE by branch
B5 fwd_sales_slope coverage artefact: imputed share per date, coverage-indicator IC, model gain on slope block
B6 DELL / spin-off tg_upside rows at the +5 clip
B7 execution-lag sanity; listing dates for VRT / de-SPAC names
"""
import json, pickle, sys, warnings, gc
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
AI_PORT = Path(r"C:/Users/westl/PycharmProjects/pythonProject/venv_vf_new/machine/re_study/c2/ai_port")
sys.path.insert(0, str(AI_PORT))
SCRATCH = Path(__file__).resolve().parent
OUT = SCRATCH / "probe_b_results.json"
res = {}

raw = pickle.load(open(SCRATCH / "probe_a_raw.pkl", "rb"))
listing = {k: pd.Timestamp(v) for k, v in raw["listing"].items()}
print("[B] loading pkl ...", flush=True)
r = pickle.load(open(AI_PORT / "outputs/s17_5_nominal_price/backtest_result.pkl", "rb"))
print("[B] pkl loaded", flush=True)
panel = r.panel
preds = r.predictions            # post-overlay, lagged
pre_exec = getattr(r, "pre_execution_predictions", None)
rawp = r.raw_predictions
targets = r.targets
W = r.portfolio_weights          # {date: Series}
rebal_dates = sorted(W)
tickers = list(W[rebal_dates[0]].index)
port = r.portfolio_returns; bm = r.benchmark_returns
ic = r.ic_series
dq = r.data_quality
res["B7_listing_VRT_and_friends"] = {t: dq["listing_mask"]["dates"].get(t) for t in ["VRT", "TKO", "HWM", "LIN", "VST", "285A", "SNDK", "GEV", "ARM"]}
res["B7_listing_sources"] = dq["listing_mask"]["source_counts"]

# ---------------------------------------------------------------- B7 lag sanity
if pre_exec is not None:
    d = (preds.shift(-1) - pre_exec).abs().stack().dropna()
    res["B7_lag_sanity_max_abs_diff_shifted"] = float(d.max())

# ---------------------------------------------------------------- B1 tilt x negative equity
print("[B1]", flush=True)
vol = panel["idio_vol_63d"].unstack("ticker"); q = panel["best_roe_level_z"].unstack("ticker")
roe_raw = raw["BEST_ROE"].reindex(index=vol.index, columns=vol.columns).ffill()
eps_raw = raw["BEST_EPS"].reindex(index=vol.index, columns=vol.columns).ffill()
neg_eq = (roe_raw < -50) & (eps_raw > 0)      # negative-equity signature: strongly negative ROE with positive EPS
def _wz(s):
    s = s.dropna()
    if len(s) < 3: return pd.Series(dtype=float)
    sw = s.clip(s.quantile(0.01), s.quantile(0.99)); sd = sw.std(ddof=1)
    return (sw - sw.mean()) / sd if sd > 0 else pd.Series(dtype=float)
pre_tilt = pre_exec if pre_exec is not None else preds
hits = []; per_name = {}
for dt in rebal_dates:
    # tilt was applied to the UNLAGGED prediction row at dt-1; use pre_exec at dt-1 to mimic
    src_dt = pre_tilt.index[pre_tilt.index.get_loc(dt) - 1] if pre_exec is not None else dt
    s = pre_tilt.loc[src_dt]; scored = s.index[s.notna()]
    if len(scored) < 30 or src_dt not in vol.index: continue
    vz = _wz(vol.loc[src_dt].reindex(scored)); top = vz.sort_values(ascending=False).head(len(vz) // 3).index
    zq = _wz(q.loc[src_dt].reindex(top)); sd = float(s[scored].std(ddof=1))
    for t in zq.index:
        if neg_eq.loc[src_dt, t] if (src_dt in neg_eq.index and t in neg_eq.columns) else False:
            hits.append((str(src_dt.date()), t, float(zq[t]), float(0.25 * sd * zq[t])))
            per_name.setdefault(t, []).append(float(zq[t]))
res["B1_tilt_negative_equity"] = {
    "rebalance_dates_with_neg_equity_in_top_vol_tercile": int(len({h[0] for h in hits})),
    "name_rebalance_pairs": int(len(hits)),
    "names": {t: {"n": len(v), "mean_zq": round(float(np.mean(v)), 2), "min_zq": round(float(np.min(v)), 2)} for t, v in sorted(per_name.items(), key=lambda kv: -len(kv[1]))},
    "mean_score_shift_lam_sd_zq": round(float(np.mean([h[3] for h in hits])), 3) if hits else None,
    "neg_equity_names_overall": sorted(set(neg_eq.columns[neg_eq.mean() > 0.5])),
}

# ---------------------------------------------------------------- B2 confidence reconstruction
print("[B2]", flush=True)
spreads = []; conf = []; etas = []; ic_vals = list(ic.items())
for i, dt in enumerate(rebal_dates):
    if rawp is None or dt not in rawp.index: continue
    rv = rawp.loc[dt].dropna()
    if len(rv) < 5: continue
    tail_n = max(3, len(rv) // 10)
    sp = float(rv.sort_values(ascending=False).head(tail_n).mean() - rv.sort_values().head(tail_n).mean())
    past = [v for d, v in ic_vals if d < dt][-6:]
    tic = float(np.nanmean(past)) if len(past) >= 2 else 0.0
    spread_score = float(np.clip(sp / 0.20, 0.2, 1.0)); ic_score = float(np.clip((tic + 0.01) / 0.04, 0.2, 1.0))
    cf = float(np.clip(spread_score * ic_score, 0.1, 1.0))
    spreads.append(sp); conf.append(cf); etas.append(float(np.clip(0.5 * cf ** 0.5, 0.05, 0.95)))
ic_s = ic.dropna()
res["B2_execution_confidence"] = {
    "raw_spread_top_bottom_decile": {"min": round(min(spreads), 2), "median": round(float(np.median(spreads)), 2), "max": round(max(spreads), 2), "spread_scale": 0.20,
                                     "share_spread_score_saturated_at_1": float(np.mean(np.array(spreads) / 0.2 >= 1.0))},
    "confidence": {"min": round(min(conf), 3), "median": round(float(np.median(conf)), 3), "max": round(max(conf), 3), "share_at_floor_0.2": float(np.mean(np.array(conf) <= 0.2 + 1e-9))},
    "eta_effective": {"min": round(min(etas), 3), "median": round(float(np.median(etas)), 3), "max": round(max(etas), 3)},
    "ic_series": {"n": int(len(ic_s)), "mean": round(float(ic_s.mean()), 4), "std": round(float(ic_s.std()), 4),
                  "lag1_autocorr": round(float(ic_s.autocorr(1)), 3), "lag2_autocorr": round(float(ic_s.autocorr(2)), 3),
                  "corr_trailing6_mean_vs_next_ic": round(float(pd.concat([ic_s.rolling(6).mean().shift(1), ic_s], axis=1).dropna().corr().iloc[0, 1]), 3)},
}

# ---------------------------------------------------------------- B3 turnover
print("[B3]", flush=True)
to = r.turnover
res["B3_turnover_two_way_per_rebalance"] = {"n": int(len(to)), "mean": round(float(to.mean()), 4), "median": round(float(to.median()), 4),
    "p90": round(float(to.quantile(0.9)), 4), "max": round(float(to.max()), 4), "share_gt_0.10": float((to > 0.10).mean()),
    "first_rebalance": round(float(to.iloc[0]), 4), "active_share_mean": round(float(r.active_share_series.mean()), 4)}

# ---------------------------------------------------------------- B4 covariance branch & ex-ante TE
print("[B4]", flush=True)
from src.portfolio_optimizer import _pairwise_covariance
from sklearn.covariance import LedoitWolf
ret_local = raw["Daily_Returns"].astype(float)
dates_all = port.index
# rebuild the risk panel the backtest used: raw local returns -> USD? we lack FX here; use local (cov branch selection only needs the NaN pattern; TE ratio uses local as approximation)
rr = ret_local.reindex(index=dates_all, columns=tickers)
for t, d in listing.items():
    if t in rr.columns:
        rr.loc[rr.index <= d, t] = np.nan
mc = raw["CUR_MKT_CAP"].reindex(index=dates_all, columns=tickers).ffill()
rows = []
for dt in rebal_dates:
    i = dates_all.get_loc(dt)
    win = rr.iloc[max(0, i - 126):i]
    if len(win) < 30: continue
    has_nan = bool(win.isna().any().any())
    if has_nan:
        cov = _pairwise_covariance(win, enforce_diag_min_obs=True)
    else:
        lw = LedoitWolf().fit(win.values); cov = lw.covariance_.copy()
    # mega-cap shrink as production
    bmw = mc.loc[dt].values.astype(float); bmw = np.where(np.isfinite(bmw) & (bmw > 0), bmw, 0.0); bmw = bmw / bmw.sum()
    n = len(bmw); vols = np.sqrt(np.diag(cov)); avg = vols.mean(); scale = np.ones(n)
    for k in range(n):
        if bmw[k] > 2.0 / n and vols[k] > 0: scale[k] = (0.5 * avg + 0.5 * vols[k]) / vols[k]
    cov = np.diag(scale) @ cov @ np.diag(scale)
    a = W[dt].reindex(tickers).values - bmw
    ex_ante = float(np.sqrt(max(a @ cov @ a, 0) * 252))
    fwd = (port - bm).iloc[i + 1:i + 22]
    realized = float(fwd.std(ddof=1) * np.sqrt(252)) if len(fwd) >= 15 else np.nan
    eig_min = float(np.linalg.eigvalsh(cov).min()); eig_max = float(np.linalg.eigvalsh(cov).max())
    rows.append({"date": str(dt.date()), "pairwise": has_nan, "ex_ante_te": ex_ante, "realized_21d_te": realized,
                 "cond": eig_max / max(eig_min, 1e-300), "n_nan_cols": int(win.isna().any().sum())})
df4 = pd.DataFrame(rows)
def _grp(g):
    return {"n": int(len(g)), "ex_ante_te_median": round(float(g.ex_ante_te.median()), 4), "realized_te_median": round(float(g.realized_21d_te.median()), 4),
            "ratio_realized_over_ex_ante_median": round(float((g.realized_21d_te / g.ex_ante_te).median()), 3),
            "cond_median": float(g.cond.median()), "n_nan_cols_median": float(g.n_nan_cols.median())}
res["B4_cov_branch"] = {"pairwise_dates": int(df4.pairwise.sum()), "lw_dates": int((~df4.pairwise).sum()),
                       "pairwise": _grp(df4[df4.pairwise]), "ledoit_wolf": _grp(df4[~df4.pairwise]),
                       "note": "local-currency returns used for the risk panel (FX leg omitted); optvol/scale-fix scaling omitted"}
df4.to_csv(SCRATCH / "probe_b_cov_branch.csv", index=False)

# ---------------------------------------------------------------- B5 slope coverage artefact
print("[B5]", flush=True)
slope = panel["fwd_sales_slope_level"].unstack("ticker")
# imputed cells sit exactly at the per-date median of the RAW slope -> after CS z they share one value per date: detect via modal value share
def modal_share(row):
    v = row.dropna().round(6)
    return float(v.value_counts().iloc[0] / len(v)) if len(v) else np.nan
ms = slope.apply(modal_share, axis=1)
res["B5_slope_modal_share_by_year"] = {int(y): round(float(v), 3) for y, v in ms.groupby(ms.index.year).median().items()}
# coverage indicator = not at the modal value
mode_val = slope.apply(lambda row: row.dropna().round(6).mode().iloc[0] if row.notna().any() else np.nan, axis=1)
covered = (slope.round(6).ne(mode_val, axis=0)) & slope.notna()
tg = targets.reindex(index=slope.index, columns=slope.columns)
ic_cov = []
for dt in slope.index[::21]:
    c_row = covered.loc[dt].astype(float); t_row = tg.loc[dt]
    v = c_row.notna() & t_row.notna()
    if v.sum() > 30 and c_row[v].std() > 0:
        ic_cov.append((dt, c_row[v].corr(t_row[v], method="spearman")))
icc = pd.Series(dict(ic_cov))
res["B5_coverage_indicator_ic_vs_target_by_year"] = {int(y): {"mean_ic": round(float(g.mean()), 4), "t": round(float(g.mean() / g.std() * np.sqrt(len(g))), 2), "n": int(len(g))}
                                                       for y, g in icc.groupby(icc.index.year) if len(g) > 2}
# model gain on the slope block per model
gains = {}
for mdt, m in sorted(r.models.items()):
    feats = getattr(m, "_active_features", None); booster = getattr(m, "booster_", None)
    if feats is None or booster is None: continue
    g = booster.feature_importance(importance_type="gain"); g = g / g.sum() if g.sum() > 0 else g
    gm = dict(zip(feats, g))
    gains[str(pd.Timestamp(mdt).date())] = {k: round(float(gm.get(k, 0.0)), 4) for k in ["fwd_sales_slope_level", "fwd_sales_slope_chg_63d", "nl_fslope_rev_confirm", "nl_fslope_growth_confirm", "tg_upside", "idio_vol_63d"]}
res["B5_slope_block_gain_share_by_model"] = gains
blk = pd.DataFrame(gains).T
res["B5_slope_block_gain_summary"] = {"slope4_sum_median": round(float(blk[["fwd_sales_slope_level", "fwd_sales_slope_chg_63d", "nl_fslope_rev_confirm", "nl_fslope_growth_confirm"]].sum(axis=1).median()), 4),
                                     "slope4_sum_by_model": {k: round(float(v), 4) for k, v in blk[["fwd_sales_slope_level", "fwd_sales_slope_chg_63d", "nl_fslope_rev_confirm", "nl_fslope_growth_confirm"]].sum(axis=1).items()}}

# ---------------------------------------------------------------- B6 DELL clip rows
print("[B6]", flush=True)
tgu = panel["tg_upside"].unstack("ticker")
clip = (tgu >= 4.99)
res["B6_tg_upside_clip_rows_by_ticker"] = {t: int(v) for t, v in clip.sum().sort_values(ascending=False).head(12).items()}
res["B6_tg_upside_clip_rows_by_year_total"] = {int(y): int(v) for y, v in clip.sum(axis=1).groupby(clip.index.year).sum().items()}
dell = tgu["DELL"] if "DELL" in tgu.columns else None
if dell is not None:
    res["B6_DELL_tg_upside_z_by_year"] = {int(y): round(float(v), 2) for y, v in dell.groupby(dell.index.year).median().items()}

json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, default=str)
print("[B] done ->", OUT, flush=True)
