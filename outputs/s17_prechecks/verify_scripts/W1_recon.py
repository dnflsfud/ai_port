"""W1 T-03 / T-05 / T-07: approximate offline reconstruction of the production rebalance
inputs (bm, prev book, Sigma with optvol diag) from the M2 sheet cache + pkl, WITHOUT loading
UniverseData (full workbook load is forbidden for verifiers).

Approximations (declared):
  * USD conversion uses Index.xlsx FX only (production prefers Factor_PX_LAST where observed).
  * dense returns for the daily drift: raw USD returns with NaN->0 (production: ffill+row-median
    imputed panel; only affects names with genuinely missing cells).
  * CUR_MKT_CAP: ffill only (production also row-median-fills missing cells before re-mask).
  * iv30_z sheet read directly (MODE=prep, OPTVOL=1) and imputed the loader way (ffill + row median).
MODE=prep  -> builds inputs and caches W1_recon_inputs.pkl
MODE=run   -> loops rebalance dates (SUBSET=anchors|all), writes W1_recon_rows.csv + W1_recon.json
"""
import os, sys, json, pickle, time
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, yaml

V = os.environ["V"]
MODE = os.environ.get("MODE", "prep")
SUBSET = os.environ.get("SUBSET", "anchors")
OPTVOL = os.environ.get("OPTVOL", "1") == "1"
PKL = "outputs/s16_7_name_risk_cap/backtest_result.pkl"
INP = os.path.join(V, "W1_recon_inputs.pkl")

from src.harness import build_override_config, inject_config
manifest = yaml.safe_load(open("variants/codex_causal_rank_65.yaml", encoding="utf-8"))
cfg = build_override_config(dict(manifest["overrides"]))
inject_config(cfg)

t0 = time.time()
r = pickle.load(open(PKL, "rb"))
rebal_dates = sorted(r.portfolio_weights.keys())
tickers = list(r.portfolio_weights[rebal_dates[0]].index)
print("pkl loaded %.1fs, tickers %d, rebal %d" % (time.time() - t0, len(tickers), len(rebal_dates)))

if MODE == "prep":
    from src.data_loader import _rename_bloomberg_equity_columns, mask_pre_listing
    sheets = pickle.load(open(os.path.join(V, "m2_sheets.pkl"), "rb"))
    meta = sheets["Universe_Meta"]
    listing = json.load(open("outputs/s16_7_name_risk_cap/metrics.json"))["data_quality"]["listing_mask"]["dates"]
    cal = sheets["Daily_Returns"].index
    bdays = cal[cal.weekday < 5]
    pidx = r.predictions.index
    assert pidx.isin(bdays).all(), "predictions index not subset of weekday grid"
    # FX usd-per-local on weekday grid
    fx = sheets["Index_PX_LAST_subset"].reindex(bdays).ffill()
    upl = pd.DataFrame(index=bdays)
    upl["USD"] = 1.0
    upl["JPY"] = 1.0 / fx["USDJPY Curncy"]; upl["KRW"] = 1.0 / fx["USDKRW Curncy"]; upl["CHF"] = 1.0 / fx["USDCHF Curncy"]
    upl["DKK"] = 1.0 / fx["USDDKK Curncy"]; upl["EUR"] = fx["EURUSD Curncy"]; upl["GBP"] = fx["GBPUSD Curncy"]
    cur = meta["currency"].astype(str).str.upper().reindex(tickers)
    rates = pd.DataFrame({t: upl[cur[t]] for t in tickers}, index=bdays)
    fx_ret = rates.pct_change(fill_method=None).fillna(0.0)
    raw_local = sheets["Daily_Returns"].reindex(bdays)[tickers].apply(pd.to_numeric, errors="coerce")
    raw_usd = (1.0 + raw_local) * (1.0 + fx_ret) - 1.0
    raw_usd = mask_pre_listing(raw_usd, listing, inclusive=True)
    dense_usd = raw_usd.fillna(0.0)
    # market cap (already USD in the sheet) -> ffill -> pre-listing NaN
    mc = sheets["CUR_MKT_CAP"].reindex(bdays)[tickers].apply(pd.to_numeric, errors="coerce").ffill()
    mc = mask_pre_listing(mc, listing, inclusive=False)
    sector_map = {t: str(meta.loc[t, "sector"]) for t in tickers if t in meta.index}
    optvol_scale = None
    optvol_info = {"enabled": OPTVOL}
    if OPTVOL:
        from src.option_vol_cov import build_option_vol_scale, OPTION_VOL_SHEET
        t1 = time.time()
        iv_raw = pd.read_excel(cfg.data_path, sheet_name=[OPTION_VOL_SHEET])[OPTION_VOL_SHEET]
        optvol_info["sheet_read_sec"] = round(time.time() - t1, 1)
        dcol = "date" if "date" in iv_raw.columns else iv_raw.columns[0]
        iv_raw = iv_raw.set_index(pd.to_datetime(iv_raw[dcol])).drop(columns=[dcol]).sort_index()
        iv_raw = _rename_bloomberg_equity_columns(iv_raw)
        iv_raw.columns = [str(c).strip() for c in iv_raw.columns]
        iv_raw = iv_raw.reindex(columns=tickers).apply(pd.to_numeric, errors="coerce")
        iv_raw = iv_raw.reindex(bdays)
        iv_masked = mask_pre_listing(iv_raw, listing, inclusive=False)
        observed = iv_masked.notna()
        iv_imp = iv_masked.ffill()
        med = iv_imp.median(axis=1)
        iv_imp = iv_imp.apply(lambda col: col.fillna(med))
        iv_imp = mask_pre_listing(iv_imp, listing, inclusive=False)
        optvol_info["observed_frac"] = float(observed.values.mean())
        t1 = time.time()
        optvol_scale = build_option_vol_scale(raw_usd, iv_imp, observed_mask=observed)
        optvol_info["build_sec"] = round(time.time() - t1, 1)
        optvol_info["nontrivial_frac"] = float((optvol_scale.values != 1.0).mean())
        optvol_info["scale_q"] = [float(x) for x in np.nanpercentile(optvol_scale.values, [1, 25, 50, 75, 99])]
    pickle.dump({"bdays": bdays, "raw_usd": raw_usd, "dense_usd": dense_usd, "mc": mc, "sector_map": sector_map,
                 "listing": listing, "optvol_scale": optvol_scale, "optvol_info": optvol_info, "cur": cur.to_dict()},
                open(INP, "wb"), protocol=4)
    print("prep done", json.dumps(optvol_info))
    print("raw_usd shape", raw_usd.shape, "nan frac", float(raw_usd.isna().values.mean()))
    sys.exit(0)

# ------------------------------------------------------------------ run
from src.portfolio_optimizer import (estimate_covariance, optimize_portfolio, project_portfolio_weights,
                                     _name_risk_shares, NAME_RISK_CAP_TOL)
from src.backtest import compute_signal_confidence, apply_dynamic_execution, _drift_weights, _listing_eligibility_mask
inp = pickle.load(open(INP, "rb"))
bdays, raw_usd, dense_usd, mc, sector_map, listing, optvol_scale = (inp[k] for k in
    ["bdays", "raw_usd", "dense_usd", "mc", "sector_map", "listing", "optvol_scale"])
print("inputs loaded; optvol", inp["optvol_info"])
cap = float(cfg.max_name_active_risk_share)
cov_lookback = int(cfg.cov_lookback)
mega_thr = float(cfg.mega_cap_bm_threshold); funding_k = int(cfg.mega_cap_funding_k)
funding_smax = float(cfg.mega_cap_funding_score_max); floor_mult = float(cfg.bm_weight_floor)
thr = float(getattr(cfg, "score_threshold_for_ow", 0.0))
ic = r.ic_series.sort_index()
dw = r.daily_weights; sim_dates = sorted(dw.keys())
anchors = [pd.Timestamp(x) for x in ["2018-11-26", "2018-12-25", "2019-11-13", "2020-04-08", "2024-11-08", "2026-04-22", "2026-05-21", "2026-06-19", "2026-08-18"]]
if SUBSET == "anchors":
    dates = anchors
elif SUBSET.startswith("range"):
    _, a_, b_ = SUBSET.split(":")
    dates = rebal_dates[int(a_):int(b_)]
else:
    dates = rebal_dates
rows = []
TAG = SUBSET.replace(":", "_")
csv_path = os.path.join(V, "W1_recon_rows_%s.csv" % TAG)
for d in dates:
    t_run = time.time()
    t_idx = bdays.get_loc(d)
    hist = raw_usd.iloc[t_idx - cov_lookback:t_idx]
    # benchmark
    elig = _listing_eligibility_mask(d, tickers, listing)
    row = mc.loc[d].values.astype(float)
    row = np.where(np.isfinite(row) & (row > 0) & elig, row, 0.0)
    bm_w = row / row.sum()
    cov = np.asarray(estimate_covariance(hist, bm_weights=bm_w, config=cfg), dtype=float)
    optvol_applied = False
    if optvol_scale is not None and d in optvol_scale.index:
        s = optvol_scale.loc[d].reindex(tickers).fillna(1.0).to_numpy(dtype=float)
        cov = np.diag(s) @ cov @ np.diag(s); optvol_applied = True
    # executed book + Euler shares (T-03)
    w_exec = r.portfolio_weights[d].reindex(tickers).fillna(0.0).to_numpy(dtype=float)
    shares, var = _name_risk_shares(w_exec, bm_w, cov)
    i_top = int(np.argmax(shares)); max_share = float(shares[i_top])
    abs_top = float(np.max(np.abs(shares)))
    te_exec = float(np.sqrt(max(var, 0.0) * 252.0))
    as_exec = 0.5 * float(np.abs(w_exec - bm_w).sum())
    as_pkl = float(r.active_share_series.get(d, np.nan))
    # entering (prev) book
    si = sim_dates.index(d)
    if si == 0:
        prev = bm_w.copy()
    else:
        prev = _drift_weights(dw[sim_dates[si - 1]].reindex(tickers).fillna(0.0).to_numpy(dtype=float), dense_usd.loc[d].to_numpy(dtype=float))
    l1_exec_prev = float(np.abs(w_exec - prev).sum())
    turn_pkl = float(r.turnover.get(d, np.nan))
    # T-07 forced trades
    mu_s = r.predictions.loc[d, tickers].astype(float)
    mu = mu_s.to_numpy(); nan_mu = ~np.isfinite(mu); mu0 = np.where(nan_mu, 0.0, mu)
    act_prev = prev - bm_w
    gate_sell = float(np.sum(np.where((~nan_mu) & (mu0 <= thr) & (act_prev > 0), act_prev, 0.0)))
    nan_pin = float(np.sum(np.abs(act_prev[nan_mu])))
    mega = [i for i in range(len(tickers)) if bm_w[i] >= mega_thr]
    scored = sorted([(i, mu0[i]) for i in mega if mu0[i] < funding_smax], key=lambda x: x[1])
    funding = {i for i, _ in scored[:funding_k]}
    mega_sell = float(sum(max(act_prev[i], 0.0) for i in funding))
    mega_buy = float(sum(max(-act_prev[i], 0.0) for i in mega if i not in funding))
    floor_buy = float(np.sum(np.maximum(bm_w * floor_mult - prev, 0.0)))
    forced = gate_sell + nan_pin + mega_sell + mega_buy + floor_buy
    n_gate = int(np.sum((~nan_mu) & (mu0 <= thr) & (act_prev > 1e-6)))
    # T-05 re-solve MVO target
    diag = {}
    try:
        target = optimize_portfolio(mu_s, cov, prev, sector_map, bm_w, config=cfg, diagnostics=diag)
        sh_t, var_t = _name_risk_shares(target, bm_w, cov)
        max_share_target = float(np.max(sh_t)) if var_t > 0 else 0.0
        l1_target_prev = float(np.abs(target - prev).sum())
    except Exception as e:
        target = None; max_share_target = np.nan; l1_target_prev = np.nan; diag["exception"] = repr(e)[:200]
    # projection stage replication (fidelity check vs executed book)
    pdiag = {}; l1_proj_exec = np.nan; conf = np.nan; max_share_proj = np.nan; l1_proj_prev = np.nan
    if target is not None:
        try:
            prior_ic = ic[ic.index < d]
            trailing = float(np.nanmean(prior_ic.iloc[-int(cfg.trailing_ic_window):])) if len(prior_ic) >= 2 else 0.0
            raw_row = r.raw_predictions.loc[d, tickers] if d in r.raw_predictions.index else None
            conf = compute_signal_confidence(mu_s, raw_row, trailing, spread_scale=float(getattr(cfg, "confidence_spread_scale", 0.20)))
            cand = apply_dynamic_execution(prev.copy(), target.copy(), conf, cfg)
            proj = project_portfolio_weights(candidate_weights=cand, expected_returns=mu_s, cov_matrix=cov, prev_weights=prev,
                                             sector_map=sector_map, bm_weights=bm_w, max_te_annual=cfg.max_te_annual,
                                             sector_deviation=cfg.sector_deviation, config=cfg, fallback_weights=target, diagnostics=pdiag)
            proj = np.asarray(proj, dtype=float)
            l1_proj_exec = float(np.abs(proj - w_exec).sum()); l1_proj_prev = float(np.abs(proj - prev).sum())
            sh_p, var_p = _name_risk_shares(proj, bm_w, cov)
            max_share_proj = float(np.max(sh_p)) if var_p > 0 else 0.0
        except Exception as e:
            pdiag["exception"] = repr(e)[:200]
    rec = {"date": str(d.date()), "optvol_applied": optvol_applied, "max_share_exec": max_share, "top_exec": tickers[i_top],
           "abs_max_share_exec": abs_top, "te_exec_exante": te_exec, "as_exec": as_exec, "as_pkl": as_pkl, "as_absdiff": abs(as_exec - as_pkl),
           "l1_exec_prev": l1_exec_prev, "turnover_pkl": turn_pkl, "l1_vs_pkl_turn_diff": l1_exec_prev - turn_pkl,
           "forced_l1": forced, "gate_sell": gate_sell, "nan_pin": nan_pin, "mega_sell": mega_sell, "mega_buy": mega_buy, "floor_buy": floor_buy,
           "n_gate_forced_names": n_gate, "n_nan_mu": int(nan_mu.sum()), "n_mega": len(mega), "n_funding": len(funding),
           "mvo_status": diag.get("status"), "mvo_used_fallback": diag.get("used_fallback"), "mvo_solver": diag.get("solver"),
           "mvo_cap_iter": diag.get("name_risk_cap_iterations"), "mvo_cap_max_share": diag.get("name_risk_cap_max_share"),
           "mvo_cap_converged": diag.get("name_risk_cap_converged"), "mvo_max_share_target": max_share_target, "l1_target_prev": l1_target_prev,
           "confidence": conf, "proj_status": pdiag.get("status"), "proj_used_fallback": pdiag.get("used_fallback"),
           "proj_cap_iter": pdiag.get("name_risk_cap_iterations"), "proj_cap_max_share": pdiag.get("name_risk_cap_max_share"),
           "proj_cap_converged": pdiag.get("name_risk_cap_converged"), "max_share_proj": max_share_proj, "l1_proj_prev": l1_proj_prev,
           "l1_proj_vs_exec": l1_proj_exec, "mvo_exc": diag.get("exception"), "proj_exc": pdiag.get("exception"), "sec": round(time.time() - t_run, 1)}
    rows.append(rec)
    print(json.dumps(rec, default=str))
    pd.DataFrame(rows).to_csv(csv_path, index=False)
df = pd.DataFrame(rows)
summary = {"subset": SUBSET, "n": len(df), "optvol": inp["optvol_info"],
           "as_absdiff_max": float(df["as_absdiff"].max()), "as_absdiff_median": float(df["as_absdiff"].median()),
           "l1_vs_pkl_turn_diff_maxabs": float(df["l1_vs_pkl_turn_diff"].abs().max()),
           "te_exec_exante_max": float(df["te_exec_exante"].max()), "te_exec_exante_median": float(df["te_exec_exante"].median()),
           "n_max_share_exec_gt_0.35": int((df["max_share_exec"] > 0.35).sum()),
           "n_max_share_exec_in_(0.35,0.36]": int(((df["max_share_exec"] > 0.35) & (df["max_share_exec"] <= 0.36 + 1e-9)).sum()),
           "n_max_share_exec_gt_0.36": int((df["max_share_exec"] > 0.36 + 1e-9).sum()),
           "n_abs_share_exec_gt_0.35": int((df["abs_max_share_exec"] > 0.35).sum()),
           "max_share_exec_top5": df.nlargest(5, "max_share_exec")[["date", "max_share_exec", "top_exec"]].values.tolist(),
           "forced_l1_max": float(df["forced_l1"].max()), "forced_l1_argmax": df.loc[df["forced_l1"].idxmax(), "date"],
           "forced_l1_q": [float(x) for x in df["forced_l1"].quantile([0.5, 0.9, 0.99]).values],
           "n_forced_gt_0.12": int((df["forced_l1"] > 0.12).sum()), "n_forced_gt_0.15": int((df["forced_l1"] > 0.15).sum()),
           "forced_2020-04-08": (df.set_index("date")["forced_l1"].get("2020-04-08")),
           "mvo_cap_iter_dist": df["mvo_cap_iter"].value_counts(dropna=False).to_dict(),
           "mvo_cap_nonconverged": int((df["mvo_cap_converged"] == False).sum()),
           "proj_cap_iter_dist": df["proj_cap_iter"].value_counts(dropna=False).to_dict(),
           "proj_cap_nonconverged": int((df["proj_cap_converged"] == False).sum()),
           "mvo_status_counts": df["mvo_status"].value_counts(dropna=False).to_dict(),
           "proj_status_counts": df["proj_status"].value_counts(dropna=False).to_dict(),
           "n_mvo_fallback": int((df["mvo_used_fallback"] == True).sum()), "n_proj_fallback": int((df["proj_used_fallback"] == True).sum()),
           "l1_proj_vs_exec_median": float(df["l1_proj_vs_exec"].median()), "l1_proj_vs_exec_max": float(df["l1_proj_vs_exec"].max()),
           "l1_target_prev_max": float(df["l1_target_prev"].max()), "l1_target_prev_binding_n(>=0.1499)": int((df["l1_target_prev"] >= 0.1499).sum()),
           "l1_proj_prev_max": float(df["l1_proj_prev"].max()), "l1_proj_prev_binding_n(>=0.1499)": int((df["l1_proj_prev"] >= 0.1499).sum())}
json.dump(summary, open(os.path.join(V, "W1_recon_%s.json" % TAG), "w"), indent=1, default=str)
print("SUMMARY", json.dumps(summary, indent=1, default=str))
