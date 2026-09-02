"""M3_negative_equity - empirical refutation attempt (scratch-only).
Uses cached sheets from M3_cache_sheets.py and the S0' pkl.
"""
import sys, pickle, time, json
sys.path.insert(0, r"C:/Users/westl/PycharmProjects/pythonProject/venv_vf_new/machine/re_study/c2/ai_port")
import numpy as np, pandas as pd
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
SP = r"C:/Users/westl/AppData/Local/Temp/claude/C--Users-westl-PycharmProjects-pythonProject-venv-vf-new-machine-re-study-c2-ai-port/0caf960f-9660-4a4b-a91e-809a2f4ab53d/scratchpad/w3/verify"
CACHE = SP + "/cache"
OUT = {}


def ld(n):
    with open(f"{CACHE}/{n}.pkl", "rb") as fh:
        return pickle.load(fh)


t0 = time.time()
with open(r"C:/Users/westl/PycharmProjects/pythonProject/venv_vf_new/machine/re_study/c2/ai_port/outputs/s16_7_name_risk_cap/backtest_result.pkl", "rb") as fh:
    R = pickle.load(fh)
print("pkl loaded", round(time.time() - t0, 1), "s")
TG = R.targets; PR = R.predictions; PW = R.portfolio_weights
dates = TG.index; tickers = list(TG.columns)
panel = R.panel

roe = ld("BEST_ROE").reindex(index=dates, columns=tickers).ffill()
pb = ld("BEST_PX_BPS_RATIO").reindex(index=dates, columns=tickers).ffill()
pe = ld("BEST_PE_RATIO").reindex(index=dates, columns=tickers).ffill()
eps = ld("BEST_EPS").reindex(index=dates, columns=tickers).ffill()
mc = ld("CUR_MKT_CAP").reindex(index=dates, columns=tickers).ffill()
elig = TG.notna()  # eligibility mask (listed + target available)

# ---------------- 1. reproduce probe (2014+) ----------------
rep = pd.DataFrame({
    "roe_med": roe.median(), "roe_abs_gt100_pct": (roe.abs() > 100).mean() * 100,
    "roe_neg_pct": (roe < 0).mean() * 100, "pb_med": pb.median(),
    "pb_gt50_pct": (pb > 50).mean() * 100, "pb_nan_pct": pb.isna().mean() * 100,
    "eps_pos_pct": (eps > 0).mean() * 100,
})
print("\n[1] ROE unit check: universe median of ticker-median ROE =", round(rep.roe_med.median(), 2), "(percent units)")
rep["flag_roe_extreme"] = rep.roe_abs_gt100_pct > 25
rep["flag_negeq_profitable"] = (rep.roe_neg_pct > 50) & (rep.eps_pos_pct > 90)
rep["flag_pb_hi"] = rep.pb_gt50_pct > 40
flag_any = rep[rep.flag_roe_extreme | rep.flag_negeq_profitable | rep.flag_pb_hi]
print(flag_any.sort_values("roe_med", ascending=False).round(2).to_string())
EXT10 = rep.roe_med.abs().sort_values(ascending=False).index[:10].tolist()
NEGEQ = rep.index[rep.flag_negeq_profitable].tolist()
EXT_ALL = sorted(set(rep.index[rep.flag_roe_extreme | rep.flag_negeq_profitable | rep.flag_pb_hi]))
print("EXT10 (top10 |median ROE|):", EXT10)
print("NEGEQ (ROE<0 >50% dates & EPS>0 >90%):", NEGEQ)
print("EXT_ALL n=", len(EXT_ALL), EXT_ALL)
OUT["ext10"] = EXT10; OUT["negeq"] = NEGEQ; OUT["ext_all_n"] = len(EXT_ALL)
ext_cell = (roe.abs() > 100) | ((roe < 0) & (eps > 0)) | (pb > 50) | (pb < 0)
ext_cell_roe = (roe.abs() > 100) | ((roe < 0) & (eps > 0))
print("extreme-cell share among eligible cells: any=%.3f roe-only=%.3f" % (
    (ext_cell & elig).sum().sum() / elig.sum().sum(), (ext_cell_roe & elig).sum().sum() / elig.sum().sum()))

# ---------------- 2a. compression ----------------
def cs_std(df, mask):
    return df.where(mask).std(axis=1)


valid_dates = elig.sum(axis=1) >= 150
m_all = elig & valid_dates.values[:, None]
def tick_mask(S):
    return pd.DataFrame(np.isin(tickers, S)[None, :].repeat(len(dates), 0), index=dates, columns=tickers)
m_ex10 = m_all & ~tick_mask(EXT10)
m_exall = m_all & ~tick_mask(EXT_ALL)
m_excell = m_all & ~ext_cell_roe
s_all = cs_std(roe, m_all); s_ex10 = cs_std(roe, m_ex10); s_exall = cs_std(roe, m_exall); s_excell = cs_std(roe, m_excell)
r10 = (s_all / s_ex10).dropna(); rall = (s_all / s_exall).dropna(); rcell = (s_all / s_excell).dropna()
print("\n[2a] raw ROE cs-std ratio (all / excl.) per date:")
for nm, r in [("excl EXT10", r10), ("excl EXT_ALL(%d)" % len(EXT_ALL), rall), ("excl extreme cells(date-varying)", rcell)]:
    print("  %-34s median %.2f  p10 %.2f  p90 %.2f  CV %.3f" % (nm, r.median(), r.quantile(.1), r.quantile(.9), r.std() / r.mean()))
OUT["raw_std_ratio_ex10_median"] = round(float(r10.median()), 3); OUT["raw_std_ratio_ex10_cv"] = round(float(r10.std() / r10.mean()), 3)
OUT["raw_std_ratio_excell_median"] = round(float(rcell.median()), 3)
Z = panel["best_roe_level_z"].unstack("ticker").reindex(index=dates, columns=tickers)
z_all = cs_std(Z, m_all); z_ex10 = cs_std(Z, m_ex10); z_excell = cs_std(Z, m_excell)
print("  panel best_roe_level_z cs-std: all median %.3f | excl EXT10 %.3f | excl extreme cells %.3f  -> compression %.0f%% / %.0f%%" % (
    z_all.median(), z_ex10.median(), z_excell.median(), 100 * (1 - z_ex10.median() / z_all.median()), 100 * (1 - z_excell.median() / z_all.median())))
OUT["z_std_all"] = round(float(z_all.median()), 3); OUT["z_std_ex10"] = round(float(z_ex10.median()), 3); OUT["z_std_excell"] = round(float(z_excell.median()), 3)
iqr240 = (Z.where(m_ex10).quantile(.75, axis=1) - Z.where(m_ex10).quantile(.25, axis=1))
print("  IQR of z among non-EXT10 names: median %.3f ; share of |z|>=4.99 (clipped) per date median %.3f" % (iqr240.median(), (Z.abs() >= 4.99).where(m_all).mean(axis=1).median()))
yr = r10.groupby(r10.index.year).median()
print("  ratio excl EXT10 by year:", yr.round(2).to_dict())
print("  ticker median z (panel) for EXT10:", Z[EXT10].median().round(2).to_dict())
print("  ticker median z (panel) for NEGEQ:", Z[NEGEQ].median().round(2).to_dict())

# ---------------- 2b. predictions vs targets ----------------
pm = PR.notna() & TG.notna()
pdates = pm.sum(axis=1) >= 150
pmask = pm & pdates.values[:, None]
pred_pct = PR.where(pmask).rank(axis=1, pct=True); tgt_pct = TG.where(pmask).rank(axis=1, pct=True)
tk = pd.DataFrame({"pred_pct": pred_pct.mean(), "tgt_pct": tgt_pct.mean(), "n": pmask.sum(),
                   "pred_mean": PR.where(pmask).mean(), "pred_within_std": PR.where(pmask).std()})
tk["gap"] = tk.tgt_pct - tk.pred_pct  # >0: model under-ranks the name relative to realised
tk["ts_corr"] = [pred_pct[t].corr(tgt_pct[t], method="spearman") for t in tickers]
uni = tk[tk.n >= 500]
print("\n[2b] per-ticker mean percentile (pred vs realised target), universe n=%d" % len(uni))
print("  universe: gap mean %.3f std %.3f | |pred_mean| median %.3f | within-ticker pred std median %.3f" % (uni.gap.mean(), uni.gap.std(), uni.pred_mean.abs().median(), uni.pred_within_std.median()))
for nm, S in [("EXT10", EXT10), ("NEGEQ", NEGEQ), ("EXT_ALL", EXT_ALL)]:
    g = uni.loc[[t for t in S if t in uni.index]]
    print("  %-8s n=%2d pred_pct %.3f tgt_pct %.3f gap %.3f (z vs universe %.2f) |pred_mean| %.3f within_std %.3f ts_corr %.3f" % (
        nm, len(g), g.pred_pct.mean(), g.tgt_pct.mean(), g.gap.mean(), (g.gap.mean() - uni.gap.mean()) / (uni.gap.std() / np.sqrt(len(g))), g.pred_mean.abs().median(), g.pred_within_std.median(), g.ts_corr.mean()))
    OUT[f"{nm}_pred_pct"] = round(float(g.pred_pct.mean()), 3); OUT[f"{nm}_tgt_pct"] = round(float(g.tgt_pct.mean()), 3); OUT[f"{nm}_gap"] = round(float(g.gap.mean()), 3)
print(uni.loc[[t for t in EXT_ALL if t in uni.index]].sort_values("pred_pct").round(3).to_string())
print("  universe pred_pct percentile of NEGEQ names:", {t: round(float((uni.pred_pct < uni.pred_pct[t]).mean()), 2) for t in NEGEQ if t in uni.index})

# ---------------- 2c-i. gain share ----------------
FOI = ["best_roe_level_z", "fin_roe_level_z", "fin_roe_pb_gap", "fin_roe_pe_gap", "best_px_bps_ratio_level_z", "fin_pb_level_z", "best_roe_chg_63d", "fin_roe_chg_63d", "fin_roe_chg_252d"]
rows = []
for k in sorted(R.models):
    m = R.models[k]; af = list(m._active_features)
    g = m.booster_.feature_importance(importance_type="gain"); g = g / g.sum()
    s = pd.Series(g, index=af); rk = s.rank(ascending=False)
    rows.append({"model": k.date(), **{f: (round(float(s.get(f, np.nan)), 4)) for f in FOI}, **{f + "_rank": (int(rk.get(f, -1)) if f in s else -1) for f in FOI}})
gs = pd.DataFrame(rows).set_index("model")
print("\n[2c-i] gain share of ROE/PB features per model (share, rank among active):")
print(gs[FOI].describe().loc[["mean", "50%", "min", "max"]].round(4).to_string())
print("  rank median:", {f: int(gs[f + "_rank"].replace(-1, np.nan).median()) for f in FOI if gs[f + "_rank"].replace(-1, np.nan).notna().any()})
print("  models where feature inactive (dropped):", {f: int((gs[f + "_rank"] == -1).sum()) for f in FOI})
print("  combined ROE/PB-family gain share mean = %.4f" % gs[FOI].fillna(0).sum(axis=1).mean())
OUT["gain_share_best_roe_level_z_mean"] = round(float(gs["best_roe_level_z"].fillna(0).mean()), 4)
OUT["gain_share_fin_roe_pb_gap_mean"] = round(float(gs["fin_roe_pb_gap"].fillna(0).mean()), 4)
OUT["gain_share_fin_roe_pe_gap_mean"] = round(float(gs["fin_roe_pe_gap"].fillna(0).mean()), 4)
OUT["gain_share_family_mean"] = round(float(gs[FOI].fillna(0).sum(axis=1).mean()), 4)

# ---------------- 2c-ii. pred_contrib on extreme names ----------------
print("\n[2c-ii] LightGBM pred_contrib (SHAP-like) of ROE family for extreme names vs universe")


def contrib_on(date_key, sample_dates):
    m = R.models[date_key]; af = list(m._active_features)
    out = []
    for d in sample_dates:
        X = panel.loc[d].reindex(columns=af)
        el = PR.loc[d].notna() if (d in PR.index and PR.loc[d].notna().sum() >= 150) else elig.loc[d]
        X = X.loc[el[el].index.intersection(X.index)]
        if len(X) == 0:
            continue
        C = m.booster_.predict(X.values, pred_contrib=True)
        C = pd.DataFrame(C[:, :-1], index=X.index, columns=af)
        out.append(C)
    return pd.concat(out, keys=sample_dates)


ctr_rows = []
for mk in [pd.Timestamp("2020-05-06"), pd.Timestamp("2022-07-08"), pd.Timestamp("2024-12-06"), pd.Timestamp("2026-08-17")]:
    if mk not in R.models:
        continue
    ds = [d for d in dates if d >= mk and d in PR.index and PR.loc[d].notna().sum() >= 150][:60:20][:3]
    if not ds:
        continue
    C = contrib_on(mk, ds)
    tot = C.sum(axis=1)
    fam = [f for f in FOI if f in C.columns]
    sub = C[fam].sum(axis=1)
    for nm, S in [("EXT10", EXT10), ("NEGEQ", NEGEQ)]:
        idx = [i for i in C.index if i[1] in S]
        ctr_rows.append({"model": mk.date(), "set": nm, "fam_contrib_mean": sub.loc[idx].mean(), "fam_contrib_absmean": sub.loc[idx].abs().mean(),
                         "uni_fam_absmean": sub.abs().mean(), "total_score_std": tot.std(),
                         "roe_z_contrib_mean": C.loc[idx, "best_roe_level_z"].mean() if "best_roe_level_z" in C else np.nan,
                         "pb_gap_contrib_mean": C.loc[idx, "fin_roe_pb_gap"].mean() if "fin_roe_pb_gap" in C else np.nan,
                         "pe_gap_contrib_mean": C.loc[idx, "fin_roe_pe_gap"].mean() if "fin_roe_pe_gap" in C else np.nan,
                         "total_mean_set": tot.loc[idx].mean(), "total_mean_uni": tot.mean()})
ctr = pd.DataFrame(ctr_rows)
print(ctr.round(4).to_string())
OUT["contrib_negeq_family_mean"] = round(float(ctr[ctr.set == "NEGEQ"].fam_contrib_mean.mean()), 4)
OUT["contrib_ext10_family_mean"] = round(float(ctr[ctr.set == "EXT10"].fam_contrib_mean.mean()), 4)
OUT["contrib_total_score_std"] = round(float(ctr.total_score_std.mean()), 4)

# ---------------- 2c-iii. IC of alternatives ----------------
def cs_corr(A, B, mask, method="spearman"):
    a = A.where(mask); b = B.where(mask)
    if method == "spearman":
        a = a.rank(axis=1); b = b.rank(axis=1)
    m2 = a.notna() & b.notna()
    a = a.where(m2); b = b.where(m2)
    am = a.sub(a.mean(axis=1), axis=0); bm = b.sub(b.mean(axis=1), axis=0)
    num = (am * bm).sum(axis=1); den = np.sqrt((am ** 2).sum(axis=1) * (bm ** 2).sum(axis=1))
    ic = num / den.replace(0, np.nan)
    ic[m2.sum(axis=1) < 100] = np.nan
    return ic


def summ(ic, label, base=None):
    ic = ic.dropna(); ic = ic[ic.index >= "2015-01-13"]
    sub = ic.iloc[::21]
    t = sub.mean() / sub.std() * np.sqrt(len(sub))
    line = "  %-44s IC mean %+.4f  t(stride21,n=%d) %+.2f" % (label, ic.mean(), len(sub), t)
    if base is not None:
        d = (ic - base.reindex(ic.index)).dropna(); ds = d.iloc[::21]
        line += "  | dIC vs base %+.4f paired t %+.2f" % (d.mean(), ds.mean() / ds.std() * np.sqrt(len(ds)))
    print(line); return ic


def medfill(df):
    return df.T.fillna(df.median(axis=1)).T


roe_e = roe.where(m_all)
roe_nan = roe_e.mask(ext_cell_roe)
roe_nanmed = medfill(roe_nan)
roe_adj = roe_e.mask((roe < 0) & (eps > 0), np.inf)
roe_adj2 = medfill(roe_e.mask(roe.abs() > 100, np.nan).mask((roe < 0) & (eps > 0), np.inf))
print("\n[2c-iii] Spearman IC vs 20d targets (all dates >=2015-01-13, eligible cells)")
ic0 = summ(cs_corr(Z, TG, m_all), "F0 panel best_roe_level_z")
summ(cs_corr(roe_e, TG, m_all), "F_raw ROE (sanity == F0)", ic0)
summ(cs_corr(roe_nanmed, TG, m_all), "F_nanmed extremes->per-date median", ic0)
summ(cs_corr(roe_e, TG, m_excell), "F0 evaluated on non-extreme cells only", ic0)
summ(cs_corr(roe_adj, TG, m_all), "F_adj neg-eq&EPS>0 -> top rank", ic0)
summ(cs_corr(roe_adj2, TG, m_all), "F_adj2 |ROE|>100->median, neg-eq->top", ic0)
icp0 = summ(cs_corr(Z, TG, m_all, "pearson"), "Pearson: panel z (clipped 5)", None)
summ(cs_corr(roe_e.rank(axis=1), TG, m_all, "pearson"), "Pearson: rank(ROE)", icp0)
lo = roe_e.quantile(.02, axis=1); hi = roe_e.quantile(.98, axis=1)
roe_w = roe_e.clip(lower=lo, upper=hi, axis=0)
summ(cs_corr(roe_w, TG, m_all, "pearson"), "Pearson: winsor 2/98 ROE", icp0)
OUT["ic_F0"] = round(float(ic0.mean()), 4)
# gap features
pb_e = pb.where(m_all)
G0 = panel["fin_roe_pb_gap"].unstack("ticker").reindex(index=dates, columns=tickers)
g_raw = medfill(roe_e.rank(axis=1, pct=True) - pb_e.rank(axis=1, pct=True))
pb_nanmed = medfill(pb_e.mask((pb > 50) | (pb < 0)))
g_nan = roe_nanmed.rank(axis=1, pct=True) - pb_nanmed.rank(axis=1, pct=True)
g_adj = medfill(roe_adj.rank(axis=1, pct=True) - pb_nanmed.rank(axis=1, pct=True))
print("  -- fin_roe_pb_gap --")
ig0 = summ(cs_corr(G0, TG, m_all), "G0 panel fin_roe_pb_gap")
summ(cs_corr(g_raw, TG, m_all), "G_raw rank(roe)-rank(pb) (sanity)", ig0)
summ(cs_corr(g_nan, TG, m_all), "G_nanmed extremes->median both legs", ig0)
summ(cs_corr(g_adj, TG, m_all), "G_adj neg-eq->top ROE, P/B>50->median", ig0)
summ(cs_corr(G0, TG, m_all & ~ext_cell), "G0 on non-extreme cells only", ig0)
OUT["ic_G0"] = round(float(ig0.mean()), 4)
pe_e = pe.where(m_all)
P0 = panel["fin_roe_pe_gap"].unstack("ticker").reindex(index=dates, columns=tickers)
p_raw = medfill(roe_e.rank(axis=1, pct=True) - pe_e.rank(axis=1, pct=True))
p_adj = medfill(roe_adj.rank(axis=1, pct=True) - pe_e.rank(axis=1, pct=True))
p_nan = medfill(roe_nanmed.rank(axis=1, pct=True) - pe_e.rank(axis=1, pct=True))
print("  -- fin_roe_pe_gap --")
ip0 = summ(cs_corr(P0, TG, m_all), "P0 panel fin_roe_pe_gap")
summ(cs_corr(p_raw, TG, m_all), "P_raw (sanity)", ip0)
summ(cs_corr(p_nan, TG, m_all), "P_nanmed ROE extremes->median", ip0)
summ(cs_corr(p_adj, TG, m_all), "P_adj neg-eq->top ROE", ip0)
OUT["ic_P0"] = round(float(ip0.mean()), 4)
tp_all = TG.where(m_all).rank(axis=1, pct=True)
print("  realised target percentile (all dates) NEGEQ:", tp_all[NEGEQ].mean().round(3).to_dict())
print("  realised target percentile (all dates) EXT10:", tp_all[EXT10].mean().round(3).to_dict())

# ---------------- 3. active weights ----------------
print("\n[3] active weight vs cap-weight (CUR_MKT_CAP, USD upstream) on 97 rebalance dates")
act_rows = []; w_rows = []
for d in sorted(PW):
    w = PW[d].reindex(tickers).fillna(0.0)
    el = PR.loc[d].notna() if d in PR.index else pd.Series(True, index=tickers)
    cap = mc.loc[d].where(el & (mc.loc[d] > 0), 0.0)
    bm = cap / cap.sum()
    act_rows.append(w - bm); w_rows.append(w)
ACT = pd.DataFrame(act_rows, index=sorted(PW)); WW = pd.DataFrame(w_rows, index=sorted(PW))
tw = pd.DataFrame({"act_mean": ACT.mean(), "act_neg_frac": (ACT < -1e-6).mean(), "w_mean": WW.mean(), "bm_mean": (WW - ACT).mean()})
print("  universe: share of names with mean active<0: %.2f ; mean |active| %.4f ; names held >0.1%% avg %.0f" % ((tw.act_mean < 0).mean(), tw.act_mean.abs().mean(), (WW > 0.001).sum(axis=1).mean()))
for nm, S in [("EXT10", EXT10), ("NEGEQ", NEGEQ), ("EXT_ALL", EXT_ALL)]:
    g = tw.loc[S]
    print("  %-8s mean active %+.4f (sum %+.4f) neg-frac %.2f ; mean w %.4f vs bm %.4f" % (nm, g.act_mean.mean(), g.act_mean.sum(), g.act_neg_frac.mean(), g.w_mean.mean(), g.bm_mean.mean()))
    OUT[f"{nm}_active_mean"] = round(float(g.act_mean.mean()), 4); OUT[f"{nm}_active_sum"] = round(float(g.act_mean.sum()), 4); OUT[f"{nm}_active_negfrac"] = round(float(g.act_neg_frac.mean()), 3)
print(tw.loc[EXT_ALL].sort_values("act_mean").round(4).to_string())
tw2 = tw.join(uni[["pred_pct", "tgt_pct"]], how="inner")
print("  corr(ticker mean pred_pct, mean active) universe = %.3f" % tw2.pred_pct.corr(tw2.act_mean))
contrib = (ACT * TG.reindex(ACT.index)).sum(axis=0)
print("  approx realised active P&L (sum over rebal dates of act*20d target): NEGEQ %+.4f EXT10 %+.4f EXT_ALL %+.4f | universe total %+.4f" % (contrib[NEGEQ].sum(), contrib[EXT10].sum(), contrib[EXT_ALL].sum(), contrib.sum()))
OUT["realised_active_pnl_negeq"] = round(float(contrib[NEGEQ].sum()), 4); OUT["realised_active_pnl_ext_all"] = round(float(contrib[EXT_ALL].sum()), 4); OUT["realised_active_pnl_total"] = round(float(contrib.sum()), 4)
with open(SP + "/M3_numbers.json", "w") as fh:
    json.dump(OUT, fh, indent=1, default=str)
print("\nDONE", round(time.time() - t0, 1), "s")
