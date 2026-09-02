"""M3 part 2 - direct tests of the 'ticker dummy' mechanism (scratch-only).
A. split-threshold distribution of ROE-family features in the 13 unique boosters
B. between-ticker variance share for all 65 panel features (context)
C. per-unique-model pred_contrib sign consistency for the pinned names
"""
import sys, pickle, time, json
sys.path.insert(0, r"C:/Users/westl/PycharmProjects/pythonProject/venv_vf_new/machine/re_study/c2/ai_port")
import numpy as np, pandas as pd
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
SP = r"C:/Users/westl/AppData/Local/Temp/claude/C--Users-westl-PycharmProjects-pythonProject-venv-vf-new-machine-re-study-c2-ai-port/0caf960f-9660-4a4b-a91e-809a2f4ab53d/scratchpad/w3/verify"
OUT = {}
t0 = time.time()
with open(r"C:/Users/westl/PycharmProjects/pythonProject/venv_vf_new/machine/re_study/c2/ai_port/outputs/s16_7_name_risk_cap/backtest_result.pkl", "rb") as fh:
    R = pickle.load(fh)
TG = R.targets; PR = R.predictions; panel = R.panel
dates = TG.index; tickers = list(TG.columns)
elig = TG.notna()
valid = elig.sum(axis=1) >= 150
m_all = elig & valid.values[:, None]
FOI = ["best_roe_level_z", "fin_roe_pb_gap", "fin_roe_pe_gap", "best_px_bps_ratio_level_z"]
PINNED = ["HCA", "MCD", "MSCI", "SBUX", "VRSN", "MSI", "LYV", "MAR"]
NEGEQ = ["PLTR", "MCD", "PM", "SBUX", "VRSN", "MSCI", "HCA", "RDDT"]

# unique boosters
uniq = {}
for k in sorted(R.models):
    uniq.setdefault(id(R.models[k]), (k, R.models[k]))
print("unique boosters:", len(uniq))

# ---- A. split thresholds ----
print("\n[A] split thresholds on ROE-family features (all unique boosters pooled, gain-weighted)")
feat_panels = {f: panel[f].unstack("ticker").reindex(index=dates, columns=tickers).where(m_all) for f in FOI}
rowsA = []
for f in FOI:
    thr_all = []; gain_all = []
    for oid, (k, m) in uniq.items():
        af = list(m._active_features)
        if f not in af:
            continue
        col = "Column_%d" % af.index(f)
        df = m.booster_.trees_to_dataframe()
        s = df[df.split_feature == col]
        thr_all += s.threshold.astype(float).tolist(); gain_all += s.split_gain.astype(float).tolist()
    thr = np.array(thr_all); g = np.array(gain_all)
    if len(thr) == 0:
        continue
    X = feat_panels[f]
    # share of eligible cells beyond each threshold (how many names a split isolates), gain-weighted
    frac_beyond = []
    for t in thr:
        fb = (X <= t).mean(axis=1).median() if t < 0 else (X > t).mean(axis=1).median()
        frac_beyond.append(min(fb, 1 - fb))
    frac_beyond = np.array(frac_beyond)
    w = g / g.sum()
    rowsA.append({"feature": f, "n_splits": len(thr), "thr_p10": np.percentile(thr, 10), "thr_p50": np.percentile(thr, 50), "thr_p90": np.percentile(thr, 90),
                  "gainw_share_|thr|>=1.5": w[np.abs(thr) >= 1.5].sum(), "gainw_share_|thr|<0.5": w[np.abs(thr) < 0.5].sum(),
                  "gainw_median_minor_side_frac": float(np.sum(w * frac_beyond)), "share_splits_isolating_<=5%": float(w[frac_beyond <= 0.05].sum())})
A = pd.DataFrame(rowsA).set_index("feature")
print(A.round(3).to_string())
for f in A.index:
    OUT[f"splits_{f}_gainw_share_abs_thr_ge_1p5"] = round(float(A.loc[f, "gainw_share_|thr|>=1.5"]), 3)
    OUT[f"splits_{f}_gainw_share_isolating_le5pct"] = round(float(A.loc[f, "share_splits_isolating_<=5%"]), 3)

# ---- B. between-ticker variance share for all features ----
print("\n[B] between-ticker variance share (ticker fixed effect) per feature, eligible cells 2015+")
sub = panel.loc[(slice(pd.Timestamp("2015-01-13"), None), slice(None)), :]
mask_long = m_all.stack()
mask_long = mask_long[mask_long]
sub = sub.loc[sub.index.intersection(mask_long.index)]
tot_var = sub.var()
tick_mean = sub.groupby(level="ticker").transform("mean")
between = tick_mean.var()
share = (between / tot_var).sort_values(ascending=False)
print("  top 15:", share.head(15).round(3).to_dict())
print("  ROE family:", {f: round(float(share[f]), 3) for f in FOI + ["fin_roe_level_z", "fin_pb_level_z"] if f in share})
print("  median across 65 features: %.3f ; rank of best_roe_level_z = %d/%d ; fin_roe_pb_gap = %d" % (share.median(), int(share.rank(ascending=False)["best_roe_level_z"]), len(share), int(share.rank(ascending=False)["fin_roe_pb_gap"])))
OUT["between_share_best_roe_level_z"] = round(float(share["best_roe_level_z"]), 3)
OUT["between_share_fin_roe_pb_gap"] = round(float(share["fin_roe_pb_gap"]), 3)
OUT["between_share_median_65"] = round(float(share.median()), 3)
OUT["between_share_rank_best_roe_level_z"] = int(share.rank(ascending=False)["best_roe_level_z"])
# within-ticker std of best_roe_level_z for the 240 vs extreme names
Z = feat_panels["best_roe_level_z"]
wstd = Z.std()
print("  within-ticker std of best_roe_level_z: median %.3f ; n tickers < 0.15 = %d ; cross-sectional std median %.3f" % (wstd.median(), int((wstd < 0.15).sum()), Z.std(axis=1).median()))

# ---- C. pred_contrib sign consistency across unique models for pinned/NEGEQ names ----
print("\n[C] pred_contrib of fin_roe_pb_gap / best_roe_level_z for pinned names, one date per unique booster")
rowsC = []
for oid, (k, m) in uniq.items():
    af = list(m._active_features)
    ds = [d for d in dates if d >= k and d in PR.index and PR.loc[d].notna().sum() >= 150][:1]
    if not ds:
        continue
    d = ds[0]
    X = panel.loc[d].reindex(columns=af)
    el = PR.loc[d].notna()
    X = X.loc[el[el].index.intersection(X.index)]
    C = pd.DataFrame(m.booster_.predict(X.values, pred_contrib=True)[:, :-1], index=X.index, columns=af)
    tot = C.sum(axis=1)
    pin = [t for t in PINNED if t in C.index]; neg = [t for t in NEGEQ if t in C.index]
    rowsC.append({"model": k.date(), "date": d.date(), "score_std": tot.std(),
                  "pb_gap_pinned": C.loc[pin, "fin_roe_pb_gap"].mean(), "pb_gap_uni_abs": C["fin_roe_pb_gap"].abs().mean(),
                  "roe_z_negeq": C.loc[neg, "best_roe_level_z"].mean(), "roe_z_uni_abs": C["best_roe_level_z"].abs().mean(),
                  "pe_gap_negeq": C.loc[neg, "fin_roe_pe_gap"].mean(),
                  "family_negeq": C.loc[neg, [f for f in FOI if f in C]].sum(axis=1).mean(),
                  "family_negeq_in_score_sd": C.loc[neg, [f for f in FOI if f in C]].sum(axis=1).mean() / tot.std(),
                  "negeq_total_pct": float((tot.rank(pct=True).loc[neg]).mean())})
Cdf = pd.DataFrame(rowsC)
print(Cdf.round(4).to_string())
print("  sign of pb_gap contribution for pinned names across unique models: +%d / -%d" % (int((Cdf.pb_gap_pinned > 0).sum()), int((Cdf.pb_gap_pinned < 0).sum())))
print("  family contribution for NEGEQ in units of score sd: mean %.3f, max |.| %.3f" % (Cdf.family_negeq_in_score_sd.mean(), Cdf.family_negeq_in_score_sd.abs().max()))
OUT["pbgap_pinned_sign_pos_models"] = int((Cdf.pb_gap_pinned > 0).sum()); OUT["pbgap_pinned_sign_neg_models"] = int((Cdf.pb_gap_pinned < 0).sum())
OUT["family_negeq_in_score_sd_mean"] = round(float(Cdf.family_negeq_in_score_sd.mean()), 3)
OUT["family_negeq_in_score_sd_maxabs"] = round(float(Cdf.family_negeq_in_score_sd.abs().max()), 3)
with open(SP + "/M3_numbers_part2.json", "w") as fh:
    json.dump(OUT, fh, indent=1, default=str)
print("\nDONE", round(time.time() - t0, 1), "s")
