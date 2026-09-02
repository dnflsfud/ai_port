import pandas as pd, numpy as np, json, os
V = os.environ["V"]
df = pd.concat([pd.read_csv(os.path.join(V, "W1_recon_rows_range_0_49.csv")), pd.read_csv(os.path.join(V, "W1_recon_rows_range_49_97.csv"))], ignore_index=True)
df.to_csv(os.path.join(V, "W1_recon_rows_all97.csv"), index=False)
n = len(df)
official = {"2026-06-19": 0.3594, "2019-11-13": 0.3579, "2026-04-22": 0.3577, "2018-12-25": 0.3564, "2026-05-21": 0.3559, "2026-08-18": 0.3439}
mine = df.set_index("date")["max_share_exec"]
cmp = {k: {"official": v, "mine": round(float(mine[k]), 4), "diff": round(float(mine[k] - v), 4)} for k, v in official.items()}
out = {"n": n,
 "fidelity": {"as_absdiff_max": float(df["as_absdiff"].max()), "l1_vs_pkl_turn_diff_maxabs": float(df["l1_vs_pkl_turn_diff"].abs().max()),
              "l1_proj_vs_exec_median": float(df["l1_proj_vs_exec"].median()), "l1_proj_vs_exec_q90": float(df["l1_proj_vs_exec"].quantile(0.9)),
              "l1_proj_vs_exec_max": float(df["l1_proj_vs_exec"].max()), "l1_proj_vs_exec_argmax": df.loc[df["l1_proj_vs_exec"].idxmax(), "date"],
              "n_l1_proj_vs_exec_gt_0.01": int((df["l1_proj_vs_exec"] > 0.01).sum()),
              "te_exec_exante_max": float(df["te_exec_exante"].max()), "n_te_exec_gt_0.0351": int((df["te_exec_exante"] > 0.0351).sum()),
              "official_worst5_vs_mine": cmp, "optvol_applied_n": int(df["optvol_applied"].sum())},
 "T03": {"n_max_share_exec_gt_0.35": int((df["max_share_exec"] > 0.35).sum()),
         "n_in_(0.35,0.36]": int(((df["max_share_exec"] > 0.35) & (df["max_share_exec"] <= 0.36 + 1e-9)).sum()),
         "n_gt_0.36": int((df["max_share_exec"] > 0.36 + 1e-9).sum()),
         "n_abs_share_gt_0.35": int((df["abs_max_share_exec"] > 0.35).sum()),
         "n_in_(0.345,0.35]": int(((df["max_share_exec"] > 0.345) & (df["max_share_exec"] <= 0.35)).sum()),
         "dates_gt_0.35": df[df["max_share_exec"] > 0.35][["date", "max_share_exec", "top_exec"]].round(4).values.tolist(),
         "top8": df.nlargest(8, "max_share_exec")[["date", "max_share_exec", "top_exec"]].round(4).values.tolist(),
         "max_share_exec_median": float(df["max_share_exec"].median()),
         "n_top_share_negative_dominant": int((df["abs_max_share_exec"] > df["max_share_exec"] + 1e-9).sum())},
 "T05": {"mvo_cap_iter_dist": {str(k): int(v) for k, v in df["mvo_cap_iter"].value_counts(dropna=False).sort_index().items()},
         "mvo_cap_nonconverged": int((df["mvo_cap_converged"] == False).sum()),
         "mvo_cap_max_share_max": float(df["mvo_cap_max_share"].max()),
         "proj_cap_iter_dist": {str(k): int(v) for k, v in df["proj_cap_iter"].value_counts(dropna=False).sort_index().items()},
         "proj_cap_nonconverged": int((df["proj_cap_converged"] == False).sum()),
         "proj_nonconverged_dates": df[df["proj_cap_converged"] == False][["date", "proj_cap_max_share", "max_share_exec", "l1_proj_vs_exec"]].round(4).values.tolist(),
         "proj_cap_max_share_max": float(df["proj_cap_max_share"].max()),
         "mvo_status": df["mvo_status"].value_counts(dropna=False).to_dict(), "proj_status": df["proj_status"].value_counts(dropna=False).to_dict(),
         "n_mvo_fallback": int((df["mvo_used_fallback"] == True).sum()), "n_proj_fallback": int((df["proj_used_fallback"] == True).sum()),
         "n_mvo_exception": int(df["mvo_exc"].notna().sum()), "n_proj_exception": int(df["proj_exc"].notna().sum()),
         "l1_target_prev_binding_n": int((df["l1_target_prev"] >= 0.1499).sum()), "l1_target_prev_min": float(df["l1_target_prev"].min()),
         "l1_proj_prev_binding_n": int((df["l1_proj_prev"] >= 0.1499).sum()), "l1_proj_prev_max": float(df["l1_proj_prev"].max()),
         "pkl_turnover_binding_n": int((df["turnover_pkl"] >= 0.1499).sum()), "pkl_turnover_max": float(df["turnover_pkl"].max()),
         "pkl_turnover_argmax": df.loc[df["turnover_pkl"].idxmax(), "date"],
         "proj_iter_ge1_n": int((df["proj_cap_iter"] >= 1).sum()), "mvo_iter_ge1_n": int((df["mvo_cap_iter"] >= 1).sum()),
         "sec_total": float(df["sec"].sum())},
 "T07": {"forced_l1_max": float(df["forced_l1"].max()), "forced_l1_argmax": df.loc[df["forced_l1"].idxmax(), "date"],
         "forced_l1_q50_90_99": [float(x) for x in df["forced_l1"].quantile([0.5, 0.9, 0.99]).values],
         "n_forced_gt_0.12": int((df["forced_l1"] > 0.12).sum()), "n_forced_gt_0.075": int((df["forced_l1"] > 0.075).sum()),
         "forced_2020-04-08": float(df.set_index("date")["forced_l1"]["2020-04-08"]),
         "turnover_pkl_2020-04-08": float(df.set_index("date")["turnover_pkl"]["2020-04-08"]),
         "components_max": {c: float(df[c].max()) for c in ["gate_sell", "nan_pin", "mega_sell", "mega_buy", "floor_buy"]},
         "n_gate_forced_names_max": int(df["n_gate_forced_names"].max()), "n_gate_forced_names_median": float(df["n_gate_forced_names"].median()),
         "n_nan_mu_max": int(df["n_nan_mu"].max()),
         "forced_over_cap_max": float(df["forced_l1"].max() / 0.15),
         "top5_forced": df.nlargest(5, "forced_l1")[["date", "forced_l1", "gate_sell", "mega_sell", "mega_buy", "n_gate_forced_names"]].round(4).values.tolist()}}
json.dump(out, open(os.path.join(V, "W1_recon_all97.json"), "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
