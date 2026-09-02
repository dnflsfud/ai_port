"""M1 aux: LightGBM gain share of affected features across the 33 production models (pkl)."""
import pickle, json, numpy as np, pandas as pd, os
REPO = r"C:/Users/westl/PycharmProjects/pythonProject/venv_vf_new/machine/re_study/c2/ai_port"
VER = r"C:/Users/westl/AppData/Local/Temp/claude/C--Users-westl-PycharmProjects-pythonProject-venv-vf-new-machine-re-study-c2-ai-port/0caf960f-9660-4a4b-a91e-809a2f4ab53d/scratchpad/w3/verify"
r = pickle.load(open(os.path.join(REPO, "outputs/s16_7_name_risk_cap/backtest_result.pkl"), "rb"))
aff = ["tg_upside", "fin_pe_level_z", "fin_pb_level_z", "best_px_bps_ratio_level_z", "best_peg_ratio_level_z",
       "fin_roe_pb_gap", "fin_roe_pe_gap", "fin_pe_chg_63d", "fin_pb_chg_63d", "cash_conversion_z"]
rows = []
for k, m in sorted(r.models.items()):
    feats = list(getattr(m, "_active_features", None) or r.feature_names)
    g = m.booster_.feature_importance(importance_type="gain")
    s = pd.Series(g, index=feats[:len(g)]); s = s / s.sum()
    rank = s.rank(ascending=False)
    rows.append({"model": str(k)[:10], **{f: round(float(s.get(f, 0.0)), 4) for f in aff},
                 "tg_upside_rank": int(rank.get("tg_upside", -1)), "sum_affected": round(float(sum(s.get(f, 0.0) for f in aff)), 4)})
df = pd.DataFrame(rows).set_index("model")
out = {"per_model_head": df.head(3).to_dict("index"), "per_model_tail": df.tail(3).to_dict("index"),
       "median_share": df.median().round(4).to_dict(), "mean_share": df.mean().round(4).to_dict(),
       "tg_upside_rank_median": float(df["tg_upside_rank"].median()), "n_models": int(len(df))}
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(VER, "M1_gain_share.json"), "w"), indent=1)
