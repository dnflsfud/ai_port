"""W1 empirical batch - pkl-only checks: T-01, T-10, T-12, T-09(b), T-05 turnover slack.
Run from ai_port with PYTHONPATH=. ; env V = scratch verify dir.
"""
import pickle, sys, os, json
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
V = os.environ["V"]
out = {}
r = pickle.load(open("outputs/s16_7_name_risk_cap/backtest_result.pkl", "rb"))
m = json.load(open("outputs/s16_7_name_risk_cap/metrics.json"))["metrics"]
panel = r.panel

# ---------------- T-10 duplicate columns ----------------
t10 = {}
for a, b in [("fin_roe_level_z", "best_roe_level_z"), ("fin_pb_level_z", "best_px_bps_ratio_level_z")]:
    x = panel[a].to_numpy(); y = panel[b].to_numpy()
    eq = np.array_equal(x, y, equal_nan=True)
    md = float(np.nanmax(np.abs(x - y)))
    fin = np.isfinite(x) & np.isfinite(y)
    corr = float(np.corrcoef(x[fin], y[fin])[0, 1])
    t10[a + "==" + b] = {"array_equal_nan": bool(eq), "max_abs_diff": md, "corr": corr, "n_finite": int(fin.sum())}
names_by_model = {}
four = ["fin_roe_level_z", "best_roe_level_z", "fin_pb_level_z", "best_px_bps_ratio_level_z"]
for k, mdl in r.models.items():
    fn = list(mdl.booster_.feature_name()) if hasattr(mdl, "booster_") else []
    af = list(getattr(mdl, "_active_features", []))
    names_by_model[str(k.date())] = {"n_feature_name": len(fn), "n_active": len(af), "has4": [n for n in four if n in af]}
t10["models_with_all4_active"] = sum(1 for v in names_by_model.values() if len(v["has4"]) == 4)
t10["n_models"] = len(names_by_model)
first = list(r.models)[0]
t10["feature_name_sample"] = list(r.models[first].booster_.feature_name())[:3]
t10["active_features_first_model"] = names_by_model[str(first.date())]
# active feature index positions of the 4 (Column_k mapping)
af0 = list(getattr(r.models[first], "_active_features", []))
t10["positions_in_first_model"] = {n: (af0.index(n) if n in af0 else None) for n in four}
out["T-10"] = t10
print("T-10", json.dumps(t10, default=str)[:900])

# ---------------- T-12 IR definitions ----------------
port = r.portfolio_returns.dropna(); bm = r.benchmark_returns.reindex(port.index).ffill().fillna(0)
a = port - bm; n = len(a)
te = a.std() * np.sqrt(252)
ir_geo = ((1 + a).prod() ** (252 / n) - 1) / te
ir_arith = a.mean() / a.std() * np.sqrt(252)
cagr_p = (1 + port).prod() ** (252 / n) - 1; cagr_b = (1 + bm).prod() ** (252 / n) - 1
ir_cagr = (cagr_p - cagr_b) / te
from src.harness import sub_ir, sub_period_irs
sub_full = sub_ir(port, bm, str(port.index[0].date()), str(port.index[-1].date()))
t12 = {"n_days": n, "te": float(te), "metrics_information_ratio": m["information_ratio"],
       "ir_geometric_active_compound": float(ir_geo), "abs_diff_vs_metrics": float(abs(ir_geo - m["information_ratio"])),
       "ir_arithmetic_mean_x252": float(ir_arith), "ir_cagr_diff": float(ir_cagr),
       "harness_sub_ir_full": float(sub_full), "geo_minus_arith": float(ir_geo - ir_arith),
       "sub_period_irs": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in sub_period_irs(port, bm).items()},
       "metrics_sub_period": {k: m.get(k) for k in ["P1_ir", "P2_ir", "P3_ir", "P4_tail_ir"] if k in m},
       "metrics_active_return": m.get("active_return"), "active_return_geo": float((1 + a).prod() ** (252 / n) - 1),
       "active_return_arith": float(a.mean() * 252)}
out["T-12"] = t12
print("T-12", json.dumps(t12, default=str))

# ---------------- T-09(b) predictions on holiday rows ----------------
t9 = {}
hol = ["2018-12-25", "2024-02-19", "2025-12-25", "2026-06-19"]
rebal_dates = sorted(r.portfolio_weights.keys())
pidx = r.predictions.index
for h in hol:
    h = pd.Timestamp(h)
    i = pidx.get_loc(h)
    prev = pidx[i - 1]
    cur = r.predictions.loc[h]; pre_prev = r.pre_execution_predictions.loc[prev]; pre_cur = r.pre_execution_predictions.loc[h]
    both = cur.notna() & pre_prev.notna()
    t9[str(h.date())] = {"is_rebal": bool(h in r.portfolio_weights), "prev_row": str(prev.date()),
                         "pred_h_minus_preexec_hm1_maxabs": float((cur - pre_prev).abs().max()), "n_compared": int(both.sum()),
                         "pred_h_minus_preexec_h_maxabs": float((cur - pre_cur).abs().max()),
                         "ic_series_has_h": bool(h in r.ic_series.index)}
sh = (r.predictions - r.pre_execution_predictions.shift(1)).abs().max().max()
t9["global_pred_equals_preexec_shift1_maxabs"] = float(sh)
t9["n_rebal"] = len(rebal_dates)
out["T-09"] = t9
print("T-09", json.dumps(t9, default=str))

# ---------------- T-05 turnover slack from pkl books ----------------
dw = r.daily_weights; dw_dates = sorted(dw.keys())
rows = []
for d in rebal_dates:
    i = dw_dates.index(d)
    w = r.portfolio_weights[d]
    if i == 0:
        rows.append({"date": str(d.date()), "l1": np.nan}); continue
    prev = dw[dw_dates[i - 1]]
    l1 = float((w.reindex(prev.index).fillna(0) - prev).abs().sum())
    rows.append({"date": str(d.date()), "l1": l1})
df5 = pd.DataFrame(rows)
df5["slack_vs_0.15"] = 0.15 - df5["l1"]
t5 = {"l1_max": float(df5["l1"].max()), "l1_argmax": df5.loc[df5["l1"].idxmax(), "date"], "slack_min": float(df5["slack_vs_0.15"].min()),
      "n_l1_gt_0.14": int((df5["l1"] > 0.14).sum()), "n_l1_gt_0.12": int((df5["l1"] > 0.12).sum()),
      "pkl_turnover_series_max": float(r.turnover.max()), "pkl_turnover_argmax": str(r.turnover.idxmax().date()),
      "l1_vs_2x_pkl_turnover_maxabs": float((df5.set_index("date")["l1"] - 2 * r.turnover.rename(lambda x: str(x.date()))).abs().max()),
      "daily_weights_vs_rebal_same_day_maxdiff": float((dw[rebal_dates[5]] - r.portfolio_weights[rebal_dates[5]]).abs().max())}
out["T-05_slack"] = t5
print("T-05", json.dumps(t5, default=str))
df5.to_csv(os.path.join(V, "W1_T05_l1_by_rebal.csv"), index=False)

# ---------------- T-01 tg_upside imputation artefact ----------------
t1 = {}
tg = panel["tg_upside"].unstack("ticker"); tgm = panel["tg_mom_63d"].unstack("ticker")
gaps = {"VRT": ("2018-08-01", "2020-02-24", "2020-02-25", "2020-05-22"), "VST": ("2016-10-05", "2017-03-30", "2017-03-31", "2017-06-28"),
        "PLTR": ("2020-09-30", "2020-10-21", "2020-10-22", "2021-01-20"), "BE": ("2018-07-25", "2018-08-15", "2018-08-16", "2018-11-13"),
        "RACE": ("2015-10-21", "2015-11-11", "2015-11-12", "2016-02-10"), "CRWD": ("2019-06-12", "2019-07-02", "2019-07-03", "2019-10-01"),
        "DDOG": ("2019-09-19", "2019-10-09", "2019-10-10", "2020-01-08"), "KEYS": ("2014-10-21", "2014-10-30", "2014-10-31", "2015-01-29"),
        "LITE": ("2015-07-27", "2015-07-30", "2015-07-31", "2015-10-28"), "285A": ("2024-12-18", "2024-12-18", "2024-12-19", "2025-03-19")}
for tk, (g0, g1, m0, m1) in gaps.items():
    if tk not in tg.columns:
        t1[tk] = "not in panel"; continue
    s = tg.loc[g0:g1, tk].dropna()
    others = tg.loc[g0:g1].drop(columns=[tk])
    q95_gap = float(np.nanpercentile(others.abs().to_numpy(), 95)) if len(others) else float("nan")
    mm = tgm.loc[m0:m1, tk].dropna()
    t1[tk] = {"gap_rows": int(len(s)), "tg_upside_mean": float(s.mean()) if len(s) else None, "tg_upside_min": float(s.min()) if len(s) else None,
              "frac_eq_5": float((s >= 4.999).mean()) if len(s) else None, "others_abs_q95_in_gap": q95_gap,
              "post_gap_tg_mom_mean": float(mm.mean()) if len(mm) else None, "post_gap_tg_mom_min": float(mm.min()) if len(mm) else None,
              "post_gap_tg_mom_frac_le_-4.99": float((mm <= -4.99).mean()) if len(mm) else None}
clean = tg.loc["2020-06-01":"2026-08-25"]
t1["others_abs_q95_2020-06_onward"] = float(np.nanpercentile(clean.abs().to_numpy(), 95))
t1["others_abs_q95_2017-04_2018-07"] = float(np.nanpercentile(tg.loc["2017-04-01":"2018-07-31"].abs().to_numpy(), 95))
mx = tg.abs().max(axis=1)
t1["n_dates_with_abs_z_ge_4.99"] = int((mx >= 4.99).sum()); t1["n_dates_total"] = int(len(mx))
hit = mx[mx >= 4.99]
t1["dates_ge_4.99_by_year"] = {str(k): int(v) for k, v in hit.groupby(hit.index.year).size().items()}
clip_counts = (tg >= 4.99).sum().sort_values(ascending=False)
t1["top_clip_tickers"] = {k: int(v) for k, v in clip_counts.head(12).items()}
t1["median_abs_z_others_gap_VRT"] = float(np.nanmedian(tg.loc["2018-08-01":"2020-02-24"].drop(columns=["VRT"]).abs().to_numpy()))
t1["median_abs_z_all_clean"] = float(np.nanmedian(clean.abs().to_numpy()))
# per-date q95 of others during VRT gap vs same length window after
q95_by_date_gap = tg.loc["2018-08-01":"2020-02-24"].drop(columns=["VRT"]).abs().quantile(0.95, axis=1)
q95_by_date_post = tg.loc["2020-06-01":"2021-12-31"].abs().quantile(0.95, axis=1)
t1["q95_by_date_gap_median"] = float(q95_by_date_gap.median()); t1["q95_by_date_post_median"] = float(q95_by_date_post.median())
t1["VRT_sample"] = {str(k.date()): round(float(v), 3) for k, v in tg["VRT"].dropna().iloc[::100].items()} if "VRT" in tg else None
t1["VST_sample"] = {str(k.date()): round(float(v), 3) for k, v in tg["VST"].dropna().iloc[::100].items()} if "VST" in tg else None
out["T-01"] = t1
print("T-01", json.dumps(t1, default=str)[:3500])
json.dump(out, open(os.path.join(V, "W1_pkl_cheap.json"), "w"), indent=1, default=str)
print("saved")
