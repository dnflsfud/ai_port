"""M4_dead_features 반박 검증 스크립트.

pkl(models 33 + panel)만 사용. 재학습 없음. 출력: 같은 폴더 M4_out.json.
"""
import json
import pickle
import sys
import time
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd

WD = "C:/Users/westl/PycharmProjects/pythonProject/venv_vf_new/machine/re_study/c2/ai_port"
PKL = WD + "/outputs/s16_7_name_risk_cap/backtest_result.pkl"
OUT = ("C:/Users/westl/AppData/Local/Temp/claude/C--Users-westl-PycharmProjects-pythonProject-"
       "venv-vf-new-machine-re-study-c2-ai-port/0caf960f-9660-4a4b-a91e-809a2f4ab53d/scratchpad/"
       "w3/verify/M4_out.json")
sys.path.insert(0, WD)

BCAST = ["cal_is_Q1", "regime_mkt_ret_21d", "fac_F_Quality_mom_63d", "fac_F_Growth_mom_63d",
         "fac_F_Value_mom_63d", "fac_value_growth_63d", "fac_yield_slope"]
DUP_PAIRS = [("fin_roe_level_z", "best_roe_level_z"), ("fin_pb_level_z", "best_px_bps_ratio_level_z")]
NEAR_PAIRS = [("oper_margin_chg_63d", "op_leverage_63d"), ("best_eps_chg_252d", "earnings_quality_252d")]

t0 = time.time()
with open(PKL, "rb") as f:
    r = pickle.load(f)
print(f"loaded in {time.time()-t0:.1f}s", flush=True)

feature_names = list(r.feature_names)
print("n feature_names", len(feature_names))
models = r.models
if isinstance(models, dict):
    items = list(models.items())
else:
    items = list(enumerate(models))
print("models type", type(models).__name__, "n", len(items))

out = OrderedDict()
out["n_feature_names"] = len(feature_names)
out["bcast_idx_in_feature_names"] = {b: feature_names.index(b) for b in BCAST}

# ---------------------------------------------------------------- 1. split==0 by name
rows = []
seen_ids = {}
per_model_gain = {}
for key, m in items:
    af = list(getattr(m, "_active_features", feature_names))
    booster = m.booster_
    split = booster.feature_importance(importance_type="split").astype(float)
    gain = booster.feature_importance(importance_type="gain").astype(float)
    assert len(af) == len(split) == booster.num_feature(), (key, len(af), len(split))
    zs = [af[i] for i in range(len(af)) if split[i] == 0]
    zg = [af[i] for i in range(len(af)) if gain[i] == 0]
    dropped = [f for f in feature_names if f not in set(af)]
    gshare = gain / gain.sum() if gain.sum() > 0 else gain
    per_model_gain[str(key)] = dict(zip(af, gshare.tolist()))
    oid = id(m)
    fresh = oid not in seen_ids
    seen_ids.setdefault(oid, str(key))
    rows.append(OrderedDict(
        key=str(key), obj_id=oid, fresh=fresh, n_active=len(af), n_trees=booster.num_trees(),
        dropped=dropped, zero_split=zs, zero_gain=zg,
        bcast_active=[b for b in BCAST if b in af],
        bcast_active_split=[float(split[af.index(b)]) for b in BCAST if b in af],
        nonbcast_zero_split=[f for f in zs if f not in BCAST],
        dropped_all_bcast=all(d in BCAST for d in dropped),
        booster_feature_names_sample=booster.feature_name()[:3],
    ))
out["models"] = rows
uniq = [x for x in rows if x["fresh"]]
out["n_models"] = len(rows)
out["n_unique_models"] = len(uniq)
out["bcast_split0_in_all_models_where_active"] = all(
    all(s == 0 for s in x["bcast_active_split"]) for x in rows)
out["n_models_bcast_active_split0"] = sum(
    all(s == 0 for s in x["bcast_active_split"]) for x in rows)
out["nonbcast_zero_split_counter_all"] = Counter(f for x in rows for f in x["nonbcast_zero_split"])
out["nonbcast_zero_split_counter_unique"] = Counter(f for x in uniq for f in x["nonbcast_zero_split"])
out["dropped_sets"] = {"|".join(k): v for k, v in Counter(tuple(x["dropped"]) for x in rows).items()}
out["dropped_all_subset_of_bcast"] = all(x["dropped_all_bcast"] for x in rows)
out["n_models_with_drop"] = sum(1 for x in rows if x["dropped"])

# ---------------------------------------------------------------- 2. duplicates in panel
panel = r.panel
print("panel", panel.shape, flush=True)
dup = {}
for a, b in DUP_PAIRS:
    A = panel[a].to_numpy(dtype=float)
    B = panel[b].to_numpy(dtype=float)
    nan_mismatch = int((np.isnan(A) != np.isnan(B)).sum())
    d = np.abs(A - B)
    dup[f"{a}~{b}"] = dict(max_abs_diff=float(np.nanmax(d)), nan_mismatch=nan_mismatch,
                           n=int(len(A)), exact_equal_frac=float(np.mean((A == B) | (np.isnan(A) & np.isnan(B)))))
out["dup_pairs"] = dup
near = {}
for a, b in NEAR_PAIRS + DUP_PAIRS:
    sub = panel[[a, b]].dropna()
    near[f"{a}~{b}"] = dict(pearson=float(sub[a].corr(sub[b])), spearman=float(sub[a].corr(sub[b], method="spearman")))
out["pair_corr"] = near


def mean_gain_share(fname, only_unique=True):
    vals = []
    for x in rows:
        if only_unique and not x["fresh"]:
            continue
        g = per_model_gain[x["key"]]
        if fname in g:
            vals.append(g[fname])
    return float(np.mean(vals)) if vals else None, len(vals)


gs = {}
for a, b in NEAR_PAIRS + DUP_PAIRS:
    for f in (a, b):
        mu, n = mean_gain_share(f)
        gs[f] = dict(mean_gain_share_unique_models=mu, n_models=n)
for f in BCAST:
    mu, n = mean_gain_share(f)
    gs[f] = dict(mean_gain_share_unique_models=mu, n_models=n)
out["gain_share"] = gs

# ---------------------------------------------------------------- 3. EWMA replication
from src.config import PipelineConfig  # noqa: E402
from src.model_trainer import EWMAFeatureTracker  # noqa: E402

cfg = PipelineConfig()
cfg.ewma_enabled = True
cfg.ewma_alpha = 0.3
cfg.ewma_min_retrains = 2
cfg.ewma_drop_pct = 0.05
cfg.ewma_min_features = 60
cfg.ewma_importance_type = "split"
cfg.ewma_full_refresh_interval = 0
tr = EWMAFeatureTracker(cfg)
tr.init_full_features(feature_names)
repl = []
prev_oid = None
for key, m in items:
    oid = id(m)
    predicted_active = tr.get_active_features(feature_names)
    actual_active = list(getattr(m, "_active_features", feature_names))
    if oid != prev_oid:  # fresh (accepted) model -> tracker was updated with it
        match = predicted_active == actual_active
        repl.append(dict(key=str(key), n_updates_before=tr.n_updates, predicted_n=len(predicted_active),
                         actual_n=len(actual_active), match=match,
                         predicted_dropped=[f for f in feature_names if f not in set(predicted_active)]))
        tr.update(m, actual_active, pd.Timestamp(str(key)) if not isinstance(key, int) else pd.Timestamp("2000-01-01"))
    prev_oid = oid
out["ewma_replication"] = repl
out["ewma_replication_all_match"] = all(x["match"] for x in repl)
# tie structure among dead features at final state
imp = tr.ewma_importance
dead_imp = {f: float(imp[feature_names.index(f)]) for f in BCAST}
out["ewma_final_importance_bcast"] = dead_imp
out["ewma_final_importance_min_nonbcast"] = float(min(imp[i] for i, f in enumerate(feature_names) if f not in BCAST))
out["ewma_final_importance_rank_of_bcast"] = {f: int((imp > imp[feature_names.index(f)]).sum()) + 1 for f in BCAST}
# feature weights (scaling) for bcast vs others
fw = tr.get_feature_weights(feature_names)
out["ewma_feature_weight_bcast"] = {f: float(fw[feature_names.index(f)]) for f in BCAST}
out["ewma_feature_weight_nonbcast_min_max"] = [float(min(fw[i] for i, f in enumerate(feature_names) if f not in BCAST)),
                                               float(max(fw[i] for i, f in enumerate(feature_names) if f not in BCAST))]


def drop_arith(n_total, drop_pct=0.05, min_features=60):
    n_drop = int(n_total * drop_pct)
    n_keep = max(n_total - n_drop, min_features)
    n_keep = min(n_keep, n_total)
    return dict(n_total=n_total, n_drop_budget=n_drop, n_keep=n_keep, effective_drops=n_total - n_keep)


out["drop_arith"] = {str(n): drop_arith(n) for n in (65, 62, 58, 56)}
out["min_features_needed_for_any_drop_at_56"] = max(n for n in range(0, 57) if min(max(56 - int(56 * 0.05), n), 56) < 56)

# ---------------------------------------------------------------- 4. colsample theory
def colsample(n_total, n_dead, frac=0.8):
    k_int = int(frac * n_total)
    k_round = int(np.floor(frac * n_total + 0.5))
    live = n_total - n_dead
    return dict(n_total=n_total, n_dead=n_dead, n_live=live,
                k_int=k_int, k_round=k_round,
                incl_prob_int=k_int / n_total, incl_prob_round=k_round / n_total,
                exp_live_per_tree_int=k_int * live / n_total, exp_live_per_tree_round=k_round * live / n_total)


out["colsample"] = {
    "65_with_7_dead": colsample(65, 7),
    "62_with_4_dead": colsample(62, 4),
    "58_no_dead": colsample(58, 0),
    "56_no_dead_no_dup": colsample(56, 0),
}

# ---------------------------------------------------------------- 5. within-date constancy of bcast (panel)
cs = {}
for f in BCAST:
    s = panel[f]
    g = s.groupby(level=0)
    cs[f] = dict(dates_cs_std_eq0_frac=float((g.std(ddof=0).fillna(0) == 0).mean()),
                 n_unique_over_time=int(g.first().nunique()),
                 ts_std=float(g.first().std()))
out["bcast_within_date_constancy"] = cs

# top-5 gain features by name (unique models mean)
agg = Counter()
cnt = Counter()
for x in uniq:
    for f, v in per_model_gain[x["key"]].items():
        agg[f] += v
        cnt[f] += 1
mean_gs = {f: agg[f] / cnt[f] for f in agg}
out["top10_mean_gain_share_unique"] = sorted(mean_gs.items(), key=lambda kv: -kv[1])[:10]
out["bottom10_mean_gain_share_unique"] = sorted(mean_gs.items(), key=lambda kv: kv[1])[:10]


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, Counter):
        return {str(k): v for k, v in o.items()}
    if isinstance(o, tuple):
        return list(o)
    return str(o)


with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, default=_default, ensure_ascii=False)
print("wrote", OUT, f"elapsed {time.time()-t0:.1f}s")
# print the key results
for k in ["n_models", "n_unique_models", "bcast_split0_in_all_models_where_active", "n_models_bcast_active_split0",
          "nonbcast_zero_split_counter_all", "nonbcast_zero_split_counter_unique", "dropped_sets",
          "dropped_all_subset_of_bcast", "n_models_with_drop", "dup_pairs", "pair_corr", "gain_share",
          "ewma_replication_all_match", "ewma_final_importance_bcast", "ewma_final_importance_min_nonbcast",
          "ewma_final_importance_rank_of_bcast", "ewma_feature_weight_bcast", "ewma_feature_weight_nonbcast_min_max",
          "drop_arith", "min_features_needed_for_any_drop_at_56", "colsample", "bcast_within_date_constancy",
          "top10_mean_gain_share_unique", "bottom10_mean_gain_share_unique", "bcast_idx_in_feature_names"]:
    print(k, "=", json.dumps(out[k], default=_default, ensure_ascii=False))
for x in repl:
    print("EWMA repl", x["key"], "n_updates_before", x["n_updates_before"], "pred", x["predicted_n"], "act", x["actual_n"],
          "match", x["match"], "pred_dropped", x["predicted_dropped"])
