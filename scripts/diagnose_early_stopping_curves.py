# -*- coding: utf-8 -*-
"""S13.8 diagnosis: WHY does early stopping fire at round 0 in 18/32 retrains?

Four arms (min_child 30, val_window 252, lr 0.03, mh blend) all swept
hyper-parameters and all left degenerate_rate at 56.25%. Hypothesis: the
early-stopping METRIC (ndcg@5 over ~126 validation dates) is noise-dominated,
so no learning-rate/capacity change can help.

Test: read the stored validation curves straight out of the fitted models
(sklearn LGBM keeps evals_result_), and for every retrain compare the
achieved improvement against the round-to-round noise of the same curve.

  gain  = best_score - score_at_round_0
  noise = std of first differences of the curve
  snr   = gain / noise

If BOTH degenerate and healthy retrains have snr ~ O(1), the metric is
uninformative and the fix is metric design, not hyper-parameters.
If only degenerate ones do, those periods genuinely lack learnable signal
and reusing the previous model is correct behaviour.
"""
import pickle

import numpy as np
import pandas as pd

PKL = "outputs/codex_causal_rank_65/backtest_result.pkl"
MIN_TREES = 10

print("loading pkl ...", flush=True)
with open(PKL, "rb") as fh:
    r = pickle.load(fh)

rows = []
missing = 0
for key, model in r.models.items():
    ev = getattr(model, "evals_result_", None)
    booster = getattr(model, "booster_", None)
    n_trees = booster.num_trees() if booster is not None else np.nan
    if not ev:
        missing += 1
        continue
    valid = ev.get("valid_0") or next(iter(ev.values()))
    rec = {"key": str(key), "n_trees": n_trees,
           "best_iter": getattr(model, "best_iteration_", None),
           "n_rounds_run": None}
    for metric, curve in valid.items():
        c = np.asarray(curve, dtype=float)
        rec["n_rounds_run"] = len(c)
        d = np.diff(c)
        gain = float(c.max() - c[0])
        noise = float(np.std(d)) if len(d) > 1 else np.nan
        rec[f"{metric}_first"] = float(c[0])
        rec[f"{metric}_best"] = float(c.max())
        rec[f"{metric}_argmax"] = int(np.argmax(c)) + 1
        rec[f"{metric}_gain"] = gain
        rec[f"{metric}_noise"] = noise
        rec[f"{metric}_snr"] = gain / noise if noise and noise > 0 else np.nan
    rows.append(rec)

del r
df = pd.DataFrame(rows)
print(f"models with stored curves: {len(df)} (missing evals_result_: {missing})\n")
if df.empty:
    raise SystemExit("no validation curves stored — need a re-run with curves kept")

df["degenerate"] = df.n_trees < MIN_TREES
metrics = sorted({c.rsplit("_", 1)[0] for c in df.columns if c.endswith("_snr")})
pd.set_option("display.width", 200)

print("=" * 78)
print("PER-RETRAIN VALIDATION CURVES")
print("=" * 78)
m0 = metrics[0]
cols = ["key", "n_trees", "best_iter", "n_rounds_run",
        f"{m0}_first", f"{m0}_best", f"{m0}_argmax",
        f"{m0}_gain", f"{m0}_noise", f"{m0}_snr", "degenerate"]
print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.5f}"))

print("\n" + "=" * 78)
print("DEGENERATE vs HEALTHY")
print("=" * 78)
for m in metrics:
    g = df.groupby("degenerate")[[f"{m}_first", f"{m}_best", f"{m}_gain",
                                  f"{m}_noise", f"{m}_snr"]].mean()
    g.index = g.index.map({True: "degenerate", False: "healthy"})
    print(f"\n--- {m} (mean) ---")
    print(g.to_string(float_format=lambda v: f"{v:.5f}"))

print("\n--- rounds actually run before early stopping ---")
print(df.groupby("degenerate")["n_rounds_run"].describe()
        .rename(index={True: "degenerate", False: "healthy"}).to_string())

print("\n" + "=" * 78)
print("VERDICT INPUTS")
print("=" * 78)
for m in metrics:
    d = df[df.degenerate][f"{m}_snr"].dropna()
    h = df[~df.degenerate][f"{m}_snr"].dropna()
    print(f"{m}: snr degenerate median={d.median():.2f} (n={len(d)}), "
          f"healthy median={h.median():.2f} (n={len(h)})")

df.to_csv("outputs/s13_8_degeneracy_curves.csv", index=False)
print("\nwrote outputs/s13_8_degeneracy_curves.csv")
