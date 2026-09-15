"""Compare preserved pre-fix runs with freshly generated verified runs."""
import json
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.export_operating_data import build_rebalance_metadata

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--role", choices=["production", "challenger", "both"], default="both")
args = parser.parse_args()
summary = {}
for role, label in (("production", "codex_causal_rank_65"), ("challenger", "iter15_65tkr_reb21_vtg")):
    if args.role != "both" and args.role != role:
        continue
    with (OUT / f"{role}_before.pkl").open("rb") as fh:
        before = pickle.load(fh)
    with (ROOT / "outputs" / label / "backtest_result.pkl").open("rb") as fh:
        after = pickle.load(fh)
    old_meta = json.loads((OUT / f"{role}_before_metrics.json").read_text(encoding="utf-8"))
    new_meta = json.loads((ROOT / "outputs" / label / "metrics.json").read_text(encoding="utf-8"))
    old_metrics, new_metrics = before.compute_metrics(), after.compute_metrics()
    metric_changes = {key: {"before": old_metrics.get(key), "after": new_metrics.get(key)} for key in
                      ("information_ratio", "tracking_error", "avg_annual_turnover", "avg_ic", "max_drawdown", "active_return")}
    rebalances = sorted(after.portfolio_weights)
    dates = after.portfolio_returns.index
    all_dates = after.predictions.index
    mismatch_known, mismatch_exchange = [], []
    for last, following in zip(rebalances, rebalances[1:]):
        for known, mismatches in ((all_dates, mismatch_known), (all_dates[all_dates <= last], mismatch_exchange)):
            meta = build_rebalance_metadata([last], dates[dates <= last], 21,
                                           calendar_dates=known, calendar_name="XNYS")
            if meta["next_expected_rebalance_date"] != str(following)[:10]:
                mismatches.append({"last": str(last)[:10], "expected": str(following)[:10],
                                   "actual": meta["next_expected_rebalance_date"]})
    pred_delta = after.predictions - before.predictions
    finite_delta = pred_delta.to_numpy()
    finite_delta = finite_delta[np.isfinite(finite_delta)]
    state = after.data_quality["currency"]
    summary[role] = {
        "same_recorded_vintage": old_meta["data_vintage"] == new_meta["data_vintage"],
        "comparison_limit": "Old artifacts have mtime/size only; content hashes are verified for the new run.",
        "metrics": metric_changes,
        "pre_overlay_predictions_identical": after.pre_overlay_predictions.equals(before.pre_overlay_predictions),
        "model_panel_identical": after.panel.equals(before.panel),
        "post_overlay_changed_cells": int(np.count_nonzero(np.abs(finite_delta) > 1e-12)),
        "post_overlay_max_abs_change": float(np.max(np.abs(finite_delta))),
        "optimizer_solver_counts": after.optimizer_solver_counts,
        "optimizer_failures_before": before.optimizer_failures,
        "optimizer_fallback_reasons_before": before.optimizer_fallback_reason_counts,
        "optimizer_failures": after.optimizer_failures,
        "earnings_calendar": after.data_quality.get("earnings_calendar"),
        "run_contract_version": after.run_contract["schema_version"],
        "tg_basis_guard_ok": state["tg_basis_guard_ok"],
        "pending_tg_names": list(state["tg_px_ratio_jump_vs_prev"]),
        "schedule_intervals": len(rebalances) - 1,
        "schedule_mismatches_known_calendar": mismatch_known,
        "schedule_mismatches_exchange_extension": mismatch_exchange,
    }
    del before, after
filename = "fix_validation.json" if args.role == "both" else f"fix_validation_{args.role}.json"
(OUT / filename).write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
if args.role != "both":
    combined = {}
    for role in ("production", "challenger"):
        part = OUT / f"fix_validation_{role}.json"
        if part.exists():
            combined.update(json.loads(part.read_text(encoding="utf-8")))
    if len(combined) == 2:
        (OUT / "fix_validation.json").write_text(
            json.dumps(combined, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
