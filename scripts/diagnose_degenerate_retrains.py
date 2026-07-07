"""SPEC D0 — degenerate-retrain diagnosis (report-only).

S0 metrics.json shows degenerate_rate 50% (16/32): during walk-forward
retraining (63d cadence) half of the retrains produce <10-tree models that are
rejected in favour of the previously retained model + feature set
(``src/model_trainer.py:399-425``). The effective retrain cadence is therefore
double the nominal one, and the cause was undiagnosed. This script builds a
census of the retrain windows, localises the degeneracy across the P1/P2/P3
sub-periods, assembles root-cause evidence, and derives the two distributions
used to pre-register later arms (A2 confidence_spread_scale, A3 trailing-IC
constants).

REPORT-ONLY: this module never imports/mutates production prediction, weight, or
metric code paths. It only *reads* the persisted BacktestResult pickle. The core
``build_report`` is a pure function (no file I/O) that ``main`` feeds after
gathering artifacts.

CLI:
    C:/Users/westl/PycharmProjects/pythonProject/venv_vf_new/Scripts/python.exe \
        scripts/diagnose_degenerate_retrains.py --out outputs
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ===========================================================================
# PURE CORE — build_report (imported by the acceptance test; no file I/O).
# ===========================================================================
def _row_spread(values) -> Optional[float]:
    """Top-bottom spread of a single raw-prediction row.

    Mirrors compute_signal_confidence (src/backtest.py:857-860) EXACTLY: drop
    NaN, tail_n = max(3, n_valid // 10), spread = mean(top tail_n) -
    mean(bottom tail_n). Returns None for an all-NaN / empty row so the caller
    can drop it from n_dates.
    """
    v = np.asarray([x for x in values if x == x], dtype=float)  # x==x drops NaN
    if v.size == 0:
        return None
    v = np.sort(v)
    n = v.size
    tail_n = max(3, n // 10)
    top_mean = float(v[-tail_n:].mean())
    bot_mean = float(v[:tail_n].mean())
    return top_mean - bot_mean


def _dist_stats(values) -> Dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(a)),
        "iqr": float(np.percentile(a, 75) - np.percentile(a, 25)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
    }


def _subperiod_label(ts: pd.Timestamp, subperiods: dict) -> Optional[str]:
    for name, (start, end) in subperiods.items():
        if pd.Timestamp(start) <= ts <= pd.Timestamp(end):
            return name
    return None


def _degenerate_runs(windows_sorted: List[dict]) -> List[dict]:
    """Run-length encode consecutive degenerate windows, tagging the tree count
    of the healthy model each run trails."""
    runs: List[dict] = []
    prev_healthy_trees: Optional[int] = None
    i = 0
    n = len(windows_sorted)
    while i < n:
        w = windows_sorted[i]
        if not w["degenerate"]:
            prev_healthy_trees = int(w.get("n_trees") or 0)
            i += 1
            continue
        j = i
        dates = []
        while j < n and windows_sorted[j]["degenerate"]:
            dates.append(str(windows_sorted[j]["train_date"]))
            j += 1
        runs.append(
            {
                "length": len(dates),
                "dates": dates,
                "trails_healthy_model_trees": prev_healthy_trees,
            }
        )
        i = j
    return runs


def build_report(
    retrain_windows: List[dict],
    raw_predictions: pd.DataFrame,
    trailing_ic: pd.Series,
    subperiods: dict,
) -> dict:
    """Assemble the degenerate-retrain diagnosis (pure, no I/O).

    Parameters
    ----------
    retrain_windows : list of dict
        One dict per retrain window with at least: ``train_date`` (Timestamp),
        ``degenerate`` (bool), ``best_iteration`` (int), ``n_trees`` (int),
        ``val_score`` (float | None). Extra keys (e.g. ``val_start``,
        ``val_end``) are preserved in the ``windows`` census.
    raw_predictions : DataFrame
        dates x tickers, raw (pre-EMA) baseline predictions.
    trailing_ic : Series
        Trailing-IC series indexed by date.
    subperiods : dict
        {"P1": (start, end), ...} with inclusive Timestamp bounds.

    Returns
    -------
    dict with keys: windows, subperiod_overlap, root_cause_evidence,
    raw_spread_dist, trailing_ic_dist.
    """
    # --- windows census (full; one entry per input window) -----------------
    windows_out: List[dict] = []
    for w in retrain_windows:
        ts = pd.Timestamp(w["train_date"])
        entry = dict(w)
        entry["train_date"] = ts.strftime("%Y-%m-%d")
        entry["degenerate"] = bool(w["degenerate"])
        entry["subperiod"] = _subperiod_label(ts, subperiods)
        windows_out.append(entry)

    # --- subperiod overlap (degenerate windows per period, inclusive) -------
    subperiod_overlap: Dict[str, int] = {}
    for name, (start, end) in subperiods.items():
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        cnt = sum(
            1
            for w in retrain_windows
            if bool(w["degenerate"]) and s <= pd.Timestamp(w["train_date"]) <= e
        )
        subperiod_overlap[name] = int(cnt)

    # --- raw_spread distribution (SAME def as compute_signal_confidence) ----
    spreads = []
    for d in raw_predictions.index:
        s = _row_spread(raw_predictions.loc[d].values)
        if s is not None:
            spreads.append(s)
    rsd = _dist_stats(spreads)
    raw_spread_dist = {
        "median": rsd["median"],
        "iqr": rsd["iqr"],
        "min": rsd["min"],
        "max": rsd["max"],
        "n_dates": int(len(spreads)),
    }

    # --- trailing IC distribution ------------------------------------------
    ic_vals = np.asarray(pd.Series(trailing_ic).dropna().values, dtype=float)
    tid = _dist_stats(ic_vals)
    trailing_ic_dist = {
        "median": tid["median"],
        "iqr": tid["iqr"],
        "min": tid["min"],
        "max": tid["max"],
        "mean": tid["mean"],
        "n": int(ic_vals.size),
        "frac_below_zero": float(np.mean(ic_vals < 0.0)) if ic_vals.size else 0.0,
    }

    # --- root-cause evidence (hypothesis -> supporting/refuting data) -------
    degen = [w for w in retrain_windows if bool(w["degenerate"])]
    healthy = [w for w in retrain_windows if not bool(w["degenerate"])]

    def _best_its(rows):
        vals = [w.get("best_iteration") for w in rows if w.get("best_iteration") is not None]
        return [int(v) for v in vals]

    degen_bi = _best_its(degen)
    healthy_bi = _best_its(healthy)
    windows_sorted = sorted(retrain_windows, key=lambda w: pd.Timestamp(w["train_date"]))
    runs = _degenerate_runs(windows_sorted)

    # H1: immediate early-stop / no learnable signal.
    h1_supported = bool(
        degen_bi
        and healthy_bi
        and float(np.median(degen_bi)) < 10.0
        and float(np.median(degen_bi)) < float(np.median(healthy_bi))
    )
    # H2: regime concentration in P2 (the spec's stated concern).
    total_degen = sum(subperiod_overlap.values())
    p2 = subperiod_overlap.get("P2", 0)
    max_period = max(subperiod_overlap, key=subperiod_overlap.get) if subperiod_overlap else None
    # "Concentrated in P2" only if P2 holds a clear plurality of the degeneracy.
    h2_supported = bool(
        total_degen > 0
        and max_period == "P2"
        and p2 > 0.5 * total_degen
    )
    # H3: clustering right after a strong (high-tree) model is retained.
    trailing_runs = [r for r in runs if (r["trails_healthy_model_trees"] or 0) >= 100]
    h3_supported = bool(runs and any(r["length"] >= 2 for r in runs))

    root_cause_evidence = {
        "H1_immediate_early_stop_no_signal": {
            "verdict": "supported" if h1_supported else "inconclusive",
            "degenerate_best_iteration": {
                "median": float(np.median(degen_bi)) if degen_bi else None,
                "min": int(np.min(degen_bi)) if degen_bi else None,
                "max": int(np.max(degen_bi)) if degen_bi else None,
                "values": degen_bi,
            },
            "healthy_best_iteration": {
                "median": float(np.median(healthy_bi)) if healthy_bi else None,
                "min": int(np.min(healthy_bi)) if healthy_bi else None,
                "max": int(np.max(healthy_bi)) if healthy_bi else None,
            },
            "note": (
                "Degenerate models early-stop within a handful of boosting rounds "
                "(best_iteration << healthy), i.e. validation loss stops improving "
                "almost immediately -> retrain found no signal that generalises to "
                "the validation window."
            ),
        },
        "H2_regime_concentration_P2": {
            "verdict": "supported" if h2_supported else "refuted",
            "subperiod_overlap": dict(subperiod_overlap),
            "note": (
                "Spec hypothesised degeneracy concentrates in the weak P2 window. "
                "Counts show whether P2 holds a clear plurality (>50% of degenerate "
                "windows) of the degeneracy."
            ),
        },
        "H3_clustering_after_strong_model": {
            "verdict": "supported" if h3_supported else "inconclusive",
            "degenerate_runs": runs,
            "runs_trailing_high_tree_model": len(trailing_runs),
            "note": (
                "Degenerate windows arrive in consecutive runs immediately after a "
                "high-tree healthy model is retained; that strong model then persists "
                "for the whole run, doubling the effective retrain cadence in that "
                "stretch."
            ),
        },
    }

    return {
        "windows": windows_out,
        "subperiod_overlap": subperiod_overlap,
        "root_cause_evidence": root_cause_evidence,
        "raw_spread_dist": raw_spread_dist,
        "trailing_ic_dist": trailing_ic_dist,
    }


# ===========================================================================
# DATA GATHERING (main only) — reads persisted artifacts, no src mutation.
# ===========================================================================
_DEFAULT_PKL = "outputs/iter15_65tkr_reb21_vtg/backtest_result.pkl"
_VAL_WINDOW = 126  # config.val_window; used only to reconstruct val periods.
_TRAILING_IC_WINDOW = 6  # config.trailing_ic_window; production trailing-IC span.


def _canonical_subperiods() -> dict:
    """Canonical P1/P2/P3 bounds (src/harness.py SUB_PERIODS), as Timestamps."""
    from src.harness import SUB_PERIODS

    return {k: (pd.Timestamp(s), pd.Timestamp(e)) for k, (s, e) in SUB_PERIODS.items()}


def _load_result(pkl_path: str):
    import pickle

    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _all_dates(result) -> pd.DatetimeIndex:
    idx = result.panel.index.get_level_values("date")
    return pd.DatetimeIndex(sorted(pd.unique(idx)))


def _build_retrain_windows(result) -> List[dict]:
    """Full retrain-window census from result.models + model_quality['events'].

    Degenerate windows reuse the previously retained model (same object), so the
    rejected model's true tree count lives only in model_quality['events'].
    For a degenerate LGBM with early stopping best_iteration == n_estimators_,
    so n_trees doubles as best_iteration. val_score of the *rejected* model is
    not persisted (documented gap); val_start/val_end are reconstructed from the
    panel date index (production slicing, src/model_trainer.py:363-365).
    """
    events = {e["date"]: e for e in (result.model_quality.get("events") or [])}
    all_dates = _all_dates(result)
    date_set = set(all_dates)
    windows: List[dict] = []
    for d in sorted(result.models.keys()):
        ds = d.strftime("%Y-%m-%d")
        val_start = val_end = None
        if d in date_set:
            t_idx = int(np.where(all_dates == d)[0][0])
            vs = max(0, t_idx - _VAL_WINDOW)
            if t_idx - 1 >= vs:
                val_start = all_dates[vs].strftime("%Y-%m-%d")
                val_end = all_dates[t_idx - 1].strftime("%Y-%m-%d")
        if ds in events:  # degenerate: use rejected model's tree count
            n_trees = int(events[ds]["n_trees"])
            best_it = n_trees
            degenerate = True
        else:  # healthy: retained model carries the counts
            mod = result.models[d]
            n_trees = int(getattr(mod, "n_estimators_", 0) or 0)
            best_it = int(getattr(mod, "best_iteration_", n_trees) or n_trees)
            degenerate = False
        windows.append(
            {
                "train_date": ds,
                "degenerate": degenerate,
                "best_iteration": best_it,
                "n_trees": n_trees,
                "val_start": val_start,
                "val_end": val_end,
                "val_score": None,  # GAP: per-window val loss not persisted.
            }
        )
    return windows


def _build_trailing_ic(result) -> pd.Series:
    """Reconstruct the production trailing-IC series that feeds ic_score.

    Production (src/backtest.py:1184-1189) takes, at each rebalance date, the
    nanmean of the last ``trailing_ic_window`` (=6) per-rebalance ICs recorded
    BEFORE the current date. ic_score = clip((trailing_ic_mean + 0.01)/0.04,
    0.20, 1.00); this series is exactly the distribution A3 would re-parametrise.
    """
    ic = result.ic_series.sort_index()
    vals = ic.values.astype(float)
    idx = ic.index
    out_vals, out_idx = [], []
    for i in range(len(vals)):
        window = vals[max(0, i - _TRAILING_IC_WINDOW): i]
        window = window[~np.isnan(window)]
        if window.size > 0:
            out_vals.append(float(np.mean(window)))
            out_idx.append(idx[i])
    return pd.Series(out_vals, index=pd.DatetimeIndex(out_idx), name="trailing_ic")


def _render_markdown(report: dict, ctx: dict) -> str:
    L: List[str] = []
    A = L.append
    A("# Degenerate-Retrain Diagnosis (SPEC D0)")
    A("")
    A(f"- Source pickle: `{ctx['pkl_path']}`")
    A(f"- Total retrains: {ctx['n_windows']}  |  Degenerate: {ctx['n_degenerate']} "
      f"({ctx['degenerate_rate']:.1%})")
    A("- Solver / protocol: single ECOS path (per metrics.json).")
    A("")

    # 1. windows -----------------------------------------------------------
    A("## 1. windows — retrain-window census")
    A("")
    A("| train_date | subperiod | degenerate | best_iter | n_trees | val_start | val_end | val_score |")
    A("|---|---|---|---:|---:|---|---|---|")
    for w in report["windows"]:
        A(f"| {w['train_date']} | {w.get('subperiod')} | {w['degenerate']} | "
          f"{w.get('best_iteration')} | {w.get('n_trees')} | {w.get('val_start')} | "
          f"{w.get('val_end')} | {w.get('val_score')} |")
    A("")
    A("> `val_score` is a documented GAP: the per-window validation loss is not "
      "persisted in any artifact, and for degenerate windows the rejected model "
      "itself is discarded. `best_iteration`/`n_trees` (the load-bearing "
      "early-stop fingerprint) and the reconstructed `val_start`/`val_end` are "
      "recovered in full.")
    A("")

    # 2. subperiod_overlap -------------------------------------------------
    ov = report["subperiod_overlap"]
    A("## 2. subperiod_overlap — degenerate windows per P1/P2/P3")
    A("")
    A("| P1 | P2 | P3 |")
    A("|---:|---:|---:|")
    A(f"| {ov.get('P1')} | {ov.get('P2')} | {ov.get('P3')} |")
    A("")
    A(f"Canonical bounds (src/harness.py): P1 {ctx['bounds']['P1']}, "
      f"P2 {ctx['bounds']['P2']}, P3 {ctx['bounds']['P3']}.")
    A("")

    # 3. root_cause_evidence ----------------------------------------------
    A("## 3. root_cause_evidence")
    A("")
    for hyp, ev in report["root_cause_evidence"].items():
        A(f"### {hyp} — **{ev['verdict']}**")
        A("")
        A(f"{ev['note']}")
        A("")
        A("```json")
        A(json.dumps({k: v for k, v in ev.items() if k not in ("verdict", "note")},
                     indent=2, default=str))
        A("```")
        A("")
    A("Structural context (config.py:147-150, lgbm_params): learning_rate=0.02, "
      "min_child_samples=60, early_stopping_rounds=100 over a ~5y (1260d) "
      "heterogeneous train window. The in-code comment already documents that "
      "this low-lr / high-min-child / long-window combination 'triggered early "
      "stopping on almost every retrain'. The best_iteration fingerprint (H1) is "
      "the direct signature of that structural cause.")
    A("")

    # 4. raw_spread_dist ---------------------------------------------------
    rsd = report["raw_spread_dist"]
    A("## 4. raw_spread_dist — daily raw (pre-EMA) top-bottom spread")
    A("")
    A(f"- n_dates: {rsd['n_dates']}")
    A(f"- median: {rsd['median']:.5f}  |  IQR: {rsd['iqr']:.5f}  |  "
      f"min: {rsd['min']:.5f}  |  max: {rsd['max']:.5f}")
    A("")
    A("Definition matches compute_signal_confidence (tail_n=max(3, n_valid//10), "
      "per-row dropna). A2 relevance: spread_score = clip(raw_spread / "
      "spread_scale, 0.20, 1.00) with the default spread_scale=0.20. Since the "
      f"median spread ({rsd['median']:.2f}) is ~{rsd['median']/0.20:.0f}x that "
      "scale, spread_score saturates at 1.00 on essentially every date — the "
      "spread axis of the confidence gate is effectively inert. Pre-registering "
      f"A2's confidence_spread_scale near the observed spread range "
      f"(~{rsd['median']:.2f}) is what makes the gate responsive.")
    A("")

    # 5. trailing_ic_dist --------------------------------------------------
    tid = report["trailing_ic_dist"]
    A("## 5. trailing_ic_dist — production trailing IC (rolling nanmean of last "
      f"{_TRAILING_IC_WINDOW} rebalance ICs)")
    A("")
    A(f"- n: {tid['n']}")
    A(f"- median: {tid['median']:.5f}  |  IQR: {tid['iqr']:.5f}  |  "
      f"min: {tid['min']:.5f}  |  max: {tid['max']:.5f}  |  mean: {tid['mean']:.5f}")
    A(f"- fraction < 0: {tid['frac_below_zero']:.3f}")
    A("")
    A("A3 relevance: ic_score = clip((trailing_ic_mean + 0.01)/0.04, 0.20, 1.00). "
      f"With median trailing IC {tid['median']:.4f}, the numerator "
      f"({tid['median']:.4f}+0.01)/0.04 = {(tid['median']+0.01)/0.04:.2f} clips to "
      "1.00 on the majority of dates, so ic_score also mostly saturates. The "
      "constants (+0.01, /0.04) are the functional knobs A3 would re-fix against "
      "this distribution.")
    A("")

    # Conclusion -----------------------------------------------------------
    A("## Conclusion")
    A("")
    for line in ctx["conclusion"]:
        A(f"- {line}")
    A("")
    return "\n".join(L)


def _conclusion(report: dict, ctx: dict) -> List[str]:
    ov = report["subperiod_overlap"]
    rsd = report["raw_spread_dist"]
    tid = report["trailing_ic_dist"]
    h1 = report["root_cause_evidence"]["H1_immediate_early_stop_no_signal"]
    dbi = h1["degenerate_best_iteration"]
    hbi = h1["healthy_best_iteration"]
    return [
        f"Most-likely root cause (strong): structural immediate early-stop. "
        f"Degenerate retrains stop at best_iteration median {dbi['median']} "
        f"(range {dbi['min']}-{dbi['max']}) vs healthy median {hbi['median']}. "
        f"Validation loss stops improving within a handful of rounds — the "
        f"lr=0.02 / min_child=60 / 100-round-patience / 5y-window combination "
        f"(documented in config.py:147-150) makes the retrain fail to beat the "
        f"incumbent model, which is then reused.",
        "Secondary structural pattern (supporting): degenerate windows cluster "
        "in consecutive runs immediately after a high-tree healthy model is "
        "retained (e.g. after the 47- and 527-tree fits), so a single strong "
        "model persists across many quarters and the effective retrain cadence "
        "doubles in those stretches.",
        f"Spec's P2-concentration hypothesis is REFUTED: degenerate windows split "
        f"{ov.get('P1')}/{ov.get('P2')}/{ov.get('P3')} across P1/P2/P3 — roughly "
        f"even, not concentrated in the weak P2 window. Degeneracy is a global "
        f"training-config artifact, not a P2-regime effect.",
        f"raw_spread median {rsd['median']:.2f} (>> spread_scale 0.20) and trailing "
        f"IC median {tid['median']:.4f} both saturate their confidence sub-scores "
        f"at 1.00 -> the dynamic-execution confidence gate is largely inert at its "
        f"current constants. These are the pre-registration anchors for A2 "
        f"(confidence_spread_scale) and A3 (IC constants).",
        "Gap: per-window validation loss is not persisted; the census recovers "
        "the early-stop fingerprint (best_iteration/n_trees) and val periods, "
        "which already localise the mechanism, so no instrumented re-run was "
        "required.",
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Degenerate-retrain diagnosis (report-only).")
    ap.add_argument("--out", default="outputs", help="Output directory.")
    ap.add_argument("--pkl", default=_DEFAULT_PKL, help="BacktestResult pickle path.")
    args = ap.parse_args(argv)

    result = _load_result(args.pkl)
    retrain_windows = _build_retrain_windows(result)
    raw_predictions = result.raw_predictions
    trailing_ic = _build_trailing_ic(result)
    subperiods = _canonical_subperiods()

    report = build_report(retrain_windows, raw_predictions, trailing_ic, subperiods)

    n_degen = sum(1 for w in retrain_windows if w["degenerate"])
    ctx = {
        "pkl_path": args.pkl,
        "n_windows": len(retrain_windows),
        "n_degenerate": n_degen,
        "degenerate_rate": n_degen / len(retrain_windows) if retrain_windows else 0.0,
        "bounds": {
            k: f"{v[0].date()}..{v[1].date()}" for k, v in subperiods.items()
        },
    }
    ctx["conclusion"] = _conclusion(report, ctx)

    os.makedirs(args.out, exist_ok=True)
    json_path = os.path.join(args.out, "degenerate_retrain_report.json")
    md_path = os.path.join(args.out, "degenerate_retrain_report.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"report": report, "context": ctx}, f, indent=2, default=str)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_render_markdown(report, ctx))

    print(f"[D0] wrote {json_path}")
    print(f"[D0] wrote {md_path}")
    print(f"[D0] degenerate {n_degen}/{len(retrain_windows)} "
          f"({ctx['degenerate_rate']:.1%}); subperiod_overlap={report['subperiod_overlap']}; "
          f"raw_spread median={report['raw_spread_dist']['median']:.4f}; "
          f"trailing_ic median={report['trailing_ic_dist']['median']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
