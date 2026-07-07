#!/usr/bin/env python
"""A4 — LightGBM seed-ensemble (k=5) arm driver (2-pass injection, src/ untouched).

The production prediction engine trains LightGBM with a single fixed seed
(random_state=42, `src/config.py:158-172`). A single-seed walk-forward carries
estimation noise; averaging k independent seeds both shrinks that noise and lets
us quantify seed luck (per-seed IR spread). This arm evaluates the pre-registered
k=5 ensemble {42, 43, 44, 45, 46} WITHOUT touching production code:

    seed 42     : reuse the certified S0 harvest (its raw_predictions).
    seeds 43-46 : harvest 4 full runs via `run_variant.py` on generated
                  variants/exp_seed{n}.yaml (canonical variant + lgbm seed only).

    combine     : cell-wise finite-value mean of the k raw z-panels, then a
                  per-date cross-sectional RE-STANDARDIZATION (the model_trainer z
                  idiom), then the standard prediction EMA (alpha=0.5,
                  apply_prediction_ema), then inject the pre-overlay panel into the
                  UNCHANGED production MVO via run_backtest(precomputed_predictions=
                  ...). Overlays run exactly once (no double overlay).

Two pure functions (`combine_seed_panels`, `nan_mask_mismatch_rate`) carry the
whole combine contract and are import-only for the acceptance tests; the heavy
harvest/backtest plumbing lives in main() with lazy imports (so `--help` / test
import stay fast).

Usage
-----
  python scripts/run_seed_ensemble_arm.py                 # harvest 43-46 + combine + identity + arm
  python scripts/run_seed_ensemble_arm.py --skip-harvest  # reuse existing per-seed pkls
  python scripts/run_seed_ensemble_arm.py --identity-only  # only the alpha=0.5 identity of S0
"""
from __future__ import annotations

import os
# Single-thread BLAS BEFORE numpy import (memory + determinism), matching
# scripts/run_adaptive_ema_arm.py / scripts/run_dr_alpha.py.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# UTF-8, line-buffered stdout so unicode (em-dash, Delta) survives a cp949
# console / redirect and progress flushes promptly.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# (heavy imports — src.backtest / UniverseData / yaml / pickle — are done lazily
# in main() so importing this module for the pure functions stays cheap.)

SEEDS = (42, 43, 44, 45, 46)          # pinned k=5 ensemble (spec §사전등록)
HARVEST_SEEDS = (43, 44, 45, 46)      # 42 reuses the certified S0 harvest
EMA_ALPHA = 0.5                       # standard prediction EMA (spec §사전등록)
NAN_MISMATCH_GATE = 0.001             # halt-and-report threshold (spec §13)

VARIANT_PATH = ROOT / "variants" / "iter15_65tkr_reb21_vtg.yaml"
PKL_PATH = ROOT / "outputs" / "iter15_65tkr_reb21_vtg" / "backtest_result.pkl"
S0_METRICS_PATH = ROOT / "outputs" / "iter15_65tkr_reb21_vtg" / "metrics.json"
VARIANTS_DIR = ROOT / "variants"
OUT_DIR = ROOT / "outputs" / "exp_seed_ensemble"
OOS_CUTOFF = pd.Timestamp("2024-12-31")


# ===========================================================================
# Pure functions — the pinned contract (acceptance tests import exactly these)
# ===========================================================================
def combine_seed_panels(panels: "list[pd.DataFrame]") -> pd.DataFrame:
    """Combine k seed raw-prediction panels into one (NO EMA — that is a later
    stage applied by main via src.model_trainer.apply_prediction_ema).

    All panels are assumed aligned on the SAME date index and ticker columns.

    STEP 1 — cell-wise finite-value mean across the k panels: each cell is the
        arithmetic mean of the FINITE seed values there (NaNs skipped); a cell
        that is NaN in EVERY seed stays NaN.
    STEP 2 — per-DATE cross-sectional re-standardization applying the
        `src.model_trainer.predict_cross_sectional` z idiom (lines 240-245) to
        each date row across tickers:
            mean = row.mean()   # pandas skipna=True
            std  = row.std()    # pandas ddof=1 (sample), skipna=True
            if std > 0:         # strict guard
                row = (row - mean) / std
        A constant row (std==0), a single-finite-value row (std==NaN), and an
        all-NaN row (std==NaN) are all left UNCHANGED (no divide-by-zero). NaN
        cells stay NaN under the affine transform, so the output NaN mask equals
        the STEP-1 all-seed-NaN mask.
    """
    idx, cols = panels[0].index, panels[0].columns
    # STEP 1: finite-value mean (sum of finite / count of finite; all-NaN -> NaN).
    arr = np.stack(
        [p.reindex(index=idx, columns=cols).values.astype(float) for p in panels],
        axis=0,
    )  # (k, D, T)
    finite = np.isfinite(arr)
    cnt = finite.sum(axis=0)                        # (D, T) number of finite seeds
    s = np.where(finite, arr, 0.0).sum(axis=0)      # (D, T) sum of finite seeds
    with np.errstate(invalid="ignore"):
        mean = np.where(cnt > 0, s / np.where(cnt > 0, cnt, 1.0), np.nan)
    combined = pd.DataFrame(mean, index=idx, columns=cols)

    # STEP 2: per-date cross-sectional re-standardization (model_trainer z idiom,
    # applied row by row on the pandas Series so mean/std match byte-for-byte:
    # skipna mean, ddof=1 sample std, strict `if std > 0` guard).
    for d in combined.index:
        row = combined.loc[d]
        std = row.std()          # ddof=1, skipna
        if std > 0:              # constant / single-value / all-NaN rows -> untouched
            combined.loc[d] = (row - row.mean()) / std
    return combined


def nan_mask_mismatch_rate(panels: "list[pd.DataFrame]") -> float:
    """Fraction of shared-grid cell positions whose per-seed NaN status is NOT
    unanimous — NaN in >=1 panel AND finite in >=1 panel. A cell NaN in ALL
    panels is unanimous (legitimately absent) and is NOT a mismatch.

        rate = (# non-unanimous cells) / (total cells in the grid)   in [0, 1]

    Panels are assumed aligned on the same date index and ticker columns; fully
    consistent masks -> 0.0. The rate is symmetric in the panel list.
    """
    idx, cols = panels[0].index, panels[0].columns
    masks = np.stack(
        [p.reindex(index=idx, columns=cols).isna().values for p in panels], axis=0
    )  # (k, D, T) bool
    all_nan = masks.all(axis=0)
    any_nan = masks.any(axis=0)
    mismatched = any_nan & ~all_nan                 # NaN in some seeds, finite in others
    return float(mismatched.sum()) / float(mismatched.size)


# ===========================================================================
# main() — seed harvest + combine + identity + arm (lazy heavy imports)
# ===========================================================================
def _ir(active: pd.Series) -> float:
    a = active.dropna()
    if len(a) < 2:
        return float("nan")
    sd = a.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return float("nan")
    return float(a.mean() / sd * np.sqrt(252))


def _oos_split(result) -> dict:
    port = result.portfolio_returns.dropna()
    bm = result.benchmark_returns.reindex(port.index)
    active = (port - bm).dropna()
    ins = active[active.index <= OOS_CUTOFF]
    oos = active[active.index > OOS_CUTOFF]
    return {
        "full_ir": _ir(active), "full_n": int(len(active)),
        "insample_ir": _ir(ins), "insample_n": int(len(ins)),
        "oos_ir": _ir(oos), "oos_n": int(len(oos)),
    }


def _headline(result) -> dict:
    from src.harness import sub_period_irs
    m = result.compute_metrics()
    port = result.portfolio_returns.dropna()
    bm = result.benchmark_returns.dropna()
    m["sub_periods"] = sub_period_irs(port, bm)
    m["oos_split"] = _oos_split(result)
    return m


def _seed_out_dir(seed: int) -> Path:
    return ROOT / "outputs" / f"exp_seed{seed}"


def _seed_variant_path(seed: int) -> Path:
    return VARIANTS_DIR / f"exp_seed{seed}.yaml"


def write_seed_variant(seed: int) -> Path:
    """Derive variants/exp_seed{seed}.yaml from the canonical variant, changing
    ONLY the label, out_dir, and lgbm_params.random_state.

    build_override_config uses dataclasses.replace, which REPLACES lgbm_params
    wholesale (no deep-merge) — so the seed variant must carry the FULL
    lgbm_params block (canonical + random_state=seed), not a bare
    {random_state: seed}. Deriving from the canonical manifest guarantees that.
    """
    import copy
    import yaml
    with VARIANT_PATH.open("r", encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh) or {}
    manifest = copy.deepcopy(manifest)
    manifest["label"] = f"exp_seed{seed}"
    manifest["out_dir"] = f"outputs/exp_seed{seed}"
    manifest["description"] = (
        f"A4 seed-ensemble member. Canonical iter15_65tkr_reb21_vtg with "
        f"lgbm_params.random_state={seed} (full harvest, no cache). Generated by "
        f"scripts/run_seed_ensemble_arm.py — do not hand-edit."
    )
    overrides = dict(manifest.get("overrides") or {})
    lgbm = dict(overrides.get("lgbm_params") or {})
    lgbm["random_state"] = seed
    overrides["lgbm_params"] = lgbm
    manifest["overrides"] = overrides
    path = _seed_variant_path(seed)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(manifest, fh, sort_keys=False, default_flow_style=False)
    return path


def _harvest_seed(seed: int, no_cache: bool = True) -> None:
    """Run one FULL seed harvest via run_variant.py as a single foreground
    subprocess (isolated config; no parallel spawn — zombie-hang history)."""
    variant = write_seed_variant(seed)
    cmd = [sys.executable, str(ROOT / "run_variant.py"), "--variant", str(variant)]
    if no_cache:
        cmd.append("--no-cache")
    print(f"[a4] harvesting seed {seed}: {' '.join(cmd)}")
    t0 = time.time()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"seed {seed} harvest failed (rc={proc.returncode})")
    print(f"[a4] seed {seed} harvested in {time.time()-t0:.0f}s")


def _load_raw_panel(seed: int):
    """Load the raw (pre-EMA) z-prediction panel for a seed from its pkl."""
    import pickle
    pkl = PKL_PATH if seed == 42 else _seed_out_dir(seed) / "backtest_result.pkl"
    with pkl.open("rb") as fh:
        res = pickle.load(fh)
    return res.raw_predictions


def _load_seed_ir(seed: int) -> float:
    """Full-period IR for a seed (from its metrics.json; seed 42 == S0)."""
    mpath = S0_METRICS_PATH if seed == 42 else _seed_out_dir(seed) / "metrics.json"
    with mpath.open("r", encoding="utf-8") as fh:
        return float(json.load(fh)["metrics"].get("information_ratio"))


def _compose_variant_config():
    """Rebuild the S0 variant config exactly as run_variant.compose_config does
    (the pkl carries no config), so the injected identity/arm runs use the same
    optimizer/overlay settings that produced S0."""
    import yaml
    from src.harness import build_override_config
    with VARIANT_PATH.open("r", encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh) or {}
    overrides = dict(manifest.get("overrides") or {})
    tuning_mode = manifest.get("tuning_mode", "production")
    if tuning_mode == "tuning":
        overrides["enforce_oos_holdout"] = True
    else:  # production / oos_verify
        overrides["enforce_oos_holdout"] = False
    return build_override_config(overrides)


def _inject_run(data, base, cfg, predictions, raw_predictions):
    """Run the UNCHANGED production MVO on an injected prediction panel, mirroring
    run_variant's checkpoint-reuse injection (overlays applied exactly once)."""
    from src.backtest import run_backtest
    return run_backtest(
        data,
        precomputed_panel=base.panel,
        precomputed_feature_names=base.feature_names,
        precomputed_feature_groups=base.feature_groups,
        precomputed_targets=base.targets,
        precomputed_models=base.models,
        precomputed_predictions=predictions,
        precomputed_raw_predictions=raw_predictions,
        config=cfg,
    )


def _panel_diagnostics(panels, combined) -> dict:
    """Prediction spread pre/post ensembling: mean pairwise cross-seed correlation
    (over shared finite cells) and the cross-sectional variance-reduction ratio
    (mean per-seed CS variance vs the combined panel's), averaged over dates."""
    idx, cols = panels[0].index, panels[0].columns
    arr = np.stack(
        [p.reindex(index=idx, columns=cols).values.astype(float) for p in panels],
        axis=0,
    )  # (k, D, T)
    k = arr.shape[0]
    corrs = []
    for i in range(k):
        for j in range(i + 1, k):
            a, b = arr[i].ravel(), arr[j].ravel()
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() >= 2:
                c = np.corrcoef(a[m], b[m])[0, 1]
                if np.isfinite(c):
                    corrs.append(float(c))
    comb = combined.values.astype(float)
    per_seed_var, comb_var = [], []
    for d in range(arr.shape[1]):
        sv = [np.nanvar(arr[s, d]) for s in range(k)]
        sv = [v for v in sv if np.isfinite(v)]
        cv = np.nanvar(comb[d])
        if sv and np.isfinite(cv):
            per_seed_var.append(float(np.mean(sv)))
            comb_var.append(float(cv))
    mean_seed_var = float(np.mean(per_seed_var)) if per_seed_var else float("nan")
    mean_comb_var = float(np.mean(comb_var)) if comb_var else float("nan")
    return {
        "mean_pairwise_seed_corr": float(np.mean(corrs)) if corrs else float("nan"),
        "n_seed_pairs": len(corrs),
        "mean_per_seed_cs_var": mean_seed_var,
        "mean_combined_cs_var": mean_comb_var,
        "cs_var_reduction_ratio": (
            (1.0 - mean_comb_var / mean_seed_var)
            if (mean_seed_var and np.isfinite(mean_seed_var) and mean_seed_var != 0)
            else float("nan")
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--identity-only", action="store_true",
                    help="run only the alpha=0.5 identity reproduction of S0")
    ap.add_argument("--skip-harvest", action="store_true",
                    help="reuse existing per-seed pkls (skip the 43-46 harvest)")
    args = ap.parse_args()

    import pickle
    from src.harness import inject_config
    from src.model_trainer import apply_prediction_ema

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load the certified S0 harvest (seed 42): raw predictions + reusable stages.
    print(f"[a4] loading S0 harvest (seed 42): {PKL_PATH}")
    with PKL_PATH.open("rb") as fh:
        base = pickle.load(fh)
    raw42 = base.raw_predictions
    calendar = raw42.index

    cfg = _compose_variant_config()
    inject_config(cfg)
    print(f"[a4] raw_predictions (seed 42) {raw42.shape} "
          f"span {calendar[0].date()}..{calendar[-1].date()}")

    # UniverseData is the one stage the pkl does not carry (needed by run_backtest
    # for the simulation loop / benchmark). Everything else is injected precomputed.
    from src.data_loader import UniverseData
    data = UniverseData(cfg.data_path, config=cfg)

    with S0_METRICS_PATH.open("r", encoding="utf-8") as fh:
        s0 = json.load(fh)["metrics"]

    # --- Pass 1: identity (alpha == 0.5 on seed 42) must reproduce S0.
    print("[a4] identity pass: apply_prediction_ema(raw42, 0.5) -> production MVO")
    t0 = time.time()
    pred_half = apply_prediction_ema(raw42, EMA_ALPHA)
    res_id = _inject_run(data, base, cfg, pred_half, raw42)
    m_id = _headline(res_id)
    id_dir = OUT_DIR / "identity"
    id_dir.mkdir(parents=True, exist_ok=True)
    keys = ("information_ratio", "tracking_error", "avg_annual_turnover",
            "active_return")
    deltas = {k: (m_id.get(k), s0.get(k),
                  (None if (m_id.get(k) is None or s0.get(k) is None)
                   else float(m_id[k] - s0[k]))) for k in keys}
    print(f"[a4] identity done in {time.time()-t0:.0f}s — vs S0:")
    for k, (got, ref, d) in deltas.items():
        print(f"      {k}: {got} vs S0 {ref}  (Δ {d})")
    with (id_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump({"metrics": m_id, "s0_metrics": s0, "deltas": deltas},
                  fh, indent=2, default=str)
    max_abs_delta = max(abs(d[2]) for d in deltas.values() if d[2] is not None)
    print(f"[a4] identity max |Δ| over {list(keys)} = {max_abs_delta:.3e}")
    if max_abs_delta > 1e-6:
        print("[a4] WARNING: identity did NOT reproduce S0 within 1e-6 — "
              "investigate injection path before trusting the arm.")

    if args.identity_only:
        print("[a4] --identity-only set; skipping harvest + arm.")
        return 0

    # --- Harvest seeds 43-46 (full, sequential foreground) unless reusing pkls.
    if not args.skip_harvest:
        for seed in HARVEST_SEEDS:
            _harvest_seed(seed, no_cache=True)
    else:
        print("[a4] --skip-harvest set; reusing existing per-seed pkls.")

    # --- Load the k=5 raw panels + per-seed full-run IR (seed luck).
    panels = [_load_raw_panel(s) for s in SEEDS]
    per_seed_ir = {s: _load_seed_ir(s) for s in SEEDS}
    print(f"[a4] per-seed full-run IR: {per_seed_ir}")

    # NaN-mask consistency gate (structural assumption — spec §13).
    mismatch = nan_mask_mismatch_rate(panels)
    print(f"[a4] NaN-mask mismatch rate = {mismatch:.5f} (gate {NAN_MISMATCH_GATE})")
    if mismatch > NAN_MISMATCH_GATE:
        print(f"[a4] HALT: NaN-mask mismatch {mismatch:.5f} > {NAN_MISMATCH_GATE} "
              "— seeds do not share the data-availability grid; investigate "
              "before combining (CLAUDE.md §9).")
        return 2

    # --- Combine -> EMA -> inject arm.
    combined = combine_seed_panels(panels)
    diag = _panel_diagnostics(panels, combined)
    print(f"[a4] ensemble diag: mean_pairwise_seed_corr="
          f"{diag['mean_pairwise_seed_corr']:.4f} "
          f"cs_var_reduction_ratio={diag['cs_var_reduction_ratio']:.4f}")
    pred_arm = apply_prediction_ema(combined, EMA_ALPHA)

    print("[a4] arm pass: seed-ensemble combined -> EMA -> production MVO")
    t1 = time.time()
    res_arm = _inject_run(data, base, cfg, pred_arm, combined)
    m_arm = _headline(res_arm)
    arm_dir = OUT_DIR / "arm"
    arm_dir.mkdir(parents=True, exist_ok=True)
    b, r = m_id, m_arm
    print(f"[a4] arm done in {time.time()-t1:.0f}s — "
          f"IR {b['information_ratio']:.3f} -> {r['information_ratio']:.3f} "
          f"(Δ{r['information_ratio']-b['information_ratio']:+.3f}) | "
          f"TE {b['tracking_error']:.4f} -> {r['tracking_error']:.4f} | "
          f"turn {b['avg_annual_turnover']:.3f} -> {r['avg_annual_turnover']:.3f}")
    with (arm_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump({"metrics": m_arm, "identity_metrics": m_id, "s0_metrics": s0,
                   "seeds": list(SEEDS), "per_seed_full_run_ir": per_seed_ir,
                   "nan_mask_mismatch_rate": mismatch,
                   "ensemble_diagnostics": diag,
                   "ema_alpha": EMA_ALPHA},
                  fh, indent=2, default=str)
    print(f"[a4] artifacts: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
