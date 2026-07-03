#!/usr/bin/env python
"""Pictet portfolio-logic evaluation — self-contained, end-to-end reproducible runner.

This project is a SELF-CONTAINED fork: the cc2_rl signal/backtest engine is
vendored under ./src (mirror layout), so every stage runs from THIS folder with
NO dependency on the cc2_rl checkout (no PYTHONPATH=CC2). Each stage is a single
FOREGROUND subprocess (no background spawn — avoids the venv_vf_new pandas
zombie-hang). All artifacts land under ./outputs and ./logs. Gate verdicts
(CLAUDE.md §2-§4) are applied from the stage outputs and written to
outputs/adoption_summary.json.

Stages
------
  0  S0 ECOS baseline         run_variant.py        -> outputs/iter15_65tkr_reb21_vtg/metrics.json
        gate: realized_beta ~1.0 => P2 (beta-neutral) SHELVED before any code
  1  Alpha attribution A/B/C  scripts/run_alpha_attribution.py -> outputs/alpha_attribution/summary.json
  2  Overlay ablation 2^3     scripts/run_overlay_ablation.py  -> outputs/overlay_ablation/summary.json
  3  Factor-neutral ablation  scripts/run_factor_ablation.py   -> outputs/factor_ablation/summary.json
  4  Selection-bias / DSR      run_selection_bias.py            -> outputs/csv/selection_bias_metrics.csv

Usage (run FROM this folder with the venv python)
-------------------------------------------------
  python run_pictet_adoption.py                 # all stages, then verdicts
  python run_pictet_adoption.py --stages 0 1    # a subset, in order
  python run_pictet_adoption.py --summary-only  # re-derive verdicts from existing outputs

Notes
-----
* The engine under ./src is a vendored SNAPSHOT of cc2_rl (see ENGINE_PROVENANCE.md).
  It is now THIS project's own source — edit here, not the cc2_rl checkout.
* Market data is the shared workbook referenced by src/config.py:data_path
  (outside both repos); it is not copied. Stage 0 recomputes from scratch.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Windows consoles default to cp949 here; the JSON we print carries non-ASCII.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = Path(__file__).resolve().parent  # self-contained project root (== CWD for stages)
PY = sys.executable
VARIANT = HERE / "variants" / "iter15_65tkr_reb21_vtg.yaml"
LOGS = HERE / "logs"
OUT = HERE / "outputs"

SE_IR = 0.36  # 1 standard error on IR (CLAUDE.md §2-4 adoption bar)

STAGES = {
    0: ("S0 ECOS baseline", [str(HERE / "run_variant.py"), "--variant", str(VARIANT)]),
    1: ("alpha attribution (A/B/C)", [str(HERE / "scripts" / "run_alpha_attribution.py")]),
    2: ("overlay ablation 2^3", [str(HERE / "scripts" / "run_overlay_ablation.py")]),
    3: ("factor-neutral ablation", [str(HERE / "scripts" / "run_factor_ablation.py")]),
    4: (
        "selection-bias / DSR",
        [str(HERE / "run_selection_bias.py"), "--auto", "--label", "iter15_65tkr_reb21_vtg"],
    ),
}


def _preflight() -> None:
    if not (HERE / "src").is_dir():
        sys.exit(f"[adoption] vendored engine not found: {HERE / 'src'} (see ENGINE_PROVENANCE.md)")
    if not VARIANT.exists():
        sys.exit(f"[adoption] production variant not found: {VARIANT}")
    LOGS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    # Confirm ECOS is the active solver (single-protocol invariant, §2-2).
    env = _env()
    r = subprocess.run(
        [PY, "-c", "import cvxpy as cp; print('ECOS' in cp.installed_solvers())"],
        cwd=str(HERE), env=env, capture_output=True, text=True,
    )
    if "True" not in r.stdout:
        # Surface rc + stderr so a swallowed cvxpy IMPORT error isn't misread as
        # a missing solver (review L5). ASCII-only (cp949 stderr safety).
        sys.exit(f"[adoption] ECOS check failed (rc={r.returncode}); cvxpy import or "
                 f"solver query did not return True (sec.2-2 single ECOS protocol). "
                 f"stderr={(r.stderr or '')[-400:]!r}")
    print("[adoption] preflight OK - ECOS present, vendored engine + variant found.")


def _env() -> dict:
    env = dict(os.environ)
    # Self-contained: put THIS project root on PYTHONPATH so `from src.X import …`
    # resolves to ./src (the vendored engine) — no cc2_rl checkout needed.
    env["PYTHONPATH"] = str(HERE) + os.pathsep + env.get("PYTHONPATH", "")
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env.setdefault(v, "1")
    return env


def _run_stage(idx: int) -> None:
    title, args = STAGES[idx]
    log = LOGS / f"stage{idx}.log"
    print(f"\n[adoption] === stage {idx}: {title} === (log: {log})")
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        fh.write(f"START {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.flush()
        proc = subprocess.run(
            [PY, *args], cwd=str(HERE), env=_env(),
            stdout=fh, stderr=subprocess.STDOUT,
        )
        fh.write(f"\nEND rc={proc.returncode} ({time.time()-t0:.0f}s)\n")
    if proc.returncode != 0:
        sys.exit(f"[adoption] stage {idx} FAILED (rc={proc.returncode}) — see {log}")
    print(f"[adoption] stage {idx} done in {time.time()-t0:.0f}s")


# ----------------------------------------------------------------------------
# Verdicts — read stage outputs and apply the CLAUDE.md gates.
# ----------------------------------------------------------------------------
def _load(path: Path):
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _load_selection_bias_csv() -> dict:
    path = OUT / "csv" / "selection_bias_metrics.csv"
    if not path.exists():
        return {
            "status": "missing",
            "report_path": str(OUT / "reports" / "selection_bias_report.md"),
        }
    rows = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rows[row.get("metric")] = row.get("value")

    def _num(key):
        val = rows.get(key)
        if val is None:
            return None
        try:
            f = float(val)
            return int(f) if key in {"N_trials", "Late_Entrants"} else f
        except ValueError:
            return val

    verdict = rows.get("Verdict")
    return {
        "observed_sr": _num("Observed_SR"),
        "deflated_sr": _num("Deflated_SR"),
        "dsr_p_value": _num("DSR_p_value"),
        "n_trials": _num("N_trials"),
        "adjusted_sr": _num("Adjusted_SR"),
        "grid_haircut": _num("Grid_Haircut"),
        "min_trl_years": _num("MinTRL_years"),
        "available_years": _num("Available_years"),
        "late_entrants": _num("Late_Entrants"),
        "sub_period_ir": {
            "P1_ir": _num("Sub_Period_1_IR"),
            "P2_ir": _num("Sub_Period_2_IR"),
            "P3_ir": _num("Sub_Period_3_IR"),
        },
        "verdict": verdict,
        "gate": "PASS" if verdict == "PASS" else "FAIL",
        "report_path": str(OUT / "reports" / "selection_bias_report.md"),
    }


def _verdict_baseline() -> dict:
    m = _load(OUT / "iter15_65tkr_reb21_vtg" / "metrics.json")
    if m is None:
        return {"status": "missing"}
    m = m.get("metrics", m)
    beta = m.get("realized_beta")
    # Proceed with P2 only in CLAUDE.md sec.3's window (~0.90-0.93); else SHELVE.
    # NaN/None -> SHELVED (safe default). (review L4: align to the contract band)
    p2 = "PROCEED" if (beta is not None and 0.90 <= beta <= 0.93) else "SHELVED"
    return {
        "information_ratio": m.get("information_ratio"),
        "tracking_error": m.get("tracking_error"),
        "avg_annual_turnover": m.get("avg_annual_turnover"),
        "realized_beta": beta,
        "realized_active_beta": m.get("realized_active_beta"),
        "P2_beta_neutral_gate": p2,
    }


def _verdict_attribution() -> dict:
    s = _load(OUT / "alpha_attribution" / "summary.json")
    if s is None:
        return {"status": "missing"}
    h = (s.get("legA_B") or {}).get("headline") or {}
    rt = s.get("roundtrip_full_vs_s0_abs_diff")
    return {
        "linear_share": h.get("linear_share"),
        "nonlinear_share_upper_bound": h.get("nonlinear_share_upper_bound"),
        "legC_construction_active_delta": s.get("legC_construction_active_delta"),
        "roundtrip_full_vs_s0_abs_diff": rt,
        "roundtrip_ok": (rt is not None and abs(rt) < 1e-6),
        "weight_invariant": True,  # attribution changes no weights (read-only)
        "production": "OFF-default, on-demand (EXPENSIVE SHAP) - no weight change",
    }


def _verdict_overlay() -> dict:
    rows = _load(OUT / "overlay_ablation" / "summary.json")
    if rows is None:
        return {"status": "missing"}
    full = rows.get("vtg1_grw1_pead1")
    if full is None:
        # L2: a partial/older summary without the (1,1,1) arm must not silently
        # read as "keep all" (empty decisions -> all([]) == True). Flag it.
        return {"status": "incomplete",
                "summary": "overlay summary missing the (1,1,1) on-baseline arm - rerun stage 2"}
    loo = {"vtg": "vtg0_grw1_pead1", "growth": "vtg1_grw0_pead1", "pead": "vtg1_grw1_pead0"}
    out = {}
    decisions = {}
    for name, key in loo.items():
        r = rows.get(key)
        if not r:
            continue
        dIR = full["information_ratio"] - r["information_ratio"]
        fsp, rsp = full["sub_periods"], r["sub_periods"]
        dsub = [fsp[f"{p}_ir"] - rsp[f"{p}_ir"] for p in ("P1", "P2", "P3")]
        removing_helps = -dIR  # +ve if dropping the overlay raises full-period IR
        # M2: REMOVE only if dropping helps full-period AND every sub-period also
        # shows dropping raises IR (dsub < 0 in the full-minus-dropped convention).
        # Direction-TIED, not the old direction-agnostic all(<0) or all(>0).
        removal_sign_consistent = all(x < 0 for x in dsub)
        remove = (removing_helps > SE_IR) and removal_sign_consistent
        out[name] = {"marginal_dIR": dIR, "dsub": dsub}
        decisions[name] = "REMOVE" if remove else "KEEP"
    if not decisions:
        return {"status": "incomplete",
                "on_baseline_IR": full.get("information_ratio"),
                "summary": "overlay summary has (1,1,1) but no leave-one-out arms - rerun stage 2"}
    return {
        "on_baseline_IR": full["information_ratio"],
        "all_off_IR": (rows.get("vtg0_grw0_pead0") or {}).get("information_ratio"),
        "leave_one_out": out,
        "decisions": decisions,
        "summary": "no overlay shows clear (>1SE) sign-consistent harm => keep all"
        if all(v == "KEEP" for v in decisions.values()) else "review: a removal cleared the bar",
    }


def _verdict_factor() -> dict:
    s = _load(OUT / "factor_ablation" / "summary.json")
    if s is None:
        return {"status": "missing"}
    # GAP2: if stage 3 flagged the OFF arm != S0, the exposure verdict is built on
    # a broken harvest — refuse to judge. (The script also returns rc=1 so stage 3
    # aborts; guard here too for --summary-only re-derivation from a stale file.)
    rt = s.get("roundtrip_off_vs_s0_abs_diff")
    if rt is not None and rt > 1e-3:
        return {"status": "harvest_invalid",
                "roundtrip_off_vs_s0_abs_diff": rt,
                "decision": "FAIL - OFF arm does not reproduce S0 (harvest confound); rerun stage 3"}
    drops = s.get("exposure_drop_pct") or {}
    vals = [v for v in drops.values() if isinstance(v, (int, float))]
    binds = bool(vals) and all(v > 20.0 for v in vals)
    mo = (s.get("metrics") or {}).get("off") or {}
    mn = (s.get("metrics") or {}).get("on") or {}
    dIR = (mo.get("information_ratio") or 0) - (mn.get("information_ratio") or 0)
    dTE = (mo.get("tracking_error") or 0) - (mn.get("tracking_error") or 0)
    # M3: sec.2-5 collapse guard. A too-strong penalty can drive the book toward
    # the benchmark; exposure_drop alone can't tell a healthy bind from a collapse.
    # Flag FAIL (independent of IR) if, vs the OFF arm (== S0), the ON arm's TE or
    # active share falls below half, or its fallback-to-bm rate jumps materially.
    ash = s.get("active_share") or {}
    ash_off, ash_on = ash.get("off"), ash.get("on")
    te_off, te_on = mo.get("tracking_error"), mn.get("tracking_error")
    fr_off, fr_on = mo.get("optimizer_failure_rate"), mn.get("optimizer_failure_rate")
    collapse = []
    if te_off and te_on is not None and te_on < 0.5 * te_off:
        collapse.append(f"TE {te_on:.4f} < 0.5*off {te_off:.4f}")
    if ash_off and ash_on is not None and ash_on < 0.5 * ash_off:
        collapse.append(f"active_share {ash_on:.4f} < 0.5*off {ash_off:.4f}")
    if fr_off is not None and fr_on is not None and fr_on > fr_off + 0.10:
        collapse.append(f"fallback_rate {fr_on:.3f} >> off {fr_off:.3f}")
    # Factor-neutral is a risk-reduction lever: when it binds it cuts BOTH the
    # active style exposure and (usually) IR/TE, because the concentrated style
    # bets are intentional alpha (sec.2-5). It is NOT a do-no-harm change, so it is
    # never auto-activated on an IR-noise test — it stays a confirmed OFF-default
    # lever, enabled only under an explicit style-risk-budget mandate (sec.8).
    if collapse:
        decision = ("FAIL - book collapsed toward benchmark (sec.2-5): "
                    + "; ".join(collapse) + ". Do NOT activate regardless of IR.")
    elif not binds:
        decision = "SHELVE (non-binding - TE-var already neutralizes style)"
    else:
        decision = ("CONFIRMED LEVER - keep OFF-default. Binds (style exposure "
                    "down) but trades IR/TE; activate only under an explicit "
                    "style-risk-budget mandate (sec.8, sec.2-5).")
    return {
        "applied_dates": s.get("applied_dates"),
        "impute_frac": s.get("impute_frac"),
        "exposure_drop_pct": drops,
        "binds": binds,
        "collapsed": bool(collapse),
        "active_share_off_on": [ash_off, ash_on],
        "fallback_rate_off_on": [fr_off, fr_on],
        "IR_cost_off_minus_on": dIR,
        "TE_drop_off_minus_on": dTE,
        "off_IR": mo.get("information_ratio"),
        "on_IR": mn.get("information_ratio"),
        "roundtrip_off_vs_s0_abs_diff": rt,
        "decision": decision,
    }


def _summary() -> dict:
    verdicts = {
        "stage0_baseline": _verdict_baseline(),
        "stage1_attribution": _verdict_attribution(),
        "stage2_overlay": _verdict_overlay(),
        "stage3_factor": _verdict_factor(),
        "selection_bias": _load_selection_bias_csv(),
        "solver": "ECOS (single-protocol, sec.2-2 - do not compare to legacy SCS)",
        "project_root": str(HERE),
        "engine": "vendored ./src (self-contained; see ENGINE_PROVENANCE.md)",
    }
    path = OUT / "adoption_summary.json"
    path.write_text(json.dumps(verdicts, indent=2, default=str), encoding="utf-8")
    print(f"\n[adoption] wrote {path}")
    print(json.dumps(verdicts, indent=2, default=str))
    return verdicts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stages", nargs="*", type=int, default=sorted(STAGES),
                    help="stage indices to run, in order (default: all)")
    ap.add_argument("--summary-only", action="store_true",
                    help="skip runs; only re-derive verdicts from existing outputs")
    args = ap.parse_args()

    if args.summary_only:
        _summary()
        return 0

    _preflight()
    for idx in args.stages:
        if idx not in STAGES:
            sys.exit(f"[adoption] unknown stage {idx} (valid: {sorted(STAGES)})")
        _run_stage(idx)
    _summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
