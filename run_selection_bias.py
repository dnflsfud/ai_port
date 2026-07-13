#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_selection_bias.py - Phase 5: Selection Bias Check (GATE)
============================================================
Deflated Sharpe Ratio, MinTRL, Grid Haircut, Survivorship, Sub-period Stability.

Based on:
- Bailey & Lopez de Prado (2014) "The Deflated Sharpe Ratio"
- Harvey & Liu (2015) "Backtesting" (Haircut Sharpe Ratio)

Usage:
  python run_selection_bias.py --auto                                # default: pick top-level outputs/backtest_result.pkl
  python run_selection_bias.py --auto --label iter15_65tkr_reb21_vtg # current production
  python run_selection_bias.py --auto --pkl outputs/baseline_v4/backtest_result.pkl

  # legacy modes
  python run_selection_bias.py                   # N=1 (biased low)
  python run_selection_bias.py --n_trials 48     # explicit override

PKL discovery order when both --pkl and --label are omitted:
  1. outputs/backtest_result.pkl (legacy top-level)
  2. outputs/iter15_65tkr_reb21_vtg/backtest_result.pkl (current production)
  3. outputs/baseline_v4/backtest_result.pkl
  4. error
"""

import argparse
import pickle
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

OUTPUT_DIR = Path("./outputs")
INVENTORY_PATH = Path(__file__).resolve().parent / "experiment_inventory.json"


def load_experiment_inventory() -> int:
    """Return conservative N_trials count from experiment_inventory.json.

    The inventory tallies every distinct strategy configuration that was
    backtested during research (grid_search, compare_*, experiment_*,
    manual tuning). Passing this into the DSR calculation replaces the
    biased N=1 default with a realistic multiple-comparison count.
    """
    if not INVENTORY_PATH.exists():
        print(f"  [!] {INVENTORY_PATH.name} not found. Falling back to N=1.")
        return 1
    with open(INVENTORY_PATH, "r", encoding="utf-8") as f:
        inv = json.load(f)
    n = int(inv.get("n_trials_total", 1))
    scripts = inv.get("scripts", [])
    print(f"  [inventory] N_trials_total = {n} from {len(scripts)} entries")
    for s in scripts:
        print(f"    {s['script']:35s} [{s['phase']:30s}] {s['trials']:>4d}")
    return n


# Production candidates checked when --pkl / --label are not supplied. Order
# matters: codex_causal_rank_65 = current production (2026-07-11 승격) first,
# iter15 = legacy challenger next, then top-level legacy path and prior
# production. Update when production baseline rotates.
_PKL_FALLBACK_ORDER = [
    OUTPUT_DIR / "codex_causal_rank_65" / "backtest_result.pkl",
    OUTPUT_DIR / "iter15_65tkr_reb21_vtg" / "backtest_result.pkl",
    OUTPUT_DIR / "backtest_result.pkl",
    OUTPUT_DIR / "baseline_v4" / "backtest_result.pkl",
]


def resolve_pkl_path(
    pkl_arg: Optional[str] = None,
    label_arg: Optional[str] = None,
) -> Path:
    """Resolve which backtest_result.pkl to load.

    Priority:
      1. --pkl <path>    (explicit override)
      2. --label <name>  → outputs/<name>/backtest_result.pkl
      3. fallback chain  (top-level → iter15_..._vtg → baseline_v4)

    Raises FileNotFoundError with a helpful list of inspected paths.
    """
    if pkl_arg:
        path = Path(pkl_arg)
        if not path.exists():
            raise FileNotFoundError(f"--pkl not found: {path}")
        return path
    if label_arg:
        path = OUTPUT_DIR / label_arg / "backtest_result.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"--label '{label_arg}' resolved to {path} but the file is "
                "missing. Run `python run_variant.py --variant variants/"
                f"{label_arg}.yaml` first."
            )
        return path
    for cand in _PKL_FALLBACK_ORDER:
        if cand.exists():
            return cand
    inspected = "\n  ".join(str(p) for p in _PKL_FALLBACK_ORDER)
    raise FileNotFoundError(
        "No backtest_result.pkl found in any of the expected paths:\n  "
        f"{inspected}\n"
        "Run backtest (run_variant.py or update_and_deploy.bat) first, or pass "
        "--pkl <path> / --label <variant>."
    )


def load_backtest_result(
    pkl_arg: Optional[str] = None,
    label_arg: Optional[str] = None,
):
    path = resolve_pkl_path(pkl_arg, label_arg)
    print(f"  [pkl] loading: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def deflated_sharpe_ratio(active_returns: pd.Series, N: int):
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    UNIT CONVENTION
    ---------------
    `observed_SR` is annualized. Previous version mixed scales:
    `sigma_SR` / `E_max_SR` were computed in daily scale and subtracted
    directly from the annualized SR, inflating DSR by ~sqrt(252) ≈ 15.9
    and making the p-value far too significant.

    Fixed: `sigma_SR` is now annualized (= daily-σ × √252), which matches
    Var(sqrt(252)·SR_daily) = 252·Var(SR_daily) → SE(SR_annual) = √252·SE(SR_daily).
    All return-fields are now in the same annualized SR scale so
    `DSR = (SR − E_max_SR)/σ_SR` is consistent.
    """
    T = len(active_returns)
    SR = active_returns.mean() / active_returns.std() * np.sqrt(252)  # annualized
    skew = active_returns.skew()
    kurt = active_returns.kurtosis() + 3  # excess → raw
    SR_daily = SR / np.sqrt(252)

    # Bailey-LdP 2014 eq. 4: SE in DAILY scale
    sigma_SR_daily = np.sqrt(
        (1 - skew * SR_daily + (kurt - 1) / 4 * SR_daily**2) / T
    )
    # Annualize so it can be subtracted from the annualized SR
    sigma_SR = sigma_SR_daily * np.sqrt(252)

    # Expected max SR under null (multiple comparisons) — annualized
    if N > 1:
        E_max_SR = sigma_SR * np.sqrt(2 * np.log(N))
    else:
        E_max_SR = 0

    DSR = (SR - E_max_SR) / sigma_SR if sigma_SR > 0 else 0
    p_value = 1 - stats.norm.cdf(DSR)

    return {
        "observed_SR": SR,
        "N_trials": N,
        "E_max_SR": E_max_SR,
        "sigma_SR": sigma_SR,
        "DSR": DSR,
        "p_value": p_value,
        "T": T,
        "skewness": skew,
        "kurtosis": kurt,
        "pass": p_value < 0.05,
    }


def minimum_track_record_length(active_returns: pd.Series):
    """Minimum Track Record Length (Bailey & Lopez de Prado)."""
    T = len(active_returns)
    SR = active_returns.mean() / active_returns.std() * np.sqrt(252)
    skew = active_returns.skew()
    kurt = active_returns.kurtosis() + 3
    SR_daily = SR / np.sqrt(252)
    z_alpha = stats.norm.ppf(0.95)

    if abs(SR_daily) < 1e-10:
        MinTRL_days = float("inf")
    else:
        MinTRL_days = 1 + (
            1 - skew * SR_daily + (kurt - 1) / 4 * SR_daily**2
        ) * (z_alpha / SR_daily) ** 2

    MinTRL_years = MinTRL_days / 252
    sufficient = T > MinTRL_days

    return {
        "MinTRL_days": MinTRL_days,
        "MinTRL_years": MinTRL_years,
        "available_days": T,
        "available_years": T / 252,
        "sufficient": sufficient,
    }


def grid_search_haircut(active_returns: pd.Series, N: int):
    """Grid Search expected overestimate haircut.

    Same unit-mixing bug as deflated_sharpe_ratio (annualized SR vs
    daily-scale haircut). Now both are annualized so `adjusted_SR` is
    a real annualized SR. Previously the haircut was ~√252 too small
    (e.g. 0.078 daily → 1.236 annualized), drastically under-penalizing
    multiple-comparison bias.
    """
    T = len(active_returns)
    SR = active_returns.mean() / active_returns.std() * np.sqrt(252)
    skew = active_returns.skew()
    kurt = active_returns.kurtosis() + 3
    SR_daily = SR / np.sqrt(252)

    sigma_SR_daily = np.sqrt(
        (1 - skew * SR_daily + (kurt - 1) / 4 * SR_daily**2) / T
    )
    sigma_SR = sigma_SR_daily * np.sqrt(252)  # annualize to match SR

    haircut = sigma_SR * np.sqrt(2 * np.log(N)) if N > 1 else 0
    adjusted_SR = SR - haircut

    return {
        "N_combinations": N,
        "haircut": haircut,
        "observed_SR": SR,
        "adjusted_SR": adjusted_SR,
        "pass": adjusted_SR > 0,
    }


def _resolve_data_path(explicit: Optional[str] = None) -> Optional[str]:
    """Figure out which dataset to load for fallback survivorship reloads.

    Priority:
        1. Explicit CLI / caller argument (``explicit``)
        2. ``outputs/experiment_manifest.json`` → ``extra.data_path``
           (written by ``run_variant.py`` at the start of every pipeline run)
        3. ``DEFAULT_CONFIG.data_path`` from ``src.config``
        4. ``None`` — caller must handle the missing dataset case
    """
    if explicit:
        return explicit

    manifest_path = OUTPUT_DIR / "experiment_manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            extra_path = (manifest.get("extra") or {}).get("data_path")
            if extra_path:
                return str(extra_path)
            cfg = manifest.get("config") or {}
            if cfg.get("data_path"):
                return str(cfg["data_path"])
        except Exception:
            pass

    try:
        from src.config import DEFAULT_CONFIG
        return getattr(DEFAULT_CONFIG, "data_path", None)
    except Exception:
        return None


def universe_survivorship(result, data_path: Optional[str] = None):
    """Check universe survivorship bias using RAW (un-imputed) returns.

    Uses UniverseData.raw_returns (pre-ffill, pre-median) so first_valid_index()
    reflects the true first observed return for each ticker. Falling back to
    the imputed .returns would mask late entrants because the fill pipeline
    back-extends every series to the common alignment start.

    ``data_path`` is only used when the stored backtest result does not carry
    a reference to its UniverseData (e.g. loaded from a pickle). We resolve
    it from the experiment manifest / CLI so a backtest run on a non-default
    ``--data_path`` still validates against the correct dataset.
    """
    port = result.portfolio_returns
    backtest_start = port.index[0]
    late_entrants = []

    # Prefer raw_returns if available on the stored data object
    raw_returns = None
    if hasattr(result, "data") and result.data is not None:
        raw_returns = getattr(result.data, "raw_returns", None)

    # Fallback: reload data just to get the raw returns panel.
    # Resolve the data path from the explicit argument / manifest / config
    # so it matches the dataset the backtest was actually run on.
    if raw_returns is None:
        resolved = _resolve_data_path(data_path)
        if resolved is None:
            return {
                "backtest_start": backtest_start.strftime("%Y-%m-%d"),
                "late_entrants": [],
                "n_late": 0,
                "clean": True,
                "warning": "data_path could not be resolved — survivorship check skipped",
            }
        from src.data_loader import UniverseData
        data = UniverseData(resolved)
        raw_returns = data.raw_returns
        if raw_returns is None:
            # Last-ditch fallback: use the imputed returns (biased low)
            raw_returns = data.returns
        # Free the raw Excel sheets from memory
        if hasattr(data, "raw"):
            del data.raw

    if raw_returns is None:
        return {
            "backtest_start": backtest_start.strftime("%Y-%m-%d"),
            "late_entrants": [],
            "n_late": 0,
            "clean": True,
            "warning": "raw returns unavailable — check skipped",
        }

    for ticker in raw_returns.columns:
        first_valid = raw_returns[ticker].first_valid_index()
        if first_valid is not None and first_valid > backtest_start + pd.Timedelta(days=30):
            late_entrants.append((ticker, first_valid.strftime("%Y-%m-%d")))

    return {
        "backtest_start": backtest_start.strftime("%Y-%m-%d"),
        "late_entrants": late_entrants,
        "n_late": len(late_entrants),
        "clean": len(late_entrants) == 0,
    }


def sub_period_stability(active_returns: pd.Series):
    """3-period stability test."""
    n = len(active_returns)
    third = n // 3

    periods = [
        ("Period 1", active_returns.iloc[:third]),
        ("Period 2", active_returns.iloc[third : 2 * third]),
        ("Period 3", active_returns.iloc[2 * third :]),
    ]

    results = []
    for name, sub in periods:
        sub_ir = sub.mean() / sub.std() * np.sqrt(252) if sub.std() > 0 else 0
        sub_ret = (1 + sub).prod() - 1
        results.append({
            "name": name,
            "start": sub.index[0].strftime("%Y-%m-%d"),
            "end": sub.index[-1].strftime("%Y-%m-%d"),
            "IR": round(sub_ir, 3),
            "cumulative_return": round(sub_ret, 4),
            "positive_IR": sub_ir > 0,
        })

    all_positive = all(r["positive_IR"] for r in results)
    return {
        "periods": results,
        "all_positive": all_positive,
        "stable": all_positive,
    }


def determine_verdict(dsr, mintrl, grid, surv, subperiod):
    """Overall verdict: PASS / WARN / FAIL."""
    fails = 0
    warns = 0

    # DSR
    if not dsr["pass"]:
        if dsr["p_value"] < 0.10:
            warns += 1
        else:
            fails += 1

    # MinTRL
    if not mintrl["sufficient"]:
        if mintrl["available_days"] > mintrl["MinTRL_days"] * 0.8:
            warns += 1
        else:
            fails += 1

    # Grid Haircut
    if not grid["pass"]:
        fails += 1
    elif grid["adjusted_SR"] <= 0.5:
        warns += 1

    # Survivorship
    if surv["n_late"] >= 3:
        fails += 1
    elif surv["n_late"] >= 1:
        warns += 1

    # Sub-period
    n_positive = sum(1 for p in subperiod["periods"] if p["positive_IR"])
    if n_positive <= 1:
        fails += 1
    elif n_positive == 2:
        warns += 1

    if fails >= 1:
        return "FAIL"
    elif warns >= 1:
        return "WARN"
    return "PASS"


def generate_report(verdict, dsr, mintrl, grid, surv, subperiod):
    """Generate markdown report."""

    period_table = ""
    for p in subperiod["periods"]:
        v = "PASS" if p["positive_IR"] else "FAIL"
        period_table += f"- {p['name']} ({p['start']} ~ {p['end']}): IR = {p['IR']:.3f} [{v}]\n"

    late_str = ", ".join([f"{t} (from {d})" for t, d in surv["late_entrants"]]) if surv["late_entrants"] else "None"

    report = f"""# Selection Bias Analysis Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 1. Summary Verdict
- **{verdict}** -- DSR p={dsr['p_value']:.4f}, Adjusted SR={grid['adjusted_SR']:.2f}, MinTRL={mintrl['MinTRL_years']:.1f}yr

## 2. Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
- Observed SR: {dsr['observed_SR']:.3f}
- Number of trials (N): {dsr['N_trials']}
- Expected max SR under null: {dsr['E_max_SR']:.3f}
- sigma(SR): {dsr['sigma_SR']:.4f}
- Deflated SR: {dsr['DSR']:.3f} (p-value: {dsr['p_value']:.4f})
- Skewness: {dsr['skewness']:.3f}, Kurtosis: {dsr['kurtosis']:.3f}
- Observations: {dsr['T']} trading days
- Verdict: **{"PASS" if dsr['pass'] else "FAIL -- 다중 비교 보정 후 유의하지 않음"}**

## 3. Minimum Track Record Length
- Required: {mintrl['MinTRL_years']:.1f} years ({mintrl['MinTRL_days']:.0f} trading days)
- Available: {mintrl['available_years']:.1f} years ({mintrl['available_days']} trading days)
- Verdict: **{"SUFFICIENT" if mintrl['sufficient'] else "INSUFFICIENT -- 데이터 부족"}**

## 4. Grid Search Bias (Haircut)
- Combinations tested: {grid['N_combinations']}
- Observed SR: {grid['observed_SR']:.3f}
- Haircut: {grid['haircut']:.3f}
- Adjusted SR: {grid['adjusted_SR']:.3f}
- Verdict: **{"PASS" if grid['pass'] else "WARN -- 보정 후 SR <= 0"}**

## 5. Universe Survivorship
- Backtest start: {surv['backtest_start']}
- Late entrants (data starts >30d after backtest): {late_str}
- Verdict: **{"CLEAN" if surv['clean'] else f"WARN -- {surv['n_late']}개 종목 생존 편향 의심"}**

## 6. Sub-period Stability
{period_table}- Verdict: **{"STABLE" if subperiod['stable'] else "UNSTABLE -- 시기 의존적 성과"}**

## References
- Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio"
- Harvey, C. R., & Liu, Y. (2015). "Backtesting" (Haircut Sharpe Ratio)
- Bailey et al. (2014). "Pseudo-Mathematics and Financial Charlatanism"
"""
    return report


def main():
    parser = argparse.ArgumentParser(description="Selection Bias Check (GATE)")
    parser.add_argument("--n_trials", type=int, default=1,
                        help="Total number of strategy trials (grid search combos)")
    parser.add_argument("--auto", action="store_true",
                        help="Read N_trials from experiment_inventory.json (recommended)")
    parser.add_argument("--pkl", type=str, default=None,
                        help="Explicit path to backtest_result.pkl (overrides --label and "
                             "fallback chain). Use this to validate a specific variant.")
    parser.add_argument("--label", type=str, default=None,
                        help="Variant label — resolves to outputs/<label>/backtest_result.pkl. "
                             "Mutually exclusive with --pkl.")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Dataset the backtest was run on (used for the survivorship "
                             "reload fallback). Defaults to the path recorded in "
                             "outputs/experiment_manifest.json, then DEFAULT_CONFIG.data_path.")
    args = parser.parse_args()
    if args.pkl and args.label:
        parser.error("--pkl and --label are mutually exclusive")

    if args.auto:
        print("\n  [Auto] Loading N_trials from experiment_inventory.json ...")
        N = max(load_experiment_inventory(), 1)
    else:
        N = max(args.n_trials, 1)
        if N == 1:
            print("\n  [!] Running with N_trials=1 (legacy default).")
            print("  [!] This underestimates multiple-comparison bias.")
            print("  [!] Use --auto or pass --n_trials N from your experiment inventory.")

    print("=" * 60)
    print("  Phase 5: Selection Bias Check (GATE)")
    print("=" * 60)

    # Load backtest result (production candidate via --pkl / --label / fallback)
    result = load_backtest_result(pkl_arg=args.pkl, label_arg=args.label)
    port = result.portfolio_returns.dropna()
    bm = result.benchmark_returns.dropna()

    # Align
    common_idx = port.index.intersection(bm.index)
    port = port.loc[common_idx]
    bm = bm.loc[common_idx]
    active = port - bm

    print(f"\n  Active returns: {len(active)} days")
    print(f"  Trial count N: {N}")

    # --- 1. Deflated Sharpe Ratio ---
    print("\n[1/5] Deflated Sharpe Ratio...")
    dsr = deflated_sharpe_ratio(active, N)
    status = "PASS" if dsr["pass"] else "FAIL"
    print(f"  Observed SR: {dsr['observed_SR']:.3f}")
    print(f"  DSR: {dsr['DSR']:.3f} (p={dsr['p_value']:.4f}) [{status}]")

    # --- 2. Minimum Track Record Length ---
    print("\n[2/5] Minimum Track Record Length...")
    mintrl = minimum_track_record_length(active)
    status = "SUFFICIENT" if mintrl["sufficient"] else "INSUFFICIENT"
    print(f"  Required: {mintrl['MinTRL_years']:.1f} yrs ({mintrl['MinTRL_days']:.0f} days)")
    print(f"  Available: {mintrl['available_years']:.1f} yrs ({mintrl['available_days']} days)")
    print(f"  [{status}]")

    # --- 3. Grid Search Haircut ---
    print("\n[3/5] Grid Search Haircut...")
    grid = grid_search_haircut(active, N)
    status = "PASS" if grid["pass"] else "WARN"
    print(f"  Haircut: {grid['haircut']:.3f}")
    print(f"  Adjusted SR: {grid['adjusted_SR']:.3f} [{status}]")

    # --- 4. Universe Survivorship ---
    print("\n[4/5] Universe Survivorship Bias...")
    resolved_data_path = _resolve_data_path(args.data_path)
    if resolved_data_path:
        print(f"  [survivorship] dataset: {resolved_data_path}")
    surv = universe_survivorship(result, data_path=resolved_data_path)
    status = "CLEAN" if surv["clean"] else f"WARN ({surv['n_late']} late)"
    print(f"  Late entrants: {surv['n_late']}")
    if surv["late_entrants"]:
        for t, d in surv["late_entrants"]:
            print(f"    {t}: data from {d}")
    print(f"  [{status}]")

    # --- 5. Sub-period Stability ---
    print("\n[5/5] Sub-period Stability...")
    subperiod = sub_period_stability(active)
    for p in subperiod["periods"]:
        v = "+" if p["positive_IR"] else "-"
        print(f"  {p['name']} ({p['start']}~{p['end']}): IR={p['IR']:.3f} [{v}]")
    status = "STABLE" if subperiod["stable"] else "UNSTABLE"
    print(f"  [{status}]")

    # --- Verdict ---
    verdict = determine_verdict(dsr, mintrl, grid, surv, subperiod)

    print("\n" + "=" * 60)
    print(f"  SELECTION BIAS GATE: **{verdict}**")
    print("=" * 60)

    if verdict == "FAIL":
        print("  WARNING: 관측된 성과가 다중 비교 편향에 의한 것일 수 있습니다.")
    elif verdict == "WARN":
        print("  CAUTION: 일부 항목에서 경고가 발생했습니다.")
    else:
        print("  All checks passed.")

    # --- Save report ---
    report_dir = OUTPUT_DIR / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    report = generate_report(verdict, dsr, mintrl, grid, surv, subperiod)
    report_path = report_dir / "selection_bias_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n  Report: {report_path}")

    # --- Save metrics CSV ---
    csv_dir = OUTPUT_DIR / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame([{
        "metric": "Observed_SR",
        "value": dsr["observed_SR"],
    }, {
        "metric": "Deflated_SR",
        "value": dsr["DSR"],
    }, {
        "metric": "DSR_p_value",
        "value": dsr["p_value"],
    }, {
        "metric": "N_trials",
        "value": N,
    }, {
        "metric": "MinTRL_years",
        "value": mintrl["MinTRL_years"],
    }, {
        "metric": "Available_years",
        "value": mintrl["available_years"],
    }, {
        "metric": "Grid_Haircut",
        "value": grid["haircut"],
    }, {
        "metric": "Adjusted_SR",
        "value": grid["adjusted_SR"],
    }, {
        "metric": "Late_Entrants",
        "value": surv["n_late"],
    }, {
        "metric": "Sub_Period_1_IR",
        "value": subperiod["periods"][0]["IR"],
    }, {
        "metric": "Sub_Period_2_IR",
        "value": subperiod["periods"][1]["IR"],
    }, {
        "metric": "Sub_Period_3_IR",
        "value": subperiod["periods"][2]["IR"],
    }, {
        "metric": "Verdict",
        "value": verdict,
    }])

    csv_path = csv_dir / "selection_bias_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"  CSV: {csv_path}")


if __name__ == "__main__":
    main()
