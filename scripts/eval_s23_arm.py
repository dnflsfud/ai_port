# -*- coding: utf-8 -*-
"""§S23.1 arm 판정 (read-only) — `--arm <label>`, 기준 = outputs/s23_s0recert.

결정 로그 §S23.1 사전등록(2026-09-28)의 재실행 가능한 판정 스크립트. 사용자 결정(09-28):
정확성+무해성 채택 기준, 공통 기준 병렬 측정 + flip 후 결합 재인증.
  G0 : base·arm data_vintage 동일, 그리고 base IR 이 같은 빈티지의 09-28 12:00 스케줄
       production IR(REFERENCE_PRODUCTION_IR)을 1e-9 이내로 재현.
  기전: s23_m01_no_slope   — 모델 피처 = base − 슬로프 4피처 (그 외 변화 0);
        s23_b01_label_uncentered — 유한 타깃 셀의 ≥99% 변경 ∧ 날짜별 Spearman(base−arm,
                                  momentum_252d) 중앙값 ≤ −0.3 (제거된 반-모멘텀 항의 부호);
        s23_d04_optvol_lag  — pre_overlay_predictions 비트 동일(알파 불변) ∧ 리밸 목표비중 변화;
        s23_combined_recert — 결합 상태(G0·E2·무해성만).
  E2 : TE ≤ 4.5% · active share ±3%p · 회전율 ≤ 1.25× · solver fallback 0.
  무해성: ΔIR > −0.36 이고 3분할 전부 음은 아님. formal E1(ΔIR > +0.36 ∧ 3분할 전부 양)은 병기만.
  flip 후보 = G0 ∧ 기전 ∧ E2 ∧ 무해성.
출력: outputs/<label>/e1_summary.json
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AI_PORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_PORT))

from scripts.eval_s13_46_arm import E1_DELTA_IR, N_SPLITS, _ir, _kst, evaluate_e1  # noqa: E402
from scripts.eval_s17_arm import evaluate_e2, evaluate_no_harm  # noqa: E402
from src.features.fwd_sales_slope import FWD_SALES_SLOPE_FEATURES  # noqa: E402

BASE = "s23_s0recert"
# outputs/codex_causal_rank_65/metrics.json of the 2026-09-28 12:00 scheduled run
# (beea658; workbook 2026-09-18T05:13:24Z / Index 2026-09-28T02:23:35Z).
REFERENCE_PRODUCTION_IR = 1.693143983245076
TURNOVER_MAX = 1.25
MOMENTUM_FEATURE = "momentum_252d"
B01_CHANGED_MIN = 0.99
B01_SPEARMAN_MAX = -0.3
ARMS = ("s23_m01_no_slope", "s23_b01_label_uncentered", "s23_d04_optvol_lag", "s23_combined_recert")


def mechanism_m01(base_features, arm_features) -> dict:
    base, arm = set(base_features), set(arm_features)
    removed, added = sorted(base - arm), sorted(arm - base)
    return {"removed": removed, "added": added,
            "pass": bool(set(removed) == set(FWD_SALES_SLOPE_FEATURES) and not added)}


def _rowwise_spearman(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    ok = a.notna() & b.notna()
    ra = a.where(ok).rank(axis=1)
    rb = b.where(ok).rank(axis=1)
    ra = ra.sub(ra.mean(axis=1), axis=0)
    rb = rb.sub(rb.mean(axis=1), axis=0)
    den = np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
    return ((ra * rb).sum(axis=1) / den.replace(0, np.nan)).where(ok.sum(axis=1) >= 10)


def mechanism_b01(base_targets: pd.DataFrame, arm_targets: pd.DataFrame,
                  momentum: pd.DataFrame) -> dict:
    arm_targets = arm_targets.reindex(index=base_targets.index, columns=base_targets.columns)
    both = base_targets.notna() & arm_targets.notna()
    diff = (base_targets - arm_targets).where(both)
    n_both = int(both.values.sum())
    changed = float((diff.abs() > 1e-12).values.sum() / n_both) if n_both else 0.0
    mom = momentum.reindex(index=diff.index, columns=diff.columns)
    rho = _rowwise_spearman(diff, mom).dropna()
    med = float(rho.median()) if len(rho) else float("nan")
    return {"changed_cell_fraction": round(changed, 6), "n_dates": int(len(rho)),
            "median_spearman_diff_vs_momentum": round(med, 4),
            "pass": bool(changed >= B01_CHANGED_MIN and np.isfinite(med) and med <= B01_SPEARMAN_MAX)}


def mechanism_d04(base_pre: pd.DataFrame, arm_pre: pd.DataFrame,
                  base_weights: dict, arm_weights: dict) -> dict:
    alpha_identical = bool(base_pre.equals(arm_pre))
    max_diff = 0.0
    for date, w in base_weights.items():
        other = arm_weights.get(date)
        if other is None:
            continue
        idx = w.index.union(other.index)
        max_diff = max(max_diff, float((w.reindex(idx).fillna(0) - other.reindex(idx).fillna(0)).abs().max()))
    return {"alpha_bit_identical": alpha_identical, "max_abs_weight_change": max_diff,
            "pass": bool(alpha_identical and max_diff > 1e-8)}


def flip_verdict(g0_pass: bool, mechanism_pass: bool, e2_pass: bool, no_harm: bool) -> bool:
    return bool(g0_pass and mechanism_pass and e2_pass and no_harm)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    args = ap.parse_args()
    base_dir = AI_PORT / "outputs" / BASE
    arm_dir = AI_PORT / "outputs" / args.arm

    base_doc = json.load(open(base_dir / "metrics.json", encoding="utf-8"))
    arm_doc = json.load(open(arm_dir / "metrics.json", encoding="utf-8"))
    base_m, arm_m = base_doc["metrics"], arm_doc["metrics"]
    base_mq, arm_mq = base_doc["model_quality"], arm_doc["model_quality"]
    vintage_ok = base_doc.get("data_vintage") == arm_doc.get("data_vintage")
    base_reproduces = abs(base_m["information_ratio"] - REFERENCE_PRODUCTION_IR) < 1e-9
    g0 = {"vintage_equal": bool(vintage_ok), "base_reproduces_scheduled_production": bool(base_reproduces),
          "g0_pass": bool(vintage_ok and base_reproduces)}

    base_r = pickle.load(open(base_dir / "backtest_result.pkl", "rb"))
    arm_r = pickle.load(open(arm_dir / "backtest_result.pkl", "rb"))

    b_act = pd.Series(base_r.active_returns).dropna()
    a_act = pd.Series(arm_r.active_returns).dropna()
    common = b_act.index.intersection(a_act.index)
    b_act, a_act = b_act.loc[common], a_act.loc[common]
    splits = []
    for k, ix in enumerate(np.array_split(np.arange(len(common)), N_SPLITS)):
        ir_b, ir_a = _ir(b_act.iloc[ix]), _ir(a_act.iloc[ix])
        splits.append({"split": k + 1, "start": str(common[ix[0]].date()), "end": str(common[ix[-1]].date()),
                       "ir_base": round(ir_b, 4), "ir_arm": round(ir_a, 4), "delta": round(ir_a - ir_b, 4)})
    d_ir = arm_m["information_ratio"] - base_m["information_ratio"]
    split_deltas = [s["delta"] for s in splits]
    e1_formal = evaluate_e1(d_ir, split_deltas)
    no_harm = evaluate_no_harm(d_ir, split_deltas)

    if args.arm == "s23_m01_no_slope":
        mechanism = mechanism_m01(base_r.feature_names, arm_r.feature_names)
    elif args.arm == "s23_b01_label_uncentered":
        mom = base_r.panel[MOMENTUM_FEATURE].unstack()
        mechanism = mechanism_b01(base_r.targets, arm_r.targets, mom)
    elif args.arm == "s23_d04_optvol_lag":
        mechanism = mechanism_d04(base_r.pre_overlay_predictions, arm_r.pre_overlay_predictions,
                                  base_r.portfolio_weights, arm_r.portfolio_weights)
    else:
        mechanism = {"note": "combined flips: judged on G0 + E2 + do-no-harm", "pass": True}

    turnover_ratio = arm_m["avg_annual_turnover"] / base_m["avg_annual_turnover"]
    e2 = evaluate_e2(arm_m["tracking_error"], base_m["active_share"], arm_m["active_share"],
                     turnover_ratio, arm_doc["optimizer_solver_fallback_rate"], TURNOVER_MAX)
    e2_pass = all(e2.values())
    verdict = flip_verdict(g0["g0_pass"], mechanism["pass"], e2_pass, no_harm)

    out = {
        "arm": args.arm, "frame": "correctness",
        "adoption_basis": (f"correctness + do-no-harm (dIR > -{E1_DELTA_IR}, not all splits negative) "
                           "with mechanism, E2 and G0; formal E1 reported only"),
        "preregistration": "decision log §S23.1 (2026-09-28)",
        "base_dir": str(base_dir),
        "vintage": {"base_pkl_mtime": _kst(base_dir / "backtest_result.pkl"),
                    "arm_pkl_mtime": _kst(arm_dir / "backtest_result.pkl"),
                    "base_data_vintage": base_doc.get("data_vintage"),
                    "arm_data_vintage": arm_doc.get("data_vintage")},
        "g0": g0,
        "full_period": {"ir_base": round(base_m["information_ratio"], 4),
                        "ir_arm": round(arm_m["information_ratio"], 4),
                        "delta_ir": round(d_ir, 4), "e1_bar": E1_DELTA_IR},
        "subperiods": splits, "e1_formal_pass": e1_formal, "no_harm_pass": no_harm,
        "mechanism": mechanism, "e2": e2, "e2_pass": e2_pass, "flip_candidate": verdict,
        "companions": {
            "te_base": round(base_m["tracking_error"], 5), "te_arm": round(arm_m["tracking_error"], 5),
            "turnover_base": round(base_m["avg_annual_turnover"], 4),
            "turnover_arm": round(arm_m["avg_annual_turnover"], 4),
            "turnover_ratio": round(turnover_ratio, 4),
            "active_share_base": round(base_m["active_share"], 5),
            "active_share_arm": round(arm_m["active_share"], 5),
            "realized_beta_base": base_m.get("realized_beta"), "realized_beta_arm": arm_m.get("realized_beta"),
            "avg_ic_base": base_m["avg_ic"], "avg_ic_arm": arm_m["avg_ic"],
            "max_drawdown_base": base_m.get("max_drawdown"), "max_drawdown_arm": arm_m.get("max_drawdown"),
            "degenerate": {"base": f"{base_mq['degenerate_retrains']}/{base_mq['total_retrains']}",
                           "arm": f"{arm_mq['degenerate_retrains']}/{arm_mq['total_retrains']}"},
            "fallback_arm": arm_doc["optimizer_solver_fallback_rate"],
            "solver_counts_arm": arm_doc.get("optimizer_solver_counts"),
            "elapsed_sec_arm": arm_doc.get("elapsed_sec"),
        },
    }
    (arm_dir / "e1_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n{args.arm}: G0 {'PASS' if g0['g0_pass'] else 'FAIL'} | dIR {d_ir:+.4f} "
          f"(formal E1 {'PASS' if e1_formal else 'FAIL'}, no-harm {'PASS' if no_harm else 'FAIL'}) | "
          f"E2 {'PASS' if e2_pass else 'FAIL'} | mechanism {mechanism.get('pass')} | flip candidate: {verdict}")


if __name__ == "__main__":
    main()
