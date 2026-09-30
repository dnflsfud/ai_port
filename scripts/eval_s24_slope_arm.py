# -*- coding: utf-8 -*-
"""§S24.2 arm 판정 (read-only) — s24_slope_1bf2bf vs 기준 outputs/s24_g0_new.

결정 로그 §S24.2 사전등록(2026-09-30)의 재실행 가능한 판정 스크립트. 성과 arm(인벤토리 +1).
  G0 : base·arm data_vintage 동일 ∧ base IR 이 같은 빈티지 G0 파리티 런(REFERENCE_BASE_IR)을 1e-9 이내 재현.
  기전: 모델 피처 집합 = base + 정확히 슬로프 4피처(FWD_SALES_SLOPE_FEATURES, 그 외 증감 0)
        ∧ 4피처 모두 소비 — 재학습별 LightGBM gain>0 비율 ≥ CONSUMED_MIN (pkl `models` 의 부스터 gain 을
        `feature_names` 순서로 매핑; 부스터가 없으면 집합 검사만).
  E2 : TE ≤ 4.5% · active share ±3%p · 회전율 ≤ 1.25× · solver fallback 0.
  formal E1(채택 기준, §2.4): ΔIR > +0.36 ∧ 3분할 전부 양.
  오버라이드 후보(§S13.50 사용자 비준 정책): ΔIR > 0 ∧ 3분할 전부 양 ∧ G0·기전·E2 (DSR 해킷은 run_selection_bias.py 로 병기).
  무해성(참고): ΔIR > −0.36 ∧ 3분할 전부 음은 아님.
출력: outputs/s24_slope_1bf2bf/e1_summary.json
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

BASE = "s24_g0_new"
ARM = "s24_slope_1bf2bf"
# outputs/s24_g0_new/metrics.json — production overrides on workbook 2026-09-30 14:27:35 /
# Index 2026-09-30 11:31:05 (decision log §S24 G0, byte-identical to the HEAD-code run).
REFERENCE_BASE_IR = 1.6951803327080086
TURNOVER_MAX = 1.25
CONSUMED_MIN = 0.9


def mechanism_feature_set(base_features, arm_features) -> dict:
    base, arm = set(base_features), set(arm_features)
    removed, added = sorted(base - arm), sorted(arm - base)
    return {"removed": removed, "added": added,
            "set_pass": bool(set(added) == set(FWD_SALES_SLOPE_FEATURES) and not removed)}


def mechanism_consumption_from_gains(recs) -> dict:
    """recs = per-retrain {feature: gain}. 4피처 각각 gain>0 인 재학습 비율과 블록 gain 점유율."""
    recs = list(recs or [])
    if not recs:
        return {"recorded": False, "note": "no per-retrain feature gain available; set check only"}
    n = len(recs)
    consumed = {f: sum(1 for r in recs if float(r.get(f, 0.0) or 0.0) > 0) / n for f in FWD_SALES_SLOPE_FEATURES}
    shares = []
    for r in recs:
        tot = sum(float(v) for v in r.values() if np.isfinite(v))
        blk = sum(float(r.get(f, 0.0) or 0.0) for f in FWD_SALES_SLOPE_FEATURES)
        shares.append(blk / tot if tot > 0 else np.nan)
    return {"recorded": True, "n_retrains": n,
            "consumed_share_by_feature": {k: round(v, 3) for k, v in consumed.items()},
            "block_gain_share_mean": round(float(np.nanmean(shares)), 4),
            "consumption_pass": bool(all(v >= CONSUMED_MIN for v in consumed.values()))}


def gains_from_models(result) -> list:
    """BacktestResult.models({date: LGBMRanker}) → per-retrain {feature: gain}.

    Column_i of the booster ↔ ``model._active_features[i]`` (the exact EWMA-selected
    training columns the trainer attaches to each model); falls back to
    ``result.feature_names`` only when no active list is attached and the lengths match.
    Features absent from a retrain's active set get gain 0 (not consumed).
    """
    models = getattr(result, "models", None) or {}
    all_names = list(getattr(result, "feature_names", []) or [])
    recs = []
    for _, model in (models.items() if isinstance(models, dict) else enumerate(models)):
        booster = getattr(model, "booster_", None) or (model if hasattr(model, "feature_importance") else None)
        if booster is None:
            continue
        gain = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
        names = list(getattr(model, "_active_features", None) or [])
        if not names:
            if len(gain) != len(all_names):
                raise ValueError(f"no _active_features and gain length {len(gain)} != feature_names {len(all_names)}")
            names = all_names
        if len(gain) != len(names):
            raise ValueError(f"gain length {len(gain)} != active features {len(names)}")
        rec = {f: 0.0 for f in all_names}
        rec.update(dict(zip(names, gain)))
        recs.append(rec)
    return recs


def verdicts(g0: bool, mechanism: bool, e2: bool, d_ir: float, split_deltas) -> dict:
    e1_formal = evaluate_e1(d_ir, split_deltas)
    no_harm = evaluate_no_harm(d_ir, split_deltas)
    gates = bool(g0 and mechanism and e2)
    return {"e1_formal_pass": bool(e1_formal), "no_harm_pass": bool(no_harm),
            "adoption_candidate": bool(gates and e1_formal),
            "override_candidate": bool(gates and d_ir > 0 and all(d > 0 for d in split_deltas))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=ARM)
    ap.add_argument("--base", default=BASE)
    args = ap.parse_args()
    base_dir = AI_PORT / "outputs" / args.base
    arm_dir = AI_PORT / "outputs" / args.arm

    base_doc = json.load(open(base_dir / "metrics.json", encoding="utf-8"))
    arm_doc = json.load(open(arm_dir / "metrics.json", encoding="utf-8"))
    base_m, arm_m = base_doc["metrics"], arm_doc["metrics"]
    base_mq, arm_mq = base_doc["model_quality"], arm_doc["model_quality"]
    vintage_ok = base_doc.get("data_vintage") == arm_doc.get("data_vintage")
    base_reproduces = abs(base_m["information_ratio"] - REFERENCE_BASE_IR) < 1e-9
    g0 = {"vintage_equal": bool(vintage_ok), "base_reproduces_reference": bool(base_reproduces),
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

    fset = mechanism_feature_set(base_r.feature_names, arm_r.feature_names)
    cons = mechanism_consumption_from_gains(gains_from_models(arm_r))
    base_cons = mechanism_consumption_from_gains(gains_from_models(base_r))
    mechanism_pass = bool(fset["set_pass"] and (cons["consumption_pass"] if cons.get("recorded") else True))
    mechanism = {"feature_set": fset, "consumption": cons, "pass": mechanism_pass,
                 "base_slope_gain_share_sanity": base_cons.get("block_gain_share_mean")}

    turnover_ratio = arm_m["avg_annual_turnover"] / base_m["avg_annual_turnover"]
    e2 = evaluate_e2(arm_m["tracking_error"], base_m["active_share"], arm_m["active_share"],
                     turnover_ratio, arm_doc["optimizer_solver_fallback_rate"], TURNOVER_MAX)
    e2_pass = all(e2.values())
    v = verdicts(g0["g0_pass"], mechanism_pass, e2_pass, d_ir, split_deltas)

    out = {
        "arm": args.arm, "frame": "performance",
        "adoption_basis": (f"formal E1 (dIR > +{E1_DELTA_IR} and all {N_SPLITS} splits positive) with G0, mechanism, E2; "
                           "override candidate (dIR > 0, all splits positive, gates) reported only — user decision + DSR haircut"),
        "preregistration": "decision log §S24.2 (2026-09-30)",
        "base_dir": str(base_dir),
        "vintage": {"base_pkl_mtime": _kst(base_dir / "backtest_result.pkl"),
                    "arm_pkl_mtime": _kst(arm_dir / "backtest_result.pkl"),
                    "base_data_vintage": base_doc.get("data_vintage"),
                    "arm_data_vintage": arm_doc.get("data_vintage")},
        "g0": g0,
        "full_period": {"ir_base": round(base_m["information_ratio"], 4),
                        "ir_arm": round(arm_m["information_ratio"], 4),
                        "delta_ir": round(d_ir, 4), "e1_bar": E1_DELTA_IR},
        "subperiods": splits, **v,
        "mechanism": mechanism, "e2": e2, "e2_pass": e2_pass,
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
            "sub_periods_metrics_base": base_m.get("sub_periods"), "sub_periods_metrics_arm": arm_m.get("sub_periods"),
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
          f"(formal E1 {'PASS' if v['e1_formal_pass'] else 'FAIL'}, no-harm {'PASS' if v['no_harm_pass'] else 'FAIL'}) | "
          f"E2 {'PASS' if e2_pass else 'FAIL'} | mechanism {mechanism_pass} | "
          f"adoption candidate: {v['adoption_candidate']} | override candidate: {v['override_candidate']}")


if __name__ == "__main__":
    main()
