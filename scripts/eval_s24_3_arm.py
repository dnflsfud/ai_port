# -*- coding: utf-8 -*-
"""§S24.3 arm 판정 (read-only) — s24_3_tg_events_a (T·RTX tg_basis_events 제거) vs 기준 outputs/s24_g0_new.

결정 로그 §S24.3 사전등록(2026-09-30)의 재실행 가능한 판정 스크립트. 정확성 트랙(§S23.1 프레임, 인벤토리 비계수).
  G0 : base·arm data_vintage 동일 ∧ base IR 이 REFERENCE_BASE_IR 을 1e-9 이내 재현.
  기전(production 패널 tg_upside·tg_mom_63d z, 클립 ±5):
    ① 제거 이벤트(RTX 2020-04-03, T 2022-04-11) 전 tg_upside 중앙 z: base ≥ PRE_Z_MIN → arm |z| ≤ ARM_PRE_Z_ABS_MAX
    ② 유지 이벤트(DELL 2021-11-02, DHR 2016-07-05) 전 중앙 z 변화 |Δ| ≤ KEEP_DRIFT_MAX
    ③ POST_CUT(2022-07-15, 두 이벤트 + 63BD 창) 이후 모든 종목의 TG 파생 2피처 비트 동일
    ④ TG 외 피처는 전 구간 비트 동일 ⑤ 피처 집합 동일.
  E2 : TE ≤ 4.5% · active share ±3%p · 회전율 ≤ 1.25× · solver fallback 0.
  무해성: ΔIR > −0.36 ∧ 3분할 전부 음은 아님. formal E1(ΔIR > +0.36 ∧ 3분할 전부 양)은 병기만.
  flip 후보 = G0 ∧ 기전 ∧ E2 ∧ 무해성.
출력: outputs/s24_3_tg_events_a/e1_summary.json
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

BASE = "s24_g0_new"
ARM = "s24_3_tg_events_a"
REFERENCE_BASE_IR = 1.6951803327080086
TURNOVER_MAX = 1.25
EVENTS_REMOVED = {"RTX": "2020-04-03", "T": "2022-04-11"}
EVENTS_KEPT = {"DELL": "2021-11-02", "DHR": "2016-07-05"}
PRE_Z_MIN = {"RTX": 2.0, "T": 1.5}
ARM_PRE_Z_ABS_MAX = 1.0
KEEP_DRIFT_MAX = 0.1
POST_CUT = "2022-07-15"
TG_FEATURES = ("tg_upside", "tg_mom_63d")
TOL = 1e-9


def _max_abs_delta(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """정렬 후 최대 |Δ|; NaN 패턴이 다르면 inf."""
    a, b = a.align(b, join="outer")
    if not (a.isna() == b.isna()).all().all():
        return float("inf")
    d = (a - b).abs().to_numpy(dtype=float)
    return float(np.nanmax(d)) if d.size and not np.all(np.isnan(d)) else 0.0


def mechanism_tg_events(base_panel: pd.DataFrame, arm_panel: pd.DataFrame) -> dict:
    bz = base_panel["tg_upside"].unstack()
    az = arm_panel["tg_upside"].unstack()
    names, all_ok = {}, True
    for t, ev in EVENTS_REMOVED.items():
        pre_b = float(bz[t][bz.index < pd.Timestamp(ev)].median())
        pre_a = float(az[t][az.index < pd.Timestamp(ev)].median())
        ok = bool(pre_b >= PRE_Z_MIN[t] and abs(pre_a) <= ARM_PRE_Z_ABS_MAX)
        names[t] = {"event": ev, "base_pre_median": round(pre_b, 4), "arm_pre_median": round(pre_a, 4), "pass": ok}
        all_ok &= ok
    kept = {}
    for t, ev in EVENTS_KEPT.items():
        pre_b = float(bz[t][bz.index < pd.Timestamp(ev)].median()) if t in bz else float("nan")
        pre_a = float(az[t][az.index < pd.Timestamp(ev)].median()) if t in az else float("nan")
        drift = abs(pre_a - pre_b) if np.isfinite(pre_a) and np.isfinite(pre_b) else 0.0
        ok = bool(drift <= KEEP_DRIFT_MAX)
        kept[t] = {"event": ev, "base_pre_median": round(pre_b, 4), "arm_pre_median": round(pre_a, 4), "drift": round(drift, 4), "pass": ok}
        all_ok &= ok
    tg_cols = [c for c in TG_FEATURES if c in base_panel.columns and c in arm_panel.columns]
    dates_b = base_panel.index.get_level_values("date")
    dates_a = arm_panel.index.get_level_values("date")
    post_delta = _max_abs_delta(base_panel.loc[dates_b >= pd.Timestamp(POST_CUT), tg_cols],
                                arm_panel.loc[dates_a >= pd.Timestamp(POST_CUT), tg_cols])
    other_cols = [c for c in base_panel.columns if c not in TG_FEATURES]
    other_delta = _max_abs_delta(base_panel[other_cols], arm_panel[[c for c in other_cols if c in arm_panel.columns]]) \
        if set(other_cols) <= set(arm_panel.columns) else float("inf")
    feature_set_equal = bool(set(base_panel.columns) == set(arm_panel.columns))
    all_ok &= bool(post_delta <= TOL and other_delta <= TOL and feature_set_equal)
    return {"names": names, "kept": kept, "post_cut": POST_CUT, "post_cut_max_abs_delta": post_delta,
            "other_features_max_abs_delta": other_delta, "feature_set_equal": feature_set_equal,
            "n_features_base": int(base_panel.shape[1]), "n_features_arm": int(arm_panel.shape[1]), "pass": bool(all_ok)}


def flip_verdict(g0_pass: bool, mechanism_pass: bool, e2_pass: bool, no_harm: bool) -> bool:
    return bool(g0_pass and mechanism_pass and e2_pass and no_harm)


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
    e1_formal = evaluate_e1(d_ir, split_deltas)
    no_harm = evaluate_no_harm(d_ir, split_deltas)

    mechanism = mechanism_tg_events(base_r.panel, arm_r.panel)
    mechanism["feature_names_equal"] = bool(list(base_r.feature_names) == list(arm_r.feature_names))
    mechanism["pass"] = bool(mechanism["pass"] and mechanism["feature_names_equal"])

    turnover_ratio = arm_m["avg_annual_turnover"] / base_m["avg_annual_turnover"]
    e2 = evaluate_e2(arm_m["tracking_error"], base_m["active_share"], arm_m["active_share"],
                     turnover_ratio, arm_doc["optimizer_solver_fallback_rate"], TURNOVER_MAX)
    e2_pass = all(e2.values())
    verdict = flip_verdict(g0["g0_pass"], mechanism["pass"], e2_pass, no_harm)

    out = {
        "arm": args.arm, "frame": "correctness",
        "adoption_basis": (f"correctness + do-no-harm (dIR > -{E1_DELTA_IR}, not all splits negative) with mechanism, E2, G0; "
                           "formal E1 reported only"),
        "preregistration": "decision log §S24.3 (2026-09-30)",
        "base_dir": str(base_dir),
        "vintage": {"base_pkl_mtime": _kst(base_dir / "backtest_result.pkl"),
                    "arm_pkl_mtime": _kst(arm_dir / "backtest_result.pkl"),
                    "base_data_vintage": base_doc.get("data_vintage"),
                    "arm_data_vintage": arm_doc.get("data_vintage")},
        "g0": g0,
        "full_period": {"ir_base": round(base_m["information_ratio"], 4), "ir_arm": round(arm_m["information_ratio"], 4),
                        "delta_ir": round(d_ir, 4), "e1_bar": E1_DELTA_IR},
        "subperiods": splits, "e1_formal_pass": e1_formal, "no_harm_pass": no_harm,
        "mechanism": mechanism, "e2": e2, "e2_pass": e2_pass, "flip_candidate": verdict,
        "companions": {
            "te_base": round(base_m["tracking_error"], 5), "te_arm": round(arm_m["tracking_error"], 5),
            "turnover_base": round(base_m["avg_annual_turnover"], 4), "turnover_arm": round(arm_m["avg_annual_turnover"], 4),
            "turnover_ratio": round(turnover_ratio, 4),
            "active_share_base": round(base_m["active_share"], 5), "active_share_arm": round(arm_m["active_share"], 5),
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
          f"(formal E1 {'PASS' if e1_formal else 'FAIL'}, no-harm {'PASS' if no_harm else 'FAIL'}) | "
          f"E2 {'PASS' if e2_pass else 'FAIL'} | mechanism {mechanism['pass']} | flip candidate: {verdict}")


if __name__ == "__main__":
    main()
