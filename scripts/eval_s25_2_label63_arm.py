# -*- coding: utf-8 -*-
"""§S25.2 G4-01 arm 판정 (read-only) — s25_2_label63 vs 기준 outputs/s24_3_tg_events_a.

결정 로그 §S25.2 사전등록(2026-10-07)의 재실행 가능한 판정 스크립트. 성과 arm(인벤토리 +1).
  G0 : base·arm 워크북 빈티지 동일 ∧ base IR 이 S0′(REFERENCE_BASE_IR)을 1e-9 이내 재현 ∧ Index.xlsx 빈티지가
       다르면 arm 과 같은 (워크북, Index) 쌍의 production 런(FX_INERT_PROOF)이 같은 IR 을 재현(= FX 리프레시 불활성 증명).
  기전: (a) 분할 audit 전 재학습 forward_horizon=63 ∧ embargo_days≥63 ∧ causal_validation_ok
        (b) 63d 라벨 설명력 — 리밸일 Spearman IC(arm 예측 vs arm 의 63d PCA 잔차 라벨) 의 평균이
            같은 라벨에 대한 base(20d 모델) 예측의 IC 평균을 초과(쌍 비교, Δ>0; t 는 관측).
  E2 : TE ≤ 4.5% · active share ±3%p · 회전율 ≤ 1.25× · solver fallback 0.
  formal E1(채택 기준, §2.4): ΔIR > +0.36 ∧ 3분할 전부 양.
  해석 매트릭스(사전 고정): 기전(b) FAIL → 축 종결(G4-01 SHELVE, 2모델 μ 결합도 기각) /
        기전 PASS ∧ E1 FAIL → G4-01 불채택, 2모델 μ 결합 arm 이 다음 사전등록 /
        기전 PASS ∧ E2 ∧ E1 → 채택 후보(DSR 해킷·사용자 결정).
  오버라이드 후보(§S13.50 사용자 비준 정책): ΔIR > 0 ∧ 3분할 전부 양 ∧ G0·기전·E2 — 병기만.
출력: outputs/s25_2_label63/e1_summary.json
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

BASE = "s24_3_tg_events_a"
ARM = "s25_2_label63"
# Production run on the arm's (workbook, Index) pair; used only when Index.xlsx was refreshed after the base run.
FX_INERT_PROOF = "codex_causal_rank_65"
# outputs/s24_3_tg_events_a/metrics.json — S0' on workbook 2026-09-30 14:27:35 (decision log §S24.3 result).
REFERENCE_BASE_IR = 1.8327704100987559
ARM_HORIZON = 63
TURNOVER_MAX = 1.25
TURNOVER_REDUCED = 0.90   # observation only: the §S11.9 "slower signal" mechanism (turnover down >= 10%)
IC_MIN_NAMES = 30

_WB_KEYS = ("data_mtime_utc", "data_size_bytes")
_FX_KEYS = ("fx_mtime_utc", "fx_size_bytes")


def _sub(vintage, keys):
    return tuple((vintage or {}).get(k) for k in keys)


def evaluate_g0(base_vintage, arm_vintage, base_ir, proof_vintage=None, proof_ir=None) -> dict:
    workbook_equal = _sub(base_vintage, _WB_KEYS) == _sub(arm_vintage, _WB_KEYS)
    fx_equal = _sub(base_vintage, _FX_KEYS) == _sub(arm_vintage, _FX_KEYS)
    base_reproduces = abs(float(base_ir) - REFERENCE_BASE_IR) < 1e-9
    proof_ok = bool(
        proof_vintage is not None and proof_ir is not None
        and _sub(proof_vintage, _WB_KEYS) == _sub(arm_vintage, _WB_KEYS)
        and _sub(proof_vintage, _FX_KEYS) == _sub(arm_vintage, _FX_KEYS)
        and abs(float(proof_ir) - REFERENCE_BASE_IR) < 1e-9
    )
    fx_inert_proven = bool(fx_equal or proof_ok)
    return {"workbook_equal": bool(workbook_equal), "fx_equal": bool(fx_equal),
            "base_reproduces_reference": bool(base_reproduces), "fx_inert_proven": fx_inert_proven,
            "g0_pass": bool(workbook_equal and base_reproduces and fx_inert_proven)}


def mechanism_split_audit(split_audit, horizon: int = ARM_HORIZON) -> dict:
    rows = list(split_audit or [])
    bad = [r.get("prediction_date") for r in rows
           if int(r.get("forward_horizon", -1)) != horizon or int(r.get("embargo_days", -1)) < horizon
           or not bool(r.get("causal_validation_ok", False))]
    return {"n_retrains": len(rows), "horizon": horizon, "offending_retrains": bad,
            "pass": bool(rows) and not bad}


def label_ic_series(predictions: pd.DataFrame, targets: pd.DataFrame, dates, min_n: int = IC_MIN_NAMES) -> pd.Series:
    """날짜별 Spearman IC(예측 vs 라벨). 유효 쌍 < min_n 인 날짜는 NaN."""
    targets = targets.reindex(index=predictions.index, columns=predictions.columns)
    out = {}
    for d in dates:
        if d not in predictions.index:
            out[d] = np.nan
            continue
        p, t = predictions.loc[d], targets.loc[d]
        m = p.notna() & t.notna()
        out[d] = float(p[m].corr(t[m], method="spearman")) if int(m.sum()) >= min_n else np.nan
    return pd.Series(out, dtype=float)


def mechanism_label_ic(ic_base: pd.Series, ic_arm: pd.Series) -> dict:
    """같은 63d 라벨에 대한 base/arm 예측 IC 의 쌍 비교. PASS = 평균 Δ(arm − base) > 0."""
    both = pd.concat([ic_base.rename("base"), ic_arm.rename("arm")], axis=1).dropna()
    delta = both["arm"] - both["base"]
    n = int(len(delta))
    sd = float(delta.std(ddof=1)) if n > 1 else float("nan")
    t = float(delta.mean() / sd * np.sqrt(n)) if n > 1 and sd > 0 else float("nan")
    return {"n_dates": n,
            "ic63_base_mean": round(float(both["base"].mean()), 5) if n else None,
            "ic63_arm_mean": round(float(both["arm"].mean()), 5) if n else None,
            "delta_mean": round(float(delta.mean()), 5) if n else None,
            "paired_t": round(t, 3) if np.isfinite(t) else None,
            "share_arm_higher": round(float((delta > 0).mean()), 3) if n else None,
            "pass": bool(n > 0 and float(delta.mean()) > 0)}


def verdicts(g0: bool, mechanism: bool, e2: bool, d_ir: float, split_deltas, turnover_ratio: float) -> dict:
    e1_formal = evaluate_e1(d_ir, split_deltas)
    no_harm = evaluate_no_harm(d_ir, split_deltas)
    gates = bool(g0 and mechanism and e2)
    return {"e1_formal_pass": bool(e1_formal), "no_harm_pass": bool(no_harm),
            "adoption_candidate": bool(gates and e1_formal),
            "override_candidate": bool(gates and d_ir > 0 and all(d > 0 for d in split_deltas)),
            "two_model_followup": bool(g0 and mechanism and not e1_formal),
            "axis_closed": bool(g0 and not mechanism),
            "turnover_reduced": bool(turnover_ratio <= TURNOVER_REDUCED)}


def _load(dir_: Path):
    doc = json.load(open(dir_ / "metrics.json", encoding="utf-8"))
    return doc, pickle.load(open(dir_ / "backtest_result.pkl", "rb"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=ARM)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--fx-inert-proof", default=FX_INERT_PROOF)
    args = ap.parse_args()
    base_dir = AI_PORT / "outputs" / args.base
    arm_dir = AI_PORT / "outputs" / args.arm
    proof_dir = AI_PORT / "outputs" / args.fx_inert_proof

    base_doc, base_r = _load(base_dir)
    arm_doc, arm_r = _load(arm_dir)
    base_m, arm_m = base_doc["metrics"], arm_doc["metrics"]
    base_mq, arm_mq = base_doc["model_quality"], arm_doc["model_quality"]

    proof_vintage = proof_ir = None
    if (proof_dir / "metrics.json").exists():
        proof_doc = json.load(open(proof_dir / "metrics.json", encoding="utf-8"))
        proof_vintage, proof_ir = proof_doc.get("data_vintage"), proof_doc["metrics"]["information_ratio"]
    g0 = evaluate_g0(base_doc.get("data_vintage"), arm_doc.get("data_vintage"), base_m["information_ratio"],
                     proof_vintage, proof_ir)

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

    audit = mechanism_split_audit(arm_mq.get("split_audit"))
    reb_dates = sorted(set(arm_r.portfolio_weights) & set(base_r.portfolio_weights))
    ic_base = label_ic_series(base_r.predictions, arm_r.targets, reb_dates)
    ic_arm = label_ic_series(arm_r.predictions, arm_r.targets, reb_dates)
    label_ic = mechanism_label_ic(ic_base, ic_arm)
    mechanism_pass = bool(audit["pass"] and label_ic["pass"])
    mechanism = {"split_audit": audit, "label_ic_63d": label_ic, "pass": mechanism_pass,
                 "note": ("base predictions (20d model) and arm predictions (63d model) are both scored against the arm's "
                          "63d PCA-residual label on the common rebalance dates; the stored ic_series of the two runs use "
                          "different labels and are NOT comparable")}

    turnover_ratio = arm_m["avg_annual_turnover"] / base_m["avg_annual_turnover"]
    e2 = evaluate_e2(arm_m["tracking_error"], base_m["active_share"], arm_m["active_share"],
                     turnover_ratio, arm_doc["optimizer_solver_fallback_rate"], TURNOVER_MAX)
    e2_pass = all(e2.values())
    v = verdicts(g0["g0_pass"], mechanism_pass, e2_pass, d_ir, split_deltas, turnover_ratio)

    out = {
        "arm": args.arm, "frame": "performance",
        "adoption_basis": (f"formal E1 (dIR > +{E1_DELTA_IR} and all {N_SPLITS} splits positive) with G0, mechanism, E2; "
                           "override candidate (dIR > 0, all splits positive, gates) reported only — user decision + DSR haircut"),
        "interpretation_matrix": {
            "axis_closed": "mechanism (b) FAIL -> G4-01 SHELVE and the two-model mu combination is rejected too",
            "two_model_followup": "mechanism PASS and formal E1 missed -> G4-01 not adopted; two-model mu combination is the next pre-registration",
            "adoption_candidate": "G0, mechanism, E2 and formal E1 -> flip candidate after DSR haircut, user decision",
        },
        "preregistration": "decision log §S25.2 (2026-10-07)",
        "base_dir": str(base_dir),
        "vintage": {"base_pkl_mtime": _kst(base_dir / "backtest_result.pkl"),
                    "arm_pkl_mtime": _kst(arm_dir / "backtest_result.pkl"),
                    "base_data_vintage": base_doc.get("data_vintage"),
                    "arm_data_vintage": arm_doc.get("data_vintage"),
                    "fx_inert_proof_dir": str(proof_dir), "fx_inert_proof_vintage": proof_vintage,
                    "fx_inert_proof_ir": proof_ir},
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
            "avg_ic_base_vs_own_20d_label": base_m["avg_ic"], "avg_ic_arm_vs_own_63d_label": arm_m["avg_ic"],
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
          f"E2 {'PASS' if e2_pass else 'FAIL'} | mechanism {mechanism_pass} (IC63 base {label_ic['ic63_base_mean']} "
          f"-> arm {label_ic['ic63_arm_mean']}, t {label_ic['paired_t']}) | adoption candidate: {v['adoption_candidate']} | "
          f"override candidate: {v['override_candidate']} | two-model follow-up: {v['two_model_followup']} | "
          f"axis closed: {v['axis_closed']}")


if __name__ == "__main__":
    main()
