# -*- coding: utf-8 -*-
"""§S27 OPER_MARGIN 발표일 재스탬프 arm 판정 (read-only) — s27_om_pit_restamp vs 기준 outputs/s26_s0_base
(2026-10-30 12:14 production 런의 동결 사본, 워크북 2026-10-21 14:58 KST / Index 2026-10-26 11:14 KST 빈티지;
11-02·11-03 production 런이 IR 을 비트 단위로 재현).

결정 로그 §S27 사전등록(2026-11-03)의 재실행 가능한 판정 스크립트. **룩어헤드 클래스** 프레임:
  flip 후보 = G0 ∧ 기전 ∧ E2. 무해성(§S17 표준 · §S27 ③ 제안식)·ΔIR·avg_ic 는 관측만 하고 롤백 근거로 쓰지 않는다
  (§S25.3 "룩어헤드 복원은 선택지가 아님" — 누수 제거로 백테스트 IR 이 떨어지는 것은 예상된 방향).
  G0 : base·arm data_vintage 동일 ∧ base IR 이 REFERENCE_BASE_IR 을 1e-9 이내 재현 ∧ arm overrides 에 플래그 True.
  기전: data_quality.oper_margin_pit_restamp 의 match_rate ≥ 0.90 ∧ 분기말 ±3일 비율이 재스탬프 전 ≥ 0.50(기말 스탬프 서명)
        ∧ 재스탬프 후 ≤ 2 × 전 세션 기저율 ∧ 피처 패널 변화가 OM 파생 피처(토큰 oper_margin / op_leverage)에만 있고 비어 있지 않음
        ∧ 피처 집합 동일.
  E2 : TE ≤ 4.5% · |ΔAS| ≤ 3%p · 회전율 ≤ 1.25× · solver fallback 0.
출력: outputs/<arm>/e1_summary.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AI_PORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_PORT))

from scripts.eval_s13_46_arm import N_SPLITS, _ir, _kst, evaluate_e1  # noqa: E402
from scripts.eval_s17_arm import evaluate_e2, evaluate_no_harm  # noqa: E402
from scripts.eval_s25_4_arms import _load, changed_features, mechanism_feature_subset  # noqa: E402

ARM = "s27_om_pit_restamp"
FLAG = "oper_margin_pit_restamp_enabled"
BASE = "s26_s0_base"
REFERENCE_BASE_IR = 1.7737993717667202
TURNOVER_MAX = 1.25
MATCH_RATE_MIN = 0.90
QE_STAMP_MIN = 0.50          # the period-end signature must be present before the restamp
QE_VISIBLE_MAX_X_BASE = 2.0  # ... and gone after it (<= 2x the share of all sessions near a quarter end)
NW_LAGS = 21                 # paired Newey-West on the daily active-return difference (one rebalance period)
# Whitelist consumers of OPER_MARGIN (src/features/accounting.py): oper_margin_chg_63d / _252d,
# oper_margin_accel, op_leverage_63d (om - gm pct change). growth_at_quality_252 is not whitelisted.
_OM_TOKENS = ("oper_margin", "op_leverage")


def mechanism_restamp(data_quality) -> dict:
    rec = (data_quality or {}).get("oper_margin_pit_restamp") or {}
    match_rate = rec.get("match_rate")
    qe_before, qe_after, qe_base = (rec.get("quarter_end_3d_share_changes"), rec.get("quarter_end_3d_share_visible"),
                                    rec.get("quarter_end_3d_share_sessions"))
    checks = {
        "present": bool(rec),
        "match_rate_ok": match_rate is not None and match_rate >= MATCH_RATE_MIN,
        "stamp_signature_before": qe_before is not None and qe_before >= QE_STAMP_MIN,
        "stamp_gone_after": None not in (qe_after, qe_base) and qe_after <= QE_VISIBLE_MAX_X_BASE * qe_base,
        "cells_changed": int(rec.get("changed_cells", 0)) > 0,
    }
    return {"record": rec, "checks": checks, "pass": all(checks.values())}


def paired_nw_t(diff: pd.Series, lags: int = NW_LAGS) -> float:
    """t of the mean daily difference with a Bartlett-kernel Newey-West variance."""
    x = diff.dropna().to_numpy(dtype=float)
    n = x.size
    if n < 2 * lags + 2:
        return float("nan")
    e = x - x.mean()
    var = e @ e / n
    for k in range(1, lags + 1):
        var += 2.0 * (1.0 - k / (lags + 1.0)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / np.sqrt(var / n)) if var > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=ARM)
    ap.add_argument("--base", default=BASE)
    args = ap.parse_args()
    base_dir, arm_dir = AI_PORT / "outputs" / args.base, AI_PORT / "outputs" / args.arm
    base_doc, base_r = _load(base_dir)
    arm_doc, arm_r = _load(arm_dir)
    bm, am = base_doc["metrics"], arm_doc["metrics"]

    vintage_ok = base_doc.get("data_vintage") == arm_doc.get("data_vintage")
    base_reproduces = abs(bm["information_ratio"] - REFERENCE_BASE_IR) < 1e-9
    flag_on = bool(arm_doc["overrides"].get(FLAG) is True)
    g0 = {"vintage_equal": bool(vintage_ok), "base_reproduces_reference": bool(base_reproduces), "flag_on": flag_on,
          "g0_pass": bool(vintage_ok and base_reproduces and flag_on)}

    b_act = pd.Series(base_r.active_returns).dropna(); a_act = pd.Series(arm_r.active_returns).dropna()
    common = b_act.index.intersection(a_act.index); b_act, a_act = b_act.loc[common], a_act.loc[common]
    splits = []
    for k, ix in enumerate(np.array_split(np.arange(len(common)), N_SPLITS)):
        ir_b, ir_a = _ir(b_act.iloc[ix]), _ir(a_act.iloc[ix])
        splits.append({"split": k + 1, "start": str(common[ix[0]].date()), "end": str(common[ix[-1]].date()),
                       "ir_base": round(ir_b, 4), "ir_arm": round(ir_a, 4), "delta": round(ir_a - ir_b, 4)})
    d_ir = am["information_ratio"] - bm["information_ratio"]
    split_deltas = [s["delta"] for s in splits]
    t_nw = paired_nw_t(a_act - b_act)

    changed = changed_features(base_r.panel, arm_r.panel)
    feature_set_equal = set(base_r.feature_names) == set(arm_r.feature_names)
    mech = mechanism_restamp(arm_doc.get("data_quality") or getattr(arm_r, "data_quality", None))
    sub = mechanism_feature_subset(changed, lambda f: any(t in f for t in _OM_TOKENS))
    mech["feature_subset"] = sub
    mech["feature_set_equal"] = bool(feature_set_equal)
    mechanism_pass = bool(mech["pass"] and sub["pass"] and feature_set_equal)

    turnover_ratio = am["avg_annual_turnover"] / bm["avg_annual_turnover"]
    e2 = evaluate_e2(am["tracking_error"], bm["active_share"], am["active_share"], turnover_ratio,
                     arm_doc["optimizer_solver_fallback_rate"], TURNOVER_MAX)
    e2_pass = all(e2.values())
    flip_candidate = bool(g0["g0_pass"] and mechanism_pass and e2_pass)

    out = {
        "arm": args.arm, "flag": FLAG,
        "frame": "look-ahead class (G0 + mechanism + E2; no-harm / dIR / avg_ic observed only, no rollback)",
        "preregistration": "decision log §S27 (2026-11-03)", "base_dir": str(base_dir),
        "vintage": {"base_pkl_mtime": _kst(base_dir / "backtest_result.pkl"), "arm_pkl_mtime": _kst(arm_dir / "backtest_result.pkl"),
                    "base_data_vintage": base_doc.get("data_vintage"), "arm_data_vintage": arm_doc.get("data_vintage")},
        "g0": g0,
        "mechanism": mech, "mechanism_pass": mechanism_pass,
        "changed_features_top": dict(sorted(changed.items(), key=lambda kv: -kv[1])[:16]),
        "e2": e2, "e2_pass": e2_pass,
        "flip_candidate": flip_candidate,
        "observed_only": {
            "ir_base": round(bm["information_ratio"], 4), "ir_arm": round(am["information_ratio"], 4),
            "delta_ir": round(d_ir, 4), "subperiods": splits, "paired_nw_t": round(t_nw, 3), "nw_lags": NW_LAGS,
            "no_harm_s17_standard": bool(evaluate_no_harm(d_ir, split_deltas)),
            "no_harm_s27_proposed": bool(d_ir > -0.10 and t_nw > -1.0),
            "e1_formal_pass": bool(evaluate_e1(d_ir, split_deltas)),
            "avg_ic_base": bm["avg_ic"], "avg_ic_arm": am["avg_ic"],
        },
        "companions": {
            "te_base": round(bm["tracking_error"], 5), "te_arm": round(am["tracking_error"], 5),
            "turnover_base": round(bm["avg_annual_turnover"], 4), "turnover_arm": round(am["avg_annual_turnover"], 4),
            "turnover_ratio": round(turnover_ratio, 4),
            "active_share_base": round(bm["active_share"], 5), "active_share_arm": round(am["active_share"], 5),
            "realized_beta_base": bm.get("realized_beta"), "realized_beta_arm": am.get("realized_beta"),
            "max_drawdown_base": bm.get("max_drawdown"), "max_drawdown_arm": am.get("max_drawdown"),
            "degenerate": {"base": f"{base_doc['model_quality']['degenerate_retrains']}/{base_doc['model_quality']['total_retrains']}",
                           "arm": f"{arm_doc['model_quality']['degenerate_retrains']}/{arm_doc['model_quality']['total_retrains']}"},
            "fallback_arm": arm_doc["optimizer_solver_fallback_rate"], "solver_counts_arm": arm_doc.get("optimizer_solver_counts"),
            "elapsed_sec_arm": arm_doc.get("elapsed_sec"),
        },
    }
    (arm_dir / "e1_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    print(f"\n{args.arm}: G0 {'PASS' if g0['g0_pass'] else 'FAIL'} | mechanism {'PASS' if mechanism_pass else 'FAIL'} | "
          f"E2 {'PASS' if e2_pass else 'FAIL'} | flip candidate: {flip_candidate} || observed: dIR {d_ir:+.4f} "
          f"splits {split_deltas} paired NW t {t_nw:+.2f}")


if __name__ == "__main__":
    main()
