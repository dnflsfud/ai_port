# -*- coding: utf-8 -*-
"""§S17.1 arm 판정 (read-only) — 4개 단일 플래그 arm 공용, `--arm <label>`.

결정 로그 §S17.1 사전등록(2026-09-02)의 재실행 가능한 판정 스크립트.
  비교 기준: outputs/s17_s0_0902 (동일 빈티지 production 동결본, IR 1.7052).
  G0: data_vintage 쌍 동일. s17_2(옵티마이저 전용)는 avg_ic 비트 동일·퇴화 동일 필수.
  E1: full ΔIR 과 3분할 — 정확성 트랙(s17_1·s17_3)은 관측, 리스크 규율/위생 트랙
      (s17_2·s17_4)은 do-no-harm: ΔIR > −0.36 이고 3분할이 전부 음이 아님.
  E2: TE ≤ 4.5% · active share ±3%p · turnover ≤ 1.25×(s17_2 는 1.20×) · fallback 0.
  arm 별 기전: s17_1 패널 tg_upside +5 클립 셀(VRT) base>0 → arm 0;
      s17_4 전 모델 활성 피처에 죽은 9개 없음·피처 수 ≤ 56; s17_1/s17_3 은 사전점검
      JSON(outputs/s17_prechecks/feature_fixes_accuracy.json)의 정확성 판정을 병기.

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

from scripts.eval_s13_46_arm import (  # noqa: E402 — 공용 판정 헬퍼
    E1_DELTA_IR, N_SPLITS, TE_GUARD, _ir, _kst, evaluate_e1,
)

BASE_DIR = AI_PORT / "outputs" / "s17_s0_0902"
PRECHECK = AI_PORT / "outputs" / "s17_prechecks" / "feature_fixes_accuracy.json"
# §S17.3 (G1-01b): nominal-price arm — base = the 09-03 PX_LAST_UNADJ-vintage S0′ recert.
NOMINAL_PRECHECK = AI_PORT / "outputs" / "s17_prechecks" / "nominal_price_gates.json"
HIGH_DIV_FALLBACK = ["MO", "T", "VZ", "PFE", "XOM", "KO", "PG", "JPM", "MSFT", "AAPL"]
ACTIVE_SHARE_BAND = 0.03
FRAMES = {
    "s17_1_coverage_gap_fix": {"frame": "correctness", "turnover_max": 1.25, "alpha_identical": False},
    "s17_2_cov_corr_overlap": {"frame": "risk_discipline", "turnover_max": 1.20, "alpha_identical": True},
    "s17_3_beta_overlap": {"frame": "correctness", "turnover_max": 1.25, "alpha_identical": False},
    "s17_4_dead_feature_prune": {"frame": "hygiene", "turnover_max": 1.25, "alpha_identical": False},
    "s17_5_nominal_price": {"frame": "correctness", "turnover_max": 1.25, "alpha_identical": False,
                            "base": "s17_2_s0recert", "prereg": "decision log §S17.3 (2026-09-03)"},
}
DEAD = {"cal_is_Q1", "regime_mkt_ret_21d", "fac_yield_slope", "fac_F_Quality_mom_63d",
        "fac_F_Growth_mom_63d", "fac_F_Value_mom_63d", "fac_value_growth_63d",
        "fin_roe_level_z", "fin_pb_level_z"}


def evaluate_no_harm(delta_ir: float, split_deltas) -> bool:
    """리스크 규율/위생 프레임: ΔIR > −바 이고 3분할이 전부 음수는 아님."""
    return bool(delta_ir > -E1_DELTA_IR and not all(d < 0 for d in split_deltas))


def evaluate_e2(arm_te: float, base_as: float, arm_as: float,
                turnover_ratio: float, arm_fallback_rate: float,
                turnover_max: float) -> dict:
    return {
        "te_ok": bool(arm_te <= TE_GUARD),
        "active_share_ok": bool(abs(arm_as - base_as) <= ACTIVE_SHARE_BAND),
        "turnover_ok": bool(turnover_ratio <= turnover_max),
        "fallback_zero": bool(arm_fallback_rate == 0),
    }


def _plus5_cells(panel, feature: str, ticker: str) -> int:
    if panel is None or feature not in panel.columns:
        return -1
    try:
        col = panel[feature].xs(ticker, level="ticker")
    except (KeyError, ValueError):
        return -1
    return int((col >= 4.999).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(FRAMES))
    ap.add_argument("--base", default=None)
    args = ap.parse_args()
    spec = FRAMES[args.arm]
    base_dir = (Path(args.base) if args.base
                else AI_PORT / "outputs" / spec.get("base", BASE_DIR.name))
    arm_dir = AI_PORT / "outputs" / args.arm

    base_doc = json.load(open(base_dir / "metrics.json", encoding="utf-8"))
    arm_doc = json.load(open(arm_dir / "metrics.json", encoding="utf-8"))
    base_m, arm_m = base_doc["metrics"], arm_doc["metrics"]
    base_mq, arm_mq = base_doc["model_quality"], arm_doc["model_quality"]
    vintage_ok = base_doc.get("data_vintage") == arm_doc.get("data_vintage")
    ic_identical = bool(arm_m["avg_ic"] == base_m["avg_ic"])
    degenerate_equal = bool(arm_mq["degenerate_retrains"] == base_mq["degenerate_retrains"])
    g0 = {"vintage_equal": bool(vintage_ok), "avg_ic_bit_identical": ic_identical,
          "degenerate_equal": degenerate_equal,
          "g0_pass": bool(vintage_ok and (not spec["alpha_identical"]
                                          or (ic_identical and degenerate_equal)))}

    base_r = pickle.load(open(base_dir / "backtest_result.pkl", "rb"))
    arm_r = pickle.load(open(arm_dir / "backtest_result.pkl", "rb"))

    b_act = pd.Series(base_r.active_returns).dropna()
    a_act = pd.Series(arm_r.active_returns).dropna()
    common = b_act.index.intersection(a_act.index)
    b_act, a_act = b_act.loc[common], a_act.loc[common]
    splits = []
    for k, ix in enumerate(np.array_split(np.arange(len(common)), N_SPLITS)):
        ir_b, ir_a = _ir(b_act.iloc[ix]), _ir(a_act.iloc[ix])
        splits.append({"split": k + 1, "start": str(common[ix[0]].date()),
                       "end": str(common[ix[-1]].date()), "ir_base": round(ir_b, 4),
                       "ir_arm": round(ir_a, 4), "delta": round(ir_a - ir_b, 4)})
    d_ir = arm_m["information_ratio"] - base_m["information_ratio"]
    split_deltas = [s["delta"] for s in splits]
    e1_formal = evaluate_e1(d_ir, split_deltas)
    no_harm = evaluate_no_harm(d_ir, split_deltas)

    mechanism = {}
    if args.arm == "s17_1_coverage_gap_fix":
        cells = {"base": _plus5_cells(getattr(base_r, "panel", None), "tg_upside", "VRT"),
                 "arm": _plus5_cells(getattr(arm_r, "panel", None), "tg_upside", "VRT")}
        mechanism = {"vrt_tg_upside_plus5_cells": cells,
                     "pass": bool(cells["base"] > 0 and cells["arm"] == 0)}
    elif args.arm == "s17_4_dead_feature_prune":
        n_feats = sorted({len(list(m._active_features)) for m in arm_r.models.values()})
        leak = sorted({f for m in arm_r.models.values() for f in m._active_features if f in DEAD})
        base_n = sorted({len(list(m._active_features)) for m in base_r.models.values()})
        mechanism = {"arm_model_feature_counts": n_feats, "base_model_feature_counts": base_n,
                     "dead_features_in_arm_models": leak,
                     "pass": bool(not leak and n_feats and max(n_feats) <= 56)}
    elif args.arm == "s17_5_nominal_price":
        # 사전점검 데이터 게이트(A~D) + 패널 기전: 고배당군 2014 tg_upside 중앙값이
        # base(조정가 분모, ≈+1.0) 대비 arm(명목가 분모)에서 뚜렷이 낮아야 한다.
        gates = json.load(open(NOMINAL_PRECHECK, encoding="utf-8")) if NOMINAL_PRECHECK.exists() else {}
        high_div = gates.get("high_dividend_12") or HIGH_DIV_FALLBACK
        def _hd_2014(res):
            panel = getattr(res, "panel", None)
            if panel is None or "tg_upside" not in panel.columns:
                return None
            col = panel["tg_upside"]
            dates = col.index.get_level_values(0)
            sub = col[(dates >= "2014-01-01") & (dates <= "2014-12-31")]
            sub = sub[sub.index.get_level_values("ticker").isin(high_div)]
            return float(sub.median()) if len(sub) else None
        hd_base, hd_arm = _hd_2014(base_r), _hd_2014(arm_r)
        mechanism = {
            "precheck_gates": {"file": str(NOMINAL_PRECHECK.relative_to(AI_PORT)),
                               "pass": bool(gates.get("gates_pass", False)),
                               "detail": gates.get("gates")},
            "panel_high_div_tg_upside_2014_median": {"base": hd_base, "arm": hd_arm},
            "pass": bool(gates.get("gates_pass", False)
                         and hd_base is not None and hd_arm is not None
                         and hd_arm < hd_base - 0.25),
        }
    if args.arm in ("s17_1_coverage_gap_fix", "s17_3_beta_overlap") and PRECHECK.exists():
        pre = json.load(open(PRECHECK, encoding="utf-8"))
        key, flag = (("t01", "t01_pass") if args.arm == "s17_1_coverage_gap_fix"
                     else ("beta_overlap", "beta_pass"))
        acc = bool(pre[key][flag])
        mechanism["precheck_accuracy"] = {"file": str(PRECHECK.relative_to(AI_PORT)), "pass": acc}
        mechanism["pass"] = bool(mechanism.get("pass", True) and acc)

    turnover_ratio = arm_m["avg_annual_turnover"] / base_m["avg_annual_turnover"]
    e2 = evaluate_e2(arm_m["tracking_error"], base_m["active_share"], arm_m["active_share"],
                     turnover_ratio, arm_doc["optimizer_solver_fallback_rate"], spec["turnover_max"])
    e2_pass = all(e2.values())
    if spec["frame"] == "correctness":
        verdict = bool(g0["g0_pass"] and e2_pass and mechanism.get("pass", True))
        basis = "correctness: accuracy + do-no-harm (dIR is an observation)"
    else:
        verdict = bool(g0["g0_pass"] and e2_pass and mechanism.get("pass", True) and no_harm)
        basis = f"{spec['frame']}: mechanism + do-no-harm (dIR > -{E1_DELTA_IR}, not all splits negative)"

    out = {
        "arm": args.arm, "frame": spec["frame"], "adoption_basis": basis,
        "preregistration": spec.get("prereg", "decision log §S17.1 (2026-09-02)"),
        "vintage": {"base_pkl_mtime": _kst(base_dir / "backtest_result.pkl"),
                    "arm_pkl_mtime": _kst(arm_dir / "backtest_result.pkl"),
                    "base_data_vintage": base_doc.get("data_vintage"),
                    "arm_data_vintage": arm_doc.get("data_vintage")},
        "g0": g0,
        "full_period": {"ir_base": round(base_m["information_ratio"], 4),
                        "ir_arm": round(arm_m["information_ratio"], 4),
                        "delta_ir": round(d_ir, 4), "e1_bar": E1_DELTA_IR},
        "subperiods": splits,
        "e1_formal_pass": e1_formal,
        "no_harm_pass": no_harm,
        "mechanism": mechanism,
        "e2": e2, "e2_pass": e2_pass,
        "flip_candidate": verdict,
        "companions": {
            "te_base": round(base_m["tracking_error"], 5), "te_arm": round(arm_m["tracking_error"], 5),
            "turnover_base": round(base_m["avg_annual_turnover"], 4),
            "turnover_arm": round(arm_m["avg_annual_turnover"], 4),
            "turnover_ratio": round(turnover_ratio, 4),
            "active_share_base": round(base_m["active_share"], 5),
            "active_share_arm": round(arm_m["active_share"], 5),
            "realized_beta_base": base_m.get("realized_beta"),
            "realized_beta_arm": arm_m.get("realized_beta"),
            "avg_ic_base": base_m["avg_ic"], "avg_ic_arm": arm_m["avg_ic"],
            "degenerate": {"base": f"{base_mq['degenerate_retrains']}/{base_mq['total_retrains']}",
                           "arm": f"{arm_mq['degenerate_retrains']}/{arm_mq['total_retrains']}"},
            "fallback_arm": arm_doc["optimizer_solver_fallback_rate"],
            "solver_counts_arm": arm_doc.get("optimizer_solver_counts"),
            "elapsed_sec_arm": arm_doc.get("elapsed_sec"),
        },
    }
    (arm_dir / "e1_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n{args.arm}: G0 {'PASS' if g0['g0_pass'] else 'FAIL'} | dIR {d_ir:+.4f} "
          f"(formal E1 {'PASS' if e1_formal else 'FAIL'}, no-harm {'PASS' if no_harm else 'FAIL'}) | "
          f"E2 {'PASS' if e2_pass else 'FAIL'} | mechanism {mechanism.get('pass', 'n/a')} | "
          f"flip candidate: {verdict}")


if __name__ == "__main__":
    main()
