# -*- coding: utf-8 -*-
"""§S26 정확성 arm 판정 (read-only) — s26_stale_fund_mask vs 기준 outputs/s26_s0_base
(2026-10-30 12:14 production 런의 동결 사본, 워크북 2026-10-21 14:58 KST / Index 2026-10-26 11:14 KST 빈티지).

결정 로그 §S26 사전등록(2026-10-31)의 재실행 가능한 판정 스크립트. 정확성 트랙(인벤토리 비계수), §S23.1/§S25.4 프레임:
  flip 후보 = G0 ∧ 기전 ∧ E2 ∧ 무해성. formal E1 은 병기만.
  G0 : base·arm data_vintage 동일 ∧ base IR 이 REFERENCE_BASE_IR 을 1e-9 이내 재현 ∧ arm overrides 에 플래그 True.
  기전: data_quality.stale_run_mask 에 펀더멘털 4 시트 전부 masked_cells > 0 ∧ max_run == 126 ∧ 사전 명시 사례
        (BEST_GROSS_MARGIN/APO · BEST_CALCULATED_FCF/C · BEST_CAPEX/APO) 가 상위 종목에 포함 ∧ 피처 패널 변화가
        펀더멘털 4 시트 파생 피처(gross_margin/op_leverage/fcf/cash_conversion/capex/roe 토큰)에만 있고 비어 있지 않음
        ∧ 피처 집합 동일. 비율 4 시트(production 21행 마스크)는 보고만.
  E2 : TE ≤ 4.5% · active share ±3%p · 회전율 ≤ 1.25× · solver fallback 0.
  무해성: ΔIR > −0.36 ∧ 3분할 전부 음 아님.
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

from scripts.eval_s13_46_arm import E1_DELTA_IR, N_SPLITS, _ir, _kst  # noqa: E402
from scripts.eval_s17_arm import evaluate_e2  # noqa: E402
from scripts.eval_s25_4_arms import _load, changed_features, mechanism_feature_subset, verdicts  # noqa: E402

ARM = "s26_stale_fund_mask"
FLAG = "stale_run_fund_mask_enabled"
BASE = "s26_s0_base"
# outputs/s26_s0_base/metrics.json — frozen copy of outputs/codex_causal_rank_65 (2026-10-30 12:14 scheduled production run,
# workbook mtime 2026-10-21T05:58:08Z / Index mtime 2026-10-26T02:14:37Z).
REFERENCE_BASE_IR = 1.7737993717667202
TURNOVER_MAX = 1.25
FUND_SHEETS = ("BEST_GROSS_MARGIN", "BEST_CALCULATED_FCF", "BEST_CAPEX", "BEST_ROE")
FUND_MAX_RUN = 126
FUND_NAMED_CASES = (("BEST_GROSS_MARGIN", "APO"), ("BEST_CALCULATED_FCF", "C"), ("BEST_CAPEX", "APO"))
RATIO_SHEETS = ("BEST_PE_RATIO", "BEST_PX_BPS_RATIO", "BEST_PEG_RATIO", "BEST_EV_TO_BEST_EBITDA")
# Consumers of the four sheets in the core whitelist (src/features/accounting.py):
#   BEST_GROSS_MARGIN -> best_gross_margin_chg_63d/252d, op_leverage_63d (om - gm pct change)
#   BEST_CALCULATED_FCF -> best_calculated_fcf_level_z, cash_conversion_z
#   BEST_CAPEX -> best_capex_level_z, capex_intensity_z
#   BEST_ROE -> best_roe_level_z, fin_roe_level_z, fin_roe_chg_63d/252d, fin_roe_pe_gap, fin_roe_pb_gap
_FUND_TOKENS = ("gross_margin", "op_leverage", "fcf", "cash_conversion", "capex", "roe")


def mechanism_fund_mask(data_quality) -> dict:
    rec = (data_quality or {}).get("stale_run_mask") or {}
    missing = [s for s in FUND_SHEETS if int((rec.get(s) or {}).get("masked_cells", 0)) <= 0]
    wrong_window = [s for s in FUND_SHEETS if s in rec and int(rec[s].get("max_run", -1)) != FUND_MAX_RUN]
    named_missing = [f"{s}/{t}" for s, t in FUND_NAMED_CASES if t not in ((rec.get(s) or {}).get("top") or {})]
    return {"per_sheet": {s: {k: v for k, v in (rec.get(s) or {}).items()} for s in FUND_SHEETS},
            "ratio_sheets_reported": {s: {k: v for k, v in (rec.get(s) or {}).items() if k != "top"} for s in RATIO_SHEETS if s in rec},
            "total_masked_cells": int(sum(int((rec.get(s) or {}).get("masked_cells", 0)) for s in FUND_SHEETS)),
            "missing_sheets": missing, "wrong_window": wrong_window, "named_cases_missing": named_missing,
            "pass": bool(rec) and not missing and not wrong_window and not named_missing}


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

    changed = changed_features(base_r.panel, arm_r.panel)
    feature_set_equal = set(base_r.feature_names) == set(arm_r.feature_names)
    mech = mechanism_fund_mask(arm_doc.get("data_quality") or getattr(arm_r, "data_quality", None))
    sub = mechanism_feature_subset(changed, lambda f: any(t in f for t in _FUND_TOKENS))
    mech["feature_subset"] = sub
    mech["feature_set_equal"] = bool(feature_set_equal)
    mechanism_pass = bool(mech["pass"] and sub["pass"] and feature_set_equal)

    turnover_ratio = am["avg_annual_turnover"] / bm["avg_annual_turnover"]
    e2 = evaluate_e2(am["tracking_error"], bm["active_share"], am["active_share"], turnover_ratio,
                     arm_doc["optimizer_solver_fallback_rate"], TURNOVER_MAX)
    e2_pass = all(e2.values())
    v = verdicts(g0["g0_pass"], mechanism_pass, e2_pass, d_ir, split_deltas)

    out = {
        "arm": args.arm, "flag": FLAG, "frame": "accuracy (G0 + mechanism + E2 + no-harm; formal E1 reported only)",
        "preregistration": "decision log §S26 (2026-10-31)", "base_dir": str(base_dir),
        "vintage": {"base_pkl_mtime": _kst(base_dir / "backtest_result.pkl"), "arm_pkl_mtime": _kst(arm_dir / "backtest_result.pkl"),
                    "base_data_vintage": base_doc.get("data_vintage"), "arm_data_vintage": arm_doc.get("data_vintage")},
        "g0": g0,
        "full_period": {"ir_base": round(bm["information_ratio"], 4), "ir_arm": round(am["information_ratio"], 4),
                        "delta_ir": round(d_ir, 4), "no_harm_bar": -E1_DELTA_IR},
        "subperiods": splits, **v,
        "mechanism": mech, "mechanism_pass": mechanism_pass,
        "changed_features_top": dict(sorted(changed.items(), key=lambda kv: -kv[1])[:16]),
        "e2": e2, "e2_pass": e2_pass,
        "companions": {
            "te_base": round(bm["tracking_error"], 5), "te_arm": round(am["tracking_error"], 5),
            "turnover_base": round(bm["avg_annual_turnover"], 4), "turnover_arm": round(am["avg_annual_turnover"], 4),
            "turnover_ratio": round(turnover_ratio, 4),
            "active_share_base": round(bm["active_share"], 5), "active_share_arm": round(am["active_share"], 5),
            "realized_beta_base": bm.get("realized_beta"), "realized_beta_arm": am.get("realized_beta"),
            "avg_ic_base": bm["avg_ic"], "avg_ic_arm": am["avg_ic"],
            "max_drawdown_base": bm.get("max_drawdown"), "max_drawdown_arm": am.get("max_drawdown"),
            "sub_periods_metrics_base": bm.get("sub_periods"), "sub_periods_metrics_arm": am.get("sub_periods"),
            "degenerate": {"base": f"{base_doc['model_quality']['degenerate_retrains']}/{base_doc['model_quality']['total_retrains']}",
                           "arm": f"{arm_doc['model_quality']['degenerate_retrains']}/{arm_doc['model_quality']['total_retrains']}"},
            "fallback_arm": arm_doc["optimizer_solver_fallback_rate"], "solver_counts_arm": arm_doc.get("optimizer_solver_counts"),
            "elapsed_sec_arm": arm_doc.get("elapsed_sec"),
        },
    }
    (arm_dir / "e1_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    print(f"\n{args.arm}: G0 {'PASS' if g0['g0_pass'] else 'FAIL'} | mechanism {'PASS' if mechanism_pass else 'FAIL'} | "
          f"E2 {'PASS' if e2_pass else 'FAIL'} | dIR {d_ir:+.4f} splits {split_deltas} no-harm {'PASS' if v['no_harm_pass'] else 'FAIL'} | "
          f"flip candidate: {v['flip_candidate']}")


if __name__ == "__main__":
    main()
