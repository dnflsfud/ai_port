# -*- coding: utf-8 -*-
"""§S28.2 arm 판정 (read-only): 커버리지 시작 전 셀 마스크(컨센서스 11개 시트) vs 같은 빈티지·같은 코드 기준 런.

결정 로그 §S28.2 사전등록(2026-11-10)의 재실행 가능한 판정 스크립트. 정확성 트랙(§S17.1 arm A 선례, IR 채택 바 아님):
  G0: data_vintage 동일 ∧ arm overrides = base overrides + {pre_coverage_mask_enabled: True} 정확히 1줄.
  G1(기전): arm data_quality.pre_coverage_mask 의 마스킹 셀 합계 > 0(시트별 셀·종목 수 기록).
  G2(무해성): ΔIR > −0.10 ∧ 쌍대 NW t > −1 ∧ TE ≤ 4.5% · |ΔAS| ≤ 3%p · 회전율 ≤ 1.25× · fallback 0.
  flip 후보 = G0 ∧ G1 ∧ G2. avg_ic·3분할·변경 피처는 병기.
사용: python scripts/eval_s28_2_pre_coverage_arm.py --arm <arm label> --base <base label>
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
from scripts.eval_s17_arm import evaluate_e2  # noqa: E402
from scripts.eval_s25_4_arms import _load, changed_features  # noqa: E402
from scripts.eval_s27_om_pit_arm import paired_nw_t  # noqa: E402

FLAG = "pre_coverage_mask_enabled"
NO_HARM_DELTA_IR = -0.10
NO_HARM_T = -1.0
TURNOVER_MAX = 1.25


def overrides_differ_only_by_flag(base_overrides: dict, arm_overrides: dict) -> bool:
    """True iff the arm is the base plus exactly ``FLAG: True``."""
    if FLAG in base_overrides or arm_overrides.get(FLAG) is not True:
        return False
    return {k: v for k, v in arm_overrides.items() if k != FLAG} == dict(base_overrides)


def mechanism_mask(data_quality) -> dict:
    rec = (data_quality or {}).get("pre_coverage_mask") or {}
    by_sheet = {name: {k: int(row.get(k, 0)) for k in ("masked_cells", "tickers", "never_observed_tickers") if k in row}
                for name, row in rec.items()}
    total = int(sum(row.get("masked_cells", 0) for row in by_sheet.values()))
    return {"by_sheet": by_sheet, "total_masked_cells": total,
            "sheets_with_masked_cells": int(sum(row.get("masked_cells", 0) > 0 for row in by_sheet.values())),
            "pass": bool(total > 0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--base", required=True)
    args = ap.parse_args()
    base_dir, arm_dir = AI_PORT / "outputs" / args.base, AI_PORT / "outputs" / args.arm
    base_doc, base_r = _load(base_dir)
    arm_doc, arm_r = _load(arm_dir)
    bm, am = base_doc["metrics"], arm_doc["metrics"]

    g0 = {"vintage_equal": bool(base_doc.get("data_vintage") == arm_doc.get("data_vintage")),
          "arm_is_base_plus_flag": overrides_differ_only_by_flag(base_doc["overrides"], arm_doc["overrides"])}
    g0["g0_pass"] = bool(all(g0.values()))

    b_act, a_act = pd.Series(base_r.active_returns).dropna(), pd.Series(arm_r.active_returns).dropna()
    common = b_act.index.intersection(a_act.index)
    b_act, a_act = b_act.loc[common], a_act.loc[common]
    splits = []
    for k, ix in enumerate(np.array_split(np.arange(len(common)), N_SPLITS)):
        ir_b, ir_a = _ir(b_act.iloc[ix]), _ir(a_act.iloc[ix])
        splits.append({"split": k + 1, "start": str(common[ix[0]].date()), "end": str(common[ix[-1]].date()),
                       "ir_base": round(ir_b, 4), "ir_arm": round(ir_a, 4), "delta": round(ir_a - ir_b, 4)})
    d_ir = am["information_ratio"] - bm["information_ratio"]
    t_nw = paired_nw_t(a_act - b_act)

    g1 = mechanism_mask(arm_doc.get("data_quality"))
    changed = changed_features(base_r.panel, arm_r.panel)
    turnover_ratio = am["avg_annual_turnover"] / bm["avg_annual_turnover"]
    e2 = evaluate_e2(am["tracking_error"], bm["active_share"], am["active_share"], turnover_ratio,
                     arm_doc["optimizer_solver_fallback_rate"], TURNOVER_MAX)
    no_harm = bool(d_ir > NO_HARM_DELTA_IR and np.isfinite(t_nw) and t_nw > NO_HARM_T)
    g2 = {**e2, "no_harm": no_harm}
    g2_pass = all(g2.values())
    flip = bool(g0["g0_pass"] and g1["pass"] and g2_pass)

    out = {
        "arm": args.arm, "base": args.base, "flag": FLAG, "preregistration": "decision log §S28.2 (2026-11-10)",
        "frame": "correctness track (G0 + G1 mechanism + G2 do-no-harm); formal E1 / avg_ic reported only",
        "vintage": {"base_pkl_mtime": _kst(base_dir / "backtest_result.pkl"), "arm_pkl_mtime": _kst(arm_dir / "backtest_result.pkl"),
                    "data_vintage": arm_doc.get("data_vintage")},
        "g0": g0, "g1": g1, "g2": g2, "g2_pass": g2_pass, "flip_candidate": flip,
        "full_period": {"ir_base": bm["information_ratio"], "ir_arm": am["information_ratio"], "delta_ir": round(d_ir, 4),
                        "paired_nw_t": round(t_nw, 3), "e1_formal_pass": bool(evaluate_e1(d_ir, [s["delta"] for s in splits])),
                        "avg_ic_base": bm["avg_ic"], "avg_ic_arm": am["avg_ic"]},
        "subperiods": splits,
        "changed_features_top": dict(sorted(changed.items(), key=lambda kv: -kv[1])[:20]),
        "changed_feature_count": int(len(changed)),
        "companions": {
            "te_base": bm["tracking_error"], "te_arm": am["tracking_error"], "turnover_ratio": round(turnover_ratio, 4),
            "active_share_base": bm["active_share"], "active_share_arm": am["active_share"],
            "realized_beta_base": bm.get("realized_beta"), "realized_beta_arm": am.get("realized_beta"),
            "degenerate": {"base": f"{base_doc['model_quality']['degenerate_retrains']}/{base_doc['model_quality']['total_retrains']}",
                           "arm": f"{arm_doc['model_quality']['degenerate_retrains']}/{arm_doc['model_quality']['total_retrains']}"},
            "elapsed_sec_arm": arm_doc.get("elapsed_sec"),
        },
    }
    (arm_dir / "e1_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    print(f"\n{args.arm}: G0 {'PASS' if g0['g0_pass'] else 'FAIL'} | G1 {'PASS' if g1['pass'] else 'FAIL'} "
          f"({g1['total_masked_cells']} cells) | G2 {'PASS' if g2_pass else 'FAIL'} | dIR {d_ir:+.4f} t {t_nw:+.2f} "
          f"| flip candidate: {flip}")


if __name__ == "__main__":
    main()
