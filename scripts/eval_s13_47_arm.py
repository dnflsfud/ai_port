# -*- coding: utf-8 -*-
"""§S13.47 arm E1 판정 (read-only): rank_eval_at [20] arm vs 동일 빈티지 S0′.

결정 로그 §S13.47 사전등록 게이트의 재실행 가능한 판정 스크립트.
  E1 primary: full ΔIR > +0.36 AND 시간순 3분할 ΔIR 부호 일관(전부 양).
  병기 의무: **퇴화율(baseline 13/33 대비, 악화 여부 명시)**, avg_ic,
    TE(실현 가드 4.5%), realized_beta, turnover, fallback율.
  §S13.46과 달리 신호 경로 arm이므로 avg_ic 동일성을 요구하지 않는다.

출력: outputs/s13_47_ndcg20/e1_summary.json
"""

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

BASE_DIR = AI_PORT / "outputs" / "codex_causal_rank_65"
ARM_DIR = AI_PORT / "outputs" / "s13_47_ndcg20"


def degeneracy_comparison(base_mq: dict, arm_mq: dict) -> dict:
    """퇴화율 병기(§S13.45-D arm 위험 기록 이행): rate와 악화 여부."""
    rb = base_mq["degenerate_retrains"] / base_mq["total_retrains"]
    ra = arm_mq["degenerate_retrains"] / arm_mq["total_retrains"]
    return {
        "base": f"{base_mq['degenerate_retrains']}/{base_mq['total_retrains']}",
        "arm": f"{arm_mq['degenerate_retrains']}/{arm_mq['total_retrains']}",
        "rate_base": rb,
        "rate_arm": ra,
        "worsened": bool(ra > rb),
    }


def main() -> None:
    base_doc = json.load(open(BASE_DIR / "metrics.json", encoding="utf-8"))
    arm_doc = json.load(open(ARM_DIR / "metrics.json", encoding="utf-8"))
    base_m, arm_m = base_doc["metrics"], arm_doc["metrics"]

    base_r = pickle.load(open(BASE_DIR / "backtest_result.pkl", "rb"))
    arm_r = pickle.load(open(ARM_DIR / "backtest_result.pkl", "rb"))

    b_act = pd.Series(base_r.active_returns).dropna()
    a_act = pd.Series(arm_r.active_returns).dropna()
    common = b_act.index.intersection(a_act.index)
    b_act, a_act = b_act.loc[common], a_act.loc[common]
    vintage_ok = (
        str(pd.Timestamp(sorted(base_r.daily_weights)[-1]).date())
        == str(pd.Timestamp(sorted(arm_r.daily_weights)[-1]).date()))

    splits = []
    for k, idx in enumerate(np.array_split(np.arange(len(common)), N_SPLITS)):
        ir_b = _ir(b_act.iloc[idx])
        ir_a = _ir(a_act.iloc[idx])
        splits.append({
            "split": k + 1,
            "start": str(common[idx[0]].date()),
            "end": str(common[idx[-1]].date()),
            "ir_base": round(ir_b, 4), "ir_arm": round(ir_a, 4),
            "delta": round(ir_a - ir_b, 4),
        })

    d_ir = arm_m["information_ratio"] - base_m["information_ratio"]
    split_deltas = [s["delta"] for s in splits]
    sign_consistent = all(d > 0 for d in split_deltas)
    e1 = evaluate_e1(d_ir, split_deltas)
    te_ok = bool(arm_m["tracking_error"] <= TE_GUARD)
    degen = degeneracy_comparison(base_doc["model_quality"],
                                  arm_doc["model_quality"])

    out = {
        "preregistration": "decision log §S13.47 (2026-08-20)",
        "vintage": {
            "base_pkl_mtime": _kst(BASE_DIR / "backtest_result.pkl"),
            "arm_pkl_mtime": _kst(ARM_DIR / "backtest_result.pkl"),
            "last_daily_weights_equal": vintage_ok,
        },
        "full_period": {
            "ir_base": round(base_m["information_ratio"], 4),
            "ir_arm": round(arm_m["information_ratio"], 4),
            "delta_ir": round(d_ir, 4),
            "e1_bar": E1_DELTA_IR,
        },
        "subperiods": splits,
        "sign_consistent": sign_consistent,
        "e1_pass": e1,
        "degeneracy": {k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in degen.items()},
        "companions": {
            "te_base": round(base_m["tracking_error"], 5),
            "te_arm": round(arm_m["tracking_error"], 5),
            "te_guard_ok": te_ok,
            "beta_base": round(base_m["realized_beta"], 4),
            "beta_arm": round(arm_m["realized_beta"], 4),
            "turnover_base": round(base_m["avg_annual_turnover"], 4),
            "turnover_arm": round(arm_m["avg_annual_turnover"], 4),
            "maxdd_base": round(base_m["max_drawdown"], 4),
            "maxdd_arm": round(arm_m["max_drawdown"], 4),
            "avg_ic_base": base_m["avg_ic"],
            "avg_ic_arm": arm_m["avg_ic"],
            "fallback_base": base_doc["optimizer_solver_fallback_rate"],
            "fallback_arm": arm_doc["optimizer_solver_fallback_rate"],
        },
    }
    (ARM_DIR / "e1_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nE1: {'PASS' if e1 else 'FAIL'}  (ΔIR {d_ir:+.4f} vs bar +{E1_DELTA_IR}, "
          f"3분할 부호 일관 {sign_consistent}, TE 가드 {te_ok}, "
          f"퇴화율 {degen['base']} → {degen['arm']} "
          f"{'악화' if degen['worsened'] else '악화 아님'})")


if __name__ == "__main__":
    main()
