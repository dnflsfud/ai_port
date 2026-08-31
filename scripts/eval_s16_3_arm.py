# -*- coding: utf-8 -*-
"""§S16.3 arm E1/E2 판정 (read-only): fresh_fixed 퇴화 폴백 vs §S16.1 런.

결정 로그 §S16.3 사전등록(2026-08-31)의 재실행 가능한 판정 스크립트.
  비교 기준: outputs/s16_1_unit_fixpack (유일한 델타 = degenerate_fallback_mode).
  E1 primary: full ΔIR > +0.36 AND 시간순 3분할 ΔIR 부호 일관(전부 양).
  병기 의무: unique_models · live_model_age_retrains(0 기대) · 퇴화율 ·
    fresh_fixed 폴백 발동 수(발동 0이면 arm이 no-op — 판정 무의미).
  E2 do-no-harm: TE ≤ 4.5% · active share 캐릭터 보존(base ± 3%p) ·
    turnover ≤ 1.25× · ECOS fallback 0.

출력: outputs/s16_3_fresh_fixed/e1_summary.json
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

BASE_DIR = AI_PORT / "outputs" / "s16_1_unit_fixpack"
ARM_DIR = AI_PORT / "outputs" / "s16_3_fresh_fixed"
TURNOVER_RATIO_MAX = 1.25
ACTIVE_SHARE_BAND = 0.03


def evaluate_e2(arm_te: float, base_as: float, arm_as: float,
                turnover_ratio: float, arm_fallback_rate: float) -> dict:
    """E2 do-no-harm 4항목(사전등록 §S16.3)."""
    return {
        "te_ok": bool(arm_te <= TE_GUARD),
        "active_share_ok": bool(abs(arm_as - base_as) <= ACTIVE_SHARE_BAND),
        "turnover_ok": bool(turnover_ratio <= TURNOVER_RATIO_MAX),
        "fallback_zero": bool(arm_fallback_rate == 0),
    }


def count_fresh_fires(events) -> tuple:
    """model_quality.events에서 fresh_fixed 폴백 발동 수·날짜 목록."""
    fired = [e for e in events if e.get("fallback") == "fresh_fixed"]
    return len(fired), [e.get("date") for e in fired]


def main() -> None:
    base_doc = json.load(open(BASE_DIR / "metrics.json", encoding="utf-8"))
    arm_doc = json.load(open(ARM_DIR / "metrics.json", encoding="utf-8"))
    base_m, arm_m = base_doc["metrics"], arm_doc["metrics"]
    base_mq, arm_mq = base_doc["model_quality"], arm_doc["model_quality"]

    base_r = pickle.load(open(BASE_DIR / "backtest_result.pkl", "rb"))
    arm_r = pickle.load(open(ARM_DIR / "backtest_result.pkl", "rb"))

    b_act = pd.Series(base_r.active_returns).dropna()
    a_act = pd.Series(arm_r.active_returns).dropna()
    common = b_act.index.intersection(a_act.index)
    b_act, a_act = b_act.loc[common], a_act.loc[common]
    vintage_ok = (
        base_doc.get("data_vintage") == arm_doc.get("data_vintage")
        and str(pd.Timestamp(sorted(base_r.daily_weights)[-1]).date())
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

    n_fires, fire_dates = count_fresh_fires(arm_mq.get("events", []))
    turnover_ratio = arm_m["avg_annual_turnover"] / base_m["avg_annual_turnover"]
    e2 = evaluate_e2(arm_m["tracking_error"], base_m["active_share"],
                     arm_m["active_share"], turnover_ratio,
                     arm_doc["optimizer_solver_fallback_rate"])

    out = {
        "preregistration": "decision log §S16.3 (2026-08-31)",
        "vintage": {
            "base_pkl_mtime": _kst(BASE_DIR / "backtest_result.pkl"),
            "arm_pkl_mtime": _kst(ARM_DIR / "backtest_result.pkl"),
            "data_vintage_and_last_weights_equal": vintage_ok,
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
        "mandatory_companions": {
            "unique_models": {"base": base_mq["unique_models"],
                              "arm": arm_mq["unique_models"]},
            "live_model_fit_date": {"base": base_mq["live_model_fit_date"],
                                    "arm": arm_mq["live_model_fit_date"]},
            "live_model_age_retrains": {
                "base": base_mq["live_model_age_retrains"],
                "arm": arm_mq["live_model_age_retrains"]},
            "degenerate": {
                "base": f"{base_mq['degenerate_retrains']}/{base_mq['total_retrains']}",
                "arm": f"{arm_mq['degenerate_retrains']}/{arm_mq['total_retrains']}"},
            "fresh_fixed_fires": n_fires,
            "fresh_fixed_dates": fire_dates,
        },
        "e2": e2,
        "e2_pass": all(e2.values()),
        "companions": {
            "te_base": round(base_m["tracking_error"], 5),
            "te_arm": round(arm_m["tracking_error"], 5),
            "beta_base": round(base_m["realized_beta"], 4),
            "beta_arm": round(arm_m["realized_beta"], 4),
            "turnover_base": round(base_m["avg_annual_turnover"], 4),
            "turnover_arm": round(arm_m["avg_annual_turnover"], 4),
            "turnover_ratio": round(turnover_ratio, 4),
            "active_share_base": round(base_m["active_share"], 5),
            "active_share_arm": round(arm_m["active_share"], 5),
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
          f"3분할 부호 일관 {sign_consistent})  "
          f"E2: {'PASS' if all(e2.values()) else 'FAIL'}  "
          f"fresh_fixed 발동 {n_fires}회, "
          f"라이브 모델 나이 {base_mq['live_model_age_retrains']} → "
          f"{arm_mq['live_model_age_retrains']}")


if __name__ == "__main__":
    main()
