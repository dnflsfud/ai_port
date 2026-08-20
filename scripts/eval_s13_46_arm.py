# -*- coding: utf-8 -*-
"""§S13.46 arm E1 판정 (read-only): ICorr 비대각 arm vs 동일 빈티지 S0′.

결정 로그 §S13.46 사전등록 게이트의 재실행 가능한 판정 스크립트.
  E1 primary: full ΔIR > +0.36 AND 시간순 3분할 ΔIR 부호 일관(전부 양).
  병기 의무: TE(실현 가드 4.5%), realized_beta, turnover,
    avg_ic 동일성(Σ-only 채널 → 알파 비트 불변 증명), fallback율.

출력: outputs/s13_46_icorr_cov/e1_summary.json
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AI_PORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_PORT))

BASE_DIR = AI_PORT / "outputs" / "codex_causal_rank_65"
ARM_DIR = AI_PORT / "outputs" / "s13_46_icorr_cov"
E1_DELTA_IR = 0.36
TE_GUARD = 0.045
N_SPLITS = 3
_ANN = float(np.sqrt(252.0))


def _ir(active: pd.Series) -> float:
    sd = float(active.std())
    return float(active.mean()) / sd * _ANN if sd > 0 else float("nan")


def evaluate_e1(delta_ir: float, split_deltas) -> bool:
    """E1 primary: full ΔIR > 바(+0.36, 초과) AND 3분할 전부 양(부호 일관)."""
    return bool(delta_ir > E1_DELTA_IR and all(d > 0 for d in split_deltas))


def _kst(path: Path) -> str:
    return pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC") \
        .tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S")


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
    ic_identical = bool(arm_m["avg_ic"] == base_m["avg_ic"])
    te_ok = bool(arm_m["tracking_error"] <= TE_GUARD)

    out = {
        "preregistration": "decision log §S13.46 (2026-08-20)",
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
            "avg_ic_identical": ic_identical,
            "fallback_base": base_doc["optimizer_solver_fallback_rate"],
            "fallback_arm": arm_doc["optimizer_solver_fallback_rate"],
        },
    }
    (ARM_DIR / "e1_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nE1: {'PASS' if e1 else 'FAIL'}  (ΔIR {d_ir:+.4f} vs bar +{E1_DELTA_IR}, "
          f"3분할 부호 일관 {sign_consistent}, avg_ic 동일 {ic_identical}, "
          f"TE 가드 {te_ok})")


if __name__ == "__main__":
    main()
