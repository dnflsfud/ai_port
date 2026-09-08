# -*- coding: utf-8 -*-
"""§S18.1 arm 판정 (read-only) — `--arm <label>`, 기준 = outputs/s18_s0recert.

결정 로그 §S18.1 사전등록(2026-09-07)의 재실행 가능한 판정 스크립트.
  기준 S0′: outputs/s18_s0recert = 09-07 12:36 production 런 동결본
            (워크북 2026-09-04 05:50:52Z / Index 2026-09-07 02:04:19Z, IR 1.6106, APH 1.166).
  G0: data_vintage 쌍 동일. s18_3(집행 전용)은 avg_ic 비트 동일·퇴화 동일 필수.
  E1: full ΔIR·3분할 — 정확성 트랙(s18_1·s18_2)은 관측; 집행 트랙(s18_3)은 no-harm
      (ΔIR > −0.36 이고 3분할 전부 음 아님) 을 flip 후보 조건으로, formal E1 은 병기.
  E2: TE ≤ 4.5% · active share ±3%p · turnover ≤ 1.25× · fallback 0.
  기전(arm별):
    s18_1 — 사전 감사(§S18 B1)의 음자본 21종에서 실행 전 예측(pre_execution_predictions)
            arm−base 평균 > +0.10 (벌점 해소) 이고 그 종목들의 변경 셀 ≥ 100;
    s18_2 — 패널 tg_upside z 연중앙값: RTX 2014~19 ≥ −1.5, T 2014~21 ≥ −1.0,
            DELL 2019~21 ≤ 3.0, DHR 2014~15 ≤ 1.5, 그리고 2014~21 횡단면 z 표준편차
            연중앙값 ≥ 0.90 (base 값 병기);
    s18_3 — 회전율 비 ∈ [1.00, 1.25] 이고 avg_ic·퇴화 비트 동일.
  §S18.3 재도전(2026-09-08) s18_5_static_execution_eta042 — base = outputs/s18_4_flip2_recert
    (두 flip 재인증 S0′), 기전 = 회전율 중립 비 ∈ TURNOVER_NEUTRAL_BAND [0.85, 1.15] 이고
    avg_ic·퇴화 비트 동일; 나머지 판정 프레임(집행: G0∧E2∧기전∧no-harm, formal E1 병기)은 동일.
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
from scripts.eval_s17_arm import evaluate_e2, evaluate_no_harm  # noqa: E402

BASE = "s18_s0recert"
ACTIVE_SHARE_BAND = 0.03
FRAMES = {
    "s18_1_tilt_negative_equity": {"frame": "correctness", "turnover_max": 1.25, "alpha_identical": False},
    "s18_2_tg_basis_events": {"frame": "correctness", "turnover_max": 1.25, "alpha_identical": False},
    "s18_3_static_execution": {"frame": "execution", "turnover_max": 1.25, "alpha_identical": True},
    # §S18.3 (2026-09-08) turnover-neutral re-attempt; base = the S18.2 two-flip re-certification.
    "s18_5_static_execution_eta042": {"frame": "execution", "turnover_max": 1.25, "alpha_identical": True,
                                      "base": "s18_4_flip2_recert", "mechanism": "static_neutral",
                                      "preregistration": "decision log §S18.3 (2026-09-08)"},
}
TURNOVER_NEUTRAL_BAND = (0.85, 1.15)
# §S18 B1 (outputs/s18_prechecks/probe_b_results.json): negative-equity names that
# received the tilt penalty inside the top vol tercile on the certified S0'.
NEG_EQUITY_NAMES = [
    "DELL", "FICO", "HCA", "MSCI", "RR/", "PM", "LNG", "ABBV", "SBUX", "VRSN", "ADSK",
    "LYV", "NRG", "MSI", "ORCL", "BKNG", "LOW", "MO", "STX", "CL", "ORLY",
]
TG_BASIS_GATES = {  # ticker: (year_from, year_to, op, bound)
    "RTX": (2014, 2019, ">=", -1.5),
    "T": (2014, 2021, ">=", -1.0),
    "DELL": (2019, 2021, "<=", 3.0),
    "DHR": (2014, 2015, "<=", 1.5),
}
Z_SD_MIN = 0.90


def _year_median(panel, feature, ticker, y0, y1):
    if panel is None or feature not in panel.columns:
        return None
    try:
        col = panel[feature].xs(ticker, level="ticker")
    except (KeyError, ValueError):
        return None
    sub = col[(col.index.year >= y0) & (col.index.year <= y1)]
    return float(sub.median()) if len(sub) else None


def _cs_sd_median(panel, feature, y0=2014, y1=2021):
    if panel is None or feature not in panel.columns:
        return None
    wide = panel[feature].unstack("ticker")
    sd = wide.std(axis=1)
    sd = sd[(sd.index.year >= y0) & (sd.index.year <= y1)]
    return float(sd.median()) if len(sd) else None


def mechanism_tilt(base_r, arm_r) -> dict:
    b = getattr(base_r, "pre_execution_predictions", None)
    a = getattr(arm_r, "pre_execution_predictions", None)
    if b is None or a is None:
        return {"pass": False, "reason": "pre_execution_predictions missing"}
    common = b.index.intersection(a.index)
    cols = [t for t in NEG_EQUITY_NAMES if t in b.columns and t in a.columns]
    delta = (a.loc[common, cols] - b.loc[common, cols])
    changed = delta.abs() > 1e-9
    mean_delta = float(delta[changed].stack().mean()) if changed.to_numpy().any() else 0.0
    n_changed = int(changed.to_numpy().sum())
    return {"neg_equity_names": cols, "changed_cells": n_changed,
            "mean_delta_on_changed_cells": round(mean_delta, 4),
            "pass": bool(n_changed >= 100 and mean_delta > 0.10)}


def mechanism_tg_basis(base_r, arm_r) -> dict:
    bp, ap = getattr(base_r, "panel", None), getattr(arm_r, "panel", None)
    rows, ok = {}, True
    for t, (y0, y1, op, bound) in TG_BASIS_GATES.items():
        vb, va = _year_median(bp, "tg_upside", t, y0, y1), _year_median(ap, "tg_upside", t, y0, y1)
        passed = va is not None and ((va >= bound) if op == ">=" else (va <= bound))
        rows[t] = {"years": [y0, y1], "base": vb, "arm": va, "gate": f"{op} {bound}", "pass": bool(passed)}
        ok = ok and passed
    sd_b, sd_a = _cs_sd_median(bp, "tg_upside"), _cs_sd_median(ap, "tg_upside")
    sd_ok = sd_a is not None and sd_a >= Z_SD_MIN
    return {"names": rows, "tg_upside_cs_sd_2014_2021": {"base": sd_b, "arm": sd_a, "gate": f">= {Z_SD_MIN}"},
            "pass": bool(ok and sd_ok)}


def mechanism_static(turnover_ratio: float, g0: dict) -> dict:
    return {"turnover_ratio": round(turnover_ratio, 4),
            "pass": bool(1.0 <= turnover_ratio <= 1.25 and g0["avg_ic_bit_identical"] and g0["degenerate_equal"])}


def mechanism_static_neutral(turnover_ratio: float, g0: dict, band=TURNOVER_NEUTRAL_BAND) -> dict:
    lo, hi = band
    return {"turnover_ratio": round(turnover_ratio, 4), "neutral_band": [lo, hi],
            "pass": bool(lo <= turnover_ratio <= hi and g0["avg_ic_bit_identical"] and g0["degenerate_equal"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(FRAMES))
    ap.add_argument("--base", default=None)
    args = ap.parse_args()
    spec = FRAMES[args.arm]
    base_dir = Path(args.base) if args.base else AI_PORT / "outputs" / spec.get("base", BASE)
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
          "g0_pass": bool(vintage_ok and (not spec["alpha_identical"] or (ic_identical and degenerate_equal)))}

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

    turnover_ratio = arm_m["avg_annual_turnover"] / base_m["avg_annual_turnover"]
    if args.arm == "s18_1_tilt_negative_equity":
        mechanism = mechanism_tilt(base_r, arm_r)
    elif args.arm == "s18_2_tg_basis_events":
        mechanism = mechanism_tg_basis(base_r, arm_r)
    elif spec.get("mechanism") == "static_neutral":
        mechanism = mechanism_static_neutral(turnover_ratio, g0)
    else:
        mechanism = mechanism_static(turnover_ratio, g0)

    e2 = evaluate_e2(arm_m["tracking_error"], base_m["active_share"], arm_m["active_share"],
                     turnover_ratio, arm_doc["optimizer_solver_fallback_rate"], spec["turnover_max"])
    e2_pass = all(e2.values())
    if spec["frame"] == "correctness":
        verdict = bool(g0["g0_pass"] and e2_pass and mechanism.get("pass", False))
        basis = "correctness: mechanism + do-no-harm (dIR is an observation)"
    else:
        verdict = bool(g0["g0_pass"] and e2_pass and mechanism.get("pass", False) and no_harm)
        basis = (f"execution: mechanism + do-no-harm (dIR > -{E1_DELTA_IR}, not all splits negative); "
                 "formal E1 reported for the user's frame decision")

    out = {
        "arm": args.arm, "frame": spec["frame"], "adoption_basis": basis,
        "preregistration": spec.get("preregistration", "decision log §S18.1 (2026-09-07)"),
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
            "degenerate": {"base": f"{base_mq['degenerate_retrains']}/{base_mq['total_retrains']}",
                           "arm": f"{arm_mq['degenerate_retrains']}/{arm_mq['total_retrains']}"},
            "fallback_arm": arm_doc["optimizer_solver_fallback_rate"],
            "solver_counts_arm": arm_doc.get("optimizer_solver_counts"),
            "tg_px_ratio_suspect_arm": (arm_doc.get("data_quality", {}).get("currency", {}) or {}).get("tg_px_ratio_suspect"),
            "elapsed_sec_arm": arm_doc.get("elapsed_sec"),
        },
    }
    (arm_dir / "e1_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n{args.arm}: G0 {'PASS' if g0['g0_pass'] else 'FAIL'} | dIR {d_ir:+.4f} "
          f"(formal E1 {'PASS' if e1_formal else 'FAIL'}, no-harm {'PASS' if no_harm else 'FAIL'}) | "
          f"E2 {'PASS' if e2_pass else 'FAIL'} | mechanism {mechanism.get('pass')} | flip candidate: {verdict}")


if __name__ == "__main__":
    main()
