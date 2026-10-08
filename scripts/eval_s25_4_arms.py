# -*- coding: utf-8 -*-
"""§S25.4 정확성 arm 판정 (read-only) — 4 arm vs 기준 outputs/s25_3_s0recert (워크북 2026-10-08 11:27:51 빈티지).

결정 로그 §S25.4 사전등록(2026-10-08)의 재실행 가능한 판정 스크립트. 정확성 트랙(인벤토리 비계수), §S23.1 프레임:
  flip 후보 = G0 ∧ 기전 ∧ E2 ∧ 무해성. formal E1 은 병기만.
  G0 : base·arm data_vintage 동일 ∧ base IR 이 REFERENCE_BASE_IR 을 1e-9 이내 재현.
  기전(arm 별):
    b02 — 피처 패널 변화가 리비전 파생 피처("rev" 포함)에만 있고 비어 있지 않음.
    b03 — 날짜별 횡단면 p90−p10 폭의 arm/base 비율: 중앙값 > 1.0 ∧ 비율 ≥ 1 인 (날짜, 피처) 몫 ≥ 0.9(윈저가 압축을 푼다).
    b05 — 분할 audit 전 재학습 forward_horizon = 20 + lag(21) ∧ embargo ≥ 21 ∧ causal_ok, 유한 타깃 셀 ≥ 99% 변경, 피처 패널 비트 동일.
    stale — data_quality.stale_run_mask 에 4 시트 전부 masked_cells > 0 ∧ 사전 명시 사례(BEST_PE_RATIO/RBLX, BEST_PX_BPS_RATIO/VRSN) 포함
            ∧ 피처 패널 변화가 비율 시트 파생 피처(pe/pb/peg/ebitda)에만 있음.
  E2 : TE ≤ 4.5% · active share ±3%p · 회전율 ≤ 1.25× · solver fallback 0.
  무해성: ΔIR > −0.36 ∧ 3분할 전부 음 아님.
출력: outputs/<arm>/e1_summary.json
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

BASE = "s25_3_s0recert"
# outputs/s25_3_s0recert/metrics.json — S0' on workbook 2026-10-08 11:27:51 (decision log §S25.3 result; bit-identical to the
# 2026-10-08 13:13 production run).
REFERENCE_BASE_IR = 1.7218790679024922
TURNOVER_MAX = 1.25
SPREAD_MEDIAN_MIN = 1.0
SPREAD_SHARE_MIN = 0.9
LABEL_CHANGED_MIN = 0.99
STALE_SHEETS = ("BEST_PE_RATIO", "BEST_PX_BPS_RATIO", "BEST_PEG_RATIO", "BEST_EV_TO_BEST_EBITDA")
STALE_NAMED_CASES = (("BEST_PE_RATIO", "RBLX"), ("BEST_PX_BPS_RATIO", "VRSN"))
_RATIO_TOKENS = ("pe_", "_pe", "pb_", "_pb", "peg", "ebitda")

ARMS = {
    "s25_4_b02_no_gradual_mask": {"flag": "revision_gradual_mask_disabled", "mechanism": "b02"},
    "s25_4_b03_winsor_first": {"flag": "zscore_winsor_first_enabled", "mechanism": "b03"},
    "s25_4_b05_label_lag": {"flag": "label_execution_lag_enabled", "mechanism": "b05"},
    "s25_4_stale_run_mask": {"flag": "stale_run_mask_enabled", "mechanism": "stale"},
}


def changed_features(base_panel: pd.DataFrame, arm_panel: pd.DataFrame) -> dict:
    """feature → number of (date, ticker) cells that differ (NaN status or value, atol 1e-9) on the common index."""
    cols = [c for c in base_panel.columns if c in arm_panel.columns]
    idx = base_panel.index.intersection(arm_panel.index)
    a, b = base_panel.loc[idx, cols], arm_panel.loc[idx, cols]
    out = {}
    for c in cols:
        av, bv = a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float)
        diff = (np.isnan(av) != np.isnan(bv)) | (~np.isnan(av) & ~np.isnan(bv) & ~np.isclose(av, bv, rtol=0, atol=1e-9))
        n = int(diff.sum())
        if n:
            out[c] = n
    return out


def mechanism_feature_subset(changed: dict, allowed) -> dict:
    outside = sorted(f for f in changed if not allowed(f))
    return {"changed": sorted(changed), "n_changed_cells": int(sum(changed.values())), "outside": outside,
            "pass": bool(changed) and not outside}


def mechanism_spread(base_panel: pd.DataFrame, arm_panel: pd.DataFrame) -> dict:
    """날짜별 (p90 − p10) 폭의 arm/base 비율, (날짜, 피처) 전체."""
    cols = [c for c in base_panel.columns if c in arm_panel.columns]
    idx = base_panel.index.intersection(arm_panel.index)
    a, b = base_panel.loc[idx, cols], arm_panel.loc[idx, cols]

    def spread(p):
        g = p.groupby(level="date")
        return g.quantile(0.9) - g.quantile(0.1)

    sa, sb = spread(a), spread(b)
    ratio = (sb / sa.where(sa > 0)).stack().dropna()
    return {"n_pairs": int(len(ratio)), "median_ratio": round(float(ratio.median()), 4) if len(ratio) else None,
            "share_ge_1": round(float((ratio >= 1 - 1e-12).mean()), 4) if len(ratio) else None,
            "p10_ratio": round(float(ratio.quantile(0.1)), 4) if len(ratio) else None,
            "max_ratio": round(float(ratio.max()), 4) if len(ratio) else None,
            "pass": bool(len(ratio) and ratio.median() > SPREAD_MEDIAN_MIN and (ratio >= 1 - 1e-12).mean() >= SPREAD_SHARE_MIN)}


def mechanism_label_shift(base_t: pd.DataFrame, arm_t: pd.DataFrame, split_audit, lag: int, base_horizon: int) -> dict:
    want = int(base_horizon) + int(lag)
    rows = list(split_audit or [])
    audit_ok = bool(rows) and all(int(r.get("forward_horizon", -1)) == want and int(r.get("embargo_days", -1)) >= want
                                  and bool(r.get("causal_validation_ok", False)) for r in rows)
    idx = base_t.index.intersection(arm_t.index); cols = [c for c in base_t.columns if c in arm_t.columns]
    a, b = base_t.loc[idx, cols].to_numpy(dtype=float), arm_t.loc[idx, cols].to_numpy(dtype=float)
    both = ~np.isnan(a) & ~np.isnan(b)
    share = float((~np.isclose(a[both], b[both], rtol=0, atol=1e-12)).mean()) if both.any() else 0.0
    return {"audit_horizon_expected": want, "audit_pass": audit_ok, "n_retrains": len(rows),
            "share_changed": round(share, 4), "pass": bool(audit_ok and share >= LABEL_CHANGED_MIN)}


def mechanism_stale_mask(data_quality) -> dict:
    rec = (data_quality or {}).get("stale_run_mask") or {}
    missing = [s for s in STALE_SHEETS if int((rec.get(s) or {}).get("masked_cells", 0)) <= 0]
    named_missing = [f"{s}/{t}" for s, t in STALE_NAMED_CASES if t not in ((rec.get(s) or {}).get("top") or {})]
    return {"per_sheet": {s: {k: v for k, v in (rec.get(s) or {}).items()} for s in STALE_SHEETS},
            "total_masked_cells": int(sum(int((rec.get(s) or {}).get("masked_cells", 0)) for s in STALE_SHEETS)),
            "missing_sheets": missing, "named_cases_missing": named_missing,
            "pass": bool(rec) and not missing and not named_missing}


def verdicts(g0: bool, mechanism: bool, e2: bool, d_ir: float, split_deltas) -> dict:
    no_harm = evaluate_no_harm(d_ir, split_deltas)
    return {"no_harm_pass": bool(no_harm), "e1_formal_pass": bool(evaluate_e1(d_ir, split_deltas)),
            "flip_candidate": bool(g0 and mechanism and e2 and no_harm)}


def _load(dir_: Path):
    doc = json.load(open(dir_ / "metrics.json", encoding="utf-8"))
    return doc, pickle.load(open(dir_ / "backtest_result.pkl", "rb"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--base", default=BASE)
    args = ap.parse_args()
    base_dir, arm_dir = AI_PORT / "outputs" / args.base, AI_PORT / "outputs" / args.arm
    base_doc, base_r = _load(base_dir)
    arm_doc, arm_r = _load(arm_dir)
    bm, am = base_doc["metrics"], arm_doc["metrics"]
    spec = ARMS[args.arm]

    vintage_ok = base_doc.get("data_vintage") == arm_doc.get("data_vintage")
    base_reproduces = abs(bm["information_ratio"] - REFERENCE_BASE_IR) < 1e-9
    flag_on = bool(arm_doc["overrides"].get(spec["flag"]) is True)
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
    kind = spec["mechanism"]
    if kind == "b02":
        mech = mechanism_feature_subset(changed, lambda f: "rev" in f)
    elif kind == "b03":
        mech = mechanism_spread(base_r.panel, arm_r.panel)
        mech["changed_features"] = len(changed)
    elif kind == "b05":
        lag = int(arm_doc["overrides"].get("execution_signal_lag_days", 1))
        mech = mechanism_label_shift(base_r.targets, arm_r.targets, arm_doc["model_quality"].get("split_audit"), lag,
                                     int(arm_doc["overrides"].get("forward_horizon") or 20))
        mech["panel_identical"] = not changed
        mech["pass"] = bool(mech["pass"] and not changed)
    else:
        mech = mechanism_stale_mask(arm_doc.get("data_quality") or getattr(arm_r, "data_quality", None))
        sub = mechanism_feature_subset(changed, lambda f: any(t in f for t in _RATIO_TOKENS))
        mech["feature_subset"] = sub
        mech["pass"] = bool(mech["pass"] and sub["pass"])
    mech["feature_set_equal"] = bool(feature_set_equal)
    mechanism_pass = bool(mech["pass"] and feature_set_equal)

    turnover_ratio = am["avg_annual_turnover"] / bm["avg_annual_turnover"]
    e2 = evaluate_e2(am["tracking_error"], bm["active_share"], am["active_share"], turnover_ratio,
                     arm_doc["optimizer_solver_fallback_rate"], TURNOVER_MAX)
    e2_pass = all(e2.values())
    v = verdicts(g0["g0_pass"], mechanism_pass, e2_pass, d_ir, split_deltas)

    out = {
        "arm": args.arm, "flag": spec["flag"], "frame": "accuracy (G0 + mechanism + E2 + no-harm; formal E1 reported only)",
        "preregistration": "decision log §S25.4 (2026-10-08)", "base_dir": str(base_dir),
        "vintage": {"base_pkl_mtime": _kst(base_dir / "backtest_result.pkl"), "arm_pkl_mtime": _kst(arm_dir / "backtest_result.pkl"),
                    "base_data_vintage": base_doc.get("data_vintage"), "arm_data_vintage": arm_doc.get("data_vintage")},
        "g0": g0,
        "full_period": {"ir_base": round(bm["information_ratio"], 4), "ir_arm": round(am["information_ratio"], 4),
                        "delta_ir": round(d_ir, 4), "no_harm_bar": -E1_DELTA_IR},
        "subperiods": splits, **v,
        "mechanism": mech, "mechanism_pass": mechanism_pass, "changed_features_top": dict(sorted(changed.items(), key=lambda kv: -kv[1])[:12]),
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
