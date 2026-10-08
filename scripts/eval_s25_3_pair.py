# -*- coding: utf-8 -*-
"""§S25.3 생성기 쌍 판정 (read-only) — s25_3_s0recert(신 생성기 워크북) vs s25_3_oldgen_1002(구 생성기 워크북), 같은 2026-10-02 인출분.

결정 로그 §S25.3 사전등록(2026-10-08)의 재실행 가능한 판정 스크립트. 정확성 트랙(인벤토리 비계수).
  기전(워크북): 쌍 diff 프로브(outputs/s25_audit/11_pair_workbook_diff.py) 결과 — 시트 집합 동일 ∧ 원시 시트는 "구 값 → 신 NaN" 외 변화 0
        ∧ 그 외 변화는 파생 시트(Summary_Stats, 252D 롤링 *_z 4종)에만 있고, *_z 의 마스킹 창(접두 끝 +252행) 밖 차이는 상수 구간의
        NaN<->0.0 플립(pandas rolling std 부동소수 상태)뿐(beyond_other == 0)
        ∧ 사전 명시 사례(UBER BEST_PE_RATIO · FICO BEST_ROE · AON EQY_REC_CONS · CS OPER_MARGIN) 마스킹 존재.
  기전(모델): 두 런의 피처 집합 동일.
  E2 : TE ≤ 4.5% · active share ±3%p · 회전율 ≤ 1.25× · solver fallback 0 (신 vs 구).
  무해성(관측): ΔIR > −0.36 ∧ 3분할 전부 음 아님 — 채택 근거 아님. 결과와 무관하게 신 생성기 워크북이 정본, 신 런 = 새 S0′.
출력: outputs/s25_3_s0recert/e1_summary.json
"""

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AI_PORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_PORT))

from scripts.eval_s13_46_arm import E1_DELTA_IR, N_SPLITS, _ir, _kst  # noqa: E402
from scripts.eval_s17_arm import evaluate_e2, evaluate_no_harm  # noqa: E402

OLD = "s25_3_oldgen_1002"
NEW = "s25_3_s0recert"
TURNOVER_MAX = 1.25
# (sheet, ticker) cases named in the §S25.3 pre-registration; names as the loader standardises them.
EXPECTED_MASKED = (("BEST_PE_RATIO", "UBER"), ("BEST_ROE", "FICO"), ("EQY_REC_CONS", "AON"), ("OPER_MARGIN", "CS"))
# Sheets the generator DERIVES from masked raw sheets; value changes there are propagation, not a defect.
ALLOWED_DERIVED_CHANGE_SHEETS = frozenset({"Summary_Stats", "iv30_z", "iv_term_structure_z", "downside_skew_z", "vol_risk_premium_z"})


def parse_probe_summary(text: str) -> dict:
    m = re.search(r"sheet sets equal: (True|False)", text)
    o = re.search(r"OTHER changes \(unmasked / value-changed cells\) by sheet: (.*)", text)
    other = o.group(1).strip() if o else ""
    sheets = [] if other == "NONE" else re.findall(r"'([^']+)': \(", other)
    z = re.search(r"Z-SHEET CLASSIFICATION: (\{.*\})", text)
    zclass = json.loads(z.group(1)) if z else {}
    return {"sheet_sets_equal": bool(m and m.group(1) == "True"),
            "other_changes_none": other == "NONE",
            "other_changes_sheets": sheets,
            "other_changes_only_derived": bool(o) and set(sheets) <= set(ALLOWED_DERIVED_CHANGE_SHEETS),
            "z_beyond_other_zero": bool(zclass) and all(int(v.get("beyond_other", 1)) == 0 for v in zclass.values()),
            "z_classification": zclass,
            "shape_or_index_issues": ("[SHAPE/COLUMNS DIFFER]" in text) or ("[INDEX DIFFERS]" in text)}


def mechanism_workbook(diff: pd.DataFrame, probe: dict) -> dict:
    present = {(r.sheet, r.ticker) for r in diff.itertuples()}
    missing = [list(c) for c in EXPECTED_MASKED if c not in present]
    by_sheet = diff.groupby("sheet")["masked_cells"].sum().sort_values(ascending=False) if len(diff) else pd.Series(dtype=int)
    raw_clean = bool(probe["other_changes_none"] or (probe["other_changes_only_derived"] and probe["z_beyond_other_zero"]))
    clean = bool(probe["sheet_sets_equal"] and raw_clean and not probe["shape_or_index_issues"])
    return {"probe": probe, "total_masked_cells": int(diff["masked_cells"].sum()) if len(diff) else 0,
            "masked_sheet_ticker_pairs": int(len(diff)),
            "masked_by_sheet": {k: int(v) for k, v in by_sheet.items()},
            "all_masked_columns": [[r.sheet, r.ticker] for r in diff.itertuples() if bool(r.all_masked)] if len(diff) else [],
            "expected_cases": {f"{s}/{t}": (diff[(diff.sheet == s) & (diff.ticker == t)].iloc[0].to_dict() if (s, t) in present else None)
                               for s, t in EXPECTED_MASKED},
            "missing_expected": missing,
            "pass": bool(clean and not missing)}


def mechanism_model(old_features, new_features) -> dict:
    o, n = set(old_features), set(new_features)
    return {"only_old": sorted(o - n), "only_new": sorted(n - o), "n_old": len(o), "n_new": len(n), "pass": bool(o == n)}


def verdicts(mechanism: bool, e2: bool, d_ir: float, split_deltas) -> dict:
    return {"no_harm_pass": bool(evaluate_no_harm(d_ir, split_deltas)),
            "certified": bool(mechanism and e2),
            "frame": "correctness: mechanism and E2 certify the new-generator workbook as S0'; dIR is an observation"}


def _load(dir_: Path):
    doc = json.load(open(dir_ / "metrics.json", encoding="utf-8"))
    return doc, pickle.load(open(dir_ / "backtest_result.pkl", "rb"))


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=OLD)
    ap.add_argument("--new", default=NEW)
    ap.add_argument("--diff-csv", required=True)
    ap.add_argument("--probe-out", required=True)
    args = ap.parse_args()
    old_dir, new_dir = AI_PORT / "outputs" / args.old, AI_PORT / "outputs" / args.new
    old_doc, old_r = _load(old_dir)
    new_doc, new_r = _load(new_dir)
    om, nm = old_doc["metrics"], new_doc["metrics"]

    diff = pd.read_csv(args.diff_csv)
    probe = parse_probe_summary(Path(args.probe_out).read_text(encoding="utf-8", errors="replace"))
    wb = mechanism_workbook(diff, probe)
    model = mechanism_model(old_r.feature_names, new_r.feature_names)
    mechanism_pass = bool(wb["pass"] and model["pass"])

    o_act = pd.Series(old_r.active_returns).dropna()
    n_act = pd.Series(new_r.active_returns).dropna()
    common = o_act.index.intersection(n_act.index)
    o_act, n_act = o_act.loc[common], n_act.loc[common]
    splits = []
    for k, ix in enumerate(np.array_split(np.arange(len(common)), N_SPLITS)):
        ir_o, ir_n = _ir(o_act.iloc[ix]), _ir(n_act.iloc[ix])
        splits.append({"split": k + 1, "start": str(common[ix[0]].date()), "end": str(common[ix[-1]].date()),
                       "ir_old": round(ir_o, 4), "ir_new": round(ir_n, 4), "delta": round(ir_n - ir_o, 4)})
    d_ir = nm["information_ratio"] - om["information_ratio"]
    split_deltas = [s["delta"] for s in splits]

    turnover_ratio = nm["avg_annual_turnover"] / om["avg_annual_turnover"]
    e2 = evaluate_e2(nm["tracking_error"], om["active_share"], nm["active_share"], turnover_ratio,
                     new_doc["optimizer_solver_fallback_rate"], TURNOVER_MAX)
    e2_pass = all(e2.values())
    v = verdicts(mechanism_pass, e2_pass, d_ir, split_deltas)

    out = {
        "pair": {"old": args.old, "new": args.new}, "frame": "correctness",
        "preregistration": "decision log §S25.3 (2026-10-08)",
        "vintage": {"old_data_vintage": old_doc.get("data_vintage"), "new_data_vintage": new_doc.get("data_vintage"),
                    "old_pkl_mtime": _kst(old_dir / "backtest_result.pkl"), "new_pkl_mtime": _kst(new_dir / "backtest_result.pkl")},
        "mechanism": {"workbook": wb, "model": model, "pass": mechanism_pass},
        "e2": e2, "e2_pass": e2_pass,
        "full_period": {"ir_old": round(om["information_ratio"], 4), "ir_new": round(nm["information_ratio"], 4),
                        "delta_ir": round(d_ir, 4), "no_harm_bar": -E1_DELTA_IR},
        "subperiods": splits, **v,
        "new_s0_prime": {k: nm.get(k) for k in ("information_ratio", "tracking_error", "avg_annual_turnover", "active_share",
                                                "realized_beta", "avg_ic", "max_drawdown", "sub_periods")},
        "companions": {
            "te_old": round(om["tracking_error"], 5), "te_new": round(nm["tracking_error"], 5),
            "turnover_old": round(om["avg_annual_turnover"], 4), "turnover_new": round(nm["avg_annual_turnover"], 4),
            "turnover_ratio": round(turnover_ratio, 4),
            "active_share_old": round(om["active_share"], 5), "active_share_new": round(nm["active_share"], 5),
            "realized_beta_old": om.get("realized_beta"), "realized_beta_new": nm.get("realized_beta"),
            "avg_ic_old": om["avg_ic"], "avg_ic_new": nm["avg_ic"],
            "max_drawdown_old": om.get("max_drawdown"), "max_drawdown_new": nm.get("max_drawdown"),
            "degenerate": {"old": f"{old_doc['model_quality']['degenerate_retrains']}/{old_doc['model_quality']['total_retrains']}",
                           "new": f"{new_doc['model_quality']['degenerate_retrains']}/{new_doc['model_quality']['total_retrains']}"},
            "fallback_new": new_doc["optimizer_solver_fallback_rate"], "solver_counts_new": new_doc.get("optimizer_solver_counts"),
            "elapsed_sec": {"old": old_doc.get("elapsed_sec"), "new": new_doc.get("elapsed_sec")},
        },
    }
    out = _json_safe(out)
    (new_dir / "e1_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n{args.new} vs {args.old}: mechanism {mechanism_pass} (workbook {wb['pass']}: masked {wb['total_masked_cells']} cells / "
          f"{wb['masked_sheet_ticker_pairs']} pairs, missing {wb['missing_expected']}; model {model['pass']}) | E2 {'PASS' if e2_pass else 'FAIL'} | "
          f"dIR {d_ir:+.4f} splits {split_deltas} no-harm {'PASS' if v['no_harm_pass'] else 'FAIL'} | certified: {v['certified']} | "
          f"new S0' IR {nm['information_ratio']:.4f}")


if __name__ == "__main__":
    main()
