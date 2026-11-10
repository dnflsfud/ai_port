# -*- coding: utf-8 -*-
"""§S28.1 arm 판정 (read-only): purged K-fold 용량 선택 + 전체 창 재적합 vs 같은 빈티지 대조군.

결정 로그 §S28.1 사전등록(2026-11-10)의 재실행 가능한 판정 스크립트. 기전·무해성 프레임(IR 채택 바 아님):
  런 4개(같은 코드, 같은 Index.xlsx 구간) — 워크북 11-06: 대조 s28_s0_1106 / arm s28_1_kfold_refit_1106,
  보존 워크북 10-21: 대조 s28_ctl_1021 / arm s28_1_kfold_refit_1021.
  G0: 각 빈티지에서 arm·대조 data_vintage 동일 ∧ arm 플래그 True·대조 플래그 없음 ∧ 대조 s28_s0_1106 이
      코드 착지 전 production 런(같은 워크북)을 비트 동일 재현(REFERENCE_PROD_1106).
  G1(기전): 두 빈티지 모두 퇴화 재학습 ≤ 1 ∧ 빈티지 쌍 북 거리 비 = D(arm)/D(대조) ≤ 0.6.
      D = 네 런 공통 리밸일의 목표 북(portfolio_weights) 편도 L1 거리 평균(10-21 북 vs 11-06 북).
  G2(무해성): 두 빈티지 각각 TE ≤ 4.5% · |ΔAS| ≤ 3%p · 회전율 ≤ 1.25× · fallback 0 ∧
      두 빈티지 평균 ΔIR > −0.10 ∧ 각 빈티지 쌍대 NW t > −1.
  flip 후보 = G0 ∧ G1 ∧ G2. formal E1(ΔIR > +0.36)은 병기만(사전 기대 ΔIR ≈ 0).
출력: outputs/s28_1_kfold_refit_1106/e1_summary.json
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AI_PORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_PORT))

from scripts.eval_s13_46_arm import N_SPLITS, _ir, _kst, evaluate_e1  # noqa: E402
from scripts.eval_s17_arm import evaluate_e2  # noqa: E402
from scripts.eval_s27_om_pit_arm import paired_nw_t  # noqa: E402

FLAG = "kfold_capacity_refit_enabled"
RUNS = {
    "1106": {"ctl": "s28_s0_1106", "arm": "s28_1_kfold_refit_1106"},
    "1021": {"ctl": "s28_ctl_1021", "arm": "s28_1_kfold_refit_1021"},
}
# Pre-landing production run on the 11-06 workbook (commit 82bcb0f code): the flag-OFF control
# under the new code must reproduce it bit for bit (G0 parity).
REFERENCE_PROD_1106 = {  # 2026-11-10 11:30 scheduled run (commit 02a338e) == 2026-11-06 15:12 manual run, bit-identical
    "information_ratio": 1.6815952167604986,
    "avg_ic": 0.004595907706540644,
    "avg_annual_turnover": 0.7424257482581837,
    "tracking_error": 0.037429399443250376,
}
MAX_DEGENERATE = 1
DIST_RATIO_MAX = 0.6
NO_HARM_DELTA_IR = -0.10
NO_HARM_T = -1.0
TURNOVER_MAX = 1.25


def book_distance(books_a: dict, books_b: dict, dates) -> dict:
    """One-way L1 distance between two runs' rebalance books on ``dates`` (missing ticker = 0)."""
    by_date = []
    for d in dates:
        a, b = pd.Series(books_a[d], dtype=float), pd.Series(books_b[d], dtype=float)
        idx = a.index.union(b.index)
        by_date.append(0.5 * float((a.reindex(idx).fillna(0.0) - b.reindex(idx).fillna(0.0)).abs().sum()))
    return {"n": len(by_date), "by_date": by_date,
            "mean": float(np.mean(by_date)) if by_date else float("nan"),
            "last": by_date[-1] if by_date else float("nan"),
            "max": float(np.max(by_date)) if by_date else float("nan")}


def degenerate_flips(slots_a, degenerate_a, slots_b, degenerate_b) -> dict:
    """Retrain slots present in both vintages whose degenerate state differs."""
    common = sorted(set(slots_a) & set(slots_b))
    da, db = set(degenerate_a), set(degenerate_b)
    return {"common_slots": len(common), "flipped": [s for s in common if (s in da) != (s in db)]}


def evaluate_g1(degenerate_by_vintage: dict, dist_arm: float, dist_ctl: float) -> dict:
    ratio = dist_arm / dist_ctl if dist_ctl > 0 else float("nan")
    degenerate_ok = all(int(n) <= MAX_DEGENERATE for n in degenerate_by_vintage.values())
    return {"degenerate_by_vintage": dict(degenerate_by_vintage), "degenerate_ok": bool(degenerate_ok),
            "distance_arm": dist_arm, "distance_ctl": dist_ctl, "distance_ratio": ratio,
            "distance_ok": bool(np.isfinite(ratio) and ratio <= DIST_RATIO_MAX + 1e-12),
            "g1_pass": bool(degenerate_ok and np.isfinite(ratio) and ratio <= DIST_RATIO_MAX + 1e-12)}


def evaluate_no_harm(delta_irs, paired_ts) -> bool:
    return bool(np.mean(delta_irs) > NO_HARM_DELTA_IR
                and all(np.isfinite(t) and t > NO_HARM_T for t in paired_ts))


def _load(label: str):
    run_dir = AI_PORT / "outputs" / label
    doc = json.load(open(run_dir / "metrics.json", encoding="utf-8"))
    with open(run_dir / "backtest_result.pkl", "rb") as fh:
        result = pickle.load(fh)
    books = {pd.Timestamp(d): pd.Series(w, dtype=float) for d, w in dict(result.portfolio_weights).items()}
    active = pd.Series(result.active_returns).dropna()
    del result
    return doc, books, active, _kst(run_dir / "backtest_result.pkl")


def _model_summary(doc: dict) -> dict:
    mq = doc["model_quality"]
    audit = mq.get("split_audit") or []
    stars = [row["kfold"]["n_star"] for row in audit if "kfold" in row]
    # training end of the model that is actually LIVE (a reused model keeps its own fit slot)
    live_row = next((row for row in audit if row["prediction_date"] == mq.get("live_model_fit_date")), {})
    out = {"degenerate_retrains": int(mq["degenerate_retrains"]), "total_retrains": int(mq["total_retrains"]),
           "unique_models": mq.get("unique_models"), "live_model_fit_date": mq.get("live_model_fit_date"),
           "live_model_age_retrains": mq.get("live_model_age_retrains"),
           "live_train_end": live_row.get("train_end"),
           "slots": [row["prediction_date"] for row in audit],
           "degenerate_slots": [e["date"] for e in (mq.get("events") or [])]}
    if stars:
        out["n_star"] = {"min": int(min(stars)), "median": float(np.median(stars)), "max": int(max(stars)),
                         "capped": int(sum(bool(row["kfold"]["capped"]) for row in audit if "kfold" in row)),
                         "values": [int(s) for s in stars]}
    return out


def main() -> None:
    loaded = {v: {role: _load(label) for role, label in pair.items()} for v, pair in RUNS.items()}
    per_vintage, d_irs, ts, g0, g2_pass = {}, [], [], {}, True
    for v, pair in loaded.items():
        (c_doc, _, c_act, c_mt), (a_doc, _, a_act, a_mt) = pair["ctl"], pair["arm"]
        cm, am = c_doc["metrics"], a_doc["metrics"]
        common = c_act.index.intersection(a_act.index)
        c_act, a_act = c_act.loc[common], a_act.loc[common]
        splits = []
        for k, ix in enumerate(np.array_split(np.arange(len(common)), N_SPLITS)):
            ir_c, ir_a = _ir(c_act.iloc[ix]), _ir(a_act.iloc[ix])
            splits.append({"split": k + 1, "start": str(common[ix[0]].date()), "end": str(common[ix[-1]].date()),
                           "ir_ctl": round(ir_c, 4), "ir_arm": round(ir_a, 4), "delta": round(ir_a - ir_c, 4)})
        d_ir = am["information_ratio"] - cm["information_ratio"]
        t_nw = paired_nw_t(a_act - c_act)
        turnover_ratio = am["avg_annual_turnover"] / cm["avg_annual_turnover"]
        e2 = evaluate_e2(am["tracking_error"], cm["active_share"], am["active_share"], turnover_ratio,
                         a_doc["optimizer_solver_fallback_rate"], TURNOVER_MAX)
        g2_pass = g2_pass and all(e2.values())
        d_irs.append(d_ir)
        ts.append(t_nw)
        g0[v] = {"vintage_equal": c_doc.get("data_vintage") == a_doc.get("data_vintage"),
                 "arm_flag_on": bool(a_doc["overrides"].get(FLAG) is True),
                 "ctl_flag_absent": bool(FLAG not in c_doc["overrides"])}
        per_vintage[v] = {
            "ctl": RUNS[v]["ctl"], "arm": RUNS[v]["arm"], "ctl_pkl_mtime": c_mt, "arm_pkl_mtime": a_mt,
            "data_vintage": a_doc.get("data_vintage"),
            "ir_ctl": cm["information_ratio"], "ir_arm": am["information_ratio"], "delta_ir": round(d_ir, 4),
            "paired_nw_t": round(t_nw, 3), "subperiods": splits,
            "e1_formal_pass": bool(evaluate_e1(d_ir, [s["delta"] for s in splits])),
            "e2": e2, "turnover_ratio": round(turnover_ratio, 4),
            "te_ctl": cm["tracking_error"], "te_arm": am["tracking_error"],
            "active_share_ctl": cm["active_share"], "active_share_arm": am["active_share"],
            "avg_ic_ctl": cm["avg_ic"], "avg_ic_arm": am["avg_ic"],
            "realized_beta_ctl": cm.get("realized_beta"), "realized_beta_arm": am.get("realized_beta"),
            "elapsed_sec_ctl": c_doc.get("elapsed_sec"), "elapsed_sec_arm": a_doc.get("elapsed_sec"),
            "model_ctl": _model_summary(c_doc), "model_arm": _model_summary(a_doc),
        }

    ref = loaded["1106"]["ctl"][0]["metrics"]
    g0["parity_1106"] = {k: bool(v is not None and ref[k] == v) for k, v in REFERENCE_PROD_1106.items()}
    g0["g0_pass"] = bool(all(all(d.values()) for d in g0.values()))

    books = {(v, role): loaded[v][role][1] for v in RUNS for role in ("ctl", "arm")}
    dates = sorted(set.intersection(*(set(b) for b in books.values())))
    dist_ctl = book_distance(books[("1106", "ctl")], books[("1021", "ctl")], dates)
    dist_arm = book_distance(books[("1106", "arm")], books[("1021", "arm")], dates)
    g1 = evaluate_g1({v: per_vintage[v]["model_arm"]["degenerate_retrains"] for v in RUNS},
                     dist_arm["mean"], dist_ctl["mean"])
    flips = {role: degenerate_flips(per_vintage["1106"][f"model_{role}"]["slots"],
                                    per_vintage["1106"][f"model_{role}"]["degenerate_slots"],
                                    per_vintage["1021"][f"model_{role}"]["slots"],
                                    per_vintage["1021"][f"model_{role}"]["degenerate_slots"])
             for role in ("ctl", "arm")}
    no_harm = evaluate_no_harm(d_irs, ts)
    g2 = {"e2_all_pass": bool(g2_pass), "mean_delta_ir": round(float(np.mean(d_irs)), 4),
          "paired_nw_t": [round(t, 3) for t in ts], "no_harm_pass": no_harm,
          "g2_pass": bool(g2_pass and no_harm)}
    flip = bool(g0["g0_pass"] and g1["g1_pass"] and g2["g2_pass"])

    out = {
        "arm": "s28_1_kfold_refit", "flag": FLAG, "preregistration": "decision log §S28.1 (2026-11-10)",
        "frame": "mechanism + harmlessness (G0 + G1 + G2); formal E1 reported only",
        "g0": g0, "g1": g1, "g2": g2, "flip_candidate": flip,
        "vintage_pair_distance": {
            "n_common_rebalances": len(dates),
            "ctl": {k: dist_ctl[k] for k in ("mean", "last", "max")},
            "arm": {k: dist_arm[k] for k in ("mean", "last", "max")},
            "ctl_tail": [round(x, 5) for x in dist_ctl["by_date"][-6:]],
            "arm_tail": [round(x, 5) for x in dist_arm["by_date"][-6:]],
        },
        "degenerate_flips_between_vintages": flips,
        "per_vintage": per_vintage,
    }
    out_dir = AI_PORT / "outputs" / RUNS["1106"]["arm"]
    (out_dir / "e1_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str),
                                             encoding="utf-8")
    pd.DataFrame({"date": [str(d.date()) for d in dates], "ctl_one_way": dist_ctl["by_date"],
                  "arm_one_way": dist_arm["by_date"]}).to_csv(out_dir / "vintage_pair_distance.csv", index=False)
    print(json.dumps({k: v for k, v in out.items() if k != "per_vintage"}, indent=2, ensure_ascii=False, default=str))
    for v in RUNS:
        p = per_vintage[v]
        print(f"[{v}] IR {p['ir_ctl']:.4f} -> {p['ir_arm']:.4f} (dIR {p['delta_ir']:+.4f}, t {p['paired_nw_t']:+.2f}) | "
              f"degenerate {p['model_ctl']['degenerate_retrains']}/{p['model_ctl']['total_retrains']} -> "
              f"{p['model_arm']['degenerate_retrains']}/{p['model_arm']['total_retrains']} | "
              f"live train end {p['model_ctl']['live_train_end']} -> {p['model_arm']['live_train_end']}")
    print(f"\ns28_1_kfold_refit: G0 {'PASS' if g0['g0_pass'] else 'FAIL'} | G1 {'PASS' if g1['g1_pass'] else 'FAIL'} "
          f"(distance ratio {g1['distance_ratio']:.3f}, bar {DIST_RATIO_MAX}) | G2 {'PASS' if g2['g2_pass'] else 'FAIL'} "
          f"| flip candidate: {flip}")


if __name__ == "__main__":
    main()
