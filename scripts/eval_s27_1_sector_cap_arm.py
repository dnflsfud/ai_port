# -*- coding: utf-8 -*-
"""§S27.1 arm 판정 (read-only): 섹터 액티브 리스크 몫 상한 0.85 옵티마이저 강제 vs 새 S0′ 'OM-PIT'.

결정 로그 §S27.1 사전등록(2026-11-03)의 재실행 가능한 판정 스크립트. 리스크 규율 프레임(§S16.7 선례):
  기준: outputs/s27_om_pit_restamp (IR 1.708662580837287, 워크북 10-21 14:58 / Index 10-26 11:14 빈티지).
  G0: data_vintage 동일 ∧ avg_ic 비트 동일(옵티마이저 전용 변경 증빙) ∧ base IR 재현 ∧ arm 플래그 True.
  G1: 실행 북 리밸일별 최대 섹터 Euler 몫(게이트와 같은 방법: raw-return cov + optvol 대각, 미매핑은 "Unknown")
      ≤ 0.85 준수율 ≥ 90% ∧ 마지막 리밸일(라이브 북) ≤ 0.85 ∧ 종목 게이트 무악화(라이브 북 최대 종목 몫 ≤ 0.36
      ∧ 종목 준수율 ≥ 기준 북). 기준 북의 준수율 병기.
  G2: TE ≤ 4.5% · |ΔAS| ≤ 3%p · 회전율 ≤ 1.25× · fallback 0 · 퇴화율 동일.
  flip 후보 = G0 ∧ G1 ∧ G2 ∧ ΔIR > −0.36. formal E1·쌍대 NW t·보고서 ③ 제안 무해성은 병기만.
출력: outputs/s27_1_sector_risk_cap/e1_summary.json
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AI_PORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_PORT))

from scripts.eval_s13_46_arm import E1_DELTA_IR, N_SPLITS, _ir, _kst, evaluate_e1  # noqa: E402
from scripts.eval_s17_arm import evaluate_e2  # noqa: E402
from scripts.eval_s27_om_pit_arm import paired_nw_t  # noqa: E402

ARM = "s27_1_sector_risk_cap"
BASE = "s27_om_pit_restamp"
FLAG = "sector_risk_share_cap_enabled"
REFERENCE_BASE_IR = 1.708662580837287
SECTOR_CAP = 0.85              # = config.max_sector_active_risk_share, gate tolerance 0
NAME_GATE = 0.36               # = max_name_active_risk_share 0.35 + NAME_RISK_CAP_TOL 0.01 (gate)
G1_COMPLIANCE_MIN = 0.90
TURNOVER_MAX = 1.25


def book_risk_shares(w, bm, cov, tickers, sector_map):
    """(top |sector Euler share|, that sector, max name Euler share) — the gate's definitions."""
    a = np.asarray(w, float) - np.asarray(bm, float)
    var = float(a @ cov @ a)
    if not np.isfinite(var) or var <= 1e-18:
        return 0.0, None, 0.0
    contrib = a * (cov @ a) / var
    sectors = pd.Series(contrib, index=[str(sector_map.get(t, "Unknown")) for t in tickers]).groupby(level=0).sum()
    top = sectors.abs().idxmax()
    return float(abs(sectors[top])), str(top), float(np.max(contrib))


def evaluate_g1(arm_rows, base_rows) -> dict:
    """rows: [(date, top_sector_share, max_name_share), ...] in date order."""
    def rate(rows, k, bar):
        return float(np.mean([r[k] <= bar + 1e-9 for r in rows]))
    s_rate, b_rate = rate(arm_rows, 1, SECTOR_CAP), rate(base_rows, 1, SECTOR_CAP)
    n_rate, bn_rate = rate(arm_rows, 2, NAME_GATE), rate(base_rows, 2, NAME_GATE)
    sector_live = bool(arm_rows[-1][1] <= SECTOR_CAP + 1e-9)
    name_live = bool(arm_rows[-1][2] <= NAME_GATE + 1e-9)
    return {
        "n_rebalances": len(arm_rows),
        "sector_compliance_rate": round(s_rate, 4), "base_sector_compliance_rate": round(b_rate, 4),
        "sector_live_ok": sector_live, "live_top_sector_share": round(arm_rows[-1][1], 4),
        "base_live_top_sector_share": round(base_rows[-1][1], 4),
        "name_compliance_rate": round(n_rate, 4), "base_name_compliance_rate": round(bn_rate, 4),
        "name_live_ok": name_live,
        "g1_pass": bool(s_rate >= G1_COMPLIANCE_MIN and sector_live and name_live and n_rate >= bn_rate - 1e-12),
    }


def compute_rows(books: dict, data, cfg, tickers) -> dict:
    """books: {label: {date: weight Series}} → {label: [(date, top_sector, max_name, sector_name)]};
    cov 는 날짜당 1회 추정해 두 북에 공유(§S16.7 compute_shares_for_books 와 같은 방법)."""
    from scripts.export_operating_data import _apply_optvol_scale, _load_optvol_scale, _production_risk_source
    from src.backtest import get_benchmark_fn, get_sector_map
    from src.portfolio_optimizer import estimate_covariance

    risk_source = _production_risk_source(data, tickers)
    optvol_scale = _load_optvol_scale(data, tickers, cfg)
    bm_fn = get_benchmark_fn(data, tickers, config=cfg)
    sector_map = get_sector_map(data)
    cov_lookback = int(getattr(cfg, "cov_lookback", 126))
    out = {label: [] for label in books}
    for d in sorted({d for wbd in books.values() for d in wbd}):
        hist = risk_source.loc[risk_source.index < d].tail(cov_lookback)
        bm_w = np.asarray(bm_fn(d, tickers, len(tickers)), dtype=float)
        cov = np.asarray(estimate_covariance(hist, bm_weights=bm_w, config=cfg), dtype=float)
        cov, _ = _apply_optvol_scale(cov, optvol_scale, d, tickers)
        for label, wbd in books.items():
            if d in wbd:
                top, sector, max_name = book_risk_shares(wbd[d].reindex(tickers).fillna(0.0).values, bm_w, cov,
                                                         tickers, sector_map)
                out[label].append((str(pd.Timestamp(d).date()), round(top, 6), round(max_name, 6), sector))
    return out


def main() -> None:
    import yaml

    from src.data_loader import UniverseData
    from src.harness import build_override_config, inject_config

    base_dir, arm_dir = AI_PORT / "outputs" / BASE, AI_PORT / "outputs" / ARM
    base_doc = json.load(open(base_dir / "metrics.json", encoding="utf-8"))
    arm_doc = json.load(open(arm_dir / "metrics.json", encoding="utf-8"))
    bm_, am = base_doc["metrics"], arm_doc["metrics"]
    g0 = {"vintage_equal": base_doc.get("data_vintage") == arm_doc.get("data_vintage"),
          "avg_ic_bit_identical": bm_["avg_ic"] == am["avg_ic"],
          "base_reproduces_reference": abs(bm_["information_ratio"] - REFERENCE_BASE_IR) < 1e-9,
          "flag_on": bool(arm_doc["overrides"].get(FLAG) is True)}
    g0["g0_pass"] = bool(all(g0.values()))

    base_r = pickle.load(open(base_dir / "backtest_result.pkl", "rb"))
    arm_r = pickle.load(open(arm_dir / "backtest_result.pkl", "rb"))
    b_act, a_act = pd.Series(base_r.active_returns).dropna(), pd.Series(arm_r.active_returns).dropna()
    common = b_act.index.intersection(a_act.index)
    b_act, a_act = b_act.loc[common], a_act.loc[common]
    splits = []
    for k, ix in enumerate(np.array_split(np.arange(len(common)), N_SPLITS)):
        ir_b, ir_a = _ir(b_act.iloc[ix]), _ir(a_act.iloc[ix])
        splits.append({"split": k + 1, "start": str(common[ix[0]].date()), "end": str(common[ix[-1]].date()),
                       "ir_base": round(ir_b, 4), "ir_arm": round(ir_a, 4), "delta": round(ir_a - ir_b, 4)})
    d_ir = am["information_ratio"] - bm_["information_ratio"]
    t_nw = paired_nw_t(a_act - b_act)
    books = {"base": dict(base_r.portfolio_weights), "arm": dict(arm_r.portfolio_weights)}
    del base_r, arm_r

    manifest = yaml.safe_load((AI_PORT / "variants" / f"{ARM}.yaml").read_text(encoding="utf-8"))
    cfg = build_override_config(dict(manifest["overrides"]))
    inject_config(cfg)
    data = UniverseData(cfg.data_path, config=cfg)
    rows = compute_rows(books, data, cfg, list(data.tickers))
    g1 = evaluate_g1([r[:3] for r in rows["arm"]], [r[:3] for r in rows["base"]])

    turnover_ratio = am["avg_annual_turnover"] / bm_["avg_annual_turnover"]
    g2 = evaluate_e2(am["tracking_error"], bm_["active_share"], am["active_share"], turnover_ratio,
                     arm_doc["optimizer_solver_fallback_rate"], TURNOVER_MAX)
    bq, aq = base_doc["model_quality"], arm_doc["model_quality"]
    g2["degenerate_equal"] = bool(bq["degenerate_retrains"] == aq["degenerate_retrains"]
                                  and bq["total_retrains"] == aq["total_retrains"])
    g2_pass = all(g2.values())
    flip = bool(g0["g0_pass"] and g1["g1_pass"] and g2_pass and d_ir > -E1_DELTA_IR)

    out = {
        "arm": ARM, "flag": FLAG, "preregistration": "decision log §S27.1 (2026-11-03)",
        "frame": "risk discipline (G0 + G1 + G2 + dIR > -0.36; formal E1 / paired t reported only)",
        "vintage": {"base_pkl_mtime": _kst(base_dir / "backtest_result.pkl"), "arm_pkl_mtime": _kst(arm_dir / "backtest_result.pkl"),
                    "base_data_vintage": base_doc.get("data_vintage"), "arm_data_vintage": arm_doc.get("data_vintage")},
        "g0": g0, "g1": g1, "g2": g2, "g2_pass": g2_pass, "flip_candidate": flip,
        "full_period": {"ir_base": round(bm_["information_ratio"], 4), "ir_arm": round(am["information_ratio"], 4),
                        "delta_ir": round(d_ir, 4), "no_harm_bar": -E1_DELTA_IR, "paired_nw_t": round(t_nw, 3),
                        "no_harm_s27_proposed": bool(d_ir > -0.10 and t_nw > -1.0),
                        "e1_formal_pass": bool(evaluate_e1(d_ir, [s["delta"] for s in splits]))},
        "subperiods": splits,
        "rows_tail": {"base": rows["base"][-6:], "arm": rows["arm"][-6:]},
        "companions": {
            "te_base": round(bm_["tracking_error"], 5), "te_arm": round(am["tracking_error"], 5),
            "turnover_ratio": round(turnover_ratio, 4),
            "active_share_base": round(bm_["active_share"], 5), "active_share_arm": round(am["active_share"], 5),
            "realized_beta_base": bm_.get("realized_beta"), "realized_beta_arm": am.get("realized_beta"),
            "active_return_base": bm_.get("active_return"), "active_return_arm": am.get("active_return"),
            "max_drawdown_base": bm_.get("max_drawdown"), "max_drawdown_arm": am.get("max_drawdown"),
            "solver_counts_arm": arm_doc.get("optimizer_solver_counts"), "elapsed_sec_arm": arm_doc.get("elapsed_sec"),
        },
    }
    (arm_dir / "e1_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    pd.DataFrame([{"date": a[0], "arm_top_sector": a[1], "arm_max_name": a[2], "arm_sector": a[3],
                   "base_top_sector": b[1], "base_max_name": b[2], "base_sector": b[3]}
                  for a, b in zip(rows["arm"], rows["base"])]).to_csv(arm_dir / "risk_shares_by_rebalance.csv", index=False)
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    print(f"\n{ARM}: G0 {'PASS' if g0['g0_pass'] else 'FAIL'} | G1 {'PASS' if g1['g1_pass'] else 'FAIL'} "
          f"(sector {g1['sector_compliance_rate']} vs base {g1['base_sector_compliance_rate']}, live {g1['live_top_sector_share']}) | "
          f"G2 {'PASS' if g2_pass else 'FAIL'} | dIR {d_ir:+.4f} t {t_nw:+.2f} | flip candidate: {flip}")


if __name__ == "__main__":
    main()
