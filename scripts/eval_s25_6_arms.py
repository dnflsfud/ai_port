# -*- coding: utf-8 -*-
"""§S25.6 성과 arm 판정 (read-only) — 2 arm vs 기준 outputs/s25_4_combined_recert (워크북 2026-10-08 11:27:51 빈티지).

결정 로그 §S25.6 사전등록(2026-10-09)의 재실행 가능한 판정 스크립트. 성과 트랙(인벤토리 +1 each).
  채택 후보 = G0 ∧ 기전 ∧ E2 ∧ formal E1(ΔIR > +0.36 ∧ 3분할 전부 양). 오버라이드 후보(§S13.50 정책: ΔIR > 0 ∧ 3분할 전부 양 ∧ 게이트) 병기만.
  G0 : data_vintage 동일 ∧ base IR 이 REFERENCE_BASE_IR 을 1e-9 이내 재현 ∧ arm overrides 에 플래그 True.
  기전:
    boost  — arm 예측이 base 예측과 다른 셀 = 정확히 base 의 날짜별 상위 10%(그 밖 변화 0, 상위 셀 전부 변화) ∧ 피처 패널·타깃 비트 동일(모델층 불변)
             ∧ 리밸일 포트폴리오 비중 중 base 상위 십분위 종목 몫의 평균이 base 보다 커진다.
    sector — 예측 비트 동일(옵티마이저 전용) ∧ 최근 24회 리밸 Tech Euler 액티브 리스크 몫(§S25.5 프로브와 같은 LW Σ 126일 레시피) 평균 감소 ∧ 0.85 초과 횟수 감소.
  E2 : TE ≤ 4.5% · active share ±3%p · 회전율 ≤ 1.25× · solver fallback 0.
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

BASE = "s25_4_combined_recert"
# outputs/s25_4_combined_recert/metrics.json — S0' on workbook 2026-10-08 11:27:51 (decision log §S25.4 결합 재인증;
# bit-identical to the 2026-10-09 10:58 production run).
REFERENCE_BASE_IR = 1.8229575020692097
TURNOVER_MAX = 1.25
SECTOR_CAP = 0.85
SECTOR_N_LAST = 24

ARMS = {
    "s25_6_top_decile_boost": {"flag": "top_decile_boost_enabled", "mechanism": "boost"},
    "s25_6_sector_risk_penalty": {"flag": "sector_active_risk_penalty_enabled", "mechanism": "sector"},
}


def mechanism_boost_cells(base_preds: pd.DataFrame, arm_preds: pd.DataFrame, quantile: float) -> dict:
    idx = base_preds.index.intersection(arm_preds.index); cols = [c for c in base_preds.columns if c in arm_preds.columns]
    a, b = base_preds.loc[idx, cols], arm_preds.loc[idx, cols]
    thr = a.quantile(quantile, axis=1)
    top = (a.ge(thr, axis=0) & a.notna()).to_numpy(dtype=bool)
    av, bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    changed = (np.isnan(av) != np.isnan(bv)) | (~np.isnan(av) & ~np.isnan(bv) & ~np.isclose(av, bv, rtol=0, atol=1e-12))
    outside = int((changed & ~top).sum()); top_unchanged = int((top & ~changed).sum())
    return {"changed_cells": int(changed.sum()), "top_cells": int(top.sum()), "changed_outside_top": outside,
            "top_unchanged": top_unchanged, "pass": bool(changed.sum() > 0 and outside == 0 and top_unchanged == 0)}


def top_decile_weight_share(portfolio_weights: dict, base_preds: pd.DataFrame, quantile: float) -> pd.Series:
    """리밸일별: 포트폴리오 비중 중 base 예측 상위 십분위 종목에 실린 몫."""
    out = {}
    for d, w in portfolio_weights.items():
        if d not in base_preds.index:
            continue
        row = base_preds.loc[d].dropna()
        if row.empty:
            continue
        top = row[row >= row.quantile(quantile)].index
        ws = pd.Series(w, dtype=float)
        tot = float(ws.sum())
        out[d] = float(ws.reindex(top).fillna(0.0).sum()) / tot if tot > 0 else np.nan
    return pd.Series(out).sort_index()


def mechanism_sector_share(base_share: pd.Series, arm_share: pd.Series, cap: float = SECTOR_CAP) -> dict:
    both = pd.concat([base_share.rename("base"), arm_share.rename("arm")], axis=1).dropna()
    mb, ma = float(both["base"].mean()), float(both["arm"].mean())
    bb, ba = int((both["base"] > cap).sum()), int((both["arm"] > cap).sum())
    return {"n_dates": int(len(both)), "mean_base": round(mb, 4), "mean_arm": round(ma, 4), "mean_delta": round(ma - mb, 4),
            "breaches_base": bb, "breaches_arm": ba, "latest_base": round(float(both["base"].iloc[-1]), 4) if len(both) else None,
            "latest_arm": round(float(both["arm"].iloc[-1]), 4) if len(both) else None,
            "pass": bool(len(both) and ma < mb and ba < bb)}


def verdicts(g0: bool, mechanism: bool, e2: bool, d_ir: float, split_deltas) -> dict:
    e1 = evaluate_e1(d_ir, split_deltas); nh = evaluate_no_harm(d_ir, split_deltas); gates = bool(g0 and mechanism and e2)
    return {"e1_formal_pass": bool(e1), "no_harm_pass": bool(nh), "adoption_candidate": bool(gates and e1),
            "override_candidate": bool(gates and d_ir > 0 and all(d > 0 for d in split_deltas))}


# ---------------------------------------------------------------------------------------------------------------------
# Heavy helper (main only): Tech Euler active-risk share per rebalance, same recipe as outputs/s25_audit/12_precheck_s25_5.py
def _sector_share_series(result, raw, sector_name="Technology", n_last=SECTOR_N_LAST):
    from src import data_loader as dl
    from src.portfolio_optimizer import estimate_covariance
    from src.config import PipelineConfig
    meta = dl.load_universe_meta(raw); tickers = list(meta.index); sector = meta["sector"]
    currency = meta["currency"].fillna("USD").str.upper()

    def prep(name):
        df = dl._rename_bloomberg_equity_columns(dl._standardize_columns(raw[name].copy()))
        return dl._filter_tickers(dl._standardize_index(df), tickers=tickers).apply(pd.to_numeric, errors="coerce")
    px, mc = prep("PX_LAST"), prep("CUR_MKT_CAP")
    fpx = raw["Factor_PX_LAST"].apply(pd.to_numeric, errors="coerce"); fpx.index = pd.to_datetime(fpx.index)
    usd_per = pd.DataFrame(1.0, index=fpx.index, columns=["USD"])
    for ccy, col, inv in (("KRW", "USDKRW", True), ("JPY", "USDJPY", True), ("EUR", "EURUSD", False), ("CHF", "USDCHF", True),
                          ("GBP", "GBPUSD", False), ("DKK", "USDDKK", True)):
        if col in fpx.columns:
            usd_per[ccy] = (1.0 / fpx[col]) if inv else fpx[col]
    dates = result.predictions.index
    px = px.reindex(dates).ffill(); mc = mc.reindex(dates).ffill()
    fx = pd.DataFrame({t: usd_per.reindex(dates).ffill()[currency[t] if currency[t] in usd_per.columns else "USD"] for t in px.columns})
    ret = (px * fx).pct_change(fill_method=None)
    for t, d in result.data_quality["listing_mask"]["dates"].items():
        if t in ret.columns:
            ret.loc[ret.index <= pd.Timestamp(d), t] = np.nan
    bm_all = mc.shift(1); bm_all = bm_all.div(bm_all.sum(axis=1), axis=0).fillna(0.0)
    cfg = PipelineConfig(); out = {}
    for d in sorted(result.portfolio_weights)[-n_last:]:
        w = pd.Series(result.portfolio_weights[d]).reindex(px.columns).fillna(0.0)
        a = (w - bm_all.loc[d]).to_numpy()
        hist = ret.loc[ret.index < d].tail(126).fillna(0.0)
        cov = np.asarray(estimate_covariance(hist, bm_weights=bm_all.loc[d].to_numpy(), config=cfg), dtype=float)
        var = float(a @ cov @ a); contrib = pd.Series(a * (cov @ a), index=px.columns).groupby(sector).sum() / var
        out[d] = float(contrib.get(sector_name, np.nan))
    return pd.Series(out).sort_index()


def _load(dir_: Path):
    doc = json.load(open(dir_ / "metrics.json", encoding="utf-8"))
    return doc, pickle.load(open(dir_ / "backtest_result.pkl", "rb"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--raw-cache", default=None, help="pickled load_all_sheets() of the vintage workbook (sector arm only)")
    args = ap.parse_args()
    base_dir, arm_dir = AI_PORT / "outputs" / args.base, AI_PORT / "outputs" / args.arm
    base_doc, base_r = _load(base_dir); arm_doc, arm_r = _load(arm_dir)
    bm, am = base_doc["metrics"], arm_doc["metrics"]; spec = ARMS[args.arm]

    g0 = {"vintage_equal": base_doc.get("data_vintage") == arm_doc.get("data_vintage"),
          "base_reproduces_reference": abs(bm["information_ratio"] - REFERENCE_BASE_IR) < 1e-9,
          "flag_on": arm_doc["overrides"].get(spec["flag"]) is True}
    g0["g0_pass"] = bool(all(g0.values()))

    b_act = pd.Series(base_r.active_returns).dropna(); a_act = pd.Series(arm_r.active_returns).dropna()
    common = b_act.index.intersection(a_act.index); b_act, a_act = b_act.loc[common], a_act.loc[common]
    splits = []
    for k, ix in enumerate(np.array_split(np.arange(len(common)), N_SPLITS)):
        ir_b, ir_a = _ir(b_act.iloc[ix]), _ir(a_act.iloc[ix])
        splits.append({"split": k + 1, "start": str(common[ix[0]].date()), "end": str(common[ix[-1]].date()),
                       "ir_base": round(ir_b, 4), "ir_arm": round(ir_a, 4), "delta": round(ir_a - ir_b, 4)})
    d_ir = am["information_ratio"] - bm["information_ratio"]; split_deltas = [s["delta"] for s in splits]

    bp, ap_ = base_r.predictions, arm_r.predictions
    preds_identical = bool(bp.shape == ap_.shape and np.allclose(bp.fillna(0).to_numpy(), ap_.fillna(0).to_numpy(), atol=1e-12)
                           and bp.isna().equals(ap_.isna()))
    panel_identical = bool(base_r.panel.shape == arm_r.panel.shape and np.allclose(base_r.panel.fillna(0).to_numpy(), arm_r.panel.fillna(0).to_numpy(), atol=1e-9))
    if spec["mechanism"] == "boost":
        q = float(arm_doc["overrides"].get("top_decile_boost_quantile") or 0.9)
        cells = mechanism_boost_cells(bp, ap_, q)
        sb = top_decile_weight_share(base_r.portfolio_weights, bp, q); sa = top_decile_weight_share(arm_r.portfolio_weights, bp, q)
        both = pd.concat([sb.rename("base"), sa.rename("arm")], axis=1).dropna()
        share = {"n_dates": int(len(both)), "mean_base": round(float(both["base"].mean()), 4), "mean_arm": round(float(both["arm"].mean()), 4),
                 "share_dates_arm_higher": round(float((both["arm"] > both["base"]).mean()), 3), "pass": bool(len(both) and both["arm"].mean() > both["base"].mean())}
        mech = {"cells": cells, "top_decile_weight_share": share, "panel_identical": panel_identical,
                "targets_identical": bool(np.allclose(base_r.targets.fillna(0).to_numpy(), arm_r.targets.fillna(0).to_numpy(), atol=1e-12)),
                "pass": bool(cells["pass"] and share["pass"] and panel_identical)}
    else:
        if not args.raw_cache:
            raise SystemExit("--raw-cache is required for the sector arm (LW covariance from the vintage workbook)")
        raw = pickle.load(open(args.raw_cache, "rb"))
        sb = _sector_share_series(base_r, raw); sa = _sector_share_series(arm_r, raw)
        sec = mechanism_sector_share(sb, sa)
        sec["series"] = {str(d.date()): [round(float(sb[d]), 4), round(float(sa[d]), 4)] for d in sb.index if d in sa.index}
        mech = {"predictions_identical": preds_identical, "sector_share": sec, "pass": bool(preds_identical and sec["pass"])}
    mechanism_pass = bool(mech["pass"])

    turnover_ratio = am["avg_annual_turnover"] / bm["avg_annual_turnover"]
    e2 = evaluate_e2(am["tracking_error"], bm["active_share"], am["active_share"], turnover_ratio, arm_doc["optimizer_solver_fallback_rate"], TURNOVER_MAX)
    e2_pass = all(e2.values()); v = verdicts(g0["g0_pass"], mechanism_pass, e2_pass, d_ir, split_deltas)

    out = {"arm": args.arm, "flag": spec["flag"], "frame": "performance",
           "adoption_basis": (f"formal E1 (dIR > +{E1_DELTA_IR} and all {N_SPLITS} splits positive) with G0, mechanism, E2; override candidate "
                              "(dIR > 0, all splits positive, gates) reported only — user decision + DSR haircut"),
           "preregistration": "decision log §S25.6 (2026-10-09)", "base_dir": str(base_dir),
           "vintage": {"base_pkl_mtime": _kst(base_dir / "backtest_result.pkl"), "arm_pkl_mtime": _kst(arm_dir / "backtest_result.pkl"),
                       "base_data_vintage": base_doc.get("data_vintage"), "arm_data_vintage": arm_doc.get("data_vintage")},
           "g0": g0, "full_period": {"ir_base": round(bm["information_ratio"], 4), "ir_arm": round(am["information_ratio"], 4),
                                     "delta_ir": round(d_ir, 4), "e1_bar": E1_DELTA_IR},
           "subperiods": splits, **v, "mechanism": mech, "mechanism_pass": mechanism_pass, "e2": e2, "e2_pass": e2_pass,
           "companions": {"te_base": round(bm["tracking_error"], 5), "te_arm": round(am["tracking_error"], 5),
                          "turnover_base": round(bm["avg_annual_turnover"], 4), "turnover_arm": round(am["avg_annual_turnover"], 4),
                          "turnover_ratio": round(turnover_ratio, 4), "active_share_base": round(bm["active_share"], 5),
                          "active_share_arm": round(am["active_share"], 5), "realized_beta_base": bm.get("realized_beta"),
                          "realized_beta_arm": am.get("realized_beta"), "avg_ic_base": bm["avg_ic"], "avg_ic_arm": am["avg_ic"],
                          "max_drawdown_base": bm.get("max_drawdown"), "max_drawdown_arm": am.get("max_drawdown"),
                          "sub_periods_metrics_base": bm.get("sub_periods"), "sub_periods_metrics_arm": am.get("sub_periods"),
                          "degenerate": {"base": f"{base_doc['model_quality']['degenerate_retrains']}/{base_doc['model_quality']['total_retrains']}",
                                         "arm": f"{arm_doc['model_quality']['degenerate_retrains']}/{arm_doc['model_quality']['total_retrains']}"},
                          "fallback_arm": arm_doc["optimizer_solver_fallback_rate"], "solver_counts_arm": arm_doc.get("optimizer_solver_counts"),
                          "elapsed_sec_arm": arm_doc.get("elapsed_sec")}}
    (arm_dir / "e1_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    print(f"\n{args.arm}: G0 {'PASS' if g0['g0_pass'] else 'FAIL'} | mechanism {'PASS' if mechanism_pass else 'FAIL'} | E2 {'PASS' if e2_pass else 'FAIL'} | "
          f"dIR {d_ir:+.4f} splits {split_deltas} (formal E1 {'PASS' if v['e1_formal_pass'] else 'FAIL'}, no-harm {'PASS' if v['no_harm_pass'] else 'FAIL'}) | "
          f"adoption candidate: {v['adoption_candidate']} | override candidate: {v['override_candidate']}")


if __name__ == "__main__":
    main()
