# -*- coding: utf-8 -*-
"""§S16.5-P risk_aversion 사전점검 (read-only, 백테스트 0회).

결정 로그 §S16.5-P 사전등록(2026-09-01)의 재실행 가능한 사전점검.
  P0: 리스크항/리턴항 비율·TE 캡 바인딩률 (97 리밸일 전수, 재해 없음)
  P1a: 타깃 재해 w(λ=1) vs w(λ=0) — 리스크 항 부재 실측 확정
  P1: 등리스크 최대알파 갭 — 갭₂(타깃 대비) 중앙값 ≥ +5% 가 결정 게이트
  P2: λ* = median[ tp·L1턴오버(타깃) / (aᵀΣa)(타깃) ] (정칙자 등가 원칙)
  P3: λ* 재해 — active share 드리프트 ±3%p·TE 중앙값 ≥ 0.5×(λ=1)·전 해 optimal

표본: 재해 P1a/P1/P3 는 매 2번째 리밸(짝수 인덱스, 사전 고정).
출력: outputs/s16_5_precheck.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AI_PORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_PORT))

BASE_PKL = AI_PORT / "outputs" / "s16_7_name_risk_cap" / "backtest_result.pkl"
OUT = AI_PORT / "outputs" / "s16_5_precheck.json"

P1_GAP2_MIN_PCT = 5.0
P3_AS_DRIFT_MAX = 0.03
P3_TE_RATIO_MIN = 0.5
BINDING_FRAC = 0.995


def evaluate_p1(gap2_median_pct: float) -> dict:
    ok = bool(np.isfinite(gap2_median_pct) and gap2_median_pct >= P1_GAP2_MIN_PCT)
    return {"gap2_median_pct": round(float(gap2_median_pct), 3)
            if np.isfinite(gap2_median_pct) else None,
            "bar_pct": P1_GAP2_MIN_PCT, "p1_pass": ok}


def evaluate_p3(as_drift_max_pp: float, te_ratio_median: float,
                all_optimal: bool) -> dict:
    return {
        "as_drift_max_pp": round(float(as_drift_max_pp), 4),
        "te_ratio_median": round(float(te_ratio_median), 4),
        "all_optimal": bool(all_optimal),
        "p3_pass": bool(as_drift_max_pp <= P3_AS_DRIFT_MAX
                        and te_ratio_median >= P3_TE_RATIO_MIN
                        and all_optimal),
    }


def lambda_star_turnover_rule(tp: float, turnover_l1_list, risk_list) -> float:
    """λ* = median( tp·‖Δw‖₁ / aᵀΣa ) — 리스크 항 = turnover 정칙자 등가."""
    vals = [tp * t / r for t, r in zip(turnover_l1_list, risk_list) if r > 0]
    return float(np.median(vals))


def _solve_max_alpha_at_te(mu, cov, prev_w, sector_map, bm_w, te_annual, config):
    """max μ·a s.t. production 제약 ∧ {TE ≤ te_annual} (목적함수 페널티 없음)."""
    import cvxpy as cp

    from src.portfolio_optimizer import _build_mvo_constraints, _solve_problem

    n = len(mu)
    w = cp.Variable(n)
    raw = np.asarray(mu.values, dtype=float)
    mu_v = np.where(np.isfinite(raw), raw, 0.0)
    _, _, constraints = _build_mvo_constraints(
        w=w, expected_returns=mu, cov_matrix=cov, prev_weights=prev_w,
        sector_map=sector_map, bm_weights=bm_w,
        max_te_annual=float(te_annual), sector_deviation=config.sector_deviation,
        config=config,
    )
    prob = cp.Problem(cp.Maximize(mu_v @ (w - bm_w)), constraints)
    if (not _solve_problem(prob, None) or
            prob.status not in ("optimal", "optimal_inaccurate") or w.value is None):
        return None
    sol = np.asarray(w.value, dtype=float).flatten()
    return sol if np.all(np.isfinite(sol)) else None


def main() -> None:
    import pickle

    import yaml

    from scripts.export_operating_data import (_apply_optvol_scale,
                                               _load_optvol_scale,
                                               _production_risk_source)
    from src.backtest import get_benchmark_fn, get_sector_map
    from src.data_loader import UniverseData
    from src.harness import build_override_config, inject_config
    from src.portfolio_optimizer import estimate_covariance, optimize_portfolio

    manifest = yaml.safe_load(
        (AI_PORT / "variants" / "codex_causal_rank_65.yaml").read_text(encoding="utf-8"))
    cfg = build_override_config(dict(manifest["overrides"]))
    inject_config(cfg)
    tp = float(cfg.turnover_penalty)
    cap_daily_var = cfg.max_te_annual ** 2 / 252.0

    data = UniverseData(cfg.data_path, config=cfg)
    tickers = list(data.tickers)
    sector_map = get_sector_map(data)
    risk_source = _production_risk_source(data, tickers)
    optvol_scale = _load_optvol_scale(data, tickers, cfg)
    bm_fn = get_benchmark_fn(data, tickers, config=cfg)
    cov_lookback = int(getattr(cfg, "cov_lookback", 126))

    r = pickle.load(open(BASE_PKL, "rb"))
    dates = sorted(r.portfolio_weights.keys())
    preds = r.predictions

    # ---- P0: 전수 진단 -------------------------------------------------------
    p0_rows = []
    covs, bms, mus, prevs, execs = {}, {}, {}, {}, {}
    prev_book = None
    for d in dates:
        hist = risk_source.loc[risk_source.index < d].tail(cov_lookback)
        bm_w = np.asarray(bm_fn(d, tickers, len(tickers)), dtype=float)
        cov = np.asarray(estimate_covariance(hist, bm_weights=bm_w, config=cfg), dtype=float)
        cov, _ = _apply_optvol_scale(cov, optvol_scale, d, tickers)
        w_exec = r.portfolio_weights[d].reindex(tickers).fillna(0.0).values.astype(float)
        mu = preds.loc[d].reindex(tickers)
        a = w_exec - bm_w
        risk0 = float(a @ cov @ a)
        mu_v = np.where(np.isfinite(mu.values), mu.values, 0.0)
        ret_term = float(abs(mu_v @ a))
        te_ann = float(np.sqrt(max(risk0, 0.0) * 252.0))
        p0_rows.append({
            "date": str(pd.Timestamp(d).date()),
            "ret_term": ret_term, "risk_term_lam1": risk0,
            "ratio": ret_term / risk0 if risk0 > 0 else np.inf,
            "te_annual": te_ann,
            "binding": bool(te_ann >= BINDING_FRAC * cfg.max_te_annual),
        })
        covs[d], bms[d], mus[d], execs[d] = cov, bm_w, mu, w_exec
        prevs[d] = prev_book if prev_book is not None else bm_w.copy()
        prev_book = w_exec
    p0 = pd.DataFrame(p0_rows)
    binding_rate = float(p0["binding"].mean())

    # ---- 표본일(매 2번째) 재해: P1a / P1 / P2 / P3 ---------------------------
    sample = dates[::2]
    p1a_dinf, gap1, gap2 = [], [], []
    lam_num_l1, lam_den_risk, scale_eq = [], [], []
    targets_l1 = {}
    for d in sample:
        cov, bm_w, mu, prev_w = covs[d], bms[d], mus[d], prevs[d]
        w1 = optimize_portfolio(mu, cov, prev_weights=prev_w, sector_map=sector_map,
                                bm_weights=bm_w, config=cfg)
        w0 = optimize_portfolio(mu, cov, prev_weights=prev_w, sector_map=sector_map,
                                bm_weights=bm_w, risk_aversion=0.0, config=cfg)
        p1a_dinf.append(float(np.max(np.abs(w1 - w0))))
        targets_l1[d] = w1
        mu_v = np.where(np.isfinite(mu.values), mu.values, 0.0)
        a_t = w1 - bm_w
        risk_t = float(a_t @ cov @ a_t)
        alpha_t = float(mu_v @ a_t)
        lam_num_l1.append(float(np.abs(w1 - prev_w).sum()))
        lam_den_risk.append(risk_t)
        if risk_t > 0:
            scale_eq.append(alpha_t / risk_t)
        # 갭₂ (타깃 대비, 결정 게이트) — 비바인딩일만
        te_t = float(np.sqrt(max(risk_t, 0.0) * 252.0))
        if te_t < BINDING_FRAC * cfg.max_te_annual and alpha_t != 0:
            sol = _solve_max_alpha_at_te(mu, cov, prev_w, sector_map, bm_w, te_t, cfg)
            if sol is not None:
                gap2.append((float(mu_v @ (sol - bm_w)) - alpha_t) / abs(alpha_t) * 100.0)
        # 갭₁ (실행북 대비, 분해용)
        w_e = execs[d]
        a_e = w_e - bm_w
        te_e = float(np.sqrt(max(float(a_e @ cov @ a_e), 0.0) * 252.0))
        alpha_e = float(mu_v @ a_e)
        if te_e < BINDING_FRAC * cfg.max_te_annual and alpha_e != 0:
            sol = _solve_max_alpha_at_te(mu, cov, prev_w, sector_map, bm_w, te_e, cfg)
            if sol is not None:
                gap1.append((float(mu_v @ (sol - bm_w)) - alpha_e) / abs(alpha_e) * 100.0)

    p1 = evaluate_p1(float(np.median(gap2)) if gap2 else float("nan"))
    lam_star = lambda_star_turnover_rule(tp, lam_num_l1, lam_den_risk)

    # ---- P3: λ* 재해 ----------------------------------------------------------
    as_drift, te_ratio, alpha_keep, optimal_flags = [], [], [], []
    for d in sample:
        cov, bm_w, mu, prev_w = covs[d], bms[d], mus[d], prevs[d]
        diag = {}
        wl = optimize_portfolio(mu, cov, prev_weights=prev_w, sector_map=sector_map,
                                bm_weights=bm_w, risk_aversion=lam_star, config=cfg,
                                diagnostics=diag)
        optimal_flags.append(not diag.get("used_fallback", False))
        w1 = targets_l1[d]
        mu_v = np.where(np.isfinite(mu.values), mu.values, 0.0)
        a1, al = w1 - bm_w, wl - bm_w
        as_drift.append(abs(float(np.abs(al).sum() / 2 - np.abs(a1).sum() / 2)))
        r1 = float(a1 @ cov @ a1)
        rl = float(al @ cov @ al)
        te_ratio.append(np.sqrt(rl / r1) if r1 > 0 else np.nan)
        d1 = float(mu_v @ a1)
        alpha_keep.append(float(mu_v @ al) / d1 if d1 != 0 else np.nan)
    p3 = evaluate_p3(float(np.max(as_drift)), float(np.nanmedian(te_ratio)),
                     bool(all(optimal_flags)))

    out = {
        "preregistration": "decision log §S16.5-P (2026-09-01)",
        "base_pkl": str(BASE_PKL),
        "p0": {
            "n_rebalances": len(dates),
            "ratio_ret_over_risk_median": float(p0["ratio"].median()),
            "binding_rate": round(binding_rate, 4),
            "te_annual_median": round(float(p0["te_annual"].median()), 5),
        },
        "p1a": {"max_dw_inf_median": float(np.median(p1a_dinf)),
                "max_dw_inf_max": float(np.max(p1a_dinf)),
                "risk_term_inert": bool(np.median(p1a_dinf) < 1e-6)},
        "p1": {**p1, "n_gap2_dates": len(gap2),
               "gap1_median_pct": round(float(np.median(gap1)), 3) if gap1 else None,
               "n_gap1_dates": len(gap1)},
        "p2": {"lambda_star_turnover_rule": round(lam_star, 1),
               "scale_equalizer_median": round(float(np.median(scale_eq)), 1),
               "n_sample": len(sample)},
        "p3": {**p3, "alpha_keep_median": round(float(np.nanmedian(alpha_keep)), 4),
               "lambda_star_used": round(lam_star, 1)},
        "verdict": "PROCEED" if (p1["p1_pass"] and p3["p3_pass"]) else "SHELVE",
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nS16.5-P verdict: {out['verdict']}  "
          f"(P1 gap2 {p1['gap2_median_pct']}% vs bar {P1_GAP2_MIN_PCT}%, "
          f"P3 {'PASS' if p3['p3_pass'] else 'FAIL'}, λ* {lam_star:.0f})")


if __name__ == "__main__":
    main()
