# -*- coding: utf-8 -*-
"""§S16.7 arm 판정 (read-only): 종목당 액티브 리스크 몫 상한 0.35 vs S0′.

결정 로그 §S16.7 사전등록(2026-08-31)의 재실행 가능한 판정 스크립트.
  비교 기준: outputs/s16_2_revision_extension_cap (유일한 델타 =
  name_risk_share_cap_enabled).
  G0: data_vintage 쌍 동일 + avg_ic 비트 동일(옵티마이저-전용 변경 증빙).
  G1: 실행 북 리밸일별 Euler name share(공식 방법론: raw-return cov +
      optvol 대각)가 cap+tol(0.36) 이하 — 준수율 ≥ 90% + 라이브 북 준수.
  G2: TE ≤ 4.5% · active share base±3%p · turnover ≤ 1.2× · fallback 0 ·
      퇴화율 동일.
  E1: 표준 ΔIR 바(+0.36 & 3분할 부호 일관) — 채택 프레임은 리스크 규율
      (G1+G2 통과 & ΔIR > −0.36이면 flip 후보로 사용자 상신).

출력: outputs/s16_7_name_risk_cap/e1_summary.json
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

BASE_DIR = AI_PORT / "outputs" / "s16_2_revision_extension_cap"
ARM_DIR = AI_PORT / "outputs" / "s16_7_name_risk_cap"
VARIANT = AI_PORT / "variants" / "s16_7_name_risk_cap.yaml"

CAP = 0.35
TOL = 0.01                    # = portfolio_optimizer.NAME_RISK_CAP_TOL
G1_COMPLIANCE_MIN = 0.90
TURNOVER_RATIO_MAX = 1.20
ACTIVE_SHARE_BAND = 0.03


def euler_shares(w, bm, cov):
    """Euler 액티브 리스크 몫 a_i(Σa)_i/(aᵀΣa); 합=1."""
    a = np.asarray(w, float) - np.asarray(bm, float)
    var = float(a @ cov @ a)
    if not np.isfinite(var) or var <= 1e-18:
        return np.zeros(len(a))
    return a * (cov @ a) / var


def evaluate_g0(base_doc: dict, arm_doc: dict) -> dict:
    vintage_equal = base_doc.get("data_vintage") == arm_doc.get("data_vintage")
    ic_identical = base_doc["metrics"]["avg_ic"] == arm_doc["metrics"]["avg_ic"]
    return {
        "vintage_equal": bool(vintage_equal),
        "avg_ic_bit_identical": bool(ic_identical),
        "g0_pass": bool(vintage_equal and ic_identical),
    }


def evaluate_g1(max_shares: dict, cap: float = CAP, tol: float = TOL,
                min_rate: float = G1_COMPLIANCE_MIN) -> dict:
    """리밸일(삽입 순서=날짜 순) → max name share 딕셔너리 판정."""
    vals = list(max_shares.values())
    ok = [v <= cap + tol + 1e-9 for v in vals]
    rate = float(sum(ok)) / len(ok)
    return {
        "n_rebalances": len(ok),
        "n_breaches": int(len(ok) - sum(ok)),
        "compliance_rate": round(rate, 4),
        "live_book_ok": bool(ok[-1]),
        "g1_pass": bool(rate >= min_rate and ok[-1]),
    }


def evaluate_g2(arm_te: float, base_as: float, arm_as: float,
                turnover_ratio: float, arm_fallback_rate: float,
                degenerate_equal: bool) -> dict:
    return {
        "te_ok": bool(arm_te <= TE_GUARD),
        "active_share_ok": bool(abs(arm_as - base_as) <= ACTIVE_SHARE_BAND),
        "turnover_ok": bool(turnover_ratio <= TURNOVER_RATIO_MAX),
        "fallback_zero": bool(arm_fallback_rate == 0),
        "degenerate_equal": bool(degenerate_equal),
    }


def compute_shares_for_books(books: dict, data, cfg, tickers):
    """books: {label: {date: weight Series}} → {label: {date: (max_share, top)}}.
    cov는 날짜당 1회만 추정해 두 북에 공유."""
    from scripts.export_operating_data import (_apply_optvol_scale,
                                               _load_optvol_scale,
                                               _production_risk_source)
    from src.backtest import get_benchmark_fn
    from src.portfolio_optimizer import estimate_covariance

    risk_source = _production_risk_source(data, tickers)
    optvol_scale = _load_optvol_scale(data, tickers, cfg)
    bm_fn = get_benchmark_fn(data, tickers, config=cfg)
    cov_lookback = int(getattr(cfg, "cov_lookback", 126))

    all_dates = sorted({d for wbd in books.values() for d in wbd})
    out = {label: {} for label in books}
    for d in all_dates:
        hist = risk_source.loc[risk_source.index < d].tail(cov_lookback)
        bm_w = np.asarray(bm_fn(d, tickers, len(tickers)), dtype=float)
        cov = np.asarray(
            estimate_covariance(hist, bm_weights=bm_w, config=cfg), dtype=float)
        cov, _ = _apply_optvol_scale(cov, optvol_scale, d, tickers)
        for label, wbd in books.items():
            if d not in wbd:
                continue
            w = wbd[d].reindex(tickers).fillna(0.0).values
            s = euler_shares(w, bm_w, cov)
            i = int(np.argmax(s))
            out[label][str(pd.Timestamp(d).date())] = (
                round(float(s[i]), 4), tickers[i])
    return out


def main() -> None:
    import yaml

    from src.data_loader import UniverseData
    from src.harness import build_override_config, inject_config

    base_doc = json.load(open(BASE_DIR / "metrics.json", encoding="utf-8"))
    arm_doc = json.load(open(ARM_DIR / "metrics.json", encoding="utf-8"))
    base_m, arm_m = base_doc["metrics"], arm_doc["metrics"]
    base_mq, arm_mq = base_doc["model_quality"], arm_doc["model_quality"]

    g0 = evaluate_g0(base_doc, arm_doc)

    base_r = pickle.load(open(BASE_DIR / "backtest_result.pkl", "rb"))
    arm_r = pickle.load(open(ARM_DIR / "backtest_result.pkl", "rb"))

    # ---- E1 (표준 바) -----------------------------------------------------
    b_act = pd.Series(base_r.active_returns).dropna()
    a_act = pd.Series(arm_r.active_returns).dropna()
    common = b_act.index.intersection(a_act.index)
    b_act, a_act = b_act.loc[common], a_act.loc[common]
    splits = []
    for k, idx in enumerate(np.array_split(np.arange(len(common)), N_SPLITS)):
        splits.append({
            "split": k + 1,
            "start": str(common[idx[0]].date()),
            "end": str(common[idx[-1]].date()),
            "ir_base": round(_ir(b_act.iloc[idx]), 4),
            "ir_arm": round(_ir(a_act.iloc[idx]), 4),
            "delta": round(_ir(a_act.iloc[idx]) - _ir(b_act.iloc[idx]), 4),
        })
    d_ir = arm_m["information_ratio"] - base_m["information_ratio"]
    split_deltas = [s["delta"] for s in splits]
    e1 = evaluate_e1(d_ir, split_deltas)

    # ---- G1 (기전/바인딩: 실행 북 Euler share 오프라인 재계산) -------------
    manifest = yaml.safe_load(VARIANT.read_text(encoding="utf-8"))
    cfg = build_override_config(dict(manifest["overrides"]))
    inject_config(cfg)
    data = UniverseData(cfg.data_path, config=cfg)
    tickers = list(data.tickers)
    books = {
        "base": dict(base_r.portfolio_weights),
        "arm": dict(arm_r.portfolio_weights),
    }
    del base_r, arm_r
    shares = compute_shares_for_books(books, data, cfg, tickers)
    arm_max = {d: v[0] for d, v in shares["arm"].items()}
    base_max = {d: v[0] for d, v in shares["base"].items()}
    g1 = evaluate_g1(arm_max)
    g1_base_ref = evaluate_g1(base_max)          # baseline breach율 병기용
    last_d = sorted(shares["arm"])[-1]

    # ---- G2 ----------------------------------------------------------------
    turnover_ratio = arm_m["avg_annual_turnover"] / base_m["avg_annual_turnover"]
    degenerate_equal = (
        base_mq["degenerate_retrains"] == arm_mq["degenerate_retrains"]
        and base_mq["total_retrains"] == arm_mq["total_retrains"])
    g2 = evaluate_g2(arm_m["tracking_error"], base_m["active_share"],
                     arm_m["active_share"], turnover_ratio,
                     arm_doc["optimizer_solver_fallback_rate"],
                     degenerate_equal)

    out = {
        "preregistration": "decision log §S16.7 (2026-08-31)",
        "vintage": {
            "base_pkl_mtime": _kst(BASE_DIR / "backtest_result.pkl"),
            "arm_pkl_mtime": _kst(ARM_DIR / "backtest_result.pkl"),
        },
        "g0": g0,
        "full_period": {
            "ir_base": round(base_m["information_ratio"], 4),
            "ir_arm": round(arm_m["information_ratio"], 4),
            "delta_ir": round(d_ir, 4),
            "e1_bar": E1_DELTA_IR,
        },
        "subperiods": splits,
        "e1_pass": e1,
        "risk_discipline_frame": bool(d_ir > -E1_DELTA_IR),
        "g1": {"arm": g1, "base_reference": g1_base_ref,
               "live_book": {"date": last_d,
                             "arm": shares["arm"][last_d],
                             "base": shares["base"][last_d]},
               "arm_worst5": sorted(arm_max.items(), key=lambda x: -x[1])[:5],
               "base_worst5": sorted(base_max.items(), key=lambda x: -x[1])[:5]},
        "g2": g2,
        "g2_pass": all(g2.values()),
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
    print(f"\nG0: {'PASS' if g0['g0_pass'] else 'FAIL'}  "
          f"G1: {'PASS' if g1['g1_pass'] else 'FAIL'} "
          f"(준수율 {g1['compliance_rate']:.1%}, breach {g1['n_breaches']}/"
          f"{g1['n_rebalances']}, base 참조 breach {g1_base_ref['n_breaches']}/"
          f"{g1_base_ref['n_rebalances']})  "
          f"G2: {'PASS' if all(g2.values()) else 'FAIL'}  "
          f"E1: {'PASS' if e1 else 'FAIL'} (ΔIR {d_ir:+.4f})  "
          f"리스크규율 프레임(ΔIR>-0.36): {out['risk_discipline_frame']}")


if __name__ == "__main__":
    main()
