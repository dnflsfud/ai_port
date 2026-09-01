# -*- coding: utf-8 -*-
"""§S16.8-A arm 판정 (read-only): fwd_opcf_level_z vs S0′(1.7596).

결정 로그 §S16.8 사전등록(2026-09-01)의 재실행 가능한 판정 스크립트.
  E1: full ΔIR > +0.36 & 3분할 부호 일관 (|ΔIR| < 0.36 노이즈).
  E2: TE ≤ 4.5% · active share ±3%p · turnover ≤ 1.25× · fallback 0.
  병기 의무: 신규 피처 EWMA 생존·라이브 모델 gain(0이면 no-op — §S13.21 전례),
    퇴화율, name-risk 캡 준수(라이브 북 max share ≤ 0.36) 유지 확인.

출력: outputs/s16_8a_opcf_level/e1_summary.json
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

BASE_DIR = AI_PORT / "outputs" / "s16_7_name_risk_cap"
ARM_DIR = AI_PORT / "outputs" / "s16_8a_opcf_level"
FEATURE = "fwd_opcf_level_z"
TURNOVER_RATIO_MAX = 1.25
ACTIVE_SHARE_BAND = 0.03
NAME_CAP_LIMIT = 0.36          # cap 0.35 + tol 0.01


def evaluate_e2(arm_te: float, base_as: float, arm_as: float,
                turnover_ratio: float, arm_fallback_rate: float) -> dict:
    return {
        "te_ok": bool(arm_te <= TE_GUARD),
        "active_share_ok": bool(abs(arm_as - base_as) <= ACTIVE_SHARE_BAND),
        "turnover_ok": bool(turnover_ratio <= TURNOVER_RATIO_MAX),
        "fallback_zero": bool(arm_fallback_rate == 0),
    }


def feature_status(active: bool, gain_pct: float) -> dict:
    """EWMA 생존 + 라이브 모델 gain — 어느 쪽이든 0이면 arm은 no-op."""
    return {
        "active_in_live_model": bool(active),
        "live_gain_pct": round(float(gain_pct), 3),
        "is_noop": bool((not active) or gain_pct <= 0.0),
    }


def main() -> None:
    base_doc = json.load(open(BASE_DIR / "metrics.json", encoding="utf-8"))
    arm_doc = json.load(open(ARM_DIR / "metrics.json", encoding="utf-8"))
    base_m, arm_m = base_doc["metrics"], arm_doc["metrics"]
    base_mq, arm_mq = base_doc["model_quality"], arm_doc["model_quality"]
    vintage_ok = base_doc.get("data_vintage") == arm_doc.get("data_vintage")

    base_r = pickle.load(open(BASE_DIR / "backtest_result.pkl", "rb"))
    arm_r = pickle.load(open(ARM_DIR / "backtest_result.pkl", "rb"))

    # ---- E1 -----------------------------------------------------------------
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
    e1 = evaluate_e1(d_ir, [s["delta"] for s in splits])

    # ---- 피처 생존·gain (라이브 모델) ---------------------------------------
    live_model = arm_r.models[max(arm_r.models)]
    feats = list(live_model._active_features)
    active = FEATURE in feats
    gain_pct = 0.0
    if active:
        booster = live_model.booster_ if hasattr(live_model, "booster_") else live_model
        g = pd.Series(booster.feature_importance(importance_type="gain"), index=feats)
        gain_pct = float(g[FEATURE] / g.sum() * 100.0)
    fstat = feature_status(active, gain_pct)
    n_models_with_feature = sum(
        1 for m in arm_r.models.values() if FEATURE in list(m._active_features))

    # ---- name-risk 캡 준수 유지(라이브 북) ----------------------------------
    import yaml

    from scripts.eval_s16_7_arm import compute_shares_for_books
    from src.data_loader import UniverseData
    from src.harness import build_override_config, inject_config

    manifest = yaml.safe_load(
        (AI_PORT / "variants" / "s16_8a_opcf_level.yaml").read_text(encoding="utf-8"))
    cfg = build_override_config(dict(manifest["overrides"]))
    inject_config(cfg)
    data = UniverseData(cfg.data_path, config=cfg)
    tickers = list(data.tickers)
    last_d = max(arm_r.portfolio_weights)
    books = {"arm_live": {last_d: arm_r.portfolio_weights[last_d]}}
    del base_r, arm_r
    shares = compute_shares_for_books(books, data, cfg, tickers)
    live_key = sorted(shares["arm_live"])[-1]
    live_share, live_top = shares["arm_live"][live_key]
    cap_ok = bool(live_share <= NAME_CAP_LIMIT + 1e-9)

    turnover_ratio = arm_m["avg_annual_turnover"] / base_m["avg_annual_turnover"]
    e2 = evaluate_e2(arm_m["tracking_error"], base_m["active_share"],
                     arm_m["active_share"], turnover_ratio,
                     arm_doc["optimizer_solver_fallback_rate"])

    out = {
        "preregistration": "decision log §S16.8 (2026-09-01)",
        "vintage": {
            "base_pkl_mtime": _kst(BASE_DIR / "backtest_result.pkl"),
            "arm_pkl_mtime": _kst(ARM_DIR / "backtest_result.pkl"),
            "data_vintage_equal": bool(vintage_ok),
        },
        "full_period": {
            "ir_base": round(base_m["information_ratio"], 4),
            "ir_arm": round(arm_m["information_ratio"], 4),
            "delta_ir": round(d_ir, 4),
            "e1_bar": E1_DELTA_IR,
        },
        "subperiods": splits,
        "e1_pass": e1,
        "feature": {**fstat,
                    "models_with_feature": f"{n_models_with_feature}/33"},
        "name_risk_cap": {"live_book_date": live_key,
                          "live_max_share": live_share,
                          "live_top_name": live_top,
                          "cap_ok": cap_ok},
        "e2": e2,
        "e2_pass": all(e2.values()),
        "companions": {
            "te_base": round(base_m["tracking_error"], 5),
            "te_arm": round(arm_m["tracking_error"], 5),
            "turnover_ratio": round(turnover_ratio, 4),
            "active_share_base": round(base_m["active_share"], 5),
            "active_share_arm": round(arm_m["active_share"], 5),
            "avg_ic_base": base_m["avg_ic"],
            "avg_ic_arm": arm_m["avg_ic"],
            "degenerate": {
                "base": f"{base_mq['degenerate_retrains']}/{base_mq['total_retrains']}",
                "arm": f"{arm_mq['degenerate_retrains']}/{arm_mq['total_retrains']}"},
            "fallback_arm": arm_doc["optimizer_solver_fallback_rate"],
            "solver_counts_arm": arm_doc.get("optimizer_solver_counts"),
        },
    }
    (ARM_DIR / "e1_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nE1: {'PASS' if e1 else 'FAIL'} (ΔIR {d_ir:+.4f} vs bar +{E1_DELTA_IR})  "
          f"E2: {'PASS' if all(e2.values()) else 'FAIL'}  "
          f"feature: {'no-op!' if fstat['is_noop'] else f'gain {gain_pct:.2f}%'} "
          f"({n_models_with_feature}/33 models)  "
          f"name-cap live {live_share:.3f} ({'OK' if cap_ok else 'BREACH'})")


if __name__ == "__main__":
    main()
