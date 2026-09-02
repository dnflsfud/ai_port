# -*- coding: utf-8 -*-
"""§S17.1 Σ 채널 기전 사전점검 (read-only, 백테스트 0회).

결정 로그 §S17.1 사전등록(2026-09-02)의 G1(기전) — 동결 S0′(outputs/s17_s0_0902)의
97 리밸일 실행 북을 고정하고 production Σ(OFF) 와 5일 겹침 상관 Σ(ON) 를 같은
126d 창에서 재추정한다(옵티마이저 경로 그대로: raw 리스크 원천 + S13.41 대각 스케일).

  G1 결정 게이트:
    (i)  ASIA×US 블록 평균 상관 ON/OFF ≥ 2.0 인 리밸일 비율 ≥ 0.90
    (ii) US×US 블록 평균 상관 |ON/OFF − 1| 중앙값 < 0.10
    (iii) ON Σ 최소 고유값 ≥ 0 (PSD) 전 날짜
  관측(사전등록 · 비게이트): EU×US 비, ex-ante TE ON/OFF, 아시아 액티브 분산 점유,
    MZ 회귀 β_d·β_w(NW HAC lag 3, 실현 = 향후 63BD 주간 합 분산의 일별 환산) —
    북 수준 효과가 +4% 분산 규모라 97점 회귀는 검정력이 없음을 사전에 명시.

출력: outputs/s17_prechecks/cov_overlap_mechanism.json
"""

import gc
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AI_PORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_PORT))

BASE_PKL = AI_PORT / "outputs" / "s17_s0_0902" / "backtest_result.pkl"
OUT = AI_PORT / "outputs" / "s17_prechecks" / "cov_overlap_mechanism.json"
VARIANT = AI_PORT / "variants" / "codex_causal_rank_65.yaml"

ASIA_CODES = {"KS", "JP"}
EU_CODES = {"FP", "GR", "NA", "SW", "LN", "DC", "SM", "IM"}
G1_ASIA_RATIO_MIN = 2.0
G1_ASIA_DATE_FRAC_MIN = 0.90
G1_US_ABS_DEV_MAX = 0.10
FWD_DAYS = 63
NW_LAG = 3


def evaluate_g1(asia_ratio_frac_ge2: float, us_abs_dev_median: float,
                all_psd: bool) -> dict:
    return {
        "asia_ratio_frac_ge2": round(float(asia_ratio_frac_ge2), 4),
        "us_abs_dev_median": round(float(us_abs_dev_median), 4),
        "all_psd": bool(all_psd),
        "g1_pass": bool(asia_ratio_frac_ge2 >= G1_ASIA_DATE_FRAC_MIN
                        and us_abs_dev_median < G1_US_ABS_DEV_MAX and all_psd),
    }


def block_mean_corr(cov: np.ndarray, rows, cols, exclude_diag: bool = False) -> float:
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    sub = corr[np.ix_(rows, cols)]
    if exclude_diag:
        mask = ~np.eye(len(rows), dtype=bool)
        return float(sub[mask].mean())
    return float(sub.mean())


def nw_ols(y: np.ndarray, x: np.ndarray, lag: int) -> dict:
    """y = a + b·x, Newey-West HAC(lag) 표준오차 (Bartlett 가중)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    G = X * e[:, None]
    S = G.T @ G
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)
        gl = G[l:].T @ G[:-l]
        S += w * (gl + gl.T)
    V = XtX_inv @ S @ XtX_inv
    se = float(np.sqrt(max(V[1, 1], 0.0)))
    return {"beta": float(beta[1]), "alpha": float(beta[0]), "se": se,
            "ci95": [float(beta[1] - 1.96 * se), float(beta[1] + 1.96 * se)]}


def main() -> None:
    import yaml

    from scripts.export_operating_data import (_apply_optvol_scale,
                                               _load_optvol_scale,
                                               _production_risk_source)
    from src.backtest import get_benchmark_fn
    from src.data_loader import UniverseData
    from src.harness import build_override_config, inject_config
    from src.portfolio_optimizer import estimate_covariance

    r = pickle.load(open(BASE_PKL, "rb"))
    books = {pd.Timestamp(d): w.copy() for d, w in r.portfolio_weights.items()}
    del r
    gc.collect()

    overrides = dict(yaml.safe_load(VARIANT.read_text(encoding="utf-8"))["overrides"])
    cfg = build_override_config(overrides)
    inject_config(cfg)
    cfg_on = build_override_config({**overrides, "cov_corr_overlap_enabled": True})

    data = UniverseData(cfg.data_path, config=cfg)
    tickers = list(data.tickers)
    region = {t: ("ASIA" if str(c) in ASIA_CODES else "EU" if str(c) in EU_CODES else "US")
              for t, c in data.meta["exchange_code"].items()}
    idx = {k: [i for i, t in enumerate(tickers) if region.get(t) == k] for k in ("ASIA", "EU", "US")}
    risk_source = _production_risk_source(data, tickers)
    realized_src = risk_source.fillna(0.0)
    optvol_scale = _load_optvol_scale(data, tickers, cfg)
    bm_fn = get_benchmark_fn(data, tickers, config=cfg)
    cov_lookback = int(getattr(cfg, "cov_lookback", 126))

    rows = []
    for d in sorted(books):
        hist = risk_source.loc[risk_source.index < d].tail(cov_lookback)
        bm_w = np.asarray(bm_fn(d, tickers, len(tickers)), dtype=float)
        c_off = np.asarray(estimate_covariance(hist, bm_weights=bm_w, config=cfg), dtype=float)
        c_on = np.asarray(estimate_covariance(hist, bm_weights=bm_w, config=cfg_on), dtype=float)
        c_off, _ = _apply_optvol_scale(c_off, optvol_scale, d, tickers)
        c_on, _ = _apply_optvol_scale(c_on, optvol_scale, d, tickers)
        w = books[d].reindex(tickers).fillna(0.0).values.astype(float)
        a = w - bm_w
        var_off, var_on = float(a @ c_off @ a), float(a @ c_on @ a)
        sh_off = float(np.sum(a[idx["ASIA"]] * (c_off @ a)[idx["ASIA"]]) / var_off) if var_off > 0 else np.nan
        sh_on = float(np.sum(a[idx["ASIA"]] * (c_on @ a)[idx["ASIA"]]) / var_on) if var_on > 0 else np.nan
        au_off, au_on = block_mean_corr(c_off, idx["ASIA"], idx["US"]), block_mean_corr(c_on, idx["ASIA"], idx["US"])
        eu_off, eu_on = block_mean_corr(c_off, idx["EU"], idx["US"]), block_mean_corr(c_on, idx["EU"], idx["US"])
        uu_off = block_mean_corr(c_off, idx["US"], idx["US"], exclude_diag=True)
        uu_on = block_mean_corr(c_on, idx["US"], idx["US"], exclude_diag=True)
        fwd = realized_src.loc[realized_src.index >= d].iloc[:FWD_DAYS]
        if len(fwd) >= FWD_DAYS:
            x = fwd[tickers].values @ a
            wk = pd.Series(x).rolling(5).sum().dropna().values
            real_week, real_day = float(np.var(wk, ddof=1) / 5.0), float(np.var(x, ddof=1))
        else:
            real_week = real_day = np.nan
        rows.append({
            "date": str(d.date()),
            "asia_us_off": au_off, "asia_us_on": au_on,
            "asia_us_ratio": au_on / au_off if au_off > 0 else np.nan,
            "eu_us_ratio": eu_on / eu_off if eu_off > 0 else np.nan,
            "us_us_ratio": uu_on / uu_off if uu_off > 0 else np.nan,
            "te_off": float(np.sqrt(max(var_off, 0.0) * 252.0)),
            "te_on": float(np.sqrt(max(var_on, 0.0) * 252.0)),
            "asia_share_off": sh_off, "asia_share_on": sh_on,
            "asia_net_active": float(a[idx["ASIA"]].sum()),
            "min_eig_on": float(np.linalg.eigvalsh(c_on).min()),
            "cond_off": float(np.linalg.cond(c_off)), "cond_on": float(np.linalg.cond(c_on)),
            "pred_var_off": var_off, "pred_var_on": var_on,
            "real_var_week": real_week, "real_var_day": real_day,
        })
    df = pd.DataFrame(rows)

    g1 = evaluate_g1(float((df["asia_us_ratio"] >= G1_ASIA_RATIO_MIN).mean()),
                     float((df["us_us_ratio"] - 1.0).abs().median()),
                     bool((df["min_eig_on"] >= -1e-12).all()))

    mz = df.dropna(subset=["real_var_week"])
    mz_out = {
        "n": int(len(mz)),
        "daily_sigma": nw_ols(mz["real_var_week"].values, mz["pred_var_off"].values, NW_LAG),
        "overlap_sigma": nw_ols(mz["real_var_week"].values, mz["pred_var_on"].values, NW_LAG),
        "mean_ratio_realized_week_over_pred": {
            "off": float((mz["real_var_week"] / mz["pred_var_off"]).mean()),
            "on": float((mz["real_var_week"] / mz["pred_var_on"]).mean())},
        "mean_ratio_realized_day_over_pred_off": float((mz["real_var_day"] / mz["pred_var_off"]).mean()),
        "note": "informational — book-level effect is ~+4% variance; a 97-point regression is underpowered by design",
    }

    out = {
        "preregistration": "decision log §S17.1 (2026-09-02)",
        "base": str(BASE_PKL.relative_to(AI_PORT)),
        "n_rebalances": int(len(df)),
        "n_region": {k: len(v) for k, v in idx.items()},
        "g1": g1,
        "bars": {"asia_ratio_min": G1_ASIA_RATIO_MIN, "asia_date_frac_min": G1_ASIA_DATE_FRAC_MIN,
                 "us_abs_dev_max": G1_US_ABS_DEV_MAX},
        "block_corr_medians": {
            "asia_us_off": round(float(df["asia_us_off"].median()), 4),
            "asia_us_on": round(float(df["asia_us_on"].median()), 4),
            "asia_us_ratio_median": round(float(df["asia_us_ratio"].median()), 3),
            "eu_us_ratio_median": round(float(df["eu_us_ratio"].median()), 3),
            "us_us_ratio_median": round(float(df["us_us_ratio"].median()), 3),
        },
        "ex_ante_te": {"off_median": round(float(df["te_off"].median()), 5),
                       "on_median": round(float(df["te_on"].median()), 5),
                       "ratio_median": round(float((df["te_on"] / df["te_off"]).median()), 4),
                       "cap": float(cfg.max_te_annual)},
        "asia_active_variance_share": {"off_mean": round(float(df["asia_share_off"].mean()), 4),
                                       "on_mean": round(float(df["asia_share_on"].mean()), 4)},
        "asia_net_active_mean": round(float(df["asia_net_active"].mean()), 4),
        "condition_number_median": {"off": round(float(df["cond_off"].median()), 1),
                                    "on": round(float(df["cond_on"].median()), 1)},
        "mz_regression": mz_out,
        "per_date": df.round(6).to_dict(orient="records"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "per_date"}, indent=2, ensure_ascii=False))
    print(f"\nG1: {'PASS' if g1['g1_pass'] else 'FAIL'}  "
          f"(ASIAxUS ratio>=2 on {g1['asia_ratio_frac_ge2']:.0%} of dates, "
          f"USxUS |dev| median {g1['us_abs_dev_median']:.3f}, PSD {g1['all_psd']})")


if __name__ == "__main__":
    main()
