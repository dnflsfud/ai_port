# -*- coding: utf-8 -*-
"""§S13.42 사전점검 (read-only): skew_z·term_z 변동성 모델 증강.

결정 로그 §S13.42 사전등록(2026-08-13)의 P0'/P1'/P2'를 실행한다. arm 미실행.

  P0' (통계, vs B): 각 추정창에서 B(=lt+iv30_z)와 C(=B+skew_z+term_z)를
     **공통표본**(전 회귀변수 dropna)에 재적합해 OOS excess-QLIKE·MAE 비교.
     게이트: 둘 다 ≥5% 개선 AND 3분할 부호 일관. 시트 커버리지 보고 의무.
  P1' (바인딩, vs 현 production): 리밸런싱 12회 샘플에서 D_B(production
     스케일, src.option_vol_cov 정본으로 생성) vs D_C(후보) 재최적화,
     one-way L1 이동 중앙값 ≥ 0.005.
  P2' (방향, 비액션): corr(Δw, skew_z)·corr(Δw, term_z)·위너 유출.

출력: outputs/s13_42_skewterm_precheck/summary.json
"""

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

AI_PORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_PORT))

from src.option_vol_cov import (CLIP_HI, CLIP_LO, EST_FREQ, FIRST_EST_POS,  # noqa: E402
                                FWD, MIN_TRAIN_ROWS, OBS_STEP, TRAIL,
                                build_option_vol_scale)
from scripts.precheck_s13_41_optvol_cov import excess_qlike  # noqa: E402

S0_PKL = AI_PORT / "outputs" / "s0_recert_s13_38" / "backtest_result.pkl"
VARIANT = AI_PORT / "variants" / "s0_recert_s13_38.yaml"
OUT_DIR = AI_PORT / "outputs" / "s13_42_skewterm_precheck"

L1_GATE = 0.005
REBAL_SAMPLE_STEP = 8
ANN = float(np.sqrt(252.0))

SKEW_SHEET = "downside_skew_z"
TERM_SHEET = "iv_term_structure_z"


# ---------------------------------------------------------------------------
# 순수 헬퍼 (단위테스트 대상)
# ---------------------------------------------------------------------------

def fit_ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """절편 포함 OLS. x = (n, k) → coef 길이 k+1 [절편, ...]."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coef


def _pred(coef: np.ndarray, x: pd.DataFrame) -> pd.Series:
    return coef[0] + x.astype(float).values @ np.asarray(coef[1:], float)


def ratio_scale(coef_num: np.ndarray, x_num: pd.DataFrame,
                coef_den: np.ndarray, x_den: pd.DataFrame,
                lo: float, hi: float) -> pd.Series:
    """s = clip(exp(pred_num − pred_den), lo, hi). 비유한 입력 → inert 1.0."""
    log_ratio = pd.Series(_pred(coef_num, x_num), index=x_num.index) \
        - pd.Series(_pred(coef_den, x_den), index=x_den.index)
    s = np.exp(log_ratio).clip(lo, hi)
    return s.where(np.isfinite(s), 1.0).astype(float)


def _thirds_impr(a_col, b_col):
    x_a, x_b = np.asarray(a_col, float), np.asarray(b_col, float)
    cut = np.linspace(0, len(x_a), 4, dtype=int)
    return [round(1.0 - float(np.mean(x_b[cut[i]:cut[i + 1]]))
                  / float(np.mean(x_a[cut[i]:cut[i + 1]])), 4) for i in range(3)]


def _rank_corr(a: pd.Series, b: pd.Series) -> float:
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 10:
        return float("nan")
    return float(pair.iloc[:, 0].rank().corr(pair.iloc[:, 1].rank()))


def main() -> None:
    from run_variant import compose_config, load_manifest
    from src.backtest import get_sector_map, make_capweight_bm_fn
    from src.data_loader import UniverseData
    from src.portfolio_optimizer import estimate_covariance, optimize_portfolio

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cfg = compose_config(load_manifest(VARIANT))
    data = UniverseData(cfg.data_path, config=cfg)
    result = pickle.load(open(S0_PKL, "rb"))
    rebal_dates = sorted(pd.Timestamp(k) for k in result.portfolio_weights)
    tickers = list(result.portfolio_weights[rebal_dates[0]].index)

    ret = data.returns[tickers]
    dates = ret.index
    ivz = data.get_sheet("iv30_z").reindex(index=dates, columns=tickers)
    skz = data.get_sheet(SKEW_SHEET).reindex(index=dates, columns=tickers)
    tmz = data.get_sheet(TERM_SHEET).reindex(index=dates, columns=tickers)
    trail = ret.rolling(TRAIL, min_periods=60).std() * ANN
    fwd_vol = (ret.rolling(FWD).std() * ANN).shift(-FWD)
    lt_p = np.log(trail.where(trail > 0))
    lf_p = np.log(fwd_vol.where(fwd_vol > 0))

    # §4.3 유사 커버리지 보고
    cov_rep = {}
    for name, panel in [("iv30_z", ivz), (SKEW_SHEET, skz), (TERM_SHEET, tmz)]:
        cov_rep[name] = {
            "nonnan_share_full": round(float(panel.notna().mean().mean()), 3),
            "nonnan_share_last252": round(float(panel.iloc[-252:].notna().mean().mean()), 3),
        }
    print(f"[load] {time.time()-t0:.0f}s  coverage {json.dumps(cov_rep)}")

    # ---------------- P0': 워크포워드 B vs C (공통표본) ----------------
    models = {}   # est_pos -> dict(coef_a, coef_b, coef_c)
    eval_rows = []
    common_share = []
    for e in range(FIRST_EST_POS, len(dates), EST_FREQ):
        train_pos = [p for p in range(0, e - FWD, OBS_STEP) if p + FWD < e]
        if not train_pos:
            continue
        pool = pd.DataFrame({
            "lt": lt_p.iloc[train_pos].values.ravel(),
            "ziv": ivz.iloc[train_pos].values.ravel(),
            "zsk": skz.iloc[train_pos].values.ravel(),
            "ztm": tmz.iloc[train_pos].values.ravel(),
            "lf": lf_p.iloc[train_pos].values.ravel(),
        })
        b_sample = pool[["lt", "ziv", "lf"]].dropna()
        common = pool.dropna()
        if len(common) < MIN_TRAIN_ROWS:
            continue
        common_share.append(len(common) / max(len(b_sample), 1))
        models[e] = {
            "a": fit_ols(common[["lt"]].values, common["lf"]),
            "b": fit_ols(common[["lt", "ziv"]].values, common["lf"]),
            "c": fit_ols(common[["lt", "ziv", "zsk", "ztm"]].values, common["lf"]),
        }
        end = min(e + EST_FREQ, len(dates))
        for p in range(e, end, OBS_STEP):
            row = pd.DataFrame({
                "lt": lt_p.iloc[p], "ziv": ivz.iloc[p], "zsk": skz.iloc[p],
                "ztm": tmz.iloc[p], "lf": lf_p.iloc[p],
            }).dropna()
            if len(row) < 30:
                continue
            m = models[e]
            pb = _pred(m["b"], row[["lt", "ziv"]])
            pc = _pred(m["c"], row[["lt", "ziv", "zsk", "ztm"]])
            sig2 = np.exp(2.0 * row["lf"].values)
            eval_rows.append({
                "date": dates[p],
                "exq_b": float(np.mean(excess_qlike(np.exp(2 * pb), sig2))),
                "exq_c": float(np.mean(excess_qlike(np.exp(2 * pc), sig2))),
                "mae_b": float(np.mean(np.abs(np.exp(pb) - np.exp(row["lf"].values)))),
                "mae_c": float(np.mean(np.abs(np.exp(pc) - np.exp(row["lf"].values)))),
            })
    ev = pd.DataFrame(eval_rows).set_index("date").sort_index()
    q_impr = 1.0 - ev["exq_c"].mean() / ev["exq_b"].mean()
    m_impr = 1.0 - ev["mae_c"].mean() / ev["mae_b"].mean()
    thirds = {"qlike": _thirds_impr(ev["exq_b"], ev["exq_c"]),
              "mae": _thirds_impr(ev["mae_b"], ev["mae_c"])}
    thirds_ok = all(s > 0 for sub in thirds.values() for s in sub)
    p0 = bool(q_impr >= 0.05 and m_impr >= 0.05 and thirds_ok)
    d_med = float(np.median([m["c"][3] for m in models.values()]))
    e_med = float(np.median([m["c"][4] for m in models.values()]))
    print(f"[P0'] n_eval={len(ev)}  vs-B excess-QLIKE {q_impr:+.1%}  MAE {m_impr:+.1%}"
          f"  thirds {thirds}  -> {'PASS' if p0 else 'FAIL'}")
    print(f"      coef med d(skew)={d_med:+.4f} e(term)={e_med:+.4f}"
          f"  공통표본 비율(중앙값) {float(np.median(common_share)):.2f}")

    # ---------------- P1': D_B(production) vs D_C ----------------
    scale_b_panel = build_option_vol_scale(ret, ivz)  # production 정본 그대로

    def scale_c_at(pos: int) -> pd.Series:
        cands = [k for k in models if k <= pos]
        if not cands:
            return pd.Series(1.0, index=tickers)
        m = models[max(cands)]
        x_c = pd.DataFrame({
            "lt": lt_p.iloc[pos], "ziv": ivz.iloc[pos],
            "zsk": skz.iloc[pos].fillna(0.0), "ztm": tmz.iloc[pos].fillna(0.0),
        })
        x_a = x_c[["lt"]]
        return ratio_scale(m["c"], x_c[["lt", "ziv", "zsk", "ztm"]],
                           m["a"], x_a, CLIP_LO, CLIP_HI)

    bm_fn = make_capweight_bm_fn(data, tickers, cfg)
    sector_map = get_sector_map(data)
    daily_idx = sorted(pd.Timestamp(k) for k in result.daily_weights)
    p1_rows = []
    for d in rebal_dates[1::REBAL_SAMPLE_STEP]:
        pos = dates.get_loc(d)
        pred_row = result.predictions.loc[d, tickers]
        if pred_row.notna().sum() < 10:
            continue
        prev_day = [x for x in daily_idx if x < d][-1]
        prev_w = result.daily_weights[prev_day].reindex(tickers).fillna(0.0).values
        bm_w = bm_fn(d, tickers, len(tickers))
        hist = ret.iloc[max(0, pos - int(cfg.cov_lookback)):pos]
        cov0 = estimate_covariance(hist, bm_weights=bm_w, config=cfg)
        s_b = scale_b_panel.iloc[pos].reindex(tickers).fillna(1.0).values
        s_c = scale_c_at(pos).reindex(tickers).fillna(1.0).values
        w_b = optimize_portfolio(pred_row, np.diag(s_b) @ cov0 @ np.diag(s_b),
                                 prev_weights=prev_w, sector_map=sector_map,
                                 bm_weights=bm_w, config=cfg)
        w_c = optimize_portfolio(pred_row, np.diag(s_c) @ cov0 @ np.diag(s_c),
                                 prev_weights=prev_w, sector_map=sector_map,
                                 bm_weights=bm_w, config=cfg)
        dw = pd.Series(w_c - w_b, index=tickers)
        mu_rank = pred_row.rank(pct=True)
        p1_rows.append({
            "date": str(d.date()),
            "l1_shift": round(float(0.5 * np.abs(dw.values).sum()), 5),
            "max_scale_diff": round(float(np.max(np.abs(s_c - s_b))), 4),
            "corr_dw_skew": round(_rank_corr(dw, skz.iloc[pos]), 3),
            "corr_dw_term": round(_rank_corr(dw, tmz.iloc[pos]), 3),
            "dw_top_mu_quintile": round(float(dw[mu_rank >= 0.8].sum()), 5),
        })
        print(f"[P1'] {d.date()}  L1 {p1_rows[-1]['l1_shift']:.4f}"
              f"  maxΔs {p1_rows[-1]['max_scale_diff']:.3f}")
    p1_df = pd.DataFrame(p1_rows)
    med_l1 = float(p1_df["l1_shift"].median())
    p1 = bool(med_l1 >= L1_GATE)
    print(f"[P1'] median one-way L1 {med_l1:.4f} (게이트 {L1_GATE}) -> "
          f"{'PASS' if p1 else 'FAIL'}")

    summary = {
        "preregistration": "decision log §S13.42 (2026-08-13)",
        "workbook_mtime": pd.Timestamp(
            Path(cfg.data_path).stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S"),
        "coverage": cov_rep,
        "p0_vs_B": {
            "n_eval_dates": int(len(ev)),
            "excess_qlike_improvement": round(float(q_impr), 4),
            "mae_improvement": round(float(m_impr), 4),
            "thirds": thirds,
            "coef_median_skew": round(d_med, 4),
            "coef_median_term": round(e_med, 4),
            "common_sample_share_median": round(float(np.median(common_share)), 3),
            "pass": p0,
        },
        "p1_vs_production": {
            "n_sampled_rebalances": int(len(p1_df)),
            "median_l1_shift": round(med_l1, 5),
            "gate": L1_GATE,
            "rows": p1_rows,
            "pass": p1,
        },
        "p2_direction": {
            "mean_corr_dw_skew": round(float(p1_df["corr_dw_skew"].mean()), 3),
            "mean_corr_dw_term": round(float(p1_df["corr_dw_term"].mean()), 3),
            "mean_dw_top_mu_quintile": round(float(p1_df["dw_top_mu_quintile"].mean()), 5),
        },
        "verdict": "PROCEED-to-arm" if (p0 and p1) else "SHELVE",
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nVERDICT: {summary['verdict']}  ({time.time()-t0:.0f}s)  saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
