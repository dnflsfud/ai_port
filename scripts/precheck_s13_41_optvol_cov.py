# -*- coding: utf-8 -*-
"""§S13.41 사전점검 (read-only): iv30_z 변동성 예측 → 공분산 대각 조정.

결정 로그 §S13.41 사전등록(2026-08-13)의 P0/P1/P2를 실행한다. arm 미실행.

  P0 (통계): 워크포워드 63BD 재추정·엠바고(t+21 < 추정일)·expanding 하에서
     모델 A(log σ_fwd21 ~ log σ_trail126) vs B(+ iv30_z)의 OOS
     excess-QLIKE·MAE. 게이트: 둘 다 ≥5% 개선 AND 3분할 부호 일관.
  P1 (바인딩): S0'(s0_recert_s13_38) 리밸런싱 12회 샘플에서 실제
     estimate_covariance + optimize_portfolio 재호출. D_s = clip(σ̂_B/σ̂_A,
     0.8, 1.5) 적용 vs baseline MVO 타깃의 one-way L1 이동 중앙값 ≥ 0.005.
  P2 (방향, 비액션): corr(Δw, iv30_z)·corr(Δw, μ)·상위 μ 5분위 이탈 비중.

출력: outputs/s13_41_optvol_precheck/summary.json
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

S0_PKL = AI_PORT / "outputs" / "s0_recert_s13_38" / "backtest_result.pkl"
VARIANT = AI_PORT / "variants" / "s0_recert_s13_38.yaml"
OUT_DIR = AI_PORT / "outputs" / "s13_41_optvol_precheck"

FWD = 21              # 타깃: 향후 21BD 실현변동성
TRAIL = 126           # 베이스 예측자: trailing 126BD (cov_lookback과 동일)
EST_FREQ = 63         # 재추정 주기 (사전등록)
OBS_STEP = 5          # 학습/평가 관측 샘플링 간격
MIN_TRAIN_ROWS = 500  # 추정 최소 풀드 관측 수
FIRST_EST_POS = 252   # 최초 추정 위치 (BD)
CLIP_LO, CLIP_HI = 0.8, 1.5
L1_GATE = 0.005       # P1: one-way L1 이동 중앙값 게이트
REBAL_SAMPLE_STEP = 8  # 96회 중 12회 샘플
ANN = float(np.sqrt(252.0))


# ---------------------------------------------------------------------------
# 순수 헬퍼 (단위테스트 대상)
# ---------------------------------------------------------------------------

def excess_qlike(h: np.ndarray, sig2: np.ndarray) -> np.ndarray:
    """QLIKE 초과손실 log(h/σ²) + σ²/h − 1 ≥ 0 (h=σ²에서 0). %개선 정의 가능."""
    r = np.asarray(sig2, dtype=float) / np.asarray(h, dtype=float)
    return -np.log(r) + r - 1.0


def fit_vol_models(log_trail: np.ndarray, z: np.ndarray, log_fwd: np.ndarray):
    """풀드 OLS: A = [1, log_trail], B = [1, log_trail, z]. (coef_a, coef_b)"""
    lt = np.asarray(log_trail, float)
    zz = np.asarray(z, float)
    lf = np.asarray(log_fwd, float)
    xa = np.column_stack([np.ones_like(lt), lt])
    xb = np.column_stack([np.ones_like(lt), lt, zz])
    coef_a, *_ = np.linalg.lstsq(xa, lf, rcond=None)
    coef_b, *_ = np.linalg.lstsq(xb, lf, rcond=None)
    return coef_a, coef_b


def predict_scale(coef_a, coef_b, trail_vol: pd.Series, z: pd.Series,
                  lo: float, hi: float) -> pd.Series:
    """s = clip(σ̂_B/σ̂_A, lo, hi). 입력 결측/비양수 → inert(1.0)."""
    lt = np.log(trail_vol.where(trail_vol > 0))
    log_ratio = ((coef_b[0] - coef_a[0])
                 + (coef_b[1] - coef_a[1]) * lt
                 + coef_b[2] * z)
    s = np.exp(log_ratio).clip(lo, hi)
    return s.where(np.isfinite(s), 1.0).astype(float)


def _rank_corr(a: pd.Series, b: pd.Series) -> float:
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 10:
        return float("nan")
    return float(pair.iloc[:, 0].rank().corr(pair.iloc[:, 1].rank()))


def _thirds(vals):
    x = np.asarray(vals, dtype=float)
    cut = np.linspace(0, len(x), 4, dtype=int)
    return [x[cut[i]:cut[i + 1]] for i in range(3)]


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
    trail = ret.rolling(TRAIL, min_periods=60).std() * ANN
    fwd_vol = (ret.rolling(FWD).std() * ANN).shift(-FWD)
    print(f"[load] {time.time()-t0:.0f}s  dates {dates[0].date()}~{dates[-1].date()}"
          f"  tickers {len(tickers)}")

    lt_panel = np.log(trail.where(trail > 0))
    lf_panel = np.log(fwd_vol.where(fwd_vol > 0))

    # ---------------- P0: 워크포워드 A vs B ----------------
    est_positions = list(range(FIRST_EST_POS, len(dates), EST_FREQ))
    models = {}          # est_pos -> (coef_a, coef_b)
    eval_rows = []       # (date, exq_a, exq_b, mae_a, mae_b) 평균/관측
    for k, e in enumerate(est_positions):
        train_pos = [p for p in range(0, e - FWD, OBS_STEP) if p + FWD < e]
        if not train_pos:
            continue
        sub = pd.DataFrame({
            "lt": lt_panel.iloc[train_pos].values.ravel(),
            "z": ivz.iloc[train_pos].values.ravel(),
            "lf": lf_panel.iloc[train_pos].values.ravel(),
        }).dropna()
        if len(sub) < MIN_TRAIN_ROWS:
            continue
        coef_a, coef_b = fit_vol_models(sub["lt"], sub["z"], sub["lf"])
        models[e] = (coef_a, coef_b)
        # OOS 평가: (e, e+63] 구간의 5BD 샘플 시점
        end = min(e + EST_FREQ, len(dates))
        for p in range(e, end, OBS_STEP):
            row = pd.DataFrame({
                "lt": lt_panel.iloc[p], "z": ivz.iloc[p], "lf": lf_panel.iloc[p],
            }).dropna()
            if len(row) < 30:
                continue
            pa = coef_a[0] + coef_a[1] * row["lt"]
            pb = coef_b[0] + coef_b[1] * row["lt"] + coef_b[2] * row["z"]
            sig2 = np.exp(2.0 * row["lf"])
            eval_rows.append({
                "date": dates[p],
                "exq_a": float(np.mean(excess_qlike(np.exp(2 * pa), sig2))),
                "exq_b": float(np.mean(excess_qlike(np.exp(2 * pb), sig2))),
                "mae_a": float(np.mean(np.abs(np.exp(pa) - np.exp(row["lf"])))),
                "mae_b": float(np.mean(np.abs(np.exp(pb) - np.exp(row["lf"])))),
            })
    ev = pd.DataFrame(eval_rows).set_index("date").sort_index()
    q_impr = 1.0 - ev["exq_b"].mean() / ev["exq_a"].mean()
    m_impr = 1.0 - ev["mae_b"].mean() / ev["mae_a"].mean()
    thirds_ok = True
    thirds_detail = {}
    for name, (a_col, b_col) in {"qlike": ("exq_a", "exq_b"),
                                 "mae": ("mae_a", "mae_b")}.items():
        subs = []
        for ta, tb in zip(_thirds(ev[a_col]), _thirds(ev[b_col])):
            subs.append(round(1.0 - float(np.mean(tb)) / float(np.mean(ta)), 4))
        thirds_detail[name] = subs
        thirds_ok = thirds_ok and all(s > 0 for s in subs)
    p0 = bool(q_impr >= 0.05 and m_impr >= 0.05 and thirds_ok)
    print(f"[P0] n_eval={len(ev)}  excess-QLIKE 개선 {q_impr:+.1%}  "
          f"MAE 개선 {m_impr:+.1%}  thirds {thirds_detail}  -> {'PASS' if p0 else 'FAIL'}")

    # z 계수 안정성 (참고)
    c_z = [round(float(cb[2]), 4) for _, cb in models.values()]
    print(f"     c_z: first/median/last {c_z[0]:+.4f}/{float(np.median(c_z)):+.4f}/{c_z[-1]:+.4f}")

    def scale_at(pos: int) -> pd.Series:
        cands = [e for e in models if e <= pos]
        if not cands:
            return pd.Series(1.0, index=tickers)
        coef_a, coef_b = models[max(cands)]
        return predict_scale(coef_a, coef_b, trail.iloc[pos], ivz.iloc[pos],
                             CLIP_LO, CLIP_HI)

    # ---------------- P1: 바인딩 (실제 MVO 재호출) ----------------
    bm_fn = make_capweight_bm_fn(data, tickers, cfg)
    sector_map = get_sector_map(data)
    daily_idx = sorted(pd.Timestamp(k) for k in result.daily_weights)
    sample = rebal_dates[1::REBAL_SAMPLE_STEP]
    p1_rows = []
    for d in sample:
        pos = dates.get_loc(d)
        pred_row = result.predictions.loc[d, tickers]
        if pred_row.notna().sum() < 10:
            continue
        prev_day = [x for x in daily_idx if x < d][-1]
        prev_w = result.daily_weights[prev_day].reindex(tickers).fillna(0.0).values
        bm_w = bm_fn(d, tickers, len(tickers))
        hist = ret.iloc[max(0, pos - int(cfg.cov_lookback)):pos]
        cov0 = estimate_covariance(hist, bm_weights=bm_w, config=cfg)
        w0 = optimize_portfolio(pred_row, cov0, prev_weights=prev_w,
                                sector_map=sector_map, bm_weights=bm_w, config=cfg)
        s = scale_at(pos).reindex(tickers).fillna(1.0).values
        cov1 = np.diag(s) @ cov0 @ np.diag(s)
        w1 = optimize_portfolio(pred_row, cov1, prev_weights=prev_w,
                                sector_map=sector_map, bm_weights=bm_w, config=cfg)
        recorded = result.portfolio_weights[d].reindex(tickers).fillna(0.0).values
        dw = pd.Series(w1 - w0, index=tickers)
        a0, a1 = w0 - bm_w, w1 - bm_w
        te0 = float(np.sqrt(a0 @ cov0 @ a0) * ANN)
        te1 = float(np.sqrt(a1 @ cov0 @ a1) * ANN)  # 동일 cov0 기준 ex-ante TE
        mu_rank = pred_row.rank(pct=True)
        top_mu = mu_rank >= 0.8
        p1_rows.append({
            "date": str(d.date()),
            "l1_shift": round(float(0.5 * np.abs(dw.values).sum()), 5),
            "fidelity_l1_vs_recorded": round(float(0.5 * np.abs(w0 - recorded).sum()), 5),
            "scale_min": round(float(np.min(s)), 3),
            "scale_max": round(float(np.max(s)), 3),
            "scale_frac_clipped": round(float(np.mean((s <= CLIP_LO) | (s >= CLIP_HI))), 3),
            "te_exante_base": round(te0, 5),
            "te_exante_adj": round(te1, 5),
            "corr_dw_ivz": round(_rank_corr(dw, ivz.iloc[pos]), 3),
            "corr_dw_mu": round(_rank_corr(dw, pred_row), 3),
            "dw_top_mu_quintile": round(float(dw[top_mu].sum()), 5),
        })
        print(f"[P1] {d.date()}  L1 {p1_rows[-1]['l1_shift']:.4f}  "
              f"fid {p1_rows[-1]['fidelity_l1_vs_recorded']:.4f}  "
              f"dw@topμ {p1_rows[-1]['dw_top_mu_quintile']:+.4f}")
    p1_df = pd.DataFrame(p1_rows)
    med_l1 = float(p1_df["l1_shift"].median())
    p1 = bool(med_l1 >= L1_GATE)
    print(f"[P1] median one-way L1 {med_l1:.4f} (게이트 {L1_GATE}) -> "
          f"{'PASS' if p1 else 'FAIL'}")

    summary = {
        "preregistration": "decision log §S13.41 (2026-08-13)",
        "workbook_mtime": pd.Timestamp(
            Path(cfg.data_path).stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S"),
        "p0": {
            "n_eval_dates": int(len(ev)),
            "excess_qlike_improvement": round(float(q_impr), 4),
            "mae_improvement": round(float(m_impr), 4),
            "thirds": thirds_detail,
            "c_z_median": float(np.median(c_z)),
            "pass": p0,
        },
        "p1": {
            "n_sampled_rebalances": int(len(p1_df)),
            "median_l1_shift": round(med_l1, 5),
            "gate": L1_GATE,
            "median_fidelity_l1": round(float(p1_df["fidelity_l1_vs_recorded"].median()), 5),
            "rows": p1_rows,
            "pass": p1,
        },
        "p2_direction": {
            "mean_corr_dw_ivz": round(float(p1_df["corr_dw_ivz"].mean()), 3),
            "mean_corr_dw_mu": round(float(p1_df["corr_dw_mu"].mean()), 3),
            "mean_dw_top_mu_quintile": round(float(p1_df["dw_top_mu_quintile"].mean()), 5),
        },
        "verdict": "PROCEED-to-arm" if (p0 and p1) else "SHELVE",
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nVERDICT: {summary['verdict']}  ({time.time()-t0:.0f}s)  saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
