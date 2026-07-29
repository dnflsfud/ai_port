# -*- coding: utf-8 -*-
"""§S13.20 사전점검: 종목별 지수-EPS 감응도(loading)의 횡단면 변환 자격.

신규 백테스트 없음 — 종목별 rolling 252d OLS(일간 USD 수익률의 유니버스-평균
초과분 ~ 일간 innovation Δ1 log NDX_FWD_EPS − Δ1 log SPX_FWD_EPS)를 96개
리밸런싱 날짜에서 PIT(과거 창만) 추정하고 3게이트를 판정한다.

게이트 (결정 로그 §S13.20 사전등록 — 단일 판독, 스윕 없음):
  G1 (분산):   리밸런싱 날짜 평균 |t|>2 종목 비율 ≥ 20%
  G2 (안정성): loading 횡단면 순위 자기상관(lag 3 리밸런싱 = 63d) 평균 ≥ 0.6
  G3 (예측력): s_i = loading_i × fac_eps_tech_lead63(t)의 rank IC(타깃 =
               S0 backtest_result.targets 20d spec 패널) — mean > 0 AND
               P1/P2/P3 중 ≥2 부호 양
하나라도 실패 → arm 미설계 SHELVE.

실행: <PY> scripts/preflight_s13_20_eps_loading.py  (WD=ai_port, PYTHONPATH=.)
산출물: outputs/s13_20_eps_loading_preflight.csv
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.preflight_s13_19_carry_eps_covariation import eps_features  # noqa: E402
from src.config import PipelineConfig  # noqa: E402
from src.data_loader import UniverseData, mask_pre_listing  # noqa: E402

WINDOW = 252
MIN_OBS = 200
STATE_NAME = "fac_eps_tech_lead63"
G1_MIN_SHARE = 0.20
G2_MIN_AUTOCORR = 0.6
AUTOCORR_LAG = 3  # 리밸런싱 3회 = 63BD
MIN_CS_OBS = 10


def daily_innovation(px: pd.DataFrame) -> pd.Series:
    """일간 innovation: Δ1 log NDX_FWD_EPS − Δ1 log SPX_FWD_EPS."""
    return np.log(px["NDX_FWD_EPS"]).diff() - np.log(px["SPX_FWD_EPS"]).diff()


def window_loadings(y: pd.DataFrame, x: pd.Series,
                    min_obs: int) -> tuple[pd.Series, pd.Series]:
    """단일 창 per-종목 OLS slope·t (NaN 셀은 종목별 제외, n<min_obs → NaN)."""
    Y = y.to_numpy(dtype=float)
    xv = x.to_numpy(dtype=float)
    M = np.isfinite(Y) & np.isfinite(xv)[:, None]
    Yz = np.where(M, Y, 0.0)
    Xz = np.where(M, xv[:, None], 0.0)
    n = M.sum(axis=0).astype(float)
    sx, sy = Xz.sum(axis=0), Yz.sum(axis=0)
    sxx, sxy = (Xz * Xz).sum(axis=0), (Xz * Yz).sum(axis=0)
    syy = (Yz * Yz).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = n * sxx - sx * sx
        b = (n * sxy - sx * sy) / denom
        a = (sy - b * sx) / n
        rss = syy - a * sy - b * sxy
        s2 = rss / (n - 2.0)
        t = b / np.sqrt(s2 * n / denom)
    bad = (n < min_obs) | ~np.isfinite(b) | ~np.isfinite(t)
    return (pd.Series(np.where(bad, np.nan, b), index=y.columns),
            pd.Series(np.where(bad, np.nan, t), index=y.columns))


def rank_autocorr(loadings: pd.DataFrame, lag: int = AUTOCORR_LAG) -> float:
    """행(리밸런싱)간 lag 시차의 횡단면 스피어만 자기상관 평균."""
    vals = []
    for k in range(lag, len(loadings)):
        d = pd.concat([loadings.iloc[k], loadings.iloc[k - lag]],
                      axis=1).dropna()
        if len(d) >= MIN_CS_OBS:
            vals.append(d.iloc[:, 0].corr(d.iloc[:, 1], method="spearman"))
    return float(np.nanmean(vals)) if vals else float("nan")


def rank_ic(sig: pd.Series, tgt: pd.Series) -> float:
    d = pd.concat([sig, tgt], axis=1).dropna()
    if len(d) < MIN_CS_OBS:
        return float("nan")
    return float(d.iloc[:, 0].corr(d.iloc[:, 1], method="spearman"))


def judge_gates(share_sig: float, autocorr: float, mean_ic: float,
                sub_ics: dict) -> dict:
    g1 = bool(share_sig >= G1_MIN_SHARE)
    g2 = bool(autocorr >= G2_MIN_AUTOCORR)
    pos = sum(1 for v in sub_ics.values() if np.isfinite(v) and v > 0.0)
    g3 = bool(np.isfinite(mean_ic) and mean_ic > 0.0 and pos >= 2)
    return {"G1": g1, "G2": g2, "G3": g3,
            "verdict": "PROCEED" if (g1 and g2 and g3) else "SHELVE"}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cfg = PipelineConfig()
    per_date = pd.read_csv(
        root / "outputs/s13_12_ic_ir_transmission/per_date_S0.csv",
        parse_dates=["date"]).set_index("date")

    px = pd.read_excel(cfg.data_path, sheet_name="Factor_PX_LAST",
                       index_col=0, engine="openpyxl")
    px.index = pd.DatetimeIndex(px.index)
    innov = daily_innovation(px)
    state = eps_features(px)[STATE_NAME]

    print("loading UniverseData (USD returns + listing mask)...")
    data = UniverseData(cfg.data_path, cfg)
    returns = mask_pre_listing(data.sheets["Daily_Returns"],
                               data.listing_dates, inclusive=True)

    with open(root / "outputs/codex_causal_rank_65/backtest_result.pkl",
              "rb") as f:
        targets = pickle.load(f).targets

    tickers = [t for t in returns.columns if t in targets.columns]
    returns = returns[tickers]
    excess = returns.sub(returns.mean(axis=1), axis=0)
    x = innov.reindex(returns.index)
    state_at = state.reindex(per_date.index, method="ffill")
    print(f"panel: {len(returns)}d × {len(tickers)} tickers, "
          f"innovation finite {np.isfinite(x.to_numpy()).mean():.3f}")

    load_rows, t_rows, stats = [], [], []
    kept_dates = []
    for t_date in per_date.index:
        pos = returns.index.searchsorted(t_date)  # strictly-before 창
        if pos < WINDOW or t_date not in targets.index:
            continue
        win = slice(pos - WINDOW, pos)
        beta, tstat = window_loadings(excess.iloc[win], x.iloc[win], MIN_OBS)
        n_valid = int(tstat.notna().sum())
        share = (float((tstat.abs() > 2.0).sum()) / n_valid
                 if n_valid else float("nan"))
        sig = beta * float(state_at.loc[t_date])
        ic = rank_ic(sig, targets.loc[t_date])
        kept_dates.append(t_date)
        load_rows.append(beta)
        t_rows.append(tstat)
        stats.append({"date": t_date, "sub": per_date.loc[t_date, "sub"],
                      "n_valid": n_valid, "share_sig": share,
                      "state": float(state_at.loc[t_date]), "ic": ic})

    loadings = pd.DataFrame(load_rows, index=kept_dates)
    stat = pd.DataFrame(stats).set_index("date")
    share_sig = float(stat["share_sig"].mean())
    autocorr = rank_autocorr(loadings)
    mean_ic = float(stat["ic"].mean())
    sub_ics = {s: float(stat.loc[stat["sub"] == s, "ic"].mean())
               for s in ("P1", "P2", "P3")}
    gates = judge_gates(share_sig, autocorr, mean_ic, sub_ics)

    print(f"\nrebalances used: {len(stat)}/{len(per_date)}  "
          f"mean n_valid {stat['n_valid'].mean():.1f}")
    print(f"\n==== §S13.20 gates (1차 = daily innovation NDX−SPX, "
          f"state = {STATE_NAME}) ====")
    print(f"G1 mean share(|t|>2) {share_sig:.3f} >= {G1_MIN_SHARE}"
          f"  -> {'PASS' if gates['G1'] else 'FAIL'}")
    print(f"G2 rank autocorr(lag {AUTOCORR_LAG}) {autocorr:.3f} >= "
          f"{G2_MIN_AUTOCORR}  -> {'PASS' if gates['G2'] else 'FAIL'}")
    print(f"G3 mean IC {mean_ic:+.4f} > 0, subs "
          + " ".join(f"{k} {v:+.4f}" for k, v in sub_ics.items())
          + f"  -> {'PASS' if gates['G3'] else 'FAIL'}")
    print(f"\n§S13.20 verdict: {gates['verdict']}"
          f"{' (잔차 슬리브/오버레이 arm 설계 착수)' if gates['verdict'] == 'PROCEED' else ' (arm 미설계)'}")

    stat.to_csv(root / "outputs/s13_20_eps_loading_preflight.csv",
                encoding="utf-8-sig")
    print("saved: outputs/s13_20_eps_loading_preflight.csv")
    return 0 if gates["verdict"] == "PROCEED" else 1


if __name__ == "__main__":
    sys.exit(main())
