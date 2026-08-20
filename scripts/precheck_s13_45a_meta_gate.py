# -*- coding: utf-8 -*-
"""§S13.45-A 사전점검 (read-only): 메타 라벨링 게이트 — rank 안정성의 fwd IC 예측력.

결정 로그 §S13.45-A 사전등록의 P0/P1/P2를 실행한다. arm 미실행.

  리밸런싱 그리드 = res.portfolio_weights 시점(21BD 비중첩).
  primary rank_stab = 직전 리밸런싱 예측벡터와의 Spearman(공통 비-NaN 종목).
  trailing_ic6 = production compute_signal_confidence 입력 미러 —
    pkl ic_series에서 당일 미포함 직전 6개 평균(직전 <2개면 0.0,
    src/backtest.py trailing_ic_window 경로와 동일).
  타깃 fwd_ic = 당일 예측 행 vs 전진 21BD masked USD 수익률 Spearman.
  P0: fwd_ic ~ rank_stab 단변량 OLS, NW(lag 1) t ≥ 2.0 AND 계수 양.
  P1: fwd_ic ~ [trailing_ic6 + rank_stab] 2변량, rank_stab NW t ≥ 2.0 AND 양.
  P2: rank_stab 상위 vs 하위 tercile fwd_ic 평균차 > 0 AND
      시간순 3분할(각 분할 내 자체 tercile) ≥ 2/3 양.
  진단(비액션): pred_disp(MAD×1.4826)·mkt_vol21 단변량 NW t,
      corr(rank_stab, trailing_ic6), fwd_ic vs pkl ic_series 교차검증 상관.

출력: outputs/s13_45a_meta_gate/summary.json
"""

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

AI_PORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_PORT))

from scripts.precheck_s13_43_regime import newey_west_t  # noqa: E402
from scripts.preflight_s13_30_vol_quality import _fwd_return  # noqa: E402
from scripts.preflight_s13_36_turnover_tilt import _spearman  # noqa: E402

PKL = AI_PORT / "outputs" / "codex_causal_rank_65" / "backtest_result.pkl"
VARIANT = AI_PORT / "variants" / "codex_causal_rank_65.yaml"
OUT_DIR = AI_PORT / "outputs" / "s13_45a_meta_gate"

FWD = 21
NW_LAG = 1
TRAIL_N = 6
MIN_NAMES = 30


# ---------------------------------------------------------------------------
# 순수 헬퍼 (단위테스트 대상)
# ---------------------------------------------------------------------------

def rank_stability(prev_row: pd.Series, curr_row: pd.Series) -> float:
    """두 리밸런싱 예측벡터의 Spearman — 공통 비-NaN 종목만."""
    return _spearman(prev_row, curr_row)


def trailing_ic(ic_series: pd.Series, date, n: int = TRAIL_N) -> float:
    """당일 미포함 직전 n개 IC 평균 — backtest.py 1571-1576 미러.

    production은 직전 IC가 2개 미만이면 trailing_ic_mean=0.0을 쓴다.
    """
    prior = ic_series.loc[ic_series.index < date]
    if len(prior) < 2:
        return 0.0
    return float(np.nanmean(prior.tail(n).to_numpy()))


def mad_dispersion(row: pd.Series) -> float:
    """단면 MAD×1.4826 (robust sd)."""
    x = row.dropna()
    if len(x) < 3:
        return float("nan")
    return float(1.4826 * (x - x.median()).abs().median())


def tercile_diff(stab: pd.Series, y: pd.Series) -> float:
    """rank_stab 상위 tercile fwd_ic 평균 − 하위 tercile 평균."""
    pair = pd.concat([stab.rename("s"), y.rename("y")], axis=1).dropna()
    k = len(pair) // 3
    if k < 2:
        return float("nan")
    ordered = pair.sort_values("s")
    return float(ordered["y"].tail(k).mean() - ordered["y"].head(k).mean())


def subperiod_tercile_signs(stab: pd.Series, y: pd.Series) -> list:
    """시간순 3분할 — 각 분할 내 자체 tercile diff (S13.43 관용)."""
    diffs = []
    for block in np.array_split(np.arange(len(stab)), 3):
        diffs.append(tercile_diff(stab.iloc[block], y.iloc[block]))
    return diffs


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

def main() -> None:
    from run_variant import compose_config, load_manifest
    from src.data_loader import UniverseData

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cfg = compose_config(load_manifest(VARIANT))
    wb_epoch = Path(cfg.data_path).stat().st_mtime
    pkl_epoch = PKL.stat().st_mtime
    fmt = lambda e: pd.Timestamp(e, unit="s").strftime("%Y-%m-%d %H:%M:%S")  # noqa: E731
    workbook_mtime, pkl_mtime = fmt(wb_epoch), fmt(pkl_epoch)

    res = pickle.load(open(PKL, "rb"))

    # ---------- 실행 선행 검증 (§9: 빈티지 어긋나면 중단·보고) ----------
    for field in ("predictions", "portfolio_weights", "daily_weights",
                  "ic_series", "benchmark_returns"):
        if getattr(res, field, None) is None:
            sys.exit(f"[ABORT] pkl에 {field} 없음 — §S13.45-A 가정 파괴, 보고 요망")
    pkl_last_date = str(pd.Timestamp(max(res.daily_weights.keys())).date())
    print(f"[vintage] pkl last daily_weights = {pkl_last_date}"
          f"  pkl mtime = {pkl_mtime}  workbook mtime = {workbook_mtime} (UTC 표기)")
    if pkl_epoch < wb_epoch:
        sys.exit("[ABORT] pkl이 워크북 리프레시보다 오래됨 — 빈티지 불일치, 보고 요망")

    data = UniverseData(cfg.data_path, config=cfg)
    if getattr(data, "returns_masked", None) is None:
        sys.exit("[ABORT] returns_masked 없음 — masked USD 수익률 가정 파괴, 보고 요망")

    returns = data.returns_masked
    preds = res.predictions
    ic_series = res.ic_series.dropna().sort_index()
    bm = res.benchmark_returns.dropna()
    dates = sorted(res.portfolio_weights.keys())
    print(f"[load] {time.time()-t0:.0f}s  rebalances={len(dates)}"
          f" ({pd.Timestamp(dates[0]).date()} ~ {pd.Timestamp(dates[-1]).date()})"
          f"  returns={returns.shape}  ic_series n={len(ic_series)}")

    rows = []
    skipped = 0
    prev_dt = None
    for dt in dates:
        if dt not in preds.index:
            skipped += 1
            prev_dt = None
            continue
        curr_row = preds.loc[dt]
        scored = curr_row.dropna()
        fwd = _fwd_return(returns, dt, FWD)
        stab = (rank_stability(preds.loc[prev_dt], curr_row)
                if prev_dt is not None else float("nan"))
        prev_dt = dt
        if fwd is None or len(scored) < MIN_NAMES:
            skipped += 1
            continue
        bm_win = bm.loc[:dt].tail(FWD)
        vol21 = (float(bm_win.std(ddof=1) * np.sqrt(252))
                 if len(bm_win) == FWD else float("nan"))
        rows.append({
            "date": str(pd.Timestamp(dt).date()),
            "rank_stab": stab,
            "trailing_ic6": trailing_ic(ic_series, dt),
            "pred_disp": mad_dispersion(curr_row),
            "mkt_vol21": vol21,
            "fwd_ic": _spearman(scored, fwd),
            "pkl_ic": float(ic_series.get(dt, np.nan)),
        })

    df = pd.DataFrame(rows).set_index("date")

    # 교차검증: 자체 fwd_ic vs pkl ic_series 동일 날짜 상관 (정의 확인)
    xchk = df[["fwd_ic", "pkl_ic"]].dropna()
    ic_crosscheck = (float(xchk["fwd_ic"].corr(xchk["pkl_ic"]))
                     if len(xchk) >= 3 else float("nan"))
    print(f"[crosscheck] fwd_ic vs pkl ic_series corr={ic_crosscheck:+.3f}"
          f" (n={len(xchk)})")

    panel = df[["rank_stab", "trailing_ic6", "pred_disp",
                "mkt_vol21", "fwd_ic"]].dropna()
    n = len(panel)
    y = panel["fwd_ic"].to_numpy()

    # ---------------- P0: 단변량 rank_stab (NW lag 1) ----------------
    coef0, t0s = newey_west_t(y, panel[["rank_stab"]].values, lag=NW_LAG)
    p0 = {"coef": round(float(coef0[1]), 6), "t": round(float(t0s[1]), 3),
          "pass": bool(t0s[1] >= 2.0 and coef0[1] > 0)}

    # ---------------- P1: trailing_ic6 통제 2변량 ----------------
    coef1, t1s = newey_west_t(y, panel[["trailing_ic6", "rank_stab"]].values,
                              lag=NW_LAG)
    p1 = {"coef": round(float(coef1[2]), 6), "t": round(float(t1s[2]), 3),
          "coef_trailing_ic6": round(float(coef1[1]), 6),
          "t_trailing_ic6": round(float(t1s[1]), 3),
          "pass": bool(t1s[2] >= 2.0 and coef1[2] > 0)}

    # ---------------- P2: tercile 스프레드 + 3분할 ----------------
    diff = tercile_diff(panel["rank_stab"], panel["fwd_ic"])
    sub_diffs = subperiod_tercile_signs(panel["rank_stab"], panel["fwd_ic"])
    sub_signs = int(sum(1 for d in sub_diffs if np.isfinite(d) and d > 0))
    p2 = {"diff": round(float(diff), 6),
          "subperiod_diffs": [round(float(d), 6) for d in sub_diffs],
          "subperiod_signs": sub_signs,
          "pass": bool(np.isfinite(diff) and diff > 0 and sub_signs >= 2)}

    overall = bool(p0["pass"] and p1["pass"] and p2["pass"])
    verdict = "PASS" if overall else "SHELVE"

    # ---------------- 진단 (비액션, 판정 무관) ----------------
    _, td = newey_west_t(y, panel[["pred_disp"]].values, lag=NW_LAG)
    _, tv = newey_west_t(y, panel[["mkt_vol21"]].values, lag=NW_LAG)
    diagnostics = {
        "pred_disp_t_nw": round(float(td[1]), 3),
        "mkt_vol21_t_nw": round(float(tv[1]), 3),
        "corr_rank_stab_trailing_ic6": round(
            float(panel["rank_stab"].corr(panel["trailing_ic6"])), 3),
        "rank_stab_mean": round(float(panel["rank_stab"].mean()), 4),
        "fwd_ic_mean": round(float(panel["fwd_ic"].mean()), 4),
        "ic_crosscheck_corr": round(ic_crosscheck, 3),
        "ic_crosscheck_n": int(len(xchk)),
        "skipped_dates": skipped,
    }

    print(f"[P0] coef={p0['coef']:+.5f} t={p0['t']:+.2f}"
          f" -> {'PASS' if p0['pass'] else 'FAIL'}")
    print(f"[P1] rank_stab coef={p1['coef']:+.5f} t={p1['t']:+.2f}"
          f" (trailing_ic6 t={p1['t_trailing_ic6']:+.2f})"
          f" -> {'PASS' if p1['pass'] else 'FAIL'}")
    print(f"[P2] diff={p2['diff']:+.5f} thirds={p2['subperiod_diffs']}"
          f" signs+={sub_signs} -> {'PASS' if p2['pass'] else 'FAIL'}")
    print(f"[diag] {json.dumps(diagnostics, ensure_ascii=False)}")

    summary = {
        "preregistration": "decision log §S13.45-A",
        "harvest": str(PKL.relative_to(AI_PORT)),
        "sample_n": n,
        "p0": p0,
        "p1": p1,
        "p2": p2,
        "overall_pass": overall,
        "verdict": verdict,
        "vintage": {"pkl_last_date": pkl_last_date,
                    "pkl_mtime": pkl_mtime,
                    "workbook_mtime": workbook_mtime,
                    "pkl_after_workbook": bool(pkl_epoch >= wb_epoch)},
        "diagnostics": diagnostics,
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    df.to_csv(OUT_DIR / "windows.csv")
    print(f"\nVERDICT: {verdict}  n={n}  ({time.time()-t0:.0f}s)  saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
