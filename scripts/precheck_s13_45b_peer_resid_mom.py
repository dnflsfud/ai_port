# -*- coding: utf-8 -*-
"""§S13.45-B 사전점검 (read-only): 피어 잔차 모멘텀의 fwd 21BD 예측력.

결정 로그 §S13.45-B 사전등록의 P0/P1을 실행한다. arm 미실행.

  잔차수익률: r̃ᵢ,t = rᵢ,t − βᵢ,t·r_bm,t.
    β = rolling 126BD cov/var(vs pkl benchmark_returns, min periods 63),
    r = masked USD 일별 수익률 패널.
  피어 집합(리밸런싱 t, 종목 i): trailing 126BD 잔차 상관 상위 10종목
    (자기 제외, pairwise 유효 관측 ≥ 100인 쌍만 후보).
  신호 sᵢ,t = 피어들의 trailing 63BD 누적(합산) 잔차수익률 평균
    (피어 <10이면 가용 피어로, 0이면 NaN).
  P0 (raw IC): 리밸런싱별 단면 Spearman(s, fwd 21BD) 시계열 —
    mean > 0 AND NW(lag 1) t ≥ 2.0 AND 시간순 3분할 ≥ 2/3 양.
  P1 (잔차 직교성): 최종 score 대비 양측 잔차화 partial Spearman
    (§S13.44 residual_ic_row 미러) — 동일 판정식.
  진단(비액션): 자기 63BD 잔차 모멘텀 단면 상관, 섹터 demean IC,
    신호-score 단면 상관, 시점당 유효 신호 종목수.

출력: outputs/s13_45b_peer_resid_mom/summary.json
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
from scripts.precheck_s13_44_volspike_rev import residual_ic_row  # noqa: E402
from scripts.preflight_s13_30_vol_quality import (  # noqa: E402
    _fwd_return,
    _summarise,
)
from scripts.preflight_s13_36_turnover_tilt import _spearman  # noqa: E402

PKL = AI_PORT / "outputs" / "codex_causal_rank_65" / "backtest_result.pkl"
VARIANT = AI_PORT / "variants" / "codex_causal_rank_65.yaml"
OUT_DIR = AI_PORT / "outputs" / "s13_45b_peer_resid_mom"

FWD = 21
BETA_W, BETA_MIN = 126, 63
CORR_W, MIN_PAIR_OBS = 126, 100
PEER_N = 10
MOM_W = 63
NW_LAG = 1
MIN_NAMES = 30
EXPECTED_PKL_LAST = "2026-08-19"


# ---------------------------------------------------------------------------
# 순수 헬퍼 (단위테스트 대상)
# ---------------------------------------------------------------------------

def rolling_beta(returns: pd.DataFrame, bm: pd.Series,
                 window: int = BETA_W, min_periods: int = BETA_MIN
                 ) -> pd.DataFrame:
    """rolling cov/var β — 사전등록 정의(126BD, min 63) 그대로."""
    bm = bm.reindex(returns.index)
    cov = returns.rolling(window, min_periods=min_periods).cov(bm)
    var = bm.rolling(window, min_periods=min_periods).var().replace(0, np.nan)
    return cov.div(var, axis=0)


def residual_returns(returns: pd.DataFrame, bm: pd.Series,
                     window: int = BETA_W, min_periods: int = BETA_MIN
                     ) -> pd.DataFrame:
    """r̃ = r − β·r_bm."""
    bm = bm.reindex(returns.index)
    beta = rolling_beta(returns, bm, window, min_periods)
    return returns.sub(beta.mul(bm, axis=0))


def peer_sets(resid_window: pd.DataFrame, top_n: int = PEER_N,
              min_pair_obs: int = MIN_PAIR_OBS) -> dict:
    """종목별 피어 = trailing 잔차 상관 상위 top_n (자기 제외, 쌍 관측 게이트)."""
    corr = resid_window.corr(min_periods=min_pair_obs)
    out = {}
    for t in corr.columns:
        c = corr[t].drop(t).dropna()
        out[t] = list(c.nlargest(top_n).index) if len(c) else []
    return out


def peer_signal(peer_map: dict, mom_row: pd.Series) -> pd.Series:
    """s = 피어들의 63BD 누적 잔차수익률 평균 — 가용 피어 nanmean, 0이면 NaN."""
    out = {}
    for t, peers in peer_map.items():
        vals = mom_row.reindex(peers).dropna()
        out[t] = float(vals.mean()) if len(vals) else float("nan")
    return pd.Series(out, dtype=float)


def sector_demeaned_ic(signal_row: pd.Series, fwd: pd.Series,
                       sector_map: pd.Series) -> float:
    """섹터내 IC — 섹터 demean 후 Spearman (진단 비액션, 싱글턴 섹터 제외)."""
    df = pd.concat([signal_row.rename("s"), fwd.rename("y"),
                    sector_map.rename("g")], axis=1).dropna()
    df = df[df.groupby("g")["g"].transform("size") >= 2]
    if len(df) < 3:
        return float("nan")
    ds = df["s"] - df.groupby("g")["s"].transform("mean")
    dy = df["y"] - df.groupby("g")["y"].transform("mean")
    return _spearman(ds, dy)


def _nw_mean_t(vals) -> float:
    """시계열 평균의 NW(lag 1) t — 절편만 있는 회귀."""
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return float("nan")
    _, t = newey_west_t(v, np.empty((len(v), 0)), lag=NW_LAG)
    return float(t[0])


def _gate(rows: list, key: str) -> dict:
    """P0/P1 판정식: mean > 0 AND NW t ≥ 2.0 AND 3분할 ≥ 2/3 양."""
    s = _summarise(rows, key)
    t_nw = _nw_mean_t([r[key] for r in rows])
    return {"mean": round(s["mean"], 6), "t": round(t_nw, 3),
            "subperiod_means": [round(m, 6) if np.isfinite(m) else None
                                for m in s["subperiod_means"]],
            "subperiod_signs": s["subperiod_pos"],
            "pass": bool(np.isfinite(s["mean"]) and s["mean"] > 0
                         and np.isfinite(t_nw) and t_nw >= 2.0
                         and s["subperiod_pos"] >= 2)}


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
                  "benchmark_returns"):
        if getattr(res, field, None) is None:
            sys.exit(f"[ABORT] pkl에 {field} 없음 — §S13.45-B 가정 파괴, 보고 요망")
    pkl_last_date = str(pd.Timestamp(max(res.daily_weights.keys())).date())
    print(f"[vintage] pkl last daily_weights = {pkl_last_date}"
          f" (기대 {EXPECTED_PKL_LAST})  pkl mtime = {pkl_mtime}"
          f"  workbook mtime = {workbook_mtime} (UTC 표기)")
    if pkl_last_date != EXPECTED_PKL_LAST:
        sys.exit("[ABORT] pkl 빈티지 불일치 — 진행 없이 보고 요망")
    if pkl_epoch < wb_epoch:
        sys.exit("[ABORT] pkl이 워크북 리프레시보다 오래됨 — 빈티지 불일치, 보고 요망")

    data = UniverseData(cfg.data_path, config=cfg)
    if getattr(data, "returns_masked", None) is None:
        sys.exit("[ABORT] returns_masked 없음 — masked USD 수익률 가정 파괴, 보고 요망")

    returns = data.returns_masked
    preds = res.predictions
    bm = res.benchmark_returns.dropna()
    sector_map = data.meta["sector"].astype(str)
    dates = sorted(res.portfolio_weights.keys())
    print(f"[load] {time.time()-t0:.0f}s  rebalances={len(dates)}"
          f" ({pd.Timestamp(dates[0]).date()} ~ {pd.Timestamp(dates[-1]).date()})"
          f"  returns={returns.shape}  bm n={len(bm)}"
          f" ({bm.index[0].date()} ~ {bm.index[-1].date()})")

    # bm이 첫 리밸런싱일부터 시작 → β(min 63)·상관(≥100 쌍 관측) 번인 동안
    # 초기 리밸런싱은 신호 불가 — 사전등록 정의의 귀결로 표본에서 자연 제외.
    resid = residual_returns(returns, bm)
    mom63 = resid.rolling(MOM_W, min_periods=1).sum()

    rows, skipped, skipped_burnin = [], 0, 0
    for dt in dates:
        if dt not in preds.index or dt not in resid.index:
            skipped += 1
            continue
        score_row = preds.loc[dt]
        scored = score_row.index[score_row.notna()]
        fwd = _fwd_return(returns, dt, FWD)
        if fwd is None or len(scored) < MIN_NAMES:
            skipped += 1
            continue
        pmap = peer_sets(resid.loc[:dt].tail(CORR_W))
        s = peer_signal(pmap, mom63.loc[dt]).reindex(scored)
        if int(s.notna().sum()) < MIN_NAMES:
            skipped_burnin += 1
            continue
        row = {"date": str(pd.Timestamp(dt).date()),
               "n_valid": int(s.notna().sum()),
               **residual_ic_row(s, score_row.loc[scored], fwd),
               "self_mom_corr": _spearman(s, mom63.loc[dt].reindex(scored)),
               "sector_ic": sector_demeaned_ic(s, fwd, sector_map),
               "score_corr": _spearman(s, score_row.loc[scored])}
        rows.append(row)

    n = len(rows)
    print(f"[sample] windows={n}  skipped={skipped}  burn-in skipped={skipped_burnin}"
          f"  first={rows[0]['date'] if rows else None}")

    p0 = _gate(rows, "raw_ic")
    p1 = _gate(rows, "resid_ic")
    overall = bool(p0["pass"] and p1["pass"])
    verdict = "PASS" if overall else "SHELVE"

    sector_sum = _summarise(rows, "sector_ic")
    diagnostics = {
        "self_mom_corr_mean": round(_summarise(rows, "self_mom_corr")["mean"], 4),
        "sector_ic": {"mean": round(sector_sum["mean"], 6),
                      "t": round(_nw_mean_t([r["sector_ic"] for r in rows]), 3),
                      "subperiod_signs": sector_sum["subperiod_pos"]},
        "score_corr_mean": round(_summarise(rows, "score_corr")["mean"], 4),
        "valid_names_mean": round(float(np.mean([r["n_valid"] for r in rows])), 1),
        "valid_names_min": int(min(r["n_valid"] for r in rows)),
        "skipped_dates": skipped,
        "skipped_burnin": skipped_burnin,
    }

    print(f"[P0 raw_ic] mean={p0['mean']:+.4f} t_nw={p0['t']:+.2f}"
          f" thirds+={p0['subperiod_signs']} -> {'PASS' if p0['pass'] else 'FAIL'}")
    print(f"[P1 resid_ic] mean={p1['mean']:+.4f} t_nw={p1['t']:+.2f}"
          f" thirds+={p1['subperiod_signs']} -> {'PASS' if p1['pass'] else 'FAIL'}")
    print(f"[diag] {json.dumps(diagnostics, ensure_ascii=False)}")

    summary = {
        "preregistration": "decision log §S13.45-B",
        "harvest": str(PKL.relative_to(AI_PORT)),
        "sample_n": n,
        "p0": p0,
        "p1": p1,
        "overall_pass": overall,
        "verdict": verdict,
        "vintage": {"pkl_last_date": pkl_last_date,
                    "expected_pkl_last_date": EXPECTED_PKL_LAST,
                    "pkl_mtime": pkl_mtime,
                    "workbook_mtime": workbook_mtime,
                    "pkl_after_workbook": bool(pkl_epoch >= wb_epoch)},
        "diagnostics": diagnostics,
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT_DIR / "windows.csv", index=False)
    print(f"\nVERDICT: {verdict}  n={n}  ({time.time()-t0:.0f}s)  saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
