# -*- coding: utf-8 -*-
"""§S13.49 사전점검 (read-only): score-gate(w_i <= bm_i)의 바인딩 검정.

결정 로그 §S13.49 사전등록(2026-08-21, 측정 전)의 G0/G1/G2를 실행한다.
성능 arm이 아니라 **전제(바인딩) 검정**이다 — arm 미실행, 백테스트 없음,
src/ 무수정.

  G0 (최종 북 바인딩): 리밸일마다 게이트 대상(mu <= threshold, mega·invalid
     제외) 중 |w+ - bm| <= 1e-6 인 비율. 그 비율의 **중앙값 >= 0.05**.
  G1 (반사실 바인딩 — 결정적 검정): 리밸 그리드 24 균등 샘플에서
     `optimize_portfolio`를 `enforce_score_gated_ow` True/False만 바꿔 두 번
     호출. one-way L1 = 0.5*sum|w_on - w_off| 의 **중앙값 >= 0.005**
     (§S13.41 P1·§S13.48-B P1(ii)와 동일 바).
  G2 (게이트 폭 변동성): g_t = (mu <= threshold 인 유효 종목 수)/(유효 종목 수)
     의 **sd >= 0.04** AND (max - min) >= 0.15.

종합 = G0 ∧ G1 ∧ G2 (PASS = arm 설계 자격 / SHELVE).

게이트 대상 정의는 `src/portfolio_optimizer.py:413-422`를 미러한다
(`not np.isfinite(score_i) or score_i <= score_threshold`). 다만 비유한 mu는
`:406-407`에서 `w[i] == bm_weights[i]`로 이미 고정되고, mega(bm >= 0.04)는
`:373-392`의 mega-cap 블록이 `max_ow_per[i] = 0.0`으로 고정할 수 있어 **교락**
이므로 사전등록대로 둘 다 제외한다(G0/G1 한정; G2는 사전등록 정의대로 전
유효 종목 기준).

출력: outputs/s13_49_score_gate_binding/summary.json
"""

import dataclasses
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

AI_PORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_PORT))

VARIANT = Path("variants/codex_causal_rank_65.yaml")
PKL = Path("outputs/codex_causal_rank_65/backtest_result.pkl")
OUT_DIR = Path("outputs/s13_49_score_gate_binding")

BIND_TOL = 1e-6          # |w - bm| <= BIND_TOL 이면 경계 바인딩
G0_BAR = 0.05            # 바인딩 비율 중앙값
N_MVO_SAMPLES = 24
G1_BAR = 0.005           # one-way L1 중앙값 (S13.41/S13.48-B와 동일 바)
G2_SD_BAR = 0.04         # sd(g_t)
G2_RANGE_BAR = 0.15      # max(g_t) - min(g_t)
EXPECTED_VINTAGE_DATE = "2026-08-19"
EXPECTED_WORKBOOK_MTIME = "2026-08-19 13:49"
EXPECTED_INDEX_MTIME = "2026-08-21 11:16"

MIN_HIST_ROWS = 30       # estimate_covariance의 실질 최소 관측 수 (§S13.48-B 동일)
OW_TOL = 1e-9            # w > bm + OW_TOL 이면 실제 OW


# ---------------------------------------------------------------------------
# 순수 헬퍼 (단위테스트 대상)
# ---------------------------------------------------------------------------

def gate_targets(mu: np.ndarray, bm: np.ndarray, threshold: float,
                 mega_bm_thr: float) -> np.ndarray:
    """score-gate가 걸리는 종목의 불리언 마스크(교락 종목 제외).

    포함: isfinite(mu) 이고 mu <= threshold (`portfolio_optimizer.py:421`의
    `score_i <= score_threshold` 부등호를 그대로 미러 — 등호 포함).
    제외: bm >= mega_bm_thr (mega-cap 블록이 w를 bm에 고정할 수 있음),
          비유한 mu (`:406-407`에서 w == bm 이 별도로 강제됨).
    """
    mu = np.asarray(mu, dtype=float)
    bm = np.asarray(bm, dtype=float)
    finite = np.isfinite(mu)
    gated = np.zeros(mu.shape, dtype=bool)
    gated[finite] = mu[finite] <= float(threshold)
    return gated & ~(bm >= float(mega_bm_thr))


def binding_share(w: np.ndarray, bm: np.ndarray, mask: np.ndarray,
                  tol: float) -> float:
    """mask 종목 중 |w - bm| <= tol 인 비율. mask가 비면 nan."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return float("nan")
    gap = np.abs(np.asarray(w, dtype=float) - np.asarray(bm, dtype=float))[mask]
    return float(np.mean(gap <= float(tol)))


def gate_width(mu: np.ndarray, threshold: float) -> float:
    """유효(유한) mu 중 mu <= threshold 인 비율. 유효 0이면 nan."""
    mu = np.asarray(mu, dtype=float)
    finite = np.isfinite(mu)
    if not finite.any():
        return float("nan")
    return float(np.mean(mu[finite] <= float(threshold)))


def _mtime_kst(path) -> str:
    return (pd.Timestamp(Path(path).stat().st_mtime, unit="s", tz="UTC")
            .tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S"))


def _quantiles(values) -> dict:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {k: float("nan") for k in ("n", "min", "p10", "median", "p90", "max")}
    return {"n": int(x.size), "min": float(np.min(x)),
            "p10": float(np.percentile(x, 10)), "median": float(np.median(x)),
            "p90": float(np.percentile(x, 90)), "max": float(np.max(x))}


def _dist(series) -> dict:
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {k: float("nan")
                for k in ("min", "p10", "median", "p90", "max", "mean")}
    return {"min": float(np.min(x)), "p10": float(np.percentile(x, 10)),
            "median": float(np.median(x)), "p90": float(np.percentile(x, 90)),
            "max": float(np.max(x)), "mean": float(np.mean(x))}


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

def main() -> None:
    from run_variant import compose_config, load_manifest
    from src.backtest import get_benchmark_fn, get_sector_map
    from src.data_loader import UniverseData
    from src.portfolio_optimizer import estimate_covariance, optimize_portfolio

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cfg = compose_config(load_manifest(VARIANT))
    pkl_mtime = _mtime_kst(PKL)
    workbook_mtime = _mtime_kst(cfg.data_path)
    index_mtime = _mtime_kst(cfg.fx_source_path)
    res = pickle.load(open(PKL, "rb"))
    last_daily = str(pd.Timestamp(sorted(res.daily_weights)[-1]).date())

    # ---------- 실행 선행 검증 (빈티지 쌍 — 불일치 시 진행 금지) ----------
    print(f"[vintage] pkl {pkl_mtime}  workbook {workbook_mtime}"
          f"  Index {index_mtime}  last daily_weights {last_daily}")
    if workbook_mtime[:16] != EXPECTED_WORKBOOK_MTIME:
        sys.exit(f"[ABORT] 워크북 mtime 불일치 — 기대 {EXPECTED_WORKBOOK_MTIME}, "
                 f"실측 {workbook_mtime}. 진행하지 않고 보고만.")
    if index_mtime[:16] != EXPECTED_INDEX_MTIME:
        sys.exit(f"[ABORT] Index.xlsx mtime 불일치 — 기대 {EXPECTED_INDEX_MTIME}, "
                 f"실측 {index_mtime}. 진행하지 않고 보고만.")
    if last_daily != EXPECTED_VINTAGE_DATE:
        sys.exit(f"[ABORT] 빈티지 불일치 — 기대 {EXPECTED_VINTAGE_DATE}, "
                 f"last daily_weights {last_daily}. 진행하지 않고 보고만.")

    # ---------- production 전제 (게이트 ON · 절대 임계 0.0) ----------
    if getattr(cfg, "enforce_score_gated_ow", False) is not True:
        sys.exit(f"[ABORT] enforce_score_gated_ow={cfg.enforce_score_gated_ow} "
                 f"— production 전제(게이트 ON) 파괴, 진행하지 않고 보고만.")
    threshold = float(cfg.score_threshold_for_ow)
    if threshold != 0.0:
        sys.exit(f"[ABORT] score_threshold_for_ow={threshold} — 사전등록 전제"
                 f"(절대 0.0) 파괴, 진행하지 않고 보고만.")
    mega_thr = float(cfg.mega_cap_bm_threshold)

    data = UniverseData(cfg.data_path, config=cfg)
    if getattr(data, "returns_masked", None) is None:
        sys.exit("[ABORT] returns_masked 없음 — masked USD 수익률 가정 파괴, 보고 요망")

    rebal_dates = sorted(pd.Timestamp(k) for k in res.portfolio_weights)
    tickers = list(res.portfolio_weights[rebal_dates[0]].index)
    returns = data.returns_masked
    ret_idx = returns.index
    # 공분산 입력은 production 경로와 동일해야 한다: backtest.py:1989-1991이
    # data.raw_returns를 returns.index/tickers로 reindex해 risk_returns로 넘기고
    # :1432에서 risk_source가 된다. §S13.48-B는 명세대로 returns_masked를 썼다가
    # 독립 검증에서 SPEC-CODE MISMATCH로 지적됐다 — 여기서는 측정 전에 바로잡는다.
    risk_returns = getattr(data, "raw_returns", None)
    if risk_returns is None:
        sys.exit("[ABORT] data.raw_returns 없음 — production 공분산 입력 재현 불가, 보고 요망")
    cov_returns = risk_returns.reindex(index=ret_idx, columns=tickers)
    preds = res.predictions
    bm_fn = get_benchmark_fn(data, tickers, config=cfg)
    sector_map = get_sector_map(data)
    daily_idx = sorted(pd.Timestamp(k) for k in res.daily_weights)
    lookback = int(cfg.cov_lookback)
    print(f"[load] {time.time()-t0:.0f}s  rebalances={len(rebal_dates)}"
          f" ({rebal_dates[0].date()} ~ {rebal_dates[-1].date()})"
          f"  tickers={len(tickers)}  returns={returns.shape}"
          f"  cov_lookback={lookback}  threshold={threshold}"
          f"  mega_thr={mega_thr}")

    # 재-MVO 대상 = 리밸일 균등 24 샘플 (사전등록)
    sample_pos = set(int(i) for i in np.linspace(
        0, len(rebal_dates) - 1, N_MVO_SAMPLES, dtype=int))

    rows, skip_reasons, mvo_cache, pooled_gap = [], {}, {}, []

    def _skip(reason):
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    # ---------------- G0 / G2: 전 리밸일 ----------------
    for i, dt in enumerate(rebal_dates):
        if dt not in preds.index:
            _skip("date_not_in_predictions")
            continue
        mu = preds.loc[dt, tickers].to_numpy(dtype=float)
        bm_w = np.asarray(bm_fn(dt, tickers, len(tickers)), dtype=float)
        w_plus = res.portfolio_weights[dt].reindex(tickers).fillna(0.0).to_numpy()
        mask = gate_targets(mu, bm_w, threshold, mega_thr)
        pooled_gap.extend((bm_w - w_plus)[mask].tolist())
        rows.append({
            "date": str(dt.date()),
            "n_finite_mu": int(np.isfinite(mu).sum()),
            "n_invalid_mu": int((~np.isfinite(mu)).sum()),
            "n_mega": int((bm_w >= mega_thr).sum()),
            "n_gate_targets": int(mask.sum()),
            "share_binding": binding_share(w_plus, bm_w, mask, BIND_TOL),
            "g_t": gate_width(mu, threshold),
            "n_ow": int((w_plus > bm_w + OW_TOL).sum()),
            "n_ow_gated": int((w_plus > bm_w + OW_TOL)[mask].sum()),
            "min_gap_gated": (float(np.min((bm_w - w_plus)[mask]))
                              if mask.any() else float("nan")),
        })

        if i not in sample_pos:
            continue
        # ---- 재-MVO 샘플 준비 (§S13.48-B 재-MVO 블록과 동일 배선) ----
        if dt not in ret_idx:
            _skip("mvo_date_not_in_returns")
            continue
        pos = ret_idx.get_loc(dt)
        # backtest.py:1537-1538 축자: iloc[t_idx-cov_lookback : t_idx] — t 제외
        hist = cov_returns.iloc[max(0, pos - lookback):pos]
        if len(hist) < MIN_HIST_ROWS:
            _skip("mvo_hist_rows_lt_30")
            continue
        pred_row = preds.loc[dt, tickers]
        if pred_row.notna().sum() < 10:
            _skip("mvo_pred_coverage")
            continue
        prior = [x for x in daily_idx if x < dt]
        if not prior:
            _skip("mvo_no_prev_daily")
            continue
        prev_w = res.daily_weights[prior[-1]].reindex(tickers).fillna(0.0).values
        cov = np.asarray(
            estimate_covariance(hist, bm_weights=bm_w, config=cfg), dtype=float)
        mvo_cache[dt] = (pred_row, cov, prev_w, bm_w, mask)

    df = pd.DataFrame(rows)
    n_used = int(len(df))
    if n_used == 0:
        sys.exit("[ABORT] 사용 가능한 리밸일 0건 — 보고 요망")

    med_share = float(df["share_binding"].median())
    g0 = bool(np.isfinite(med_share) and med_share >= G0_BAR)

    g_series = df["g_t"].to_numpy(dtype=float)
    g_valid = g_series[np.isfinite(g_series)]
    g_sd = float(np.std(g_valid, ddof=1)) if g_valid.size > 1 else float("nan")
    g_range = float(g_valid.max() - g_valid.min()) if g_valid.size else float("nan")
    g_median = float(np.median(g_valid)) if g_valid.size else float("nan")
    g2 = bool(np.isfinite(g_sd) and g_sd >= G2_SD_BAR
              and np.isfinite(g_range) and g_range >= G2_RANGE_BAR)

    # ---------------- G1: 반사실 재-MVO 24 샘플 ----------------
    cfg_off = dataclasses.replace(cfg, enforce_score_gated_ow=False)
    replace_diff = [
        f.name for f in dataclasses.fields(cfg)
        if f.name != "enforce_score_gated_ow"
        and repr(getattr(cfg_off, f.name)) != repr(getattr(cfg, f.name))
    ]
    if replace_diff:
        sys.exit(f"[ABORT] dataclasses.replace가 다른 필드까지 변경: {replace_diff}"
                 f" — 반사실의 유일-차이 전제 파괴, 보고 요망.")
    if cfg_off.enforce_score_gated_ow is not False:
        sys.exit("[ABORT] cfg_off.enforce_score_gated_ow가 False가 아님 — 보고 요망.")

    mvo_rows = []
    for dt in sorted(mvo_cache):
        pred_row, cov, prev_w, bm_w, mask = mvo_cache[dt]
        d_on, d_off = {}, {}
        w_on = optimize_portfolio(pred_row, cov, prev_weights=prev_w,
                                  sector_map=sector_map, bm_weights=bm_w,
                                  config=cfg, diagnostics=d_on)
        w_off = optimize_portfolio(pred_row, cov, prev_weights=prev_w,
                                   sector_map=sector_map, bm_weights=bm_w,
                                   config=cfg_off, diagnostics=d_off)
        if d_on.get("used_fallback", False) or d_off.get("used_fallback", False):
            _skip("mvo_solver_fallback")
            continue
        w_on = np.asarray(w_on, dtype=float)
        w_off = np.asarray(w_off, dtype=float)
        mvo_rows.append({
            "date": str(dt.date()),
            "l1": float(0.5 * np.abs(w_on - w_off).sum()),
            "active_share_on": float(0.5 * np.abs(w_on - bm_w).sum()),
            "active_share_off": float(0.5 * np.abs(w_off - bm_w).sum()),
            "n_gate_targets": int(mask.sum()),
            "n_ow_released": int((w_off > bm_w + OW_TOL)[mask].sum()),
            "n_ow_on": int((w_on > bm_w + OW_TOL).sum()),
            "n_ow_off": int((w_off > bm_w + OW_TOL).sum()),
        })
        r = mvo_rows[-1]
        print(f"[G1] {r['date']}  L1 {r['l1']:.5f}  AS on/off "
              f"{r['active_share_on']:.1%}/{r['active_share_off']:.1%}"
              f"  released OW {r['n_ow_released']}/{r['n_gate_targets']}")

    mdf = pd.DataFrame(mvo_rows)
    if mdf.empty:
        sys.exit("[ABORT] 재-MVO 유효 샘플 0건 — 보고 요망")
    med_l1 = float(mdf["l1"].median())
    g1 = bool(med_l1 >= G1_BAR)

    overall = bool(g0 and g1 and g2)
    verdict = "PASS" if overall else "SHELVE"

    gap_q = _quantiles(pooled_gap)
    print(f"[G0] 바인딩 비율 중앙값 {med_share:.4f} (바 {G0_BAR})  n={n_used}"
          f"  -> {'PASS' if g0 else 'FAIL'}")
    print(f"[G1] one-way L1 중앙값 {med_l1:.5f} (바 {G1_BAR})  n={len(mdf)}"
          f"  -> {'PASS' if g1 else 'FAIL'}")
    print(f"[G2] sd(g_t) {g_sd:.4f} (바 {G2_SD_BAR})  range {g_range:.4f}"
          f" (바 {G2_RANGE_BAR})  median {g_median:.4f}"
          f"  -> {'PASS' if g2 else 'FAIL'}")
    print(f"[diag] bm-w+ (게이트 대상) min {gap_q['min']:.2e}"
          f" p10 {gap_q['p10']:.2e} median {gap_q['median']:.2e}"
          f" p90 {gap_q['p90']:.2e}"
          f"  |  게이트 대상 수 중앙값 {float(df['n_gate_targets'].median()):.1f}"
          f"  실제 OW 수 중앙값 {float(df['n_ow'].median()):.1f}")
    print(f"[gates] G0={g0} G1={g1} G2={g2}")
    print(f"[VERDICT] {verdict}")

    summary = {
        "preregistration": "decision log §S13.49 (2026-08-21)",
        "vintage": {
            "expected_workbook_mtime": EXPECTED_WORKBOOK_MTIME,
            "expected_index_mtime": EXPECTED_INDEX_MTIME,
            "expected_last_daily_weights": EXPECTED_VINTAGE_DATE,
            "pkl_mtime": pkl_mtime,
            "workbook_mtime": workbook_mtime,
            "index_mtime": index_mtime,
            "last_daily_weights": last_daily,
        },
        "config": {
            "enforce_score_gated_ow": bool(cfg.enforce_score_gated_ow),
            "score_threshold_for_ow": threshold,
            "mega_cap_bm_threshold": mega_thr,
            "mega_cap_protection_enabled": bool(cfg.mega_cap_protection_enabled),
            "mega_cap_funding_score_max": float(cfg.mega_cap_funding_score_max),
            "cov_lookback": lookback,
            "replace_field_diff": replace_diff,
        },
        "n_rebalances": int(len(rebal_dates)),
        "n_used": n_used,
        "n_skipped": int(sum(skip_reasons.values())),
        "skip_reasons": skip_reasons,
        "g0": {
            "median_binding_share": med_share,
            "mean_binding_share": float(df["share_binding"].mean()),
            "threshold": G0_BAR,
            "n_dates_any_binding": int((df["share_binding"] > 0).sum()),
            "binding_share_dist": _dist(df["share_binding"]),
            "pass": g0,
        },
        "g1": {
            "n_samples_requested": N_MVO_SAMPLES,
            "n_samples_used": int(len(mdf)),
            "median_l1": med_l1,
            "threshold": G1_BAR,
            "l1_dist": _dist(mdf["l1"]),
            "median_active_share_on": float(mdf["active_share_on"].median()),
            "median_active_share_off": float(mdf["active_share_off"].median()),
            "median_n_ow_released": float(mdf["n_ow_released"].median()),
            "total_n_ow_released": int(mdf["n_ow_released"].sum()),
            "pass": g1,
        },
        "g2": {
            "sd_g": g_sd,
            "range_g": g_range,
            "median_g": g_median,
            "sd_threshold": G2_SD_BAR,
            "range_threshold": G2_RANGE_BAR,
            "g_dist": _dist(df["g_t"]),
            "q_candidate": round(g_median, 3) if np.isfinite(g_median) else None,
            "pass": g2,
        },
        "diagnostics": {
            "bind_tol": BIND_TOL,
            "ow_tol": OW_TOL,
            "gap_bm_minus_w_gated": gap_q,
            "n_gate_targets_dist": _dist(df["n_gate_targets"]),
            "n_ow_dist": _dist(df["n_ow"]),
            "n_ow_gated_total": int(df["n_ow_gated"].sum()),
            "n_invalid_mu_dist": _dist(df["n_invalid_mu"]),
            "n_mega_dist": _dist(df["n_mega"]),
            "n_tickers": int(len(tickers)),
        },
        "notes": [
            "게이트 대상 = portfolio_optimizer.py:413-422 미러(mu <= threshold, "
            "등호 포함)에서 mega(bm >= mega_cap_bm_threshold)와 비유한 mu를 제외 "
            "— 각각 :373-392 mega-cap 블록과 :406-407 w==bm 핀이 교락시키기 "
            "때문(사전등록 G0 정의).",
            "G2의 g_t는 사전등록 정의대로 mega·invalid 제외 없이 전 유효 종목 기준.",
            "G1의 게이트 OFF cfg는 dataclasses.replace(cfg, "
            "enforce_score_gated_ow=False)이며, 그 외 전 필드가 repr 동일함을 "
            "실행 시 검증(replace_field_diff == []). 그 외 인자(mu·cov·"
            "prev_weights·bm·sector_map)는 동일 객체를 재사용.",
            "공분산 입력 = data.raw_returns를 returns.index/tickers로 reindex한 "
            "패널(production backtest.py:1989-1991 -> :1432 risk_source와 동일). "
            "§S13.48-B가 returns_masked를 써서 SPEC-CODE MISMATCH 지적을 받은 건을 "
            "측정 전에 바로잡은 것이다(결정 로그 §S13.49 측정 전 정정).",
            "hist 창 = returns.iloc[pos-cov_lookback:pos] — t 자신 제외 "
            "(backtest.py:1537-1538 hist_start:t_idx 축자 일치).",
        ],
        "overall_pass": overall,
        "verdict": verdict,
        "runtime_s": round(time.time() - t0, 1),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    df.to_csv(OUT_DIR / "by_date.csv", index=False)
    mdf.to_csv(OUT_DIR / "mvo_samples.csv", index=False)
    print(f"\nVERDICT: {verdict}  ({time.time()-t0:.0f}s)  saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
