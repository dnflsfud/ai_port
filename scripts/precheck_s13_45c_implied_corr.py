# -*- coding: utf-8 -*-
"""§S13.45-C 사전점검 (read-only): implied correlation → Σ 비대각 예측력.

결정 로그 §S13.45-C 사전등록(2026-08-19)의 P0'/P1'/P2'를 실행한다. arm 미실행.

  ICorr_t = (σ_idx² − Σw²σ²) / ((Σwσ)² − Σw²σ²), σ = iv30 시트(단위 로그 확인),
    idx = SPX Index 열, w = 당일 cap-weighted bm 비중을 IV 유효 종목으로 renorm.
    유효 bm 비중 합 ≥ 60%인 날만 유효, [0,1] 밖 → NaN(발생률 보고).
  비교자 ρ̄_real,t = 동일 항등식을 trailing 126BD 실현 vol로, 타깃 = 전진
    21BD(t+1..t+21) 실현 ρ̄ — 둘 다 동일 w renorm, 표본 = 일별 중첩(ICorr 유효일).
  P0' (증분): 전진 ρ̄ ~ [trailing ρ̄ + ICorr] OLS, ICorr NW(lag 21) t ≥ 2 AND 양.
  P1' (재료성): 시간순 4분할 expanding OOS RMSE, [trailing+ICorr] vs [trailing
     단독] — 전체(3 eval 합산) 개선 ≥ 5% AND 3개 eval 구간 전부 개선.
  P2' (표본): ICorr 유효 일수 ≥ 750. 종합 = P0'∧P1'∧P2' (PASS/SHELVE).

출력: outputs/s13_45c_implied_corr/summary.json
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

from scripts.precheck_s13_43_regime import newey_west_t  # noqa: E402

PKL = AI_PORT / "outputs" / "codex_causal_rank_65" / "backtest_result.pkl"
VARIANT = AI_PORT / "variants" / "codex_causal_rank_65.yaml"
OUT_DIR = AI_PORT / "outputs" / "s13_45c_implied_corr"

IV_SHEET = "iv30"
TRAIL = 126
FWD = 21
MIN_COVERAGE = 0.60
P2_MIN_DAYS = 750
NW_LAG = 21
T_GATE = 2.0
P1_MIN_IMPROVEMENT = 0.05
N_SPLITS = 4
EXPECTED_VINTAGE_DATE = "2026-08-19"  # 08-19 빈티지 pkl — 불일치 시 중단·보고


# ---------------------------------------------------------------------------
# 순수 헬퍼 (단위테스트 대상)
# ---------------------------------------------------------------------------

def implied_avg_corr(sig_idx: float, sig: np.ndarray, w: np.ndarray) -> float:
    """사전등록 항등식 ρ̄ = (σ_idx² − Σw²σ²)/((Σwσ)² − Σw²σ²). denom≤0 → NaN."""
    sig = np.asarray(sig, float)
    w = np.asarray(w, float)
    diag = float(np.sum(w ** 2 * sig ** 2))
    denom = float(np.sum(w * sig)) ** 2 - diag
    if not np.isfinite(denom) or denom <= 1e-12:
        return float("nan")
    return (float(sig_idx) ** 2 - diag) / denom


def realized_avg_corr(window_returns: np.ndarray, w: np.ndarray) -> float:
    """동일 항등식의 실현판: σᵢ = 창내 std(ddof=1), σ_idx = (창수익률@w)의 std.

    σ_p² = wᵀSw (S = 창 표본공분산)이므로 결과는 창 표본 pairwise corr의
    w-가중 평균과 정확히 일치한다."""
    r = np.asarray(window_returns, float)
    sig = r.std(axis=0, ddof=1)
    sig_p = float((r @ np.asarray(w, float)).std(ddof=1))
    return implied_avg_corr(sig_p, sig, w)


def renorm_valid_weights(w: np.ndarray, valid: np.ndarray,
                         min_coverage: float = MIN_COVERAGE):
    """(coverage, renormed w|None) — 유효 종목 bm 비중 합이 게이트 미달이면 None."""
    w = np.asarray(w, float)
    valid = np.asarray(valid, bool)
    coverage = float(w[valid].sum())
    if not np.isfinite(coverage) or coverage < min_coverage:
        return coverage, None
    return coverage, w[valid] / coverage


def clip_unit_interval(x: float) -> float:
    """사전등록: [0,1] 밖 값은 NaN (경계 포함 유지)."""
    return float(x) if np.isfinite(x) and 0.0 <= x <= 1.0 else float("nan")


def _fit_predict(x_tr, y_tr, x_ev):
    design = np.column_stack([np.ones(len(x_tr)), x_tr])
    coef, *_ = np.linalg.lstsq(design, y_tr, rcond=None)
    return np.column_stack([np.ones(len(x_ev)), x_ev]) @ coef


def expanding_oos_rmse(y: np.ndarray, x_base: np.ndarray, x_aug: np.ndarray,
                       n_splits: int = N_SPLITS) -> dict:
    """시간순 n분할 expanding OOS: 1→2, 1-2→3, 1-3→4 예측 RMSE (base vs aug)."""
    y = np.asarray(y, float)
    blocks = np.array_split(np.arange(len(y)), n_splits)
    per_split, se_b, se_a = [], [], []
    for k in range(1, n_splits):
        tr = np.concatenate(blocks[:k])
        ev = blocks[k]
        eb = y[ev] - _fit_predict(x_base[tr], y[tr], x_base[ev])
        ea = y[ev] - _fit_predict(x_aug[tr], y[tr], x_aug[ev])
        rb = float(np.sqrt(np.mean(eb ** 2)))
        ra = float(np.sqrt(np.mean(ea ** 2)))
        per_split.append({"n_train": int(len(tr)), "n_eval": int(len(ev)),
                          "rmse_base": round(rb, 6), "rmse_aug": round(ra, 6),
                          "improvement": round(1.0 - ra / rb, 4)})
        se_b.append(eb ** 2)
        se_a.append(ea ** 2)
    rmse_b = float(np.sqrt(np.mean(np.concatenate(se_b))))
    rmse_a = float(np.sqrt(np.mean(np.concatenate(se_a))))
    return {"rmse_base": rmse_b, "rmse_aug": rmse_a,
            "improvement": 1.0 - rmse_a / rmse_b, "per_split": per_split}


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

def _kst(path: Path) -> str:
    return pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC") \
        .tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    from run_variant import compose_config, load_manifest
    from src.backtest import get_benchmark_fn
    from src.data_loader import UniverseData

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cfg = compose_config(load_manifest(VARIANT))
    workbook_mtime = _kst(Path(cfg.data_path))
    pkl_mtime = _kst(PKL)

    res = pickle.load(open(PKL, "rb"))
    last_daily = str(pd.Timestamp(sorted(res.daily_weights)[-1]).date())
    print(f"[vintage] pkl mtime {pkl_mtime}  workbook mtime {workbook_mtime}"
          f"  last daily_weights {last_daily}")
    if last_daily != EXPECTED_VINTAGE_DATE:
        sys.exit(f"[ABORT] 빈티지 불일치 — 기대 {EXPECTED_VINTAGE_DATE}, "
                 f"daily {last_daily}. 진행하지 않고 보고만.")

    data = UniverseData(cfg.data_path, config=cfg)
    if getattr(data, "returns_masked", None) is None:
        sys.exit("[ABORT] returns_masked 없음 — masked USD 수익률 가정 파괴, 보고 요망")
    returns = data.returns_masked
    dates = returns.index
    tickers = list(data.tickers)
    bm_fn = get_benchmark_fn(data, tickers, config=cfg)

    # ---------- 데이터 가정 검증 (§9: 어긋나면 중단·보고) ----------
    if IV_SHEET not in data.raw:
        sys.exit(f"[ABORT] 시트 {IV_SHEET} 없음 — §S13.37 계층 가정 파괴, 보고 요망")
    iv_raw = data.raw[IV_SHEET].apply(pd.to_numeric, errors="coerce")
    iv_raw.index = pd.to_datetime(iv_raw.index)
    spx_cols = [c for c in iv_raw.columns if "SPX" in str(c)]
    if not spx_cols:
        sys.exit(f"[ABORT] {IV_SHEET}(raw)에 SPX 열 없음 — 보고 요망")
    stock_cols = {str(c).split(" ")[0]: c for c in iv_raw.columns
                  if c not in spx_cols}
    iv_panel = pd.DataFrame(
        {t: iv_raw[stock_cols[t]] for t in tickers if t in stock_cols}
    ).reindex(index=dates, columns=tickers)
    spx_iv = iv_raw[spx_cols[0]].reindex(dates)

    # 단위 확인·통일: % 표기(중앙값 > 3)면 소수로 변환. 항등식은 스케일
    # 불변이지만 로그·진단 일관성을 위해 소수로 통일한다.
    med = float(np.nanmedian(iv_panel.values))
    unit = "percent_annualized" if med > 3.0 else "decimal_annualized"
    if unit == "percent_annualized":
        iv_panel /= 100.0
        spx_iv /= 100.0
    n_iv_names = int(iv_panel.notna().any().sum())
    print(f"[unit] {IV_SHEET} median {med:.2f} -> {unit} (소수 통일)"
          f"  종목 열 {n_iv_names}/{len(tickers)}  SPX 열 '{spx_cols[0]}'")

    # ---------- 일별 ICorr / trailing ρ̄ / 전진 21BD ρ̄ ----------
    ret_vals = returns.values.astype(float)
    notna = returns.notna()
    dense_tr = (notna.rolling(TRAIL).sum() == TRAIL).values
    dense_fw = (notna.rolling(FWD).sum() == FWD).shift(-FWD).fillna(False).values
    iv_vals = iv_panel.values
    spx_vals = spx_iv.values.astype(float)

    rows = []
    n_coverage_ok = n_outside = 0
    weight_drop = []
    for p in range(TRAIL - 1, len(dates)):
        iv_row = iv_vals[p]
        if not np.isfinite(spx_vals[p]) or not np.isfinite(iv_row).any():
            continue
        w_full = np.asarray(bm_fn(dates[p], tickers, len(tickers)), float)
        v_iv = np.isfinite(iv_row) & (w_full > 0)
        coverage, w_iv = renorm_valid_weights(w_full, v_iv)
        if w_iv is None:
            rows.append({"date": dates[p], "coverage": coverage,
                         "icorr": np.nan, "rho_trail": np.nan, "rho_fwd": np.nan})
            continue
        n_coverage_ok += 1
        icorr_raw = implied_avg_corr(spx_vals[p], iv_row[v_iv], w_iv)
        icorr = clip_unit_interval(icorr_raw)
        if np.isfinite(icorr_raw) and np.isnan(icorr):
            n_outside += 1

        # 실현 항등식은 창내 dense 종목 교집합으로 renorm(동일 w 절차) —
        # 항등식이 정확히 성립하도록 유지하고 탈락 비중은 진단으로 보고.
        v_c = v_iv & dense_tr[p] & dense_fw[p]
        cov_c, w_c = renorm_valid_weights(w_full, v_c)
        rho_trail = rho_fwd = np.nan
        if w_c is not None:
            idx = np.where(v_c)[0]
            rho_trail = realized_avg_corr(ret_vals[p - TRAIL + 1:p + 1, idx], w_c)
            rho_fwd = realized_avg_corr(ret_vals[p + 1:p + 1 + FWD, idx], w_c)
            weight_drop.append(coverage - cov_c)
        rows.append({"date": dates[p], "coverage": coverage, "icorr": icorr,
                     "rho_trail": rho_trail, "rho_fwd": rho_fwd})

    daily = pd.DataFrame(rows).set_index("date").sort_index()
    valid = daily[daily["icorr"].notna()]
    valid_days = int(len(valid))
    first_valid = str(valid.index[0].date()) if valid_days else None
    outside_rate = n_outside / max(n_coverage_ok, 1)
    print(f"[ICorr] 유효 {valid_days}일 (시작 {first_valid})"
          f"  coverage mean {float(valid['coverage'].mean()):.3f}"
          f"  [0,1]밖 {n_outside}/{n_coverage_ok} ({outside_rate:.2%})")

    sample = valid.dropna(subset=["rho_trail", "rho_fwd"])
    n_sample = int(len(sample))
    print(f"[sample] 회귀 표본 {n_sample}일"
          + (f" ({sample.index[0].date()} ~ {sample.index[-1].date()})"
             if n_sample else ""))

    # ---------------- P2': 표본 충분성 ----------------
    p2 = bool(valid_days >= P2_MIN_DAYS)
    print(f"[P2'] valid_days={valid_days} (게이트 {P2_MIN_DAYS}) -> "
          f"{'PASS' if p2 else 'FAIL'}")

    # ---------------- P0': 증분 (NW lag21) ----------------
    x2 = sample[["rho_trail", "icorr"]].values
    coef, tstats = newey_west_t(sample["rho_fwd"].values, x2, lag=NW_LAG)
    t_ic = float(tstats[2])
    p0 = bool(t_ic >= T_GATE and coef[2] > 0)
    print(f"[P0'] ICorr coef={coef[2]:+.4f} t_nw={t_ic:+.2f}"
          f" (trailing coef={coef[1]:+.4f} t={float(tstats[1]):+.2f})"
          f" -> {'PASS' if p0 else 'FAIL'}")

    # ---------------- P1': expanding OOS RMSE ----------------
    oos = expanding_oos_rmse(sample["rho_fwd"].values,
                             sample[["rho_trail"]].values, x2)
    all_improve = all(s["improvement"] > 0 for s in oos["per_split"])
    p1 = bool(oos["improvement"] >= P1_MIN_IMPROVEMENT and all_improve)
    print(f"[P1'] RMSE base {oos['rmse_base']:.5f} aug {oos['rmse_aug']:.5f}"
          f"  개선 {oos['improvement']:+.2%}"
          f"  per-split {[s['improvement'] for s in oos['per_split']]}"
          f" -> {'PASS' if p1 else 'FAIL'}")

    overall = bool(p0 and p1 and p2)
    verdict = "PASS" if overall else "SHELVE"

    # ---------------- 진단 (비액션) ----------------
    by_year = {
        str(y): {"valid_days": int(len(g)),
                 "mean_coverage": round(float(g["coverage"].mean()), 3),
                 "mean_icorr": round(float(g["icorr"].mean()), 3)}
        for y, g in valid.groupby(valid.index.year)
    }
    diagnostics = {
        "iv_unit": unit,
        "iv_median_raw": round(med, 3),
        "n_iv_stock_columns": n_iv_names,
        "first_valid_icorr_date": first_valid,
        "coverage_mean": round(float(valid["coverage"].mean()), 4),
        "coverage_min": round(float(valid["coverage"].min()), 4),
        "coverage_by_year": by_year,
        "outside_unit_interval_rate": round(outside_rate, 4),
        "n_outside_unit_interval": int(n_outside),
        "corr_icorr_vs_trailing": round(float(
            sample["icorr"].corr(sample["rho_trail"])), 4),
        "mean_spread_icorr_minus_trailing": round(float(
            (sample["icorr"] - sample["rho_trail"]).mean()), 4),
        "mean_icorr": round(float(sample["icorr"].mean()), 4),
        "mean_rho_trail": round(float(sample["rho_trail"].mean()), 4),
        "mean_rho_fwd": round(float(sample["rho_fwd"].mean()), 4),
        "target_autocorr_lag1": round(float(sample["rho_fwd"].autocorr(1)), 4),
        "target_autocorr_lag21": round(float(sample["rho_fwd"].autocorr(21)), 4),
        "mean_bm_weight_dropped_for_realized": round(
            float(np.mean(weight_drop)) if weight_drop else float("nan"), 5),
    }
    print(f"[diag] corr(ICorr, trailing)={diagnostics['corr_icorr_vs_trailing']:+.3f}"
          f"  premium(ICorr−trailing)={diagnostics['mean_spread_icorr_minus_trailing']:+.4f}"
          f"  target AC1/AC21 {diagnostics['target_autocorr_lag1']:+.3f}/"
          f"{diagnostics['target_autocorr_lag21']:+.3f}")

    summary = {
        "preregistration": "decision log §S13.45-C (2026-08-19)",
        "vintage": {"expected_date": EXPECTED_VINTAGE_DATE,
                    "pkl_mtime": pkl_mtime,
                    "workbook_mtime": workbook_mtime,
                    "last_daily_weights": last_daily,
                    "match": last_daily == EXPECTED_VINTAGE_DATE},
        "sample_n_days": n_sample,
        "p0": {"coef": round(float(coef[2]), 6), "t": round(t_ic, 3),
               "coef_trailing": round(float(coef[1]), 6),
               "t_trailing": round(float(tstats[1]), 3),
               "nw_lag": NW_LAG, "threshold_t": T_GATE, "pass": p0},
        "p1": {"rmse_base": round(oos["rmse_base"], 6),
               "rmse_aug": round(oos["rmse_aug"], 6),
               "improvement": round(oos["improvement"], 4),
               "per_split": oos["per_split"],
               "threshold": P1_MIN_IMPROVEMENT, "pass": p1},
        "p2": {"valid_days": valid_days, "threshold": P2_MIN_DAYS, "pass": p2},
        "overall_pass": overall,
        "verdict": verdict,
        "diagnostics": diagnostics,
        "assumptions": [
            "realized comparator/target computed on the IV-valid set "
            "intersected with dense-return names (identity stays exact); "
            "dropped bm weight reported as diagnostic",
            "trailing window = t-125..t (inclusive), target window = t+1..t+21",
            "std uses ddof=1 throughout; [0,1]-NaN rule applied to ICorr only "
            "(preregistered clause); realized values reported as computed",
            "iv30 values converted percent -> decimal (identity is "
            "scale-invariant; conversion is for logging consistency)",
        ],
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    daily.to_csv(OUT_DIR / "daily_series.csv")
    print(f"\nVERDICT: {verdict}  ({time.time()-t0:.0f}s)  saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
