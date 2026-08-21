# -*- coding: utf-8 -*-
"""§S13.48-B 사전점검 (read-only): RMT 벌크 고유값 정제 (Σ 고유공간).

결정 로그 §S13.48-B 사전등록(2026-08-21, 측정 전)의 P0/P1/P2를 실행한다.
arm 미실행 — 백테스트 없음, src/ 무수정.

  P0 (예측력): 전진 21BD 실현 액티브 분산을 타깃으로 한 excess-QLIKE 개선율
     >= +5% AND 시간순 3분할 부호 일관 (§S13.41 P0와 동일 바).
  P1 (증분·2중):
     (i)  바인딩 — baseline Σ_t에서 floor(λ <= floor·(1+1e-9))에 닿은
          고유방향이 설명하는 aᵀΣa 점유율의 중앙값 >= 10%.
     (ii) 스칼라-등가 제거 — aᵀΣa가 같아지도록 baseline을 균일 배율한
          Σ_scalar 대비, Σ_RMT 재-MVO(24회 균등 샘플)의 one-way L1 이동
          중앙값 >= 0.005.
  P2 (경제성·캐릭터): 24 샘플 재-MVO 북에서 Pictet active_share >= 18%
     AND ex-ante TE >= 3.0%. 부수 필수: replaced_share 중앙값 > 0.95면
     Σ가 대각+평균상관으로 축퇴 → degenerate_warning 병기.

종합 = P0 ∧ P1(i) ∧ P1(ii) ∧ P2 (PASS/SHELVE).

**명세 vs 코드 현실 불일치 (보고용 — 추측으로 메우지 않음)**
사전등록 명세는 공분산 입력 패널을 `data.returns_masked`로 지정한다(아래
`returns`). 그러나 production 백테스트의 공분산 경로는
`backtest.py:1432` `risk_source = risk_returns if ... else returns` +
`backtest.py:1989-1991` `risk_returns = data.raw_returns`(미보간·마스킹된
USD 패널)를 쓴다 — `returns_masked`(보간 후 마스킹)와 다른 객체다.
이 스크립트는 명세를 축자 이행하되, 실제로 NaN이 있어 `_pairwise_covariance`
경로(=floor가 적용되는 유일한 경로)를 탄 리밸일 수를 진단으로 기록한다
(`diagnostics.pairwise_path_dates`). LedoitWolf 경로에서는 floor가 적용되지
않으므로 P1(i) 해석에 필수.

출력: outputs/s13_48b_rmt_denoise/summary.json
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

VARIANT = AI_PORT / "variants" / "codex_causal_rank_65.yaml"
PKL = AI_PORT / "outputs" / "codex_causal_rank_65" / "backtest_result.pkl"
OUT_DIR = AI_PORT / "outputs" / "s13_48b_rmt_denoise"

FWD = 21                      # 전진 실현 액티브 분산 창 (BD)
N_MVO_SAMPLES = 24            # 재-MVO 균등 샘플 수
P0_BAR = 0.05                 # excess-QLIKE 개선율 >= +5%
P1I_BAR = 0.10                # floor 방향 분산 점유 중앙값 >= 10%
P1II_BAR = 0.005              # one-way L1 이동 중앙값 >= 0.005
P2_AS_BAR = 0.18              # Pictet active_share >= 18%
P2_TE_BAR = 0.030             # ex-ante TE >= 3.0%
EXPECTED_VINTAGE_DATE = "2026-08-19"
EXPECTED_WORKBOOK_MTIME = "2026-08-19 13:49"
EXPECTED_INDEX_MTIME = "2026-08-21 11:16"

DEGENERATE_BAR = 0.95         # replaced_share 중앙값 경고선 (§S13.48-B 부수 필수)
MIN_HIST_ROWS = 30            # estimate_covariance의 실질 최소 관측 수
DEFAULT_VAR = 0.04 / 252.0    # portfolio_optimizer._pairwise_covariance 상수
ANN = 252.0


# ---------------------------------------------------------------------------
# 순수 헬퍼 (단위테스트 대상)
# ---------------------------------------------------------------------------

def excess_qlike(h: np.ndarray, sig2: np.ndarray) -> np.ndarray:
    """QLIKE 초과손실 log(h/σ²) + σ²/h − 1 ≥ 0 (h=σ²에서 0). %개선 정의 가능.

    `scripts/precheck_s13_41_optvol_cov.py:49-52`에서 그대로 복사(정의 재사용).
    스크립트 self-contained 원칙상 cross-script import는 하지 않는다.
    """
    r = np.asarray(sig2, dtype=float) / np.asarray(h, dtype=float)
    return -np.log(r) + r - 1.0


def _thirds(vals):
    """시간순 3분할. `scripts/precheck_s13_41_optvol_cov.py:85-88`에서 그대로 복사."""
    x = np.asarray(vals, dtype=float)
    cut = np.linspace(0, len(x), 4, dtype=int)
    return [x[cut[i]:cut[i + 1]] for i in range(3)]


def mp_lambda_plus(n_eff: int, t_obs: int, sigma_bar_sq: float) -> float:
    """Marchenko-Pastur 상단: sigma_bar_sq * (1 + sqrt(n_eff/t_obs))**2."""
    return float(sigma_bar_sq) * (1.0 + np.sqrt(float(n_eff) / float(t_obs))) ** 2


def rmt_denoise(cov: np.ndarray, t_obs: int) -> tuple:
    """벌크 고유값을 평균으로 치환한 Σ_RMT와 진단을 반환.

    절차(고정): (1) eigh 대칭 고유분해, (2) n = cov.shape[0],
    sigma_bar_sq = trace(cov)/n, lam_plus = mp_lambda_plus(n, t_obs, sigma_bar_sq),
    (3) mask = eigvals <= lam_plus — 2개 이상이면 그 부분집합 산술평균으로 치환
    (1개 이하면 무변경), (4) 재구성 후 0.5*(X+X.T) 대칭화, (5) trace 보존
    재스케일(새 trace <= 0이면 무변경), (6) 진단 dict 반환.
    비유한/비대칭 입력은 그대로 반환하고 n_replaced=0.
    """
    cov = np.asarray(cov, dtype=float)
    n = int(cov.shape[0])
    diag = {"n": n, "q": float(n) / float(t_obs),
            "sigma_bar_sq": float("nan"), "lambda_plus": float("nan"),
            "n_replaced": 0, "replaced_share": 0.0}
    if not np.all(np.isfinite(cov)) or not np.allclose(cov, cov.T):
        return cov, diag

    tr0 = float(np.trace(cov))
    sigma_bar_sq = tr0 / n
    lam_plus = mp_lambda_plus(n, t_obs, sigma_bar_sq)
    diag["sigma_bar_sq"] = sigma_bar_sq
    diag["lambda_plus"] = lam_plus

    eigvals, eigvecs = np.linalg.eigh(cov)
    mask = eigvals <= lam_plus
    if int(mask.sum()) < 2:
        return cov, diag

    new_vals = eigvals.copy()
    new_vals[mask] = float(eigvals[mask].mean())
    out = (eigvecs * new_vals) @ eigvecs.T
    out = 0.5 * (out + out.T)
    tr_new = float(np.trace(out))
    if tr_new > 0:
        out = out * (tr0 / tr_new)
    diag["n_replaced"] = int(mask.sum())
    diag["replaced_share"] = float(mask.sum()) / n
    return out, diag


def floor_direction_share(cov: np.ndarray, a: np.ndarray, floor: float) -> float:
    """baseline cov의 고유방향 중 lam <= floor*(1+1e-9) 인 방향들이 설명하는
    a^T cov a 의 점유율. 분모 a^T cov a <= 0 이면 nan."""
    cov = np.asarray(cov, dtype=float)
    a = np.asarray(a, dtype=float)
    denom = float(a @ cov @ a)
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    eigvals, eigvecs = np.linalg.eigh(cov)
    contrib = eigvals * (eigvecs.T @ a) ** 2
    mask = eigvals <= floor * (1.0 + 1e-9)
    return float(contrib[mask].sum() / denom)


def _floor_ref(hist: pd.DataFrame) -> float:
    """`portfolio_optimizer._pairwise_covariance:139-146`의 floor를 동일 창에서 재계산."""
    recent = hist.replace([np.inf, -np.inf], np.nan)
    var = recent.var(axis=0, skipna=True).reindex(recent.columns).values
    finite_var = var[np.isfinite(var) & (var > 0)]
    fallback_var = float(np.median(finite_var)) if len(finite_var) else DEFAULT_VAR
    return max(fallback_var * 1e-4, 1e-10)


def _mtime_kst(path) -> str:
    return (pd.Timestamp(Path(path).stat().st_mtime, unit="s", tz="UTC")
            .tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S"))


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

    data = UniverseData(cfg.data_path, config=cfg)
    if getattr(data, "returns_masked", None) is None:
        sys.exit("[ABORT] returns_masked 없음 — masked USD 수익률 가정 파괴, 보고 요망")

    rebal_dates = sorted(pd.Timestamp(k) for k in res.portfolio_weights)
    tickers = list(res.portfolio_weights[rebal_dates[0]].index)
    returns = data.returns_masked
    ret_idx = returns.index
    bm_fn = get_benchmark_fn(data, tickers, config=cfg)
    sector_map = get_sector_map(data)
    daily_idx = sorted(pd.Timestamp(k) for k in res.daily_weights)
    lookback = int(cfg.cov_lookback)
    print(f"[load] {time.time()-t0:.0f}s  rebalances={len(rebal_dates)}"
          f" ({rebal_dates[0].date()} ~ {rebal_dates[-1].date()})"
          f"  tickers={len(tickers)}  returns={returns.shape}"
          f"  cov_lookback={lookback}")

    # 재-MVO 대상 = 리밸일 균등 24 샘플 (사전등록)
    sample_pos = set(int(i) for i in np.linspace(
        0, len(rebal_dates) - 1, N_MVO_SAMPLES, dtype=int))

    rows, skip_reasons, mvo_cache = [], {}, {}

    def _skip(reason):
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    for i, dt in enumerate(rebal_dates):
        if dt not in ret_idx:
            _skip("date_not_in_returns")
            continue
        pos = ret_idx.get_loc(dt)
        # backtest.py:1537-1538 축자: iloc[t_idx-cov_lookback : t_idx] — t 제외
        hist = returns[tickers].iloc[max(0, pos - lookback):pos]
        if len(hist) < MIN_HIST_ROWS:
            _skip("hist_rows_lt_30")
            continue
        bm_w = np.asarray(bm_fn(dt, tickers, len(tickers)), dtype=float)
        cov_base = np.asarray(
            estimate_covariance(hist, bm_weights=bm_w, config=cfg), dtype=float)
        a = (res.portfolio_weights[dt].reindex(tickers).fillna(0.0).to_numpy()
             - bm_w)
        cov_rmt, diag = rmt_denoise(cov_base, t_obs=len(hist))

        v_base = float(a @ cov_base @ a)
        v_rmt = float(a @ cov_rmt @ a)

        # 전진 실현 액티브 분산 (일별 — ex-ante와 동일 단위)
        window = ret_idx[pos + 1:pos + 1 + FWD]
        realized = float("nan")
        if len(window) == FWD:
            r_act = (returns.loc[window, tickers] * a).sum(axis=1)
            realized = float(r_act.var(ddof=0))

        row = {
            "date": str(dt.date()),
            "hist_rows": int(len(hist)),
            "hist_has_nan": bool(hist.isna().any().any()),
            "v_base": v_base,
            "v_rmt": v_rmt,
            "realized": realized,
            "n_replaced": int(diag["n_replaced"]),
            "replaced_share": float(diag["replaced_share"]),
            "lambda_plus": float(diag["lambda_plus"]),
            "sigma_bar_sq": float(diag["sigma_bar_sq"]),
            "q": float(diag["q"]),
            "floor_ref": _floor_ref(hist),
        }
        row["floor_share"] = floor_direction_share(cov_base, a, row["floor_ref"])
        if np.isfinite(realized) and realized > 0 and v_base > 0 and v_rmt > 0:
            row["exq_base"] = float(excess_qlike(v_base, realized))
            row["exq_rmt"] = float(excess_qlike(v_rmt, realized))
        else:
            row["exq_base"] = float("nan")
            row["exq_rmt"] = float("nan")
            _skip("qlike_unavailable")
        rows.append(row)

        if i in sample_pos:
            mvo_cache[dt] = (cov_base, cov_rmt, a, bm_w, v_base, v_rmt)

    df = pd.DataFrame(rows)
    n_used = int(len(df))
    if n_used == 0:
        sys.exit("[ABORT] 사용 가능한 리밸일 0건 — 보고 요망")

    # ---------------- P0: excess-QLIKE 개선 ----------------
    qdf = df.dropna(subset=["exq_base", "exq_rmt"]).reset_index(drop=True)
    mean_base = float(qdf["exq_base"].mean())
    mean_rmt = float(qdf["exq_rmt"].mean())
    improve = (mean_base - mean_rmt) / mean_base if mean_base != 0 else float("nan")
    thirds_impr = []
    for tb, tr in zip(_thirds(qdf["exq_base"]), _thirds(qdf["exq_rmt"])):
        mb = float(np.mean(tb)) if len(tb) else float("nan")
        mr = float(np.mean(tr)) if len(tr) else float("nan")
        thirds_impr.append(round(1.0 - mr / mb, 4) if mb else float("nan"))
    p0 = bool(np.isfinite(improve) and improve >= P0_BAR
              and all(np.isfinite(s) and s > 0 for s in thirds_impr))
    print(f"[P0] n={len(qdf)}  excess-QLIKE 개선 {improve:+.1%} (바 {P0_BAR:+.0%})"
          f"  thirds {thirds_impr}  -> {'PASS' if p0 else 'FAIL'}")

    # ---------------- P1(i): floor 방향 바인딩 ----------------
    med_floor_share = float(df["floor_share"].median())
    p1i = bool(np.isfinite(med_floor_share) and med_floor_share >= P1I_BAR)
    print(f"[P1i] floor 방향 분산 점유 중앙값 {med_floor_share:.1%}"
          f" (바 {P1I_BAR:.0%})  -> {'PASS' if p1i else 'FAIL'}")

    # ---------------- P1(ii)/P2: 재-MVO 24 샘플 ----------------
    mvo_rows = []
    for dt in sorted(mvo_cache):
        cov_base, cov_rmt, a, bm_w, v_base, v_rmt = mvo_cache[dt]
        if v_base <= 0:
            _skip("mvo_v_base_nonpositive")
            continue
        pred_row = res.predictions.loc[dt, tickers]
        if pred_row.notna().sum() < 10:
            _skip("mvo_pred_coverage")
            continue
        prior = [x for x in daily_idx if x < dt]
        if not prior:
            _skip("mvo_no_prev_daily")
            continue
        prev_w = res.daily_weights[prior[-1]].reindex(tickers).fillna(0.0).values

        scale = v_rmt / v_base           # 정의상 a^T (cov_base*scale) a == v_rmt
        cov_scalar = cov_base * scale
        d_s, d_r = {}, {}
        w_scalar = optimize_portfolio(pred_row, cov_scalar, prev_weights=prev_w,
                                      sector_map=sector_map, bm_weights=bm_w,
                                      config=cfg, diagnostics=d_s)
        w_rmt = optimize_portfolio(pred_row, cov_rmt, prev_weights=prev_w,
                                   sector_map=sector_map, bm_weights=bm_w,
                                   config=cfg, diagnostics=d_r)
        if d_s.get("used_fallback", False) or d_r.get("used_fallback", False):
            _skip("mvo_solver_fallback")
            continue

        a_rmt = np.asarray(w_rmt, dtype=float) - bm_w
        mvo_rows.append({
            "date": str(dt.date()),
            "scale": float(scale),
            "l1": float(0.5 * np.abs(np.asarray(w_rmt) - np.asarray(w_scalar)).sum()),
            "active_share": float(0.5 * np.abs(np.asarray(w_rmt) - bm_w).sum()),
            "te_exante": float(np.sqrt(max(a_rmt @ cov_base @ a_rmt, 0.0)) * np.sqrt(ANN)),
        })
        print(f"[P1ii] {mvo_rows[-1]['date']}  L1 {mvo_rows[-1]['l1']:.4f}"
              f"  AS {mvo_rows[-1]['active_share']:.1%}"
              f"  TE {mvo_rows[-1]['te_exante']:.2%}")

    mdf = pd.DataFrame(mvo_rows)
    if mdf.empty:
        sys.exit("[ABORT] 재-MVO 유효 샘플 0건 — 보고 요망")
    med_l1 = float(mdf["l1"].median())
    p1ii = bool(med_l1 >= P1II_BAR)
    med_as = float(mdf["active_share"].median())
    med_te = float(mdf["te_exante"].median())
    p2 = bool(med_as >= P2_AS_BAR and med_te >= P2_TE_BAR)
    print(f"[P1ii] median one-way L1 {med_l1:.4f} (바 {P1II_BAR})"
          f"  n={len(mdf)}  -> {'PASS' if p1ii else 'FAIL'}")
    print(f"[P2] active_share 중앙값 {med_as:.1%} (바 {P2_AS_BAR:.0%})"
          f"  ex-ante TE 중앙값 {med_te:.2%} (바 {P2_TE_BAR:.1%})"
          f"  -> {'PASS' if p2 else 'FAIL'}")

    med_replaced_share = float(df["replaced_share"].median())
    degenerate_warning = bool(med_replaced_share > DEGENERATE_BAR)
    if degenerate_warning:
        print(f"[WARN] replaced_share 중앙값 {med_replaced_share:.3f} > {DEGENERATE_BAR}"
              f" — Σ가 대각+평균상관으로 축퇴, §S13.46/스칼라 채널과 구분 불가."
              f" 판정에 병기하고 arm 설계 재검토 사유로 기록.")
    else:
        print(f"[diag] replaced_share 중앙값 {med_replaced_share:.3f}"
              f" (경고선 {DEGENERATE_BAR})")

    overall = bool(p0 and p1i and p1ii and p2)
    verdict = "PASS" if overall else "SHELVE"
    pairwise_dates = int(df["hist_has_nan"].sum())
    print(f"[gates] P0={p0} P1i={p1i} P1ii={p1ii} P2={p2}")
    print(f"[VERDICT] {verdict}")

    summary = {
        "preregistration": "decision log §S13.48-B (2026-08-21)",
        "vintage": {
            "expected_workbook_mtime": EXPECTED_WORKBOOK_MTIME,
            "expected_index_mtime": EXPECTED_INDEX_MTIME,
            "expected_last_daily_weights": EXPECTED_VINTAGE_DATE,
            "pkl_mtime": pkl_mtime,
            "workbook_mtime": workbook_mtime,
            "index_mtime": index_mtime,
            "last_daily_weights": last_daily,
        },
        "n_rebalances": int(len(rebal_dates)),
        "n_used": n_used,
        "n_skipped": int(sum(skip_reasons.values())),
        "skip_reasons": skip_reasons,
        "p0": {
            "n_eval": int(len(qdf)),
            "mean_exq_base": mean_base,
            "mean_exq_rmt": mean_rmt,
            "improvement": float(improve),
            "threshold": P0_BAR,
            "thirds": thirds_impr,
            "pass": p0,
        },
        "p1i": {
            "median_floor_share": med_floor_share,
            "mean_floor_share": float(df["floor_share"].mean()),
            "threshold": P1I_BAR,
            "pass": p1i,
        },
        "p1ii": {
            "n_samples_requested": N_MVO_SAMPLES,
            "n_samples_used": int(len(mdf)),
            "median_l1": med_l1,
            "threshold": P1II_BAR,
            "median_scale": float(mdf["scale"].median()),
            "pass": p1ii,
        },
        "p2": {
            "median_active_share": med_as,
            "median_te_exante": med_te,
            "active_share_threshold": P2_AS_BAR,
            "te_threshold": P2_TE_BAR,
            "pass": p2,
        },
        "degenerate_warning": degenerate_warning,
        "diagnostics": {
            "median_replaced_share": med_replaced_share,
            "median_n_replaced": float(df["n_replaced"].median()),
            "median_q": float(df["q"].median()),
            "median_lambda_plus": float(df["lambda_plus"].median()),
            "median_sigma_bar_sq": float(df["sigma_bar_sq"].median()),
            "median_floor_ref": float(df["floor_ref"].median()),
            "median_v_base": float(df["v_base"].median()),
            "median_v_rmt": float(df["v_rmt"].median()),
            "degenerate_bar": DEGENERATE_BAR,
            "cov_input_panel": "data.returns_masked (사전등록 명세 축자)",
            "pairwise_path_dates": pairwise_dates,
            "cov_lookback": lookback,
        },
        "notes": [
            "ex-ante 분산과 전진 실현 액티브 분산 모두 일별 단위 "
            "(estimate_covariance는 일별 공분산 반환 — portfolio_optimizer.py:87 "
            "degenerate 반환값 0.04/252). excess-QLIKE는 예측/실현 비율만 쓰므로 "
            "연율화 여부에 불변.",
            "hist 창 = returns.iloc[pos-cov_lookback:pos] — t 자신 제외 "
            "(backtest.py:1537-1538 hist_start:t_idx 축자 일치).",
            "SPEC-CODE MISMATCH: 명세는 cov 입력을 data.returns_masked로 지정하나 "
            "production 경로는 data.raw_returns(backtest.py:1432/1989-1991)를 쓴다. "
            "명세를 축자 이행했고 pairwise_path_dates로 floor 경로 진입 횟수를 병기.",
        ],
        "overall_pass": overall,
        "verdict": verdict,
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    df.to_csv(OUT_DIR / "by_date.csv", index=False)
    mdf.to_csv(OUT_DIR / "mvo_samples.csv", index=False)
    print(f"\nVERDICT: {verdict}  ({time.time()-t0:.0f}s)  saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
