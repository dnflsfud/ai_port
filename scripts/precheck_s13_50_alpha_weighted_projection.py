# -*- coding: utf-8 -*-
"""§S13.50 사전점검 (read-only): 알파-가중 투영의 직접 반사실.

§S13.48-A는 투영의 **예산 성분(균일 클로백)만** 반사실 프록시로 쟀다. 본 절은
대상을 **총 투영 변위 d_t = w⁺_t − candidate_t** 전체로 넓히고, 프록시 대신
**사전약정 arm(κ=3.0 알파-가중 투영)을 실제로 풀어** 현행 북과의 전진 초과수익
차를 직접 잰다. 백테스트가 아니라 리밸일별 단면 측정이다. arm 미실행.

재구성(결정 로그 §S13.50, 측정 전 고정) — 리밸일 t마다 production 경로 축자:
  1. w⁻_t = _drift_weights(daily_weights[t_prev], returns_masked.loc[t])
  2. target_t = production optimizer(pred_row, hist, w⁻_t, sector_map, bm_w)
  3. trailing_ic_mean_t = ic_series 중 날짜 < t 의 마지막 trailing_ic_window개
     nanmean; 사용 가능 개수 < 2면 0.0 (backtest.py:1580-1584 축자)
  4. confidence_t = compute_signal_confidence(pred_row, raw_row, tic, spread_scale)
  5. candidate_t = apply_dynamic_execution(w⁻_t, target_t, conf, cfg)
  6. w_0 = 현행 투영(κ=0 — 원래 sum_squares 표현식 그대로)
  7. w_κ = 제약 동일, 목적함수만 Minimize(sum_squares(multiply(sqrt(v), w−cand))),
     v_i = 1 + κ·pct_rank(μ_i), κ=3.0 고정(스윕 금지), 비유한 μ는 v=1
  8. fwdex21 = t+1..t+21 종목 누적수익 − 동기간 벤치마크(buy-and-hold) 수익

게이트(측정 전 고정):
  E0 (하드·선판정) median_t L1(w_0, w⁺_t) ≤ 1e-6 AND ≥90% 샘플에서 ≤ 1e-5.
     미달 = 재구성 실패 → P1/P2/P3 null, verdict=ABORT_E0, 즉시 종료.
  P1 Σ_t gain_t > 0 AND 시간순 3분할 3/3 양, gain_t = (w_κ − w_0)·fwdex21_t.
  P2 Σ_t gain_t / years ≥ +0.0010 (§S13.48-A P2와 동일 바).
  P3 mean_t(‖w_κ − w⁻_t‖₁ − ‖w_0 − w⁻_t‖₁) ≤ 0.0008 (two-way L1).
  종합 = E0∧P1∧P2∧P3 (PASS / SHELVE).

출력: outputs/s13_50_alpha_weighted_projection/{summary.json,by_date.csv}
"""

import bisect
import json
import os
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
OUT_DIR = Path("outputs/s13_50_alpha_weighted_projection")
CKPT = OUT_DIR / "rows.jsonl"   # 리밸일별 체크포인트(중단 내성). 값에 영향 없음.

KAPPA = 3.0              # 사전약정 단일 값 — 스윕 금지
FWD = 21                 # 전진 영업일
E0_MEDIAN_BAR = 1e-6     # median L1(w_0, w_plus)
E0_TAIL_BAR = 1e-5       # >= E0_TAIL_FRAC 샘플이 이 이하
E0_TAIL_FRAC = 0.90
P2_BAR_ANNUAL = 0.0010   # +0.10%p/yr (§S13.48-A P2와 동일 바)
P3_BAR = 0.0008          # 리밸당 추가 회전(two-way L1 차) 상한
EXPECTED_VINTAGE_DATE = "2026-08-19"          # pkl last daily_weights
EXPECTED_WORKBOOK_MTIME = "2026-08-19 13:49"  # ai_signal_data.xlsx
EXPECTED_INDEX_MTIME = "2026-08-21 11:16"     # Index.xlsx (제2 빈티지 축)

MIN_HIST_ROWS = 30       # estimate_covariance의 실질 최소 관측 수 (§S13.48-B 동일)

# production 경로 재현이 이 스크립트가 복제하지 않은 분기를 타면 즉시 중단한다.
# (재현 실패를 조용한 E0 FAIL로 오독하지 않기 위한 가드 — CLAUDE.md §9)
UNSUPPORTED_FLAGS = (
    "implied_corr_covariance_enabled",   # backtest.py:2044-2056 (S13.46)
    "factor_neutral_enabled",            # backtest.py:2085-2104
    "winner_trim_protection_enabled",    # backtest.py:2106-2112
    "carry_te_conditioning_enabled",     # backtest.py:2017-2018 (S13.22)
    "use_score_based",                   # backtest.py:1600-1607 (다른 투영 경로)
)


# ---------------------------------------------------------------------------
# 순수 헬퍼 (단위테스트 대상)
# ---------------------------------------------------------------------------

def alpha_weight_vector(mu, kappa: float) -> np.ndarray:
    """v_i = 1 + kappa * pct_rank(mu_i); 비유한 mu는 v = 1.

    pct_rank는 **유한 mu만으로** 계산한 평균-순위 백분위 in [0, 1]:
    동점은 평균 순위, pct = (rank - 1) / (n_finite - 1) — 최소 mu가 0.0,
    최대 mu가 1.0이 되어 v가 [1, 1+kappa]를 정확히 채운다. kappa == 0이면
    전원 1.0(=inert). 유한 원소가 1개뿐이면 pct = 0.0으로 둔다(가정).
    """
    mu = np.asarray(getattr(mu, "values", mu), dtype=float).ravel()
    v = np.ones(mu.shape, dtype=float)
    k = float(kappa)
    if k == 0.0:
        return v
    finite = np.isfinite(mu)
    n = int(finite.sum())
    if n == 0:
        return v
    x = mu[finite]
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    # 그룹의 마지막 1-based 순위 = cumsum(counts); 평균 순위는 거기서 (c-1)/2 뺀 값
    avg_rank = np.cumsum(counts) - (counts - 1) / 2.0
    ranks = avg_rank[np.ravel(inv)]
    pct = np.zeros(n, dtype=float) if n == 1 else (ranks - 1.0) / (n - 1.0)
    v[finite] = 1.0 + k * pct
    return v


def trailing_ic_mean(ic_series, t, window: int) -> float:
    """backtest.py:1580-1584 축자 재현.

    원본은 `ic_values` 리스트(리밸일마다 non-NaN IC만 chronological append)에서
    `if len(ic_values) >= 2: nanmean(v for _, v in ic_values[-window:])`,
    아니면 0.0을 쓴다. 리밸일 t의 IC는 그 바의 **끝**에서 append되므로 t 시점의
    리스트는 **날짜 < t** 인 항목만 담고 있다. `res.ic_series`는 그 리스트를
    dict -> Series -> sort_index 한 것이라 원소·순서가 동일하다.
    """
    if ic_series is None or len(ic_series) == 0:
        return 0.0
    s = pd.Series(ic_series).sort_index()
    prior = s.to_numpy(dtype=float)[pd.DatetimeIndex(s.index) < pd.Timestamp(t)]
    if len(prior) < 2:
        return 0.0
    return float(np.nanmean(prior[-int(window):]))


def project_with_alpha_weight(candidate_weights, expected_returns, cov_matrix,
                              prev_weights, sector_map, bm_weights,
                              max_te_annual, sector_deviation, config,
                              fallback_weights, kappa, diagnostics=None):
    """`project_portfolio_weights`의 복제 — 제약 동일, 목적함수만 교체.

    kappa == 0.0 이면 원래 `cp.Minimize(cp.sum_squares(w - candidate))` 표현식을
    **그대로** 탄다(사전등록 §S13.50 "절차/검증 의무": 가중 이차형의
    canonicalization 차이가 E0 재현을 오염시키지 못하게 하는 명시 분기).
    kappa != 0.0 이면 `Minimize(sum_squares(multiply(sqrt(v), w - candidate)))`.
    제약은 `_build_mvo_constraints`를 원본과 동일 인자로 호출해 얻는다.
    솔버 호출·fallback·비유한 검사는 `portfolio_optimizer.py:495-566` 그대로.
    """
    import cvxpy as cp

    from src.portfolio_optimizer import (DEFAULT_CONFIG, MAX_TE_ANNUAL,
                                         SECTOR_DEVIATION,
                                         _build_mvo_constraints,
                                         _init_diagnostics, _solve_problem)

    config = config or DEFAULT_CONFIG
    if max_te_annual == MAX_TE_ANNUAL:
        max_te_annual = config.max_te_annual
    if sector_deviation == SECTOR_DEVIATION:
        sector_deviation = config.sector_deviation

    candidate = np.asarray(candidate_weights, dtype=float)
    n = len(candidate)
    if bm_weights is None:
        bm_weights = np.ones(n) / n
    bm_weights = np.asarray(bm_weights, dtype=float)
    if prev_weights is None:
        prev_weights = bm_weights.copy()
    else:
        prev_weights = np.asarray(prev_weights, dtype=float)

    fallback = np.asarray(
        fallback_weights if fallback_weights is not None else bm_weights,
        dtype=float,
    ).copy()

    diag = _init_diagnostics(diagnostics, mode="projection_mvo")
    w = cp.Variable(n)
    _, _, constraints = _build_mvo_constraints(
        w=w,
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
        prev_weights=prev_weights,
        sector_map=sector_map,
        bm_weights=bm_weights,
        max_te_annual=max_te_annual,
        sector_deviation=sector_deviation,
        config=config,
    )
    if float(kappa) == 0.0:
        objective = cp.Minimize(cp.sum_squares(w - candidate))
    else:
        v = alpha_weight_vector(expected_returns, float(kappa))
        objective = cp.Minimize(
            cp.sum_squares(cp.multiply(np.sqrt(v), w - candidate))
        )
    prob = cp.Problem(objective, constraints)
    if (
        not _solve_problem(
            prob,
            diag,
            allow_scs_fallback=getattr(config, "allow_scs_on_ecos_exception", False),
        )
        or prob.status not in ("optimal", "optimal_inaccurate")
        or w.value is None
    ):
        if diag is not None:
            diag["used_fallback"] = True
            diag["fallback_reason"] = diag.get("fallback_reason") or prob.status or "projection_failed"
        return fallback

    projected = np.asarray(w.value, dtype=float).flatten()
    if not np.all(np.isfinite(projected)):
        if diag is not None:
            diag["used_fallback"] = True
            diag["fallback_reason"] = "non_finite_projection"
        return fallback

    return projected


def thirds(values) -> list:
    """S13.41/48-A 관용: np.linspace(0, len(x), 4, dtype=int) 절단으로 3분할."""
    x = np.asarray(values, dtype=float)
    cut = np.linspace(0, len(x), 4, dtype=int)
    return [x[cut[i]:cut[i + 1]] for i in range(3)]


def _mtime_kst(path) -> str:
    return (pd.Timestamp(Path(path).stat().st_mtime, unit="s", tz="UTC")
            .tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S"))


def _dist(series) -> dict:
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {k: float("nan")
                for k in ("n", "min", "p10", "median", "mean", "p90", "max")}
    return {"n": int(x.size), "min": float(np.min(x)),
            "p10": float(np.percentile(x, 10)), "median": float(np.median(x)),
            "mean": float(np.mean(x)), "p90": float(np.percentile(x, 90)),
            "max": float(np.max(x))}


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

def load_checkpoint(path) -> dict:
    """이미 계산된 리밸일 레코드를 date -> row 로 되돌린다.

    중단 시 마지막 줄이 잘릴 수 있으므로 파싱 실패 줄은 버린다. 레코드는
    리밸일마다 독립·결정적으로 계산되므로 재사용이 값을 바꾸지 않는다.
    """
    done = {}
    path = Path(path)
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and "date" in row:
            done[row["date"]] = row
    return done


def main() -> None:
    from run_variant import compose_config, load_manifest
    from scripts.preflight_s13_30_vol_quality import _fwd_return
    from src.backtest import (_drift_weights, _sanitize_daily_ret,
                              apply_dynamic_execution,
                              compute_signal_confidence, get_benchmark_fn,
                              get_sector_map)
    from src.data_loader import UniverseData
    from src.option_vol_cov import OPTION_VOL_SHEET, build_option_vol_scale
    from src.portfolio_optimizer import estimate_covariance, optimize_portfolio

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cfg = compose_config(load_manifest(VARIANT))
    pkl_mtime = _mtime_kst(PKL)
    workbook_mtime = _mtime_kst(cfg.data_path)
    index_mtime = _mtime_kst(cfg.fx_source_path)
    res = pickle.load(open(PKL, "rb"))
    for name in ("ic_series", "raw_predictions", "predictions",
                 "portfolio_weights", "daily_weights"):
        if getattr(res, name, None) is None:
            sys.exit(f"[ABORT] pkl에 {name} 없음 — 재구성 불가, 보고 요망")
    last_daily = str(pd.Timestamp(sorted(res.daily_weights)[-1]).date())

    # ---------- 실행 선행 검증 (빈티지 3중 가드 — 불일치 시 진행 금지) ----------
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

    for flag in UNSUPPORTED_FLAGS:
        if getattr(cfg, flag, False):
            sys.exit(f"[ABORT] {flag}=True — 이 스크립트가 복제하지 않은 "
                     f"production 분기. 재현 실패를 E0 FAIL로 오독하지 않도록 "
                     f"진행하지 않고 보고만.")

    data = UniverseData(cfg.data_path, config=cfg)
    if getattr(data, "returns_masked", None) is None:
        sys.exit("[ABORT] returns_masked 없음 — masked USD 수익률 가정 파괴, 보고 요망")
    if getattr(data, "raw_returns", None) is None:
        sys.exit("[ABORT] data.raw_returns 없음 — production 공분산 입력 재현 불가, 보고 요망")

    # returns(dense) = simulate_portfolio의 P&L·위치 인덱스 패널,
    # returns_masked = 전진수익·drift 패널(§S13.30/48-A 관용). 용도를 섞지 않는다.
    returns_dense = data.returns
    returns_masked = data.returns_masked
    if not pd.DatetimeIndex(data.dates).equals(pd.DatetimeIndex(returns_dense.index)):
        sys.exit("[ABORT] data.dates != data.returns.index — backtest의 iloc 위치 "
                 "재현 불가, 보고 요망.")

    rebal_dates = sorted(pd.Timestamp(k) for k in res.portfolio_weights)
    tickers = list(res.portfolio_weights[rebal_dates[0]].index)
    ret_idx = returns_dense.index
    # 공분산 입력 = production 경로(backtest.py:1989-1991 -> :1432 risk_source)
    cov_returns = data.raw_returns.reindex(index=ret_idx, columns=tickers)
    preds = res.predictions
    raw_preds = res.raw_predictions
    bm_fn = get_benchmark_fn(data, tickers, config=cfg)
    sector_map = get_sector_map(data)
    daily_idx = sorted(pd.Timestamp(k) for k in res.daily_weights)
    lookback = int(cfg.cov_lookback)
    ic_window = int(getattr(cfg, "trailing_ic_window", 6))
    spread_scale = float(getattr(cfg, "confidence_spread_scale", 0.20))
    fallback_mode = getattr(cfg, "projection_fallback_mode", "target")

    # S13.41 대각 스케일 — production에서 ON이므로 재현하지 않으면 TE 제약이
    # 달라져 E0가 반드시 깨진다(backtest.py:2028-2039, :2061-2065).
    optvol_scale = None
    if getattr(cfg, "option_vol_covariance_enabled", False):
        try:
            iv_sheet = data.get_sheet(OPTION_VOL_SHEET)
        except KeyError:
            sys.exit(f"[ABORT] option_vol_covariance_enabled=True인데 "
                     f"{OPTION_VOL_SHEET!r} 시트 부재 — production 공분산 재현 "
                     f"불가, 보고 요망.")
        optvol_scale = build_option_vol_scale(returns_dense[tickers], iv_sheet)

    print(f"[load] {time.time()-t0:.0f}s  rebalances={len(rebal_dates)}"
          f" ({rebal_dates[0].date()} ~ {rebal_dates[-1].date()})"
          f"  tickers={len(tickers)}  returns={returns_dense.shape}"
          f"  cov_lookback={lookback}  ic_window={ic_window}"
          f"  spread_scale={spread_scale}  fallback_mode={fallback_mode}"
          f"  optvol={'ON' if optvol_scale is not None else 'OFF'}"
          f"  kappa={KAPPA}")

    rows, skip_reasons = [], {}
    done = load_checkpoint(CKPT)
    if done:
        print(f"[resume] 체크포인트 {len(done)}건 재사용 — {CKPT}")
    ckpt_fh = open(CKPT, "a", encoding="utf-8")

    def _skip(reason):
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    for dt in rebal_dates:
        key = str(dt.date())
        if key in done:
            rows.append(done[key])
            continue
        if dt not in preds.index:
            _skip("date_not_in_predictions")
            continue
        if dt not in ret_idx:
            _skip("date_not_in_returns")
            continue
        j = bisect.bisect_left(daily_idx, dt)
        if j == 0:
            _skip("no_prev_daily")
            continue
        t_prev = daily_idx[j - 1]
        pos = ret_idx.get_loc(dt)
        hist = cov_returns.iloc[max(0, pos - lookback):pos]
        if len(hist) < MIN_HIST_ROWS:
            _skip("hist_rows_lt_30")
            continue

        # --- 1. w⁻ 복원 (§S13.48-A 관용: returns_masked 기준) ---------------
        w_prev = res.daily_weights[t_prev].reindex(tickers).fillna(0.0).to_numpy()
        r_t = returns_masked.loc[dt, tickers].fillna(0.0).to_numpy()
        w_minus = _drift_weights(w_prev, r_t)
        # 진단: production은 dense returns + _sanitize_daily_ret를 쓴다
        # (backtest.py:1497, :1510). 유령 종목의 비중이 0이면 두 경로가 같다.
        w_minus_dense = _drift_weights(
            w_prev, _sanitize_daily_ret(returns_dense.loc[dt, tickers].values))
        drift_gap = float(np.abs(w_minus - w_minus_dense).sum())

        # --- 2. target (production _optimizer_fn 경로 축자) -----------------
        pred_row = preds.loc[dt, tickers]
        non_finite = ~np.isfinite(pred_row.astype(float))   # backtest.py:1519-1530 (C5)
        if non_finite.any():
            pred_row = pred_row.mask(non_finite)
        if pred_row.notna().sum() < 10:
            _skip("pred_coverage_lt_10")
            continue
        bm_w = np.asarray(bm_fn(dt, tickers, len(tickers)), dtype=float)

        opt_diag = {}
        cov_matrix = estimate_covariance(hist, bm_weights=bm_w, config=cfg)
        if optvol_scale is not None and dt in optvol_scale.index:
            _s = optvol_scale.loc[dt].reindex(pred_row.index).fillna(1.0).values
            cov_matrix = np.diag(_s) @ cov_matrix @ np.diag(_s)
        opt_diag["cov_matrix"] = cov_matrix
        opt_diag["max_te_annual"] = cfg.max_te_annual
        opt_diag["sector_deviation"] = cfg.sector_deviation
        target = optimize_portfolio(
            expected_returns=pred_row,
            cov_matrix=cov_matrix,
            prev_weights=w_minus,
            sector_map=sector_map if sector_map else None,
            bm_weights=bm_w,
            config=cfg,
            diagnostics=opt_diag,
            factor_loadings=None,
            winner_mask=None,
        )
        if opt_diag.get("used_fallback", False):
            _skip("optimizer_fallback")
            continue
        target = np.asarray(target, dtype=float)

        # --- 3~5. confidence -> dynamic execution ---------------------------
        raw_row = raw_preds.loc[dt, tickers] if dt in raw_preds.index else None
        tic = trailing_ic_mean(res.ic_series, dt, ic_window)
        conf = compute_signal_confidence(pred_row, raw_row, tic,
                                         spread_scale=spread_scale)
        candidate = apply_dynamic_execution(w_minus, target, conf, cfg)

        # --- 6~7. 두 투영 ---------------------------------------------------
        projection_fallback = (w_minus.copy() if fallback_mode == "prev" else target)
        max_te = opt_diag.get("max_te_annual", cfg.max_te_annual)
        sec_dev = opt_diag.get("sector_deviation", cfg.sector_deviation)
        cov = opt_diag.get("cov_matrix")
        if cov is None:
            cov = estimate_covariance(hist, bm_weights=bm_w, config=cfg)

        d0, dk = {}, {}
        w_0 = project_with_alpha_weight(
            candidate, pred_row, cov, w_minus, sector_map, bm_w,
            max_te, sec_dev, cfg, projection_fallback, 0.0, diagnostics=d0)
        w_k = project_with_alpha_weight(
            candidate, pred_row, cov, w_minus, sector_map, bm_w,
            max_te, sec_dev, cfg, projection_fallback, KAPPA, diagnostics=dk)
        if d0.get("used_fallback", False) or dk.get("used_fallback", False):
            _skip("projection_fallback")
            continue
        w_0 = np.asarray(w_0, dtype=float)
        w_k = np.asarray(w_k, dtype=float)

        # --- 8. 전진 초과수익 & 지표 ----------------------------------------
        w_plus = res.portfolio_weights[dt].reindex(tickers).fillna(0.0).to_numpy()
        fwd = _fwd_return(returns_masked, dt, FWD)
        if fwd is None:
            gain = float("nan")     # 꼬리 리밸 — E0에는 남기고 P1/P2에서만 제외
            _skip("no_fwd_window(gain only)")
        else:
            fwd = fwd.reindex(tickers).to_numpy()
            bm_fwd = float(np.nansum(bm_w * fwd))
            fwdex = fwd - bm_fwd
            gain = float(np.nansum((w_k - w_0) * fwdex))

        rows.append({
            "date": str(dt.date()),
            "l1_repro": float(np.abs(w_0 - w_plus).sum()),
            "gain": gain,
            "extra_turnover": float(np.abs(w_k - w_minus).sum()
                                    - np.abs(w_0 - w_minus).sum()),
            "conf": float(conf),
            "tic": float(tic),
            "proj_displacement": float(np.abs(w_0 - candidate).sum()),
            "proj_displacement_k": float(np.abs(w_k - candidate).sum()),
            "l1_w0_wk": float(np.abs(w_k - w_0).sum()),
            "turnover_0": float(np.abs(w_0 - w_minus).sum()),
            "drift_gap": drift_gap,
            "solver_opt": opt_diag.get("solver"),
            "status_opt": opt_diag.get("status"),
            "solver_proj0": d0.get("solver"),
            "status_proj0": d0.get("status"),
            "solver_projk": dk.get("solver"),
            "status_projk": dk.get("status"),
        })
        r = rows[-1]
        print(json.dumps(r, ensure_ascii=False), file=ckpt_fh, flush=True)
        os.fsync(ckpt_fh.fileno())
        print(f"[t] {r['date']}  L1repro {r['l1_repro']:.2e}"
              f"  disp {r['proj_displacement']:.5f}"
              f"  |wk-w0| {r['l1_w0_wk']:.5f}"
              f"  gain {r['gain']:+.6f}  dTO {r['extra_turnover']:+.5f}"
              f"  conf {r['conf']:.3f}  tic {r['tic']:+.4f}")

    ckpt_fh.close()
    if not rows:
        sys.exit("[ABORT] 사용 가능한 리밸일 0건 — 보고 요망")
    df = pd.DataFrame(rows)
    n_used = int(len(df))
    n_skipped = int(sum(skip_reasons.values()))
    years = (rebal_dates[-1] - rebal_dates[0]).days / 365.25

    # ---------------- E0 (하드 게이트 — 선판정) ----------------
    l1 = df["l1_repro"].to_numpy(dtype=float)
    l1_ok = l1[np.isfinite(l1)]
    med_l1 = float(np.median(l1_ok)) if l1_ok.size else float("nan")
    tail_frac = float(np.mean(l1_ok <= E0_TAIL_BAR)) if l1_ok.size else float("nan")
    e0 = bool(np.isfinite(med_l1) and med_l1 <= E0_MEDIAN_BAR
              and np.isfinite(tail_frac) and tail_frac >= E0_TAIL_FRAC)
    repro_fail_dates = df.loc[df["l1_repro"] > E0_TAIL_BAR, "date"].tolist()

    diagnostics = {
        "l1_repro": _dist(df["l1_repro"]),
        "conf": _dist(df["conf"]),
        "tic": _dist(df["tic"]),
        "proj_displacement": _dist(df["proj_displacement"]),
        "proj_displacement_kappa": _dist(df["proj_displacement_k"]),
        "l1_w0_wk": _dist(df["l1_w0_wk"]),
        "turnover_0": _dist(df["turnover_0"]),
        "drift_gap_masked_vs_dense": _dist(df["drift_gap"]),
        "solver_counts": {
            "optimizer": df["solver_opt"].value_counts().to_dict(),
            "projection_kappa0": df["solver_proj0"].value_counts().to_dict(),
            "projection_kappa": df["solver_projk"].value_counts().to_dict(),
        },
        "n_tickers": int(len(tickers)),
    }
    notes = [
        "w⁻ = _drift_weights(daily_weights[t_prev], returns_masked.loc[t]) — "
        "§S13.48-A 사전등록 관용. production simulate_portfolio는 dense "
        "data.returns + _sanitize_daily_ret를 쓴다(backtest.py:1497,:1510); "
        "두 경로 차이는 diagnostics.drift_gap_masked_vs_dense로 계량된다.",
        "공분산 입력 = data.raw_returns를 data.returns.index/tickers로 reindex한 "
        "패널(backtest.py:1989-1991 -> :1432 risk_source). hist 창은 "
        "iloc[pos-cov_lookback:pos] — t 자신 제외(:1537-1538 축자).",
        "target은 backtest.py:2058-2131의 _optimizer_fn을 복제한다: "
        "estimate_covariance -> (S13.41 ON이면) 대각 스케일 -> optimize_portfolio. "
        "S13.41이 production ON이므로 이 스케일을 빼면 TE 제약이 달라져 E0가 "
        "구조적으로 깨진다. 그 외 분기(S13.46 icorr·factor_neutral·winner_trim·"
        "carry-TE·score_based)는 UNSUPPORTED_FLAGS 가드로 실행 전 차단.",
        "cov_matrix는 production과 동일하게 optimizer diagnostics에서 받아 "
        "투영에 재사용한다(backtest.py:1613-1618의 None-분기까지 미러).",
        "전진수익 창이 없는 꼬리 리밸은 gain=NaN으로 남겨 E0 표본에는 포함하고 "
        "P1/P2에서만 제외한다. years는 §S13.48-A와 동일하게 전 리밸 구간 기준.",
        "P3는 사전등록 정의대로 two-way L1 차 "
        "mean_t(‖w_κ−w⁻‖₁ − ‖w_0−w⁻‖₁) ≤ 0.0008.",
    ]
    assumptions = [
        "pct_rank = (평균순위 - 1)/(유한 μ 수 - 1) — 최소 μ가 v=1.0, 최대 μ가 "
        "v=1+κ. 유한 μ가 1개뿐이면 pct=0.0.",
        "동점 μ는 평균 순위(numpy 구현, scipy 미사용).",
    ]

    summary = {
        "preregistration": "decision log §S13.50 (2026-08-21)",
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
            "kappa": KAPPA,
            "fwd_days": FWD,
            "cov_lookback": lookback,
            "trailing_ic_window": ic_window,
            "confidence_spread_scale": spread_scale,
            "projection_fallback_mode": fallback_mode,
            "max_te_annual": float(cfg.max_te_annual),
            "sector_deviation": float(cfg.sector_deviation),
            "no_trade_band": float(cfg.no_trade_band),
            "partial_rebalance_eta": float(cfg.partial_rebalance_eta),
            "option_vol_covariance_enabled": bool(
                getattr(cfg, "option_vol_covariance_enabled", False)),
            "allow_scs_on_ecos_exception": bool(
                getattr(cfg, "allow_scs_on_ecos_exception", False)),
        },
        "n_rebalances": int(len(rebal_dates)),
        "n_used": n_used,
        "n_skipped": n_skipped,
        "skip_reasons": skip_reasons,
        "years": float(years),
        "e0": {
            "median_l1": med_l1,
            "median_threshold": E0_MEDIAN_BAR,
            "tail_frac_le_bar": tail_frac,
            "tail_bar": E0_TAIL_BAR,
            "tail_frac_threshold": E0_TAIL_FRAC,
            "n_repro_fail": int(len(repro_fail_dates)),
            "repro_fail_dates": repro_fail_dates,
            "pass": e0,
        },
        "diagnostics": diagnostics,
        "notes": notes,
        "assumptions": assumptions,
    }

    print(f"[E0] median L1(w_0, w+) {med_l1:.3e} (바 {E0_MEDIAN_BAR:.0e})"
          f"  frac<=1e-5 {tail_frac:.3f} (바 {E0_TAIL_FRAC})"
          f"  n={n_used}  실패일 {len(repro_fail_dates)}"
          f"  -> {'PASS' if e0 else 'FAIL'}")
    dq = diagnostics["proj_displacement"]
    print(f"[diag] ‖w_0-candidate‖₁ median {dq['median']:.5f}"
          f" p10 {dq['p10']:.5f} p90 {dq['p90']:.5f} max {dq['max']:.5f}"
          f"  |  conf median {diagnostics['conf']['median']:.3f}"
          f"  tic median {diagnostics['tic']['median']:+.4f}"
          f"  drift_gap max {diagnostics['drift_gap_masked_vs_dense']['max']:.2e}")

    if not e0:
        # 사전등록: E0 미달이면 나머지 게이트 판정은 무효.
        summary["p1"] = None
        summary["p2"] = None
        summary["p3"] = None
        summary["overall_pass"] = False
        summary["verdict"] = "ABORT_E0"
        summary["runtime_s"] = round(time.time() - t0, 1)
        (OUT_DIR / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        df.to_csv(OUT_DIR / "by_date.csv", index=False)
        print("[P1] null (E0 FAIL — 사전등록상 판정 무효)")
        print("[P2] null (E0 FAIL — 사전등록상 판정 무효)")
        print("[P3] null (E0 FAIL — 사전등록상 판정 무효)")
        print("[VERDICT] ABORT_E0")
        print(f"\nVERDICT: ABORT_E0  ({time.time()-t0:.0f}s)  saved: {OUT_DIR}")
        return

    # ---------------- P1 / P2 / P3 (E0 PASS일 때만) ----------------
    gain_list = df.loc[np.isfinite(df["gain"]), "gain"].tolist()
    gain_sum = float(np.sum(gain_list))
    third_sums = [float(np.sum(b)) for b in thirds(gain_list)]
    p1 = bool(gain_sum > 0 and all(s > 0 for s in third_sums))

    gain_annual = gain_sum / years
    p2 = bool(gain_annual >= P2_BAR_ANNUAL)

    extra_to = df["extra_turnover"].to_numpy(dtype=float)
    mean_extra_to = float(np.mean(extra_to[np.isfinite(extra_to)]))
    p3 = bool(mean_extra_to <= P3_BAR)

    overall = bool(e0 and p1 and p2 and p3)
    verdict = "PASS" if overall else "SHELVE"

    summary["p1"] = {"gain_sum": gain_sum, "n_gain": int(len(gain_list)),
                     "thirds_sums": third_sums, "pass": p1}
    summary["p2"] = {"gain_annual": gain_annual, "threshold": P2_BAR_ANNUAL,
                     "years": float(years), "pass": p2}
    summary["p3"] = {"mean_extra_turnover": mean_extra_to, "threshold": P3_BAR,
                     "extra_turnover_dist": _dist(df["extra_turnover"]),
                     "pass": p3}
    summary["gain_dist"] = _dist(df["gain"])
    summary["overall_pass"] = overall
    summary["verdict"] = verdict
    summary["runtime_s"] = round(time.time() - t0, 1)

    print(f"[P1] Σgain {gain_sum:+.6f}  thirds ["
          f"{third_sums[0]:+.6f}, {third_sums[1]:+.6f}, {third_sums[2]:+.6f}]"
          f"  n={len(gain_list)}  -> {'PASS' if p1 else 'FAIL'}")
    print(f"[P2] Σgain/yr {gain_annual:+.6f} (바 {P2_BAR_ANNUAL:.4f})"
          f"  years {years:.2f}  -> {'PASS' if p2 else 'FAIL'}")
    print(f"[P3] mean 추가회전 {mean_extra_to:+.6f} (바 {P3_BAR:.4f})"
          f"  -> {'PASS' if p3 else 'FAIL'}")
    print(f"[gates] E0={e0} P1={p1} P2={p2} P3={p3}")
    print(f"[VERDICT] {verdict}")

    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    df.to_csv(OUT_DIR / "by_date.csv", index=False)
    print(f"\nVERDICT: {verdict}  ({time.time()-t0:.0f}s)  saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
