# -*- coding: utf-8 -*-
"""§S13.48-A 사전점검 (read-only): 투영 계량의 알파 맹목성(균일 클로백).

no-trade band가 죽인 거래분을 `cp.Minimize(sum_squares(w − candidate))` +
`sum(w) == 1`이 **μ가 목적함수에 없는 균일 절대 이동**으로 되살린다. 그 균일
클로백이 (P0) 97 리밸 전반에서 일반적인지, (P1) μ 순서대로 배분했을 때의
반사실 이득 Ĝ가 양이며 3분할 부호 일관인지, (P2) 연율이 +0.10%p/yr 이상인지를
측정한다. arm 미실행.

사전등록 정의(결정 로그 §S13.48-A, 측정 전 고정):
  w⁻_t = _drift_weights(daily_weights[t_prev], returns_masked.loc[t]),
  w⁺_t = res.portfolio_weights[t], δ_t = w⁺_t − w⁻_t, turnover_t = Σ|δ_t|.
  균일 클로백 군집 = δ_t를 소수 7자리 반올림한 최빈값 c_t와 종목 수 m_t.
  μ = res.predictions.loc[t] (옵티마이저가 실제로 본 값).
  fwdex21_i = t+1..t+21 누적수익 − 동기간 벤치마크 수익.
게이트: P0 = mean_t(|c_t|·m_t/turnover_t) ≥ 0.20.
  P1 = Ĝ = Σ_t |c_t|·h_t·[mean(fwdex21|상위 μ 절반) − mean(fwdex21|하위 μ 절반)]
       > 0 AND 시간순 3분할 3/3 양.
  P2 = Ĝ/years ≥ +0.0010. 종합 = P0∧P1∧P2 (PASS/SHELVE).

가정(구현 재량 기록): 사전등록식의 (m_t/2)는 홀수 m에서 정수가 아니므로 실제
분할 크기 h_t = (군집 내 유한-μ 멤버 수)//2를 쓴다(중앙 1개 폐기). μ가 비유한인
멤버는 분할 전에 제외한다 — 두 절반이 서로소·동일 크기임을 보장하기 위해서다.

출력: outputs/s13_48a_projection_metric/summary.json
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

from scripts.preflight_s13_30_vol_quality import _fwd_return  # noqa: E402

VARIANT = Path("variants/codex_causal_rank_65.yaml")
PKL = Path("outputs/codex_causal_rank_65/backtest_result.pkl")
OUT_DIR = Path("outputs/s13_48a_projection_metric")

DECIMALS = 7             # 클로백 군집 반올림 자리수 (08-19 관측: 노이즈는 9자리)
FWD = 21                 # 전진 영업일
P0_BAR = 0.20            # S >= 0.20
P2_BAR_ANNUAL = 0.0010   # +0.10%p/yr
EXPECTED_VINTAGE_DATE = "2026-08-19"          # pkl last daily_weights
EXPECTED_WORKBOOK_MTIME = "2026-08-19 13:49"  # ai_signal_data.xlsx
EXPECTED_INDEX_MTIME = "2026-08-21 11:16"     # Index.xlsx (제2 빈티지 축)


# ---------------------------------------------------------------------------
# 순수 헬퍼 (단위테스트 대상)
# ---------------------------------------------------------------------------

def clawback_cluster(delta: np.ndarray, decimals: int = 7):
    """delta를 decimals 자리 반올림했을 때의 최빈값 군집.

    반환 (c, m, mask): c=군집 대표값(반올림된 값), m=종목 수, mask=멤버 불리언.
    비유한 delta는 후보에서 제외. 최빈값이 복수면 |값|*개수가 큰 쪽, 그래도
    동률이면 값이 작은 쪽(결정론적). 유한 원소가 없으면 (nan, 0, all-False).
    """
    delta = np.asarray(delta, dtype=float)
    finite = np.isfinite(delta)
    if not finite.any():
        return float("nan"), 0, np.zeros(delta.shape, dtype=bool)
    rounded = np.round(delta, decimals)
    vals, counts = np.unique(rounded[finite], return_counts=True)
    best = min(range(len(vals)),
               key=lambda i: (-counts[i], -abs(vals[i]) * counts[i], vals[i]))
    c = float(vals[best])
    mask = finite & (rounded == c)
    return c, int(mask.sum()), mask


def split_by_mu(mu: np.ndarray, mask: np.ndarray):
    """군집 멤버를 mu 오름차순 정렬해 (하위 절반, 상위 절반) 인덱스를 반환.

    mu가 비유한인 멤버는 먼저 제외한 뒤 분할한다. h = (유효 멤버 수)//2 —
    홀수면 중앙 1개를 버려 두 절반이 서로소·동일 크기가 되게 한다.
    정렬은 np.argsort(kind='stable')로 결정론 보장. h==0이면 빈 배열 2개.
    """
    mu = np.asarray(mu, dtype=float)
    members = np.flatnonzero(np.asarray(mask, dtype=bool) & np.isfinite(mu))
    h = len(members) // 2
    if h == 0:
        empty = np.array([], dtype=int)
        return empty, empty.copy()
    order = members[np.argsort(mu[members], kind="stable")]
    return order[:h], order[len(order) - h:]


def thirds(values: list) -> list:
    """S13.41 관용: np.linspace(0, len(x), 4, dtype=int) 절단으로 3분할."""
    x = np.asarray(values, dtype=float)
    cut = np.linspace(0, len(x), 4, dtype=int)
    return [x[cut[i]:cut[i + 1]] for i in range(3)]


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

def _mtime_kst(path) -> str:
    return pd.Timestamp(Path(path).stat().st_mtime, unit="s", tz="UTC").tz_convert(
        "Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    from run_variant import compose_config, load_manifest
    from src.backtest import _drift_weights, get_benchmark_fn
    from src.data_loader import UniverseData

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cfg = compose_config(load_manifest(VARIANT))
    pkl_mtime = _mtime_kst(PKL)
    workbook_mtime = _mtime_kst(cfg.data_path)
    index_mtime = _mtime_kst(cfg.fx_source_path)
    print(f"[vintage] pkl {pkl_mtime}  workbook {workbook_mtime}"
          f"  Index {index_mtime}")
    if not workbook_mtime.startswith(EXPECTED_WORKBOOK_MTIME):
        sys.exit(f"[ABORT] 워크북 빈티지 불일치 — 기대 {EXPECTED_WORKBOOK_MTIME}, "
                 f"실제 {workbook_mtime}. 진행하지 않고 보고만.")
    if not index_mtime.startswith(EXPECTED_INDEX_MTIME):
        sys.exit(f"[ABORT] Index.xlsx 빈티지 불일치 — 기대 {EXPECTED_INDEX_MTIME}, "
                 f"실제 {index_mtime}. 진행하지 않고 보고만.")

    res = pickle.load(open(PKL, "rb"))
    for name in ("predictions", "portfolio_weights", "daily_weights"):
        if getattr(res, name, None) is None:
            sys.exit(f"[ABORT] pkl에 {name} 없음 — 보고 요망")
    daily_keys = sorted(res.daily_weights)
    last_daily = str(pd.Timestamp(daily_keys[-1]).date())
    if last_daily != EXPECTED_VINTAGE_DATE:
        sys.exit(f"[ABORT] 빈티지 불일치 — 기대 {EXPECTED_VINTAGE_DATE}, "
                 f"last daily_weights {last_daily}. 진행하지 않고 보고만.")

    data = UniverseData(cfg.data_path, config=cfg)
    if getattr(data, "returns_masked", None) is None:
        sys.exit("[ABORT] returns_masked 없음 — masked USD 수익률 가정 파괴, 보고 요망")
    returns = data.returns_masked

    dates = sorted(res.portfolio_weights)
    tickers = list(res.portfolio_weights[dates[0]].index)
    bm_fn = get_benchmark_fn(data, tickers, config=cfg)
    preds = res.predictions
    print(f"[load] {time.time()-t0:.0f}s  rebalances={len(dates)}"
          f" ({pd.Timestamp(dates[0]).date()} ~ {pd.Timestamp(dates[-1]).date()})"
          f"  returns={returns.shape}  tickers={len(tickers)}")

    rows = []
    skip_reasons = {}

    def _skip(reason):
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    for dt in dates:
        prev = [d for d in daily_keys if d < dt]
        if not prev:
            _skip("no_prev_daily")
            continue
        t_prev = prev[-1]
        if dt not in returns.index:
            _skip("date_not_in_returns")
            continue
        w_prev = res.daily_weights[t_prev].reindex(tickers).fillna(0.0).to_numpy()
        r_t = returns.loc[dt, tickers].fillna(0.0).to_numpy()
        w_minus = _drift_weights(w_prev, r_t)
        w_plus = res.portfolio_weights[dt].reindex(tickers).fillna(0.0).to_numpy()
        delta = w_plus - w_minus
        turnover = float(np.abs(delta).sum())
        if turnover <= 0:
            _skip("zero_turnover")
            continue
        c, m, mask = clawback_cluster(delta, DECIMALS)
        if m == 0:
            _skip("empty_cluster")
            continue
        s_t = abs(c) * m / turnover
        if dt not in preds.index:
            _skip("date_not_in_predictions")
            continue
        mu = preds.loc[dt, tickers].to_numpy()
        fwd = _fwd_return(returns, dt, FWD)
        if fwd is None:
            _skip("no_fwd_window")
            continue
        fwd = fwd.reindex(tickers).to_numpy()
        bm_w = np.asarray(bm_fn(dt, tickers, len(tickers)), dtype=float)
        bm_fwd = float(np.nansum(bm_w * fwd))
        fwdex = fwd - bm_fwd
        lo_idx, hi_idx = split_by_mu(mu, mask)
        h = len(hi_idx)
        if h == 0:
            _skip("split_empty")
            continue
        if not (np.isfinite(fwdex[hi_idx]).any() and np.isfinite(fwdex[lo_idx]).any()):
            _skip("no_finite_fwdex")
            continue
        g_t = float(np.nanmean(fwdex[hi_idx]) - np.nanmean(fwdex[lo_idx]))
        rows.append({
            "date": str(pd.Timestamp(dt).date()),
            "c": float(c), "m": int(m), "h": int(h),
            "turnover": turnover, "S": float(s_t), "g": g_t,
            "contrib": abs(c) * h * g_t,
            "mu_sd": float(np.nanstd(mu[mask])), "bm_fwd": bm_fwd,
        })

    if not rows:
        sys.exit("[ABORT] 사용 가능한 리밸일 0건 — 보고 요망")
    df = pd.DataFrame(rows)

    # ---------- 게이트 ----------
    s_series = df["S"].to_numpy()
    S = float(np.mean(s_series))
    p0 = bool(S >= P0_BAR)

    contrib_list = df["contrib"].tolist()
    ghat = float(np.sum(contrib_list))
    third_sums = [float(np.sum(b)) for b in thirds(contrib_list)]
    p1 = bool(ghat > 0 and all(s > 0 for s in third_sums))

    years = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days / 365.25
    ghat_annual = ghat / years
    p2 = bool(ghat_annual >= P2_BAR_ANNUAL)

    overall = bool(p0 and p1 and p2)
    verdict = "PASS" if overall else "SHELVE"

    c_sign_counts = {
        "negative": int((df["c"] < 0).sum()),
        "zero": int((df["c"] == 0).sum()),
        "positive": int((df["c"] > 0).sum()),
    }
    n_used = int(len(df))
    n_skipped = int(sum(skip_reasons.values()))

    print(f"[diag] n_used={n_used} n_skipped={n_skipped} {skip_reasons}"
          f"  m_median={float(df['m'].median()):.1f}"
          f"  mu_sd_median={float(df['mu_sd'].median()):.3f}"
          f"  c_signs={c_sign_counts}  years={years:.2f}")
    print(f"[P0] S={S:.4f} bar={P0_BAR:.2f} -> {'PASS' if p0 else 'FAIL'}")
    print(f"[P1] Ghat={ghat:+.6f} thirds=["
          f"{third_sums[0]:+.6f}, {third_sums[1]:+.6f}, {third_sums[2]:+.6f}]"
          f" -> {'PASS' if p1 else 'FAIL'}")
    print(f"[P2] Ghat_annual={ghat_annual:+.6f} bar={P2_BAR_ANNUAL:.4f}"
          f" -> {'PASS' if p2 else 'FAIL'}")
    print(f"[VERDICT] {verdict}")

    summary = {
        "preregistration": "decision log §S13.48-A (2026-08-21)",
        "vintage": {"expected_last_daily_weights": EXPECTED_VINTAGE_DATE,
                    "last_daily_weights": last_daily,
                    "pkl_mtime": pkl_mtime,
                    "workbook_mtime": workbook_mtime,
                    "index_mtime": index_mtime,
                    "expected_workbook_mtime": EXPECTED_WORKBOOK_MTIME,
                    "expected_index_mtime": EXPECTED_INDEX_MTIME},
        "n_rebalances": int(len(dates)),
        "n_used": n_used,
        "n_skipped": n_skipped,
        "skip_reasons": skip_reasons,
        "decimals": DECIMALS,
        "fwd_days": FWD,
        "years": float(years),
        "p0": {"S": S, "bar": P0_BAR, "pass": p0},
        "p1": {"Ghat": ghat, "thirds_sums": third_sums, "pass": p1},
        "p2": {"Ghat_annual": ghat_annual, "bar": P2_BAR_ANNUAL, "pass": p2},
        "overall_pass": overall,
        "verdict": verdict,
        "diagnostics": {
            "m_median": float(df["m"].median()),
            "m_min": int(df["m"].min()),
            "m_max": int(df["m"].max()),
            "c_sign_counts": c_sign_counts,
            "c_abs_median": float(df["c"].abs().median()),
            "mu_sd_median": float(df["mu_sd"].median()),
            "g_mean": float(df["g"].mean()),
            "turnover_mean": float(df["turnover"].mean()),
            "S_series": {"mean": S,
                         "median": float(np.median(s_series)),
                         "min": float(np.min(s_series)),
                         "max": float(np.max(s_series)),
                         "std": float(np.std(s_series, ddof=1))
                         if len(s_series) > 1 else float("nan"),
                         "frac_ge_bar": float(np.mean(s_series >= P0_BAR))},
        },
        "assumptions": [
            "h_t = (finite-mu cluster members)//2; odd clusters drop the median "
            "member so the halves stay disjoint and equal-sized "
            "(preregistered m_t/2 is not integral for odd m_t)",
            "non-finite mu members are dropped before the split",
            "tail rebalances without a full 21BD forward window are skipped",
        ],
        "runtime_s": round(time.time() - t0, 1),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    df.to_csv(OUT_DIR / "per_rebalance.csv", index=False)
    print(f"saved: {OUT_DIR}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
