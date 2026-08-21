# -*- coding: utf-8 -*-
"""§S13.48-C 사전점검 (read-only): mega funding_set 이산 재선택 히스테리시스.

`portfolio_optimizer.py:373-392`는 매 리밸 funding_set을 처음부터 재선택한다
(mega = {bm >= bm_thr}, eligible = {mu < score_max} 엄격부등, mu 오름차순 상위 k).
mega 간 mu 순위 치환만으로 멤버십이 뒤집히면 그 종목의 제약이 wide UW 허용 ↔
w >= bm 강제 핀 사이를 점프한다. 그 이산 재선택이 (a) 실재하고 (b) 통상
리밸런싱으로 설명되지 않는 증분 거래를 만들며 (c) 경제적으로 유의한지를
측정한다. arm 미실행 — 사전점검만.

사전등록 정의(고정, 결정 로그 §S13.48-C):
  w⁻_t = _drift_weights(daily_weights[t_prev], returns_masked.loc[t]),
  t_prev = daily_weights 인덱스 중 t보다 엄격히 작은 최대값,
  w⁺_t = res.portfolio_weights[t], δ_t = w⁺_t − w⁻_t,
  turnover_t = Σ|δ_t| (two-way L1).
  fs_t = funding_set(mu_t, bm_t) — 옵티마이저 선택 로직의 정확한 미러링.
  flip_t = fs_t △ fs_{t-1} (대칭차).

게이트(고정):
  P0     = flip이 1건 이상인 리밸 비율 f >= 0.25
  P1(i)  = (t, i∈fs_t) 중 w⁻_i < bm_i − 0.005 인 비율 >= 0.95 (프록시 타당성)
  P1(ii) = median|δ| (flip) / median|δ| (비-flip mega) >= 2.0
  P2     = Σ_flip |δ| / Σ_t turnover_t >= 0.05
종합 = 네 게이트 전부 PASS여야 PASS, 아니면 SHELVE.

출력: outputs/s13_48c_mega_hysteresis/summary.json
"""

import bisect
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

AI_PORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_PORT))

VARIANT = AI_PORT / "variants" / "codex_causal_rank_65.yaml"
PKL = AI_PORT / "outputs" / "codex_causal_rank_65" / "backtest_result.pkl"
OUT_DIR = AI_PORT / "outputs" / "s13_48c_mega_hysteresis"

UW_TOL = 0.005
P0_BAR = 0.25     # flip 발생 리밸 비율 f
P1I_BAR = 0.95    # 프록시 일치율
P1II_BAR = 2.0    # flip |delta| 중앙값 / non-flip mega |delta| 중앙값
P2_BAR = 0.05     # flip 귀속 |delta| 합 / 전체 two-way L1
EXPECTED_VINTAGE_DATE = "2026-08-19"
EXPECTED_WORKBOOK_MTIME = "2026-08-19 13:49"
EXPECTED_INDEX_MTIME = "2026-08-21 11:16"


# ---------------------------------------------------------------------------
# 순수 헬퍼 (단위테스트 대상)
# ---------------------------------------------------------------------------

def funding_set(mu: np.ndarray, bm: np.ndarray, bm_thr: float,
                score_max: float, k: int) -> set:
    """src/portfolio_optimizer.py:373-385의 funding_set 선택을 그대로 미러링.

    mega = {i : bm_i >= bm_thr}, 점수 = mu_i(비유한이면 0.0),
    eligible = {점수 < score_max} (엄격부등), 점수 오름차순 안정정렬 상위 k.
    k <= 0 이거나 mega가 비면 빈 set(원본의 `funding_k > 0 and mega_indices`
    분기가 성립하지 않아 제약이 생성되지 않는 상태).
    """
    if k <= 0:
        return set()
    mega_indices = [i for i in range(len(bm)) if bm[i] >= bm_thr]
    if not mega_indices:
        return set()
    scored = [
        (i, mu[i] if np.isfinite(mu[i]) else 0.0)
        for i in mega_indices
    ]
    eligible = [(i, s) for i, s in scored if s < score_max]
    eligible.sort(key=lambda x: x[1])
    return {i for i, _ in eligible[:k]}


def thirds(values: list) -> list:
    """시간순 절단 3분할 (§S13.41 관용: np.linspace(0, n, 4, dtype=int))."""
    x = np.asarray(values, dtype=float)
    cut = np.linspace(0, len(x), 4, dtype=int)
    return [x[cut[i]:cut[i + 1]] for i in range(3)]


def _mtime_kst(path) -> str:
    return pd.Timestamp(
        Path(path).stat().st_mtime, unit="s", tz="UTC"
    ).tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S")


def _dist(values: list) -> dict:
    """리스트의 n·최소·중앙·평균·최대 (빈 리스트는 nan)."""
    if not values:
        return {"n": 0, "min": float("nan"), "median": float("nan"),
                "mean": float("nan"), "max": float("nan")}
    a = np.asarray(values, dtype=float)
    return {"n": int(len(a)), "min": float(np.min(a)),
            "median": float(np.median(a)), "mean": float(np.mean(a)),
            "max": float(np.max(a))}


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

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

    res = pickle.load(open(PKL, "rb"))
    daily_keys = sorted(res.daily_weights)
    last_daily = str(pd.Timestamp(daily_keys[-1]).date())

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
                 f"daily {last_daily}. 진행하지 않고 보고만.")
    if not getattr(cfg, "mega_cap_protection_enabled", False):
        sys.exit("[ABORT] mega_cap_protection_enabled != True — "
                 "production 전제 파괴, 보고 요망")
    if not getattr(cfg, "mega_cap_funding_mode", False):
        sys.exit("[ABORT] mega_cap_funding_mode != True — "
                 "production 전제 파괴, 보고 요망")

    bm_thr = float(cfg.mega_cap_bm_threshold)
    score_max = float(cfg.mega_cap_funding_score_max)
    funding_k = int(cfg.mega_cap_funding_k)

    data = UniverseData(cfg.data_path, config=cfg)
    if getattr(data, "returns_masked", None) is None:
        sys.exit("[ABORT] returns_masked 없음 — masked USD 수익률 가정 파괴, 보고 요망")
    returns = data.returns_masked

    dates = sorted(res.portfolio_weights)
    tickers = list(res.portfolio_weights[dates[0]].index)
    bm_fn = get_benchmark_fn(data, tickers, config=cfg)
    print(f"[load] {time.time()-t0:.0f}s  rebalances={len(dates)}"
          f" ({pd.Timestamp(dates[0]).date()} ~ {pd.Timestamp(dates[-1]).date()})"
          f"  tickers={len(tickers)}  mega(bm_thr={bm_thr}, k={funding_k},"
          f" score_max={score_max})")

    # ---------- 리밸별 사전/사후 비중·funding_set ----------
    recs = []
    skip_reasons = Counter()
    for dt in dates:
        if dt not in res.predictions.index:
            skip_reasons["no_prediction_row"] += 1
            continue
        if dt not in returns.index:
            skip_reasons["no_return_row"] += 1
            continue
        pos = bisect.bisect_left(daily_keys, dt)
        if pos == 0:
            skip_reasons["no_prior_daily_weights"] += 1
            continue
        t_prev = daily_keys[pos - 1]

        bm_w = np.asarray(bm_fn(dt, tickers, len(tickers)), dtype=float)
        mu = res.predictions.loc[dt, tickers].to_numpy(dtype=float)
        fs = funding_set(mu, bm_w, bm_thr, score_max, funding_k)

        w_minus = _drift_weights(
            res.daily_weights[t_prev].reindex(tickers).fillna(0.0).to_numpy(),
            returns.loc[dt, tickers].fillna(0.0).to_numpy(),
        )
        w_plus = res.portfolio_weights[dt].reindex(tickers).fillna(0.0).to_numpy()
        delta = w_plus - w_minus
        recs.append({
            "date": pd.Timestamp(dt),
            "bm": bm_w, "mu": mu, "fs": fs, "delta": delta,
            "w_minus": w_minus,
            "turnover": float(np.abs(delta).sum()),
            "mega_idx": [i for i, b in enumerate(bm_w) if b >= bm_thr],
        })

    n_used = len(recs)
    if n_used < 2:
        sys.exit(f"[ABORT] 사용 가능 리밸 {n_used}건 — 판정 불가, 보고 요망")

    # ---------- flip 집계 (직전 '사용된' 리밸 대비) ----------
    for j, r in enumerate(recs):
        prev_fs = recs[j - 1]["fs"] if j > 0 else None
        r["has_prev"] = prev_fs is not None
        r["flip"] = set() if prev_fs is None else (r["fs"] ^ prev_fs)
        r["entered"] = set() if prev_fs is None else (r["fs"] - prev_fs)
        r["exited"] = set() if prev_fs is None else (prev_fs - r["fs"])
    paired = [r for r in recs if r["has_prev"]]

    # ---------- P0: flip 발생 리밸 비율 ----------
    flip_flags = [1.0 if r["flip"] else 0.0 for r in paired]
    f_rate = float(np.mean(flip_flags))
    p0 = bool(f_rate >= P0_BAR)

    # ---------- P1(i): 프록시 타당성 ----------
    proxy_hit, proxy_tot = 0, 0
    for r in recs:
        for i in r["fs"]:
            proxy_tot += 1
            if r["w_minus"][i] < r["bm"][i] - UW_TOL:
                proxy_hit += 1
    proxy_rate = float(proxy_hit / proxy_tot) if proxy_tot else float("nan")
    p1i = bool(proxy_tot > 0 and proxy_rate >= P1I_BAR)

    # ---------- P1(ii): flip |δ| vs 비-flip mega |δ| ----------
    flip_abs, nonflip_abs = [], []
    for r in paired:
        for i in r["flip"]:
            flip_abs.append(abs(float(r["delta"][i])))
        for i in r["mega_idx"]:
            if i not in r["flip"]:
                nonflip_abs.append(abs(float(r["delta"][i])))
    med_flip = float(np.median(flip_abs)) if flip_abs else float("nan")
    med_nonflip = float(np.median(nonflip_abs)) if nonflip_abs else float("nan")
    ratio = (med_flip / med_nonflip
             if (np.isfinite(med_flip) and np.isfinite(med_nonflip)
                 and med_nonflip > 0) else float("nan"))
    p1ii = bool(np.isfinite(ratio) and ratio >= P1II_BAR)

    # ---------- P2: flip 귀속 거래 점유율 ----------
    flip_l1 = float(np.sum(flip_abs)) if flip_abs else 0.0
    total_l1 = float(np.sum([r["turnover"] for r in recs]))
    share = flip_l1 / total_l1 if total_l1 > 0 else float("nan")
    p2 = bool(np.isfinite(share) and share >= P2_BAR)

    overall = bool(p0 and p1i and p1ii and p2)
    verdict = "PASS" if overall else "SHELVE"

    # ---------- 진단 (비액션) ----------
    entered_mu = [float(r["mu"][i]) for r in paired for i in r["entered"]]
    exited_mu = [float(r["mu"][i]) for r in paired for i in r["exited"]]
    year_flip, year_total = Counter(), Counter()
    for r in paired:
        y = int(r["date"].year)
        year_total[y] += 1
        if r["flip"]:
            year_flip[y] += 1
    flip_rate_thirds = [float(np.mean(b)) if len(b) else float("nan")
                        for b in thirds(flip_flags)]

    diagnostics = {
        "flip_count_per_rebalance": _dist([len(r["flip"]) for r in paired]),
        "flip_count_histogram": {str(k): int(v) for k, v in sorted(
            Counter(len(r["flip"]) for r in paired).items())},
        "mega_count_per_rebalance": _dist([len(r["mega_idx"]) for r in recs]),
        "funding_set_size_per_rebalance": _dist([len(r["fs"]) for r in recs]),
        "entered_mu": _dist(entered_mu),
        "exited_mu": _dist(exited_mu),
        "flip_rebalances_by_year": {str(y): {"flip": int(year_flip.get(y, 0)),
                                             "total": int(year_total[y])}
                                    for y in sorted(year_total)},
        "flip_rate_thirds": flip_rate_thirds,
        "turnover_per_rebalance": _dist([r["turnover"] for r in recs]),
        "median_abs_delta_flip": med_flip,
        "median_abs_delta_nonflip_mega": med_nonflip,
    }

    print(f"[P0]     f={f_rate:.3f} (flip 리밸 {int(sum(flip_flags))}/{len(paired)})"
          f"  bar {P0_BAR}  -> {'PASS' if p0 else 'FAIL'}")
    print(f"[P1(i)]  proxy {proxy_rate:.4f} ({proxy_hit}/{proxy_tot})"
          f"  bar {P1I_BAR}  -> {'PASS' if p1i else 'FAIL'}")
    print(f"[P1(ii)] ratio {ratio:.3f} (flip {med_flip:.6f} / non-flip"
          f" {med_nonflip:.6f}, n {len(flip_abs)}/{len(nonflip_abs)})"
          f"  bar {P1II_BAR}  -> {'PASS' if p1ii else 'FAIL'}")
    print(f"[P2]     share {share:.4f} ({flip_l1:.4f} / {total_l1:.4f})"
          f"  bar {P2_BAR}  -> {'PASS' if p2 else 'FAIL'}")
    print(f"[VERDICT] {verdict}")

    summary = {
        "preregistration": "decision log §S13.48-C (2026-08-21)",
        "vintage": {"expected_date": EXPECTED_VINTAGE_DATE,
                    "expected_workbook_mtime": EXPECTED_WORKBOOK_MTIME,
                    "expected_index_mtime": EXPECTED_INDEX_MTIME,
                    "pkl_mtime": pkl_mtime,
                    "workbook_mtime": workbook_mtime,
                    "index_mtime": index_mtime,
                    "last_daily_weights": last_daily},
        "mega_params_from_cfg": {
            "mega_cap_bm_threshold": bm_thr,
            "mega_cap_funding_k": funding_k,
            "mega_cap_funding_score_max": score_max,
            "mega_cap_protection_enabled": bool(cfg.mega_cap_protection_enabled),
            "mega_cap_funding_mode": bool(cfg.mega_cap_funding_mode)},
        "uw_tol": UW_TOL,
        "n_rebalances": int(len(dates)),
        "n_used": int(n_used),
        "n_paired": int(len(paired)),
        "n_skipped": int(sum(skip_reasons.values())),
        "skip_reasons": {k: int(v) for k, v in skip_reasons.items()},
        "p0": {"flip_rebalance_rate": f_rate, "n_flip": int(sum(flip_flags)),
               "n_paired": int(len(paired)), "bar": P0_BAR, "pass": p0},
        "p1_i": {"proxy_agreement": proxy_rate, "hits": int(proxy_hit),
                 "pairs": int(proxy_tot), "bar": P1I_BAR, "pass": p1i},
        "p1_ii": {"ratio": ratio, "median_abs_delta_flip": med_flip,
                  "median_abs_delta_nonflip_mega": med_nonflip,
                  "n_flip_pairs": int(len(flip_abs)),
                  "n_nonflip_pairs": int(len(nonflip_abs)),
                  "bar": P1II_BAR, "pass": p1ii},
        "p2": {"flip_l1_share": share, "flip_l1": flip_l1,
               "total_two_way_l1": total_l1, "bar": P2_BAR, "pass": p2},
        "overall_pass": overall,
        "verdict": verdict,
        "diagnostics": diagnostics,
        "assumptions": [
            "fs_{t-1} = 직전 '사용된' 리밸의 funding_set (스킵된 리밸은 건너뜀)",
            "첫 사용 리밸은 flip 정의 불가 → P0·P1(ii) 표본에서 제외; "
            "P1(i)와 P2 분모(Σ turnover_t)는 사용된 전 리밸 포함",
            "entered/exited mu는 당일 t의 mu (그 선택을 만든 값)",
            "delta는 사후 w+ 와 드리프트 사전 w- 의 차 — two-way L1",
        ],
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    pd.DataFrame([{
        "date": str(r["date"].date()),
        "n_mega": len(r["mega_idx"]),
        "funding_set_size": len(r["fs"]),
        "funding_set": "|".join(sorted(tickers[i] for i in r["fs"])),
        "n_flip": len(r["flip"]),
        "entered": "|".join(sorted(tickers[i] for i in r["entered"])),
        "exited": "|".join(sorted(tickers[i] for i in r["exited"])),
        "flip_abs_delta_sum": float(sum(abs(float(r["delta"][i]))
                                        for i in r["flip"])),
        "turnover": r["turnover"],
    } for r in recs]).to_csv(OUT_DIR / "by_date.csv", index=False)

    print(f"[done] {time.time()-t0:.0f}s  saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
