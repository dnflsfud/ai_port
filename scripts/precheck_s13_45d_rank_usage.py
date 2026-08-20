# -*- coding: utf-8 -*-
"""§S13.45-D 사전점검 (read-only): ndcg@k 사용률 감사.

production objective는 rank_xendcg + rank_eval_at [5,10] — 학습이 top-5/10에
집중하는데 실제 북은 어느 mu-랭크 구간을 쓰는가를 측정한다. arm 미실행.

사전등록 정의(고정): 각 리밸런싱 t에서 active = w_t − b_t(cap-weighted bm),
랭크 = 당일 res.predictions 행(executable mu) 내림차순(비-NaN만).
  (a) gross OW active(Σ max(active,0))의 상위랭크 버킷별 점유율(시점 평균)
  (b) gross UW active(Σ max(−active,0))의 하위랭크 버킷 동형
  (c) 버킷별 전진 21BD 실현 active 기여 = Σ(active_i × fwd21_i)
버킷: [1–5, 6–20, 21–50, 51–100, 101+] (OW=상위랭크, UW=하위랭크 역순 동형).

게이트(고정): P0 = mu-top-5 밖 gross OW 점유율(시점 평균) ≥ 40%.
P1 = OW 버킷 6–50(6–20+21–50) 기여 전체 합 > 0 AND 시간순 3분할 ≥2/3 양.
종합 = P0∧P1 (PASS/SHELVE).

가정(구현 재량 기록): (c)의 active_i는 부호 그대로(사전등록 식 축자) —
버킷 합계가 21BD 실현 active 수익 분해가 되도록 한다. (a)(b)의 max()와 달리
(c)는 식에 max가 없다. 동순위는 rank method='first'(결정적).

출력: outputs/s13_45d_rank_usage/summary.json
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

PKL = AI_PORT / "outputs" / "codex_causal_rank_65" / "backtest_result.pkl"
VARIANT = AI_PORT / "variants" / "codex_causal_rank_65.yaml"
OUT_DIR = AI_PORT / "outputs" / "s13_45d_rank_usage"

FWD = 21
BUCKETS = [("1-5", 1, 5), ("6-20", 6, 20), ("21-50", 21, 50),
           ("51-100", 51, 100), ("101+", 101, 10**9)]
P0_MIN_OUTSIDE_TOP5 = 0.40
EXPECTED_VINTAGE_DATE = "2026-08-19"  # 08-19 빈티지 pkl — 불일치 시 중단·보고


# ---------------------------------------------------------------------------
# 순수 헬퍼 (단위테스트 대상)
# ---------------------------------------------------------------------------

def mu_ranks(mu_row: pd.Series):
    """비-NaN 종목의 (top_rank, bottom_rank) — 1부터, ties method='first'."""
    mu = mu_row.dropna()
    top = mu.rank(ascending=False, method="first").astype(int)
    bottom = mu.rank(ascending=True, method="first").astype(int)
    return top, bottom


def bucket_shares(active: pd.Series, ranks: pd.Series, side: str) -> dict:
    """(a)/(b): gross OW/UW active 대비 랭크 버킷 점유율 (+unranked 잔여).

    분모는 전 종목 gross(비-NaN 랭크 밖 종목 포함) — 버킷+unranked 합=1.
    """
    part = active.clip(lower=0.0) if side == "ow" else (-active).clip(lower=0.0)
    gross = float(part.sum())
    if gross <= 0:
        shares = {lab: float("nan") for lab, _, _ in BUCKETS}
        shares["unranked"] = float("nan")
        return {"gross": gross, "shares": shares}
    shares = {}
    for lab, lo, hi in BUCKETS:
        members = ranks.index[(ranks >= lo) & (ranks <= hi)]
        shares[lab] = float(part.reindex(members).sum() / gross)
    shares["unranked"] = float(part.loc[~part.index.isin(ranks.index)].sum() / gross)
    return {"gross": gross, "shares": shares}


def bucket_contribs(active: pd.Series, ranks: pd.Series, fwd: pd.Series) -> dict:
    """(c): 버킷별 Σ(active_i × fwd21_i) — 부호 있는 active 축자 적용."""
    out = {}
    for lab, lo, hi in BUCKETS:
        members = ranks.index[(ranks >= lo) & (ranks <= hi)]
        out[lab] = float((active.reindex(members) * fwd.reindex(members))
                         .dropna().sum())
    return out


def thirds_signs(values: list) -> list:
    """시간순 3분할 각 블록 합의 부호(+1/−1/0)."""
    signs = []
    for block in np.array_split(np.asarray(values, dtype=float), 3):
        s = float(np.nansum(block)) if len(block) else float("nan")
        signs.append(0 if not np.isfinite(s) or s == 0 else (1 if s > 0 else -1))
    return signs


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

def main() -> None:
    from run_variant import compose_config, load_manifest
    from src.backtest import get_benchmark_fn
    from src.data_loader import UniverseData

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cfg = compose_config(load_manifest(VARIANT))
    workbook_mtime = pd.Timestamp(
        Path(cfg.data_path).stat().st_mtime, unit="s", tz="UTC"
    ).tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S")
    pkl_mtime = pd.Timestamp(
        PKL.stat().st_mtime, unit="s", tz="UTC"
    ).tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S")

    res = pickle.load(open(PKL, "rb"))
    daily_keys = sorted(res.daily_weights)
    last_daily = str(pd.Timestamp(daily_keys[-1]).date())
    dates = sorted(res.portfolio_weights.keys())

    # ---------- 실행 선행 검증 (08-19 빈티지 — 불일치 시 진행 금지) ----------
    print(f"[vintage] pkl mtime {pkl_mtime}  workbook mtime {workbook_mtime}"
          f"  last daily_weights {last_daily}")
    vintage_ok = (last_daily == EXPECTED_VINTAGE_DATE
                  and pkl_mtime[:10] == EXPECTED_VINTAGE_DATE)
    if not vintage_ok:
        sys.exit(f"[ABORT] 빈티지 불일치 — 기대 {EXPECTED_VINTAGE_DATE}, "
                 f"pkl {pkl_mtime[:10]} / daily {last_daily}. 진행하지 않고 보고만.")
    if getattr(res, "predictions", None) is None:
        sys.exit("[ABORT] pkl에 predictions 없음 — 보고 요망")

    data = UniverseData(cfg.data_path, config=cfg)
    if getattr(data, "returns_masked", None) is None:
        sys.exit("[ABORT] returns_masked 없음 — masked USD 수익률 가정 파괴, 보고 요망")
    returns = data.returns_masked
    tickers = list(data.tickers)
    bm_fn = get_benchmark_fn(data, tickers, config=cfg)
    preds = res.predictions

    print(f"[load] {time.time()-t0:.0f}s  rebalances={len(dates)}"
          f" ({pd.Timestamp(dates[0]).date()} ~ {pd.Timestamp(dates[-1]).date()})"
          f"  returns={returns.shape}")

    share_rows, contrib_rows = [], []
    skipped = 0
    for dt in dates:
        if dt not in preds.index:
            skipped += 1
            continue
        top, bottom = mu_ranks(preds.loc[dt])
        if len(top) < 30:
            skipped += 1
            continue
        w = res.portfolio_weights[dt].reindex(tickers).fillna(0.0)
        b = pd.Series(np.asarray(bm_fn(dt, tickers, len(tickers)), dtype=float),
                      index=tickers)
        active = w - b
        ow = bucket_shares(active, top, "ow")
        uw = bucket_shares(active, bottom, "uw")
        srow = {"date": str(pd.Timestamp(dt).date()), "n_valid": int(len(top)),
                "gross_ow": ow["gross"], "gross_uw": uw["gross"],
                "outside_top5": 1.0 - ow["shares"]["1-5"],
                "outside_top20": 1.0 - ow["shares"]["1-5"] - ow["shares"]["6-20"]}
        for lab, _, _ in BUCKETS:
            srow[f"ow_share_{lab}"] = ow["shares"][lab]
            srow[f"uw_share_{lab}"] = uw["shares"][lab]
        srow["ow_share_unranked"] = ow["shares"]["unranked"]
        srow["uw_share_unranked"] = uw["shares"]["unranked"]
        share_rows.append(srow)

        fwd = _fwd_return(returns, dt, FWD)
        if fwd is None:
            continue  # 꼬리 시점 — (c)만 제외, (a)(b)는 유지
        c_ow = bucket_contribs(active, top, fwd)
        c_uw = bucket_contribs(active, bottom, fwd)
        crow = {"date": srow["date"],
                "p1_contrib_6_50": c_ow["6-20"] + c_ow["21-50"]}
        for lab, _, _ in BUCKETS:
            crow[f"ow_contrib_{lab}"] = c_ow[lab]
            crow[f"uw_contrib_{lab}"] = c_uw[lab]
        contrib_rows.append(crow)

    sdf = pd.DataFrame(share_rows)
    cdf = pd.DataFrame(contrib_rows)

    # ---------- 게이트 ----------
    share_outside_top5 = float(sdf["outside_top5"].mean())
    p0 = bool(share_outside_top5 >= P0_MIN_OUTSIDE_TOP5)

    p1_sum = float(cdf["p1_contrib_6_50"].sum())
    p1_signs = thirds_signs(cdf["p1_contrib_6_50"].tolist())
    p1 = bool(p1_sum > 0 and sum(1 for s in p1_signs if s > 0) >= 2)
    overall = bool(p0 and p1)
    verdict = "PASS" if overall else "SHELVE"

    bucket_table = {}
    for lab, _, _ in BUCKETS:
        bucket_table[lab] = {
            "ow_share_mean": float(sdf[f"ow_share_{lab}"].mean()),
            "uw_share_mean": float(sdf[f"uw_share_{lab}"].mean()),
            "ow_contrib_sum": float(cdf[f"ow_contrib_{lab}"].sum()),
            "uw_contrib_sum": float(cdf[f"uw_contrib_{lab}"].sum()),
            "ow_contrib_thirds_signs": thirds_signs(cdf[f"ow_contrib_{lab}"].tolist()),
        }
    bucket_table["unranked"] = {
        "ow_share_mean": float(sdf["ow_share_unranked"].mean()),
        "uw_share_mean": float(sdf["uw_share_unranked"].mean()),
    }

    print(f"\n{'bucket':>8} | {'OW share':>9} {'UW share':>9} |"
          f" {'OW contrib':>11} {'UW contrib':>11}")
    for lab, _, _ in BUCKETS:
        r = bucket_table[lab]
        print(f"{lab:>8} | {r['ow_share_mean']:>9.1%} {r['uw_share_mean']:>9.1%} |"
              f" {r['ow_contrib_sum']:>+11.4f} {r['uw_contrib_sum']:>+11.4f}")
    print(f"[diag] outside_top5 {share_outside_top5:.1%}"
          f"  outside_top20 {float(sdf['outside_top20'].mean()):.1%}"
          f"  n_valid avg {float(sdf['n_valid'].mean()):.1f}"
          f" (min {int(sdf['n_valid'].min())})")
    print(f"[gates] P0={p0} (outside_top5 {share_outside_top5:.1%} >= 40%)"
          f"  P1={p1} (sum {p1_sum:+.4f}, thirds {p1_signs})")

    summary = {
        "preregistration": "decision log §S13.45-D (2026-08-19)",
        "vintage": {"expected_date": EXPECTED_VINTAGE_DATE,
                    "pkl_mtime": pkl_mtime,
                    "workbook_mtime": workbook_mtime,
                    "last_daily_weights": last_daily,
                    "match": vintage_ok},
        "sample_n": int(len(sdf)),
        "contrib_n": int(len(cdf)),
        "skipped_dates": skipped,
        "n_valid_avg": float(sdf["n_valid"].mean()),
        "p0": {"share_outside_top5": share_outside_top5,
               "threshold": P0_MIN_OUTSIDE_TOP5, "pass": p0},
        "p1": {"contribution_sum": p1_sum, "subperiod_signs": p1_signs,
               "pass": p1},
        "overall_pass": overall,
        "verdict": verdict,
        "bucket_table": bucket_table,
        "diagnostics": {
            "share_outside_top20": float(sdf["outside_top20"].mean()),
            "gross_ow_avg": float(sdf["gross_ow"].mean()),
            "gross_uw_avg": float(sdf["gross_uw"].mean()),
        },
        "assumptions": [
            "(c) contribution uses signed active_i (preregistered formula "
            "verbatim; buckets decompose the realized 21BD active return)",
            "rank ties broken by method='first' (deterministic)",
            "(a)/(b) denominators are all-name gross OW/UW; NaN-mu remainder "
            "reported as 'unranked' bucket",
        ],
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    sdf.to_csv(OUT_DIR / "shares_by_date.csv", index=False)
    cdf.to_csv(OUT_DIR / "contribs_by_date.csv", index=False)
    print(f"\nVERDICT: {verdict}  ({time.time()-t0:.0f}s)  saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
