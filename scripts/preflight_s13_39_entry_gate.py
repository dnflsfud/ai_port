# -*- coding: utf-8 -*-
"""§S13.39 사전점검 (read-only): 옵션 리스크 turnover-중립 진입 게이트.

§S13.38 arm A(피처 주입)는 빠른 옵션 신호가 랭킹을 재편해 캐리를 훼손
(E1 FAIL −0.276). 후속 후보는 **기존 포지션을 건드리지 않고, 어차피
일어나는 신규 진입/증량에만 개입**하는 turnover-중립 게이트다. 이 스크립트는
arm 구현 전 두 게이트를 S0′(s0_recert_s13_38)의 실제 리밸런싱 기록으로
실측한다:

  P1 (바인딩): 리밸런싱당 평균 진입/증량 건수 ≥ 3 AND 게이트 신호 극단
     (사전등록: 사용자 원안 iv30_z > 2 AND downside_skew_z > 2) 겹침
     ≥ 진입의 10%. 미달이면 게이트가 inert — arm 미구현 SHELVE.
  P2 (판별력): 극단 진입 vs 비극단 진입의 21d 선행수익 스프레드가
     음(−)이고 |t| ≥ 2 & 서브기간(3분할) 부호 일관. 양(+)이거나 비유의면
     컷 방향이 정당화되지 않음 — SHELVE.

진입/증량 = 연속 리밸런싱 목표가중 차 Δw > no_trade_band(0.003).
OR·단일 신호 분할은 참고 진단(비액션)으로만 병기한다.

출력: outputs/s13_39_entry_gate_preflight/summary.json
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

AI_PORT = Path(__file__).resolve().parents[1]
S0_PKL = AI_PORT / "outputs" / "s0_recert_s13_38" / "backtest_result.pkl"
OUT_DIR = AI_PORT / "outputs" / "s13_39_entry_gate_preflight"

BAND = 0.003          # no_trade_band — 실거래로 치는 최소 증량
Z_THRESHOLD = 2.0     # 사용자 원안 (Skew_Z>2 & IV_Z>2)
FWD_DAYS = 21


def detect_increases(prev: pd.Series, curr: pd.Series, band: float) -> dict:
    """직전 대비 band 초과 증량(신규 포함)된 종목 -> Δw. 감소는 무시."""
    prev = prev.reindex(curr.index).fillna(0.0)
    delta = (curr - prev).round(10)
    hit = delta[delta > band]
    return {str(k): float(v) for k, v in hit.items()}


def event_spread(events: pd.DataFrame) -> dict:
    """extreme vs rest의 fwd 평균 차이와 Welch t. 그룹이 비면 NaN."""
    ext = events.loc[events["extreme"], "fwd"].dropna()
    rest = events.loc[~events["extreme"], "fwd"].dropna()
    if len(ext) < 2 or len(rest) < 2:
        return {"n_extreme": int(len(ext)), "n_rest": int(len(rest)),
                "mean_diff": float("nan"), "t": float("nan")}
    diff = float(ext.mean() - rest.mean())
    se = float(np.sqrt(ext.var(ddof=1) / len(ext) + rest.var(ddof=1) / len(rest)))
    return {
        "n_extreme": int(len(ext)),
        "n_rest": int(len(rest)),
        "mean_diff": round(diff, 5),
        "t": round(diff / se, 2) if se > 0 else float("nan"),
    }


def main() -> None:
    from scripts.run_s13_38_option_risk_precheck import WORKBOOK, load_panel, simple

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mtime = pd.Timestamp(WORKBOOK.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")
    print(f"workbook mtime: {mtime}")

    result = pickle.load(open(S0_PKL, "rb"))
    pw = {pd.Timestamp(k): v for k, v in result.portfolio_weights.items()}
    rebalance_dates = sorted(pw.keys())
    print(f"rebalances: {len(rebalance_dates)}")

    xl = pd.ExcelFile(WORKBOOK)
    meta = pd.read_excel(xl, sheet_name="Universe_Meta")
    universe = [simple(t) for t in meta["Ticker"].dropna().astype(str)]
    returns = load_panel(xl, "Daily_Returns", universe)
    iv_z = load_panel(xl, "iv30_z", universe)
    skew_z = load_panel(xl, "downside_skew_z", universe)
    fwd = returns.rolling(FWD_DAYS).sum().shift(-FWD_DAYS)

    def asof(panel: pd.DataFrame, d: pd.Timestamp) -> pd.Series:
        sub = panel.loc[:d]
        return sub.iloc[-1] if len(sub) else pd.Series(dtype=float)

    rows = []
    for prev_d, d in zip(rebalance_dates[:-1], rebalance_dates[1:]):
        increases = detect_increases(pw[prev_d], pw[d], BAND)
        if not increases:
            continue
        iv_d, skew_d = asof(iv_z, d), asof(skew_z, d)
        fwd_d = fwd.loc[d] if d in fwd.index else pd.Series(dtype=float)
        for ticker, delta in increases.items():
            rows.append({
                "date": d,
                "ticker": ticker,
                "delta_w": delta,
                "iv_z": float(iv_d.get(ticker, np.nan)),
                "skew_z": float(skew_d.get(ticker, np.nan)),
                "fwd": float(fwd_d.get(ticker, np.nan)),
            })
    events = pd.DataFrame(rows)
    n_rebals = len(rebalance_dates) - 1
    per_rebal = len(events) / n_rebals
    print(f"increase events: {len(events)} over {n_rebals} rebalances "
          f"({per_rebal:.1f}/rebal)")

    variants = {
        "AND(iv>2 & skew>2)": (events["iv_z"] > Z_THRESHOLD) & (events["skew_z"] > Z_THRESHOLD),
        "OR(iv>2 | skew>2)": (events["iv_z"] > Z_THRESHOLD) | (events["skew_z"] > Z_THRESHOLD),
        "iv_z>2": events["iv_z"] > Z_THRESHOLD,
        "skew_z>2": events["skew_z"] > Z_THRESHOLD,
    }

    summary: dict = {
        "workbook_mtime": mtime,
        "n_rebalances": n_rebals,
        "n_increase_events": int(len(events)),
        "events_per_rebalance": round(per_rebal, 2),
        "band": BAND,
        "z_threshold": Z_THRESHOLD,
        "variants": {},
    }
    for name, mask in variants.items():
        ev = events.assign(extreme=mask.fillna(False))
        spread = event_spread(ev)
        share = spread["n_extreme"] / max(len(events), 1)
        # 서브기간(3분할) 부호 일관
        sorted_ev = ev.sort_values("date").reset_index(drop=True)
        cut = np.linspace(0, len(sorted_ev), 4, dtype=int)
        thirds = [sorted_ev.iloc[cut[i]:cut[i + 1]] for i in range(3)]
        sub = []
        for t in thirds:
            s = event_spread(t)
            sub.append(s["mean_diff"])
        signs_ok = (
            all(np.isfinite(s) for s in sub)
            and all(np.sign(s) == np.sign(spread["mean_diff"]) for s in sub)
        ) if np.isfinite(spread["mean_diff"]) else False
        entry = {
            **spread,
            "extreme_share": round(share, 3),
            "subperiod_mean_diff": [round(s, 5) if np.isfinite(s) else None for s in sub],
            "sign_consistent": bool(signs_ok),
        }
        summary["variants"][name] = entry
        print(f"\n[{name}] {json.dumps(entry, ensure_ascii=False)}")

    primary = summary["variants"]["AND(iv>2 & skew>2)"]
    p1 = per_rebal >= 3 and primary["extreme_share"] >= 0.10
    p2 = (
        np.isfinite(primary["t"])
        and primary["t"] <= -2.0
        and primary["mean_diff"] < 0
        and primary["sign_consistent"]
    )
    summary["gates"] = {"P1_binding": bool(p1), "P2_discrimination": bool(p2)}
    print(f"\nGATES: P1={p1}  P2={p2}  -> "
          f"{'PROCEED' if (p1 and p2) else 'SHELVE'}")

    events.to_csv(OUT_DIR / "increase_events.csv", index=False, encoding="utf-8")
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
