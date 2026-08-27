# -*- coding: utf-8 -*-
"""§S15 후보 B 사전점검 (read-only): 부호 안정 피처 monotone 제약.

63BD 비중첩 윈도우별 피처 CS rank-IC(윈도우 내 일별 Spearman 평균)로
부호 안정성을 측정한다. 자격 = 부호 일관성 ≥ 0.75 AND 윈도우 IC t-stat
|t| ≥ 2.5. 자격 피처 ≥ 5개면 PASS(arm 진행), 미만 SHELVE. 맵 = 자격 전
피처(부호 = IC 부호), 개수/부분집합/부호 스윕 금지 (결정 로그 §S15).

데이터원 = outputs/codex_causal_rank_65/backtest_result.pkl. IC 행렬은
§S15a와 공유 캐시(outputs/s15_prechecks/ic_matrix.csv). 읽기 전용,
인벤토리 비계수.

출력: outputs/s15_prechecks/{s15b_summary.json, s15b_feature_windows.csv,
      s15b_monotone_map.json}
"""

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

from scripts.precheck_s15a_feature_contri import per_date_ic_matrix

PKL_PATH = "outputs/codex_causal_rank_65/backtest_result.pkl"
OUT_DIR = "outputs/s15_prechecks"
IC_CACHE = os.path.join(OUT_DIR, "ic_matrix.csv")
WINDOW = 63
MIN_DATES_PER_WINDOW = 30
CONSISTENCY_MIN = 0.75
T_MIN = 2.5
MIN_QUALIFIED = 5


def window_mean_ics(ic_matrix: pd.DataFrame, window: int = WINDOW,
                    min_dates: int = MIN_DATES_PER_WINDOW) -> pd.DataFrame:
    """날짜축을 연속 비중첩 window-행 블록으로 잘라 블록별 피처 mean IC."""
    rows = []
    idx = []
    n = len(ic_matrix)
    for start in range(0, n, window):
        block = ic_matrix.iloc[start:start + window]
        if len(block) < min_dates:
            continue
        counts = block.notna().sum()
        mean = block.mean()
        mean[counts < min_dates] = np.nan
        rows.append(mean)
        idx.append(block.index[-1])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx, name="window_end"))


def qualify_features(win_ic: pd.DataFrame,
                     consistency_min: float = CONSISTENCY_MIN,
                     t_min: float = T_MIN) -> pd.DataFrame:
    """피처별 부호 일관성·t-stat 자격 판정."""
    recs = []
    for f in win_ic.columns:
        w = win_ic[f].dropna().to_numpy()
        n = len(w)
        if n < 8:
            recs.append({"feature": f, "n_windows": n, "mean_ic": np.nan,
                         "t": np.nan, "sign_consistency": np.nan,
                         "sign": 0, "qualified": False})
            continue
        mean = float(w.mean())
        std = float(w.std(ddof=1))
        t = mean / (std / np.sqrt(n)) if std > 0 else float("nan")
        sign = int(np.sign(mean)) if mean != 0 else 0
        nonzero = w[w != 0]
        consistency = (float((np.sign(nonzero) == sign).mean())
                       if len(nonzero) and sign != 0 else 0.0)
        qualified = bool(sign != 0 and consistency >= consistency_min
                         and np.isfinite(t) and abs(t) >= t_min)
        recs.append({"feature": f, "n_windows": n, "mean_ic": mean, "t": t,
                     "sign_consistency": consistency, "sign": sign,
                     "qualified": qualified})
    return pd.DataFrame(recs).set_index("feature")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(IC_CACHE):
        print(f"[S15b] loading cached IC matrix {IC_CACHE}")
        ic = pd.read_csv(IC_CACHE, index_col=0, parse_dates=True)
    else:
        print(f"[S15b] loading {PKL_PATH} ...")
        with open(PKL_PATH, "rb") as f:
            res = pickle.load(f)
        ic = per_date_ic_matrix(res.panel, res.targets)
        ic.to_csv(IC_CACHE)
    print(f"[S15b] IC matrix shape = {ic.shape}")

    win_ic = window_mean_ics(ic)
    print(f"[S15b] windows = {len(win_ic)}")
    qual = qualify_features(win_ic)
    qual.sort_values("t", key=lambda s: s.abs(), ascending=False).to_csv(
        os.path.join(OUT_DIR, "s15b_feature_windows.csv"))

    qualified = qual[qual["qualified"]]
    monotone_map = {f: int(s) for f, s in qualified["sign"].items()}
    n_q = len(qualified)
    verdict = "PASS" if n_q >= MIN_QUALIFIED else "SHELVE"
    summary = {
        "candidate": "S15-B monotone constraints on sign-stable features",
        "n_windows": int(len(win_ic)),
        "consistency_min": CONSISTENCY_MIN,
        "t_min": T_MIN,
        "min_qualified": MIN_QUALIFIED,
        "n_qualified": n_q,
        "qualified_features": {
            f: {"sign": int(qual.loc[f, "sign"]),
                "t": float(qual.loc[f, "t"]),
                "consistency": float(qual.loc[f, "sign_consistency"]),
                "mean_ic": float(qual.loc[f, "mean_ic"])}
            for f in qualified.index
        },
        "verdict": verdict,
    }
    with open(os.path.join(OUT_DIR, "s15b_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(OUT_DIR, "s15b_monotone_map.json"), "w") as f:
        json.dump(monotone_map, f, indent=2)
    print(f"[S15b] qualified features = {n_q} -> VERDICT: {verdict}")
    for f, s in monotone_map.items():
        print(f"    {f}: {'+' if s > 0 else '-'}1  (t={qual.loc[f, 't']:.2f}, "
              f"cons={qual.loc[f, 'sign_consistency']:.2f})")
    return summary


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result else 1)
