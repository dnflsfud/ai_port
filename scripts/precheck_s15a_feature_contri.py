# -*- coding: utf-8 -*-
"""§S15 후보 A 사전점검 (read-only): EWMA importance → feature_contri 이관.

production의 get_feature_weights(sqrt(ewma/mean) clip[0.5,2.0])는 X-곱으로는
no-op(§S12 입증)이므로, 같은 벡터를 LightGBM feature_contri(split gain 승수)로
이관하는 arm의 근거를 잰다. 데이터원 = outputs/codex_causal_rank_65/
backtest_result.pkl (S0(250) 08-26 산출물). 백테스트 재실행 0, 읽기 전용 —
§S13.7 선례로 인벤토리 비계수.

게이트(결정 로그 §S15, 측정 전 고정):
  G1 (지속성): 연속 비퇴화(유니크) 재훈련 간 split-importance 벡터의
     Spearman rank-autocorr(교집합 active feature 기준) 중앙값 ≥ 0.5.
  G2 (전방 정합): 재훈련 k의 EWMA(split) importance(업데이트 후 상태) vs
     (t_k, t_{k+1}] 윈도우 피처별 |CS rank-IC|의 피처 횡단 Spearman —
     윈도우 평균 > 0 AND t-stat ≥ 2 (윈도우 경계 = 유니크 모델 교체 시점,
     마지막 상태는 IC 행렬 마지막 날짜까지).
  둘 다 PASS 시에만 arm 실행. gain-importance는 참고 진단만(선택 축 아님).

출력: outputs/s15_prechecks/{s15a_summary.json, s15a_persistence.csv,
      s15a_alignment.csv, ic_matrix.csv(§S15b와 공유 캐시)}
"""

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

PKL_PATH = "outputs/codex_causal_rank_65/backtest_result.pkl"
OUT_DIR = "outputs/s15_prechecks"
IC_CACHE = os.path.join(OUT_DIR, "ic_matrix.csv")
EWMA_ALPHA = 0.3           # production ewma_alpha
MIN_TICKERS_PER_DATE = 30  # 이하면 그 날짜 IC는 NaN


def spearman(a, b) -> float:
    """pairwise-NaN 제거 Spearman. 유효쌍 < 3 이면 NaN."""
    s1 = pd.Series(np.asarray(a, dtype=float))
    s2 = pd.Series(np.asarray(b, dtype=float)).set_axis(s1.index)
    valid = s1.notna() & s2.notna()
    if int(valid.sum()) < 3:
        return float("nan")
    return float(s1[valid].corr(s2[valid], method="spearman"))


def unique_models_in_order(models: dict):
    """재훈련 날짜순 정렬 후, 퇴화 재사용(동일 객체)을 제거한 (date, model)
    목록. 같은 객체의 첫 등장만 유지."""
    seen = set()
    out = []
    for d in sorted(models.keys()):
        m = models[d]
        if id(m) in seen:
            continue
        seen.add(id(m))
        out.append((pd.Timestamp(d), m))
    return out


def map_importance(raw_imp, active_features, full_features) -> np.ndarray:
    """raw importance(active 순서)를 정규화(sum=1)해 full feature 공간으로 매핑.
    EWMAFeatureTracker.update의 매핑 단계 축자."""
    raw = np.asarray(raw_imp, dtype=float)
    total = raw.sum()
    norm = raw / total if total > 0 else np.ones(len(raw)) / len(raw)
    full = np.zeros(len(full_features))
    pos = {f: i for i, f in enumerate(full_features)}
    for i, f in enumerate(active_features):
        full[pos[f]] = norm[i]
    return full


def model_split_importance_full(model, full_features) -> np.ndarray:
    active = getattr(model, "_active_features", None)
    if active is None:
        raise ValueError("model has no _active_features attribute")
    raw = model.booster_.feature_importance(importance_type="split").astype(float)
    return map_importance(raw, active, full_features)


def consecutive_persistence(imp_list, active_sets):
    """연속 유니크 모델 쌍마다 교집합 active feature의 importance Spearman."""
    out = []
    for k in range(len(imp_list) - 1):
        common = sorted(active_sets[k] & active_sets[k + 1])
        if len(common) < 3:
            out.append(float("nan"))
            continue
        out.append(spearman(np.asarray(imp_list[k])[common],
                            np.asarray(imp_list[k + 1])[common]))
    return out


def reconstruct_ewma_states(imp_list, alpha, n_features):
    """EWMAFeatureTracker 재귀 축자: uniform 초기 → 업데이트마다
    state = alpha*imp + (1-alpha)*state 후 재정규화. 업데이트 후 상태 목록."""
    state = np.ones(n_features) / n_features
    states = []
    for imp in imp_list:
        state = alpha * np.asarray(imp, dtype=float) + (1 - alpha) * state
        total = state.sum()
        if total > 0:
            state = state / total
        states.append(state.copy())
    return states


def per_date_ic_matrix(panel: pd.DataFrame, targets: pd.DataFrame,
                       min_tickers: int = MIN_TICKERS_PER_DATE) -> pd.DataFrame:
    """날짜별 피처 vs 타깃 cross-sectional Spearman IC 행렬 (dates x features)."""
    target_stacked = targets.stack()
    target_stacked.index.names = ["date", "ticker"]
    rows = {}
    for d, x_group in panel.groupby(level="date", sort=True):
        y = target_stacked.reindex(x_group.index)
        valid = y.notna()
        if int(valid.sum()) < min_tickers:
            continue
        xg = x_group.loc[valid]
        yg = y.loc[valid]
        rows[d] = xg.corrwith(yg, method="spearman")
    return pd.DataFrame(rows).T.sort_index()


def window_alignment(states, state_dates, ic_matrix: pd.DataFrame):
    """상태 k vs (t_k, t_{k+1}] 윈도우 피처별 |mean IC|의 피처 횡단 Spearman."""
    out = []
    boundaries = list(state_dates) + [ic_matrix.index.max()]
    for k in range(len(states)):
        lo, hi = boundaries[k], boundaries[k + 1]
        if not (hi > lo):
            continue
        win = ic_matrix.loc[(ic_matrix.index > lo) & (ic_matrix.index <= hi)]
        if len(win) < 5:
            continue
        abs_ic = win.mean().abs()
        out.append({
            "state_date": str(pd.Timestamp(lo).date()),
            "window_end": str(pd.Timestamp(hi).date()),
            "n_window_dates": int(len(win)),
            "spearman": spearman(np.asarray(states[k], dtype=float),
                                 abs_ic.to_numpy()),
        })
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[S15a] loading {PKL_PATH} ...")
    with open(PKL_PATH, "rb") as f:
        res = pickle.load(f)
    full_features = list(res.feature_names)
    uniq = unique_models_in_order(res.models)
    print(f"[S15a] retained model entries {len(res.models)}, unique {len(uniq)}")

    imp_split, imp_gain, active_sets, dates = [], [], [], []
    pos = {f: i for i, f in enumerate(full_features)}
    for d, m in uniq:
        active = getattr(m, "_active_features", None)
        if active is None:
            raise SystemExit("FATAL: model without _active_features")
        imp_split.append(model_split_importance_full(m, full_features))
        raw_gain = m.booster_.feature_importance(importance_type="gain").astype(float)
        imp_gain.append(map_importance(raw_gain, active, full_features))
        active_sets.append({pos[f] for f in active})
        dates.append(pd.Timestamp(d))

    # --- G1: persistence ---
    pers_split = consecutive_persistence(imp_split, active_sets)
    pers_gain = consecutive_persistence(imp_gain, active_sets)
    g1_median = float(np.nanmedian(pers_split)) if pers_split else float("nan")
    g1_pass = bool(g1_median >= 0.5)
    pd.DataFrame({
        "pair_end_date": [str(d.date()) for d in dates[1:]],
        "spearman_split": pers_split,
        "spearman_gain_ref": pers_gain,
    }).to_csv(os.path.join(OUT_DIR, "s15a_persistence.csv"), index=False)
    print(f"[S15a] G1 median split-importance autocorr = {g1_median:.4f} "
          f"(n={len(pers_split)}) -> {'PASS' if g1_pass else 'FAIL'}")

    # --- IC matrix (shared cache) ---
    if os.path.exists(IC_CACHE):
        print(f"[S15a] loading cached IC matrix {IC_CACHE}")
        ic = pd.read_csv(IC_CACHE, index_col=0, parse_dates=True)
    else:
        print("[S15a] computing per-date IC matrix (dates x features) ...")
        ic = per_date_ic_matrix(res.panel, res.targets)
        ic.to_csv(IC_CACHE)
    print(f"[S15a] IC matrix shape = {ic.shape}")

    # --- G2: forward alignment (split-EWMA = arm이 실제 쓰는 상태) ---
    states = reconstruct_ewma_states(imp_split, EWMA_ALPHA, len(full_features))
    ic_cols = ic.reindex(columns=full_features)
    align = window_alignment(states, dates, ic_cols)
    align_df = pd.DataFrame(align)
    align_df.to_csv(os.path.join(OUT_DIR, "s15a_alignment.csv"), index=False)
    vals = align_df["spearman"].dropna().to_numpy() if len(align_df) else np.array([])
    g2_mean = float(vals.mean()) if len(vals) else float("nan")
    g2_t = (float(g2_mean / (vals.std(ddof=1) / np.sqrt(len(vals))))
            if len(vals) > 2 else float("nan"))
    g2_pass = bool(g2_mean > 0 and g2_t >= 2.0)
    print(f"[S15a] G2 mean cross-feature spearman = {g2_mean:.4f}, "
          f"t = {g2_t:.2f} (n={len(vals)}) -> {'PASS' if g2_pass else 'FAIL'}")

    # gain-EWMA 참고 진단 (선택 축 아님)
    states_gain = reconstruct_ewma_states(imp_gain, EWMA_ALPHA, len(full_features))
    align_gain = window_alignment(states_gain, dates, ic_cols)
    vg = (pd.DataFrame(align_gain)["spearman"].dropna().to_numpy()
          if align_gain else np.array([]))
    g2_gain_mean = float(vg.mean()) if len(vg) else float("nan")
    g2_gain_t = (float(g2_gain_mean / (vg.std(ddof=1) / np.sqrt(len(vg))))
                 if len(vg) > 2 else float("nan"))

    verdict = "PASS" if (g1_pass and g2_pass) else "SHELVE"
    summary = {
        "candidate": "S15-A ewma_contri (feature_contri handoff)",
        "n_model_entries": len(res.models),
        "n_unique_models": len(uniq),
        "G1_median_split_autocorr": g1_median,
        "G1_threshold": 0.5,
        "G1_pass": g1_pass,
        "G2_mean_spearman": g2_mean,
        "G2_t": g2_t,
        "G2_n_windows": int(len(vals)),
        "G2_pass": g2_pass,
        "gain_reference": {"mean_spearman": g2_gain_mean, "t": g2_gain_t},
        "verdict": verdict,
    }
    with open(os.path.join(OUT_DIR, "s15a_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[S15a] VERDICT: {verdict}")
    return summary


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result else 1)
