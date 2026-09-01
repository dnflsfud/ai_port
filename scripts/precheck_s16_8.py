# -*- coding: utf-8 -*-
"""§S16.8 사전점검 (read-only): 단위 수정 후 OCF 피처 2종의 예측력·중복 진단.

결정 로그 §S16.8 사전등록(2026-09-01)의 재실행 가능한 사전점검.
표준 데이터원 = 새 기준선 pkl(panel·targets·predictions, 재실행 0) + 워크북 시트.

  P1 (중복 진단, 비게이트): 신규 피처 vs 활성 62피처의 per-date Spearman 중앙값
     — 최대 상관 축 보고(|ρ|≥0.8이면 중복 경고).
  P2 (예측력): 일별 횡단면 rank IC(vs 21d fwd target)와 executable score 잔차
     IC. **arm B 게이트 = 잔차 IC NW(lag20) |t| ≥ 2** (§S13.36 잔차 IC 우선 +
     §S13.43 NW 관용구; 21d 타깃 중첩 자기상관 보정). arm A는 사용자 지시
     재도전이라 비게이트(결과는 병기).

출력: outputs/s16_8_precheck.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AI_PORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_PORT))

BASE_PKL = AI_PORT / "outputs" / "s16_7_name_risk_cap" / "backtest_result.pkl"
OUT = AI_PORT / "outputs" / "s16_8_precheck.json"
NW_LAG = 20
T_BAR = 2.0


def daily_rank_ic(feature: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    """일별 횡단면 Spearman IC (공통 날짜·티커, 유효쌍 10개 미만 날짜 제외)."""
    common_d = feature.index.intersection(target.index)
    common_c = feature.columns.intersection(target.columns)
    f = feature.loc[common_d, common_c].rank(axis=1)
    t = target.loc[common_d, common_c].rank(axis=1)
    out = {}
    for d in common_d:
        fr, tr = f.loc[d], t.loc[d]
        mask = fr.notna() & tr.notna()
        if int(mask.sum()) < 10:
            continue
        out[d] = float(np.corrcoef(fr[mask], tr[mask])[0, 1])
    return pd.Series(out).sort_index()


def residualize_on(feature: pd.DataFrame, score: pd.DataFrame) -> pd.DataFrame:
    """날짜별 횡단면 OLS로 score 성분 제거한 잔차 피처."""
    common_d = feature.index.intersection(score.index)
    common_c = feature.columns.intersection(score.columns)
    f = feature.loc[common_d, common_c]
    s = score.loc[common_d, common_c]
    fc = f.sub(f.mean(axis=1), axis=0)
    sc = s.sub(s.mean(axis=1), axis=0)
    beta = (fc * sc).mean(axis=1) / (sc * sc).mean(axis=1).replace(0, np.nan)
    return fc.sub(sc.mul(beta, axis=0))


def nw_tstat(series: pd.Series, lag: int = NW_LAG) -> float:
    """Newey-West(lag) 보정 평균 t-통계량."""
    x = series.dropna().values
    n = len(x)
    if n < lag + 2:
        return float("nan")
    mu = x.mean()
    e = x - mu
    var = float(e @ e) / n
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        var += 2.0 * w * float(e[:-k] @ e[k:]) / n
    return float(mu / np.sqrt(var / n))


def evaluate_precheck_b(t_resid_nw: float, bar: float = T_BAR) -> dict:
    """arm B 게이트: 잔차 IC NW |t| ≥ 2 → PROCEED (부호 무관 — 트리 모델)."""
    ok = bool(np.isfinite(t_resid_nw) and abs(t_resid_nw) >= bar)
    return {"t_resid_nw": round(float(t_resid_nw), 3), "bar": bar, "proceed": ok}


def _median_cs_corr(feature: pd.DataFrame, other: pd.DataFrame, dates) -> float:
    vals = []
    common_c = feature.columns.intersection(other.columns)
    for d in dates:
        if d not in feature.index or d not in other.index:
            continue
        a = feature.loc[d, common_c].rank()
        b = other.loc[d, common_c].rank()
        mask = a.notna() & b.notna()
        if int(mask.sum()) < 10:
            continue
        vals.append(float(np.corrcoef(a[mask], b[mask])[0, 1]))
    return float(np.median(vals)) if vals else float("nan")


def main() -> None:
    import pickle

    import yaml

    from src.data_loader import UniverseData
    from src.features.utils import (cross_sectional_zscore, rolling_tsz,
                                    safe_pct_change)
    from src.harness import build_override_config, inject_config

    manifest = yaml.safe_load(
        (AI_PORT / "variants" / "codex_causal_rank_65.yaml").read_text(encoding="utf-8"))
    cfg = build_override_config(dict(manifest["overrides"]))
    inject_config(cfg)
    data = UniverseData(cfg.data_path, config=cfg)

    opcf = data.get_sheet("Factset_Fwd_OpCashflow")
    fcf = data.get_sheet("BEST_CALCULATED_FCF")
    feat_a = cross_sectional_zscore(rolling_tsz(opcf, window=756, min_periods=252))
    feat_b = (cross_sectional_zscore(safe_pct_change(opcf, 252))
              - cross_sectional_zscore(safe_pct_change(fcf, 252)))

    r = pickle.load(open(BASE_PKL, "rb"))
    targets = r.targets
    score = r.predictions          # executable(오버레이 후) 예측
    panel = r.panel
    feats62 = list(r.models[max(r.models)]._active_features)

    result = {"preregistration": "decision log §S16.8 (2026-09-01)",
              "base_pkl": str(BASE_PKL), "nw_lag": NW_LAG}
    sample_dates = list(targets.index[::21])

    for name, feat in [("A_fwd_opcf_level_z", feat_a),
                       ("B_fwd_opcf_invest_divergence", feat_b)]:
        ic = daily_rank_ic(feat, targets)
        resid = residualize_on(feat, score)
        ic_resid = daily_rank_ic(resid, targets)
        # P1: 활성 62피처와의 중복(21일 간격 표본 날짜)
        corrs = {}
        for f62 in feats62:
            try:
                s = panel[f62]
                non_date = [n for n in s.index.names if n != "date"][0]
                other = s.unstack(non_date)          # index=date, columns=ticker
            except Exception:
                continue
            corrs[f62] = _median_cs_corr(feat, other, sample_dates)
        top_corr = sorted(corrs.items(), key=lambda x: -abs(x[1]))[:3] if corrs else []
        result[name] = {
            "coverage_days": int(len(ic)),
            "raw_ic_mean": round(float(ic.mean()), 5),
            "raw_ic_t_nw": round(nw_tstat(ic), 3),
            "resid_ic_mean": round(float(ic_resid.mean()), 5),
            "resid_ic_t_nw": round(nw_tstat(ic_resid), 3),
            "p1_top_corr": [(k, round(v, 3)) for k, v in top_corr],
        }

    result["arm_b_gate"] = evaluate_precheck_b(
        result["B_fwd_opcf_invest_divergence"]["resid_ic_t_nw"])
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\narm B gate: {'PROCEED' if result['arm_b_gate']['proceed'] else 'SHELVE'}"
          f" (잔차 IC NW t = {result['arm_b_gate']['t_resid_nw']})")


if __name__ == "__main__":
    main()
