# -*- coding: utf-8 -*-
"""§S20 사전점검 (read-only): Sales_Revision(FY2) 시트가 production 점수에 증분 예측력을 갖는가.

결정 로그 §S20 (2026-09-19). arm 없음 — 백테스트 0회, 오프라인 재해만.
표준 데이터원 = 09-18 production pkl(`outputs/codex_causal_rank_65`: targets=20d fwd,
predictions=executable score, panel·활성 62피처) + 09-18 ai_signal_data 번들의
`Factset_Sales_Revision`(FY1, production 사용 중) / `Factset_Sales_Revision_FY2`(신규) 시트.

클리닝은 production 경로 그대로(`get_cleaned_revision`: reversion_gated 15/50/0.5,
연장 상한 21BD) → `build_bounded_revision_features` 로 FY1 과 동일 피처족 생성.

게이트(측정 전 고정, §S13.36/§S16.8 관용구):
  G1  잔차 IC(vs executable score) NW(lag20) |t| ≥ 2
  G2  잔차 IC(vs executable score + FY1 대응 피처) NW |t| ≥ 2  — FY1 이 이미 production
      에 있으므로 **결정 게이트 = G2**(FY1 을 넘는 증분만 새 정보).
  P1  중복 진단(비게이트): FY1 대응 피처·활성 62피처와의 per-date Spearman 중앙값.
  P2  3분할 잔차 IC 부호 일관(비게이트, 정보성).
자기검증: 재구성한 FY1 `sales_rev_ma_63d` 가 production panel 열과 rank 동치인지.

출력: outputs/s20_precheck.json
"""
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import yaml

AI_PORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_PORT))
BUNDLE = Path(r"C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\ai_signal_data.xlsx")
PKL = AI_PORT / "outputs" / "codex_causal_rank_65" / "backtest_result.pkl"
OUT = AI_PORT / "outputs" / "s20_precheck.json"
NW_LAG, T_BAR = 20, 2.0
warnings.filterwarnings("ignore", category=RuntimeWarning)


def read_sheet(path: Path, name: str) -> pd.DataFrame:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = list(wb[name].iter_rows(values_only=True))
    wb.close()
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").apply(pd.to_numeric, errors="coerce")


def rowwise_rank_corr(a: pd.DataFrame, b: pd.DataFrame, min_n: int = 10) -> pd.Series:
    """행(날짜)별 횡단면 Spearman — 각 행의 비결측 순위 후 공통 유효쌍만 Pearson."""
    d = a.index.intersection(b.index)
    c = a.columns.intersection(b.columns)
    x, y = a.loc[d, c].rank(axis=1), b.loc[d, c].rank(axis=1)
    m = x.notna() & y.notna()
    n = m.sum(axis=1)
    x, y = x.where(m), y.where(m)
    xc = x.sub(x.mean(axis=1), axis=0)
    yc = y.sub(y.mean(axis=1), axis=0)
    rho = (xc * yc).sum(axis=1) / np.sqrt((xc ** 2).sum(axis=1) * (yc ** 2).sum(axis=1))
    return rho[n >= min_n].dropna()


def daily_rank_ic(feature: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    return rowwise_rank_corr(feature, target)


def residualize(f: pd.DataFrame, regs: list) -> pd.DataFrame:
    """날짜별 횡단면 OLS(절편 포함)로 regressors 성분을 제거한 잔차."""
    d, c = f.index, f.columns
    for r in regs:
        d, c = d.intersection(r.index), c.intersection(r.columns)
    F = f.loc[d, c].values
    R = np.stack([r.loc[d, c].values for r in regs], axis=-1)
    out = np.full(F.shape, np.nan)
    for i in range(len(d)):
        y, X = F[i], R[i]
        m = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if m.sum() < 10:
            continue
        Xm = np.column_stack([np.ones(int(m.sum())), X[m]])
        beta, *_ = np.linalg.lstsq(Xm, y[m], rcond=None)
        out[i, m] = y[m] - Xm @ beta
    return pd.DataFrame(out, index=d, columns=c)


def nw_t(s: pd.Series, lag: int = NW_LAG) -> float:
    x = s.dropna().values
    n = len(x)
    if n < lag + 2:
        return float("nan")
    e = x - x.mean()
    var = float(e @ e) / n
    for k in range(1, lag + 1):
        var += 2.0 * (1 - k / (lag + 1)) * float(e[:-k] @ e[k:]) / n
    return float(x.mean() / np.sqrt(var / n))


def median_cs_corr(a: pd.DataFrame, b: pd.DataFrame, dates) -> float:
    idx = pd.DatetimeIndex(dates)
    rho = rowwise_rank_corr(a.loc[a.index.intersection(idx)], b.loc[b.index.intersection(idx)])
    return float(rho.median()) if len(rho) else float("nan")


def thirds(s: pd.Series) -> list:
    n, k = len(s), len(s) // 3
    return [round(float(s.iloc[i * k:((i + 1) * k if i < 2 else n)].mean()), 5) for i in range(3)]


class _Shim:
    """get_cleaned_revision 이 요구하는 최소 인터페이스(get_sheet)."""

    def __init__(self, sheets):
        self.sheets = sheets

    def get_sheet(self, name):
        return self.sheets[name]


def main() -> None:
    from src.features.sellside import build_bounded_revision_features, get_cleaned_revision
    from src.harness import build_override_config

    t0 = time.time()
    manifest = yaml.safe_load(
        (AI_PORT / "variants" / "codex_causal_rank_65.yaml").read_text(encoding="utf-8"))
    cfg = build_override_config(dict(manifest["overrides"]))
    r = pickle.load(open(PKL, "rb"))
    targets, score, panel = r.targets, r.predictions, r.panel
    active = list(r.models[max(r.models)]._active_features)
    print(f"pkl ok {time.time()-t0:.0f}s targets {targets.shape} active {len(active)}")

    fy1_raw = read_sheet(BUNDLE, "Factset_Sales_Revision").reindex(index=targets.index, columns=targets.columns)
    fy2_raw = read_sheet(BUNDLE, "Factset_Sales_Revision_FY2").reindex(index=targets.index, columns=targets.columns)
    print(f"sheets ok {time.time()-t0:.0f}s")

    shim = _Shim({"Factset_Sales_Revision": fy1_raw, "Factset_Sales_Revision_FY2": fy2_raw})
    fy1 = get_cleaned_revision(shim, "Factset_Sales_Revision", config=cfg)
    fy2 = get_cleaned_revision(shim, "Factset_Sales_Revision_FY2", config=cfg)
    f1 = build_bounded_revision_features(fy1, "sales_rev")
    f2 = build_bounded_revision_features(fy2, "sales_rev_fy2")

    sample_dates = list(targets.index[::21])
    panel_wide = {f: panel[f].unstack("ticker") for f in active}

    # ---- 자기검증: 재구성 FY1 ma_63d vs production panel(per-date z-score 후) ----
    rec = f1["sales_rev_ma_63d"]
    prod = panel_wide["sales_rev_ma_63d"].reindex(index=rec.index, columns=rec.columns)
    m = (rec.notna() & prod.notna()).values
    rec_z = rec.sub(rec.mean(axis=1), axis=0).div(rec.std(axis=1), axis=0)
    selfcheck = {
        "median_cs_spearman": round(median_cs_corr(rec, prod, sample_dates), 4),
        "pooled_pearson_raw": round(float(np.corrcoef(rec.values[m], prod.values[m])[0, 1]), 4),
        "pooled_pearson_cs_zscored": round(float(np.corrcoef(rec_z.values[m], prod.values[m])[0, 1]), 4),
        "aligned_cells": int(m.sum()),
    }
    print("self-check:", selfcheck)

    cands = {
        "C1_fy2_ma_63d": (f2["sales_rev_fy2_ma_63d"], f1["sales_rev_ma_63d"]),
        "C2_fy2_level": (f2["sales_rev_fy2"], f1["sales_rev"]),
        "C3_fy2_trend": (f2["sales_rev_fy2_trend"], f1["sales_rev_trend"]),
        "C4_fy2_diff_21d": (f2["sales_rev_fy2_diff_21d"], f1["sales_rev_diff_21d"]),
        "C5_spread_ma_63d": (f2["sales_rev_fy2_ma_63d"] - f1["sales_rev_ma_63d"], f1["sales_rev_ma_63d"]),
        "C6_spread_level": (f2["sales_rev_fy2"] - f1["sales_rev"], f1["sales_rev"]),
        "REF_fy1_ma_63d_in_production": (f1["sales_rev_ma_63d"], None),
    }
    result = {
        "preregistration": "decision log §S20 (2026-09-19)",
        "pkl": str(PKL),
        "bundle_mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(BUNDLE.stat().st_mtime)),
        "target": "20d fwd (production forward_horizon)",
        "nw_lag": NW_LAG, "t_bar": T_BAR,
        "gates": {"G1": "resid IC vs executable score NW|t|>=2",
                  "G2": "resid IC vs score + FY1 counterpart NW|t|>=2 (decision gate)"},
        "selfcheck_fy1_reconstruction": selfcheck,
        "candidates": {},
    }
    for name, (feat, fy1c) in cands.items():
        ic = daily_rank_ic(feat, targets)
        icr = daily_rank_ic(residualize(feat, [score]), targets)
        row = {
            "coverage_days": int(len(ic)),
            "nonnan_share": round(float(feat.notna().mean().mean()), 3),
            "raw_ic_mean": round(float(ic.mean()), 5), "raw_ic_t_nw": round(nw_t(ic), 2),
            "resid_ic_mean": round(float(icr.mean()), 5), "resid_ic_t_nw": round(nw_t(icr), 2),
            "resid_ic_thirds": thirds(icr),
        }
        row["G1"] = "PASS" if np.isfinite(row["resid_ic_t_nw"]) and abs(row["resid_ic_t_nw"]) >= T_BAR else "FAIL"
        if fy1c is not None:
            icr2 = daily_rank_ic(residualize(feat, [score, fy1c]), targets)
            row["resid_ic_vs_score_fy1_mean"] = round(float(icr2.mean()), 5)
            row["resid_ic_vs_score_fy1_t_nw"] = round(nw_t(icr2), 2)
            row["resid_ic_vs_score_fy1_thirds"] = thirds(icr2)
            row["cs_spearman_vs_fy1_counterpart"] = round(median_cs_corr(feat, fy1c, sample_dates), 3)
            row["G2"] = ("PASS" if np.isfinite(row["resid_ic_vs_score_fy1_t_nw"])
                         and abs(row["resid_ic_vs_score_fy1_t_nw"]) >= T_BAR else "FAIL")
            row["gate"] = "PROCEED" if row["G2"] == "PASS" else "SHELVE"
        corrs = {f: median_cs_corr(feat, panel_wide[f], sample_dates) for f in active}
        ranked = sorted(corrs.items(), key=lambda kv: -abs(kv[1]) if np.isfinite(kv[1]) else 0)
        row["top3_corr_active"] = [(k, round(v, 3)) for k, v in ranked[:3]]
        result["candidates"][name] = row
        print(f"{name:30s} n={row['coverage_days']} rawIC={row['raw_ic_mean']:+.4f}(t{row['raw_ic_t_nw']:+.2f}) "
              f"G1 residIC={row['resid_ic_mean']:+.4f}(t{row['resid_ic_t_nw']:+.2f}) thirds={row['resid_ic_thirds']}"
              + (f" | G2 t{row['resid_ic_vs_score_fy1_t_nw']:+.2f} rho_fy1={row['cs_spearman_vs_fy1_counterpart']}"
                 f" -> {row['gate']}" if fy1c is not None else ""))
        print(f"{'':30s} top3 corr active: {row['top3_corr_active']}")

    proceed = [k for k, v in result["candidates"].items() if v.get("gate") == "PROCEED"]
    result["verdict"] = "PROCEED: " + ", ".join(proceed) if proceed else "SHELVE (no candidate passes G2)"
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nverdict: {result['verdict']}\ndone {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
