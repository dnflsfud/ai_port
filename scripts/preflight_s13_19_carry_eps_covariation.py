# -*- coding: utf-8 -*-
"""§S13.19 사전점검: S0 캐리(gap) 성분의 지수 EPS 사이클 공변 여부.

신규 백테스트 없음 — §S13.12 산출물(per_date_S0.csv)의 리밸런싱별
realized/spec_cap에서 gap = realized − spec_cap을 재구성하고, 리밸런싱 시점의
지수 선행-EPS 피처(index_eps.py 수식 동일) 터실로 조건화한다.

게이트 (결정 로그 §S13.19 사전등록, 1차 조건 변수 = fac_eps_g63 고정):
  G1: |Δgap(top−bottom 터실)| ≥ 1.5%/yr
  G2: 표본 반분(48/48) 모두 Δgap 부호 동일
  G3: |Δgap| > |Δspec_cap| (공변이 캐리 축에 국지화)
하나라도 실패 → arm 미설계 SHELVE. 나머지 3개 스프레드는 서술 전용.

실행: <PY> scripts/preflight_s13_19_carry_eps_covariation.py  (WD=ai_port, PYTHONPATH=.)
산출물: outputs/s13_19_carry_eps_preflight.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import PipelineConfig  # noqa: E402

EPS_SPREADS = ("fac_eps_g63", "fac_eps_us_lead63",
               "fac_eps_us_lead252", "fac_eps_tech_lead63")
PRIMARY = "fac_eps_g63"

# §S13.12 전기간 앵커 (×12 연환산) — CSV 오염/오독 가드
ANCHOR_REALIZED = 0.0560
ANCHOR_SPEC = 0.0330
ANCHOR_ATOL = 0.005

G1_MIN_ABS_DELTA = 0.015


def eps_features(px: pd.DataFrame) -> pd.DataFrame:
    """src/features/index_eps.py와 동일 수식 (Factor_PX_LAST 직접 입력)."""
    eps = {k: np.log(px[f"{k}_FWD_EPS"]) for k in ("SPX", "NDX", "MXWD")}
    return pd.DataFrame({
        "fac_eps_g63": eps["MXWD"].diff(63),
        "fac_eps_us_lead63": eps["SPX"].diff(63) - eps["MXWD"].diff(63),
        "fac_eps_us_lead252": eps["SPX"].diff(252) - eps["MXWD"].diff(252),
        "fac_eps_tech_lead63": eps["NDX"].diff(63) - eps["SPX"].diff(63),
    }, index=px.index)


def assign_terciles(cond: pd.Series) -> pd.Series:
    """전표본 터실 라벨 (rank pct ≤1/3 → bottom, >2/3 → top)."""
    pct = cond.rank(pct=True, method="average")
    return pd.Series(
        np.where(pct <= 1.0 / 3.0, "bottom",
                 np.where(pct > 2.0 / 3.0, "top", "mid")),
        index=cond.index)


def tercile_delta(vals: pd.Series, cond: pd.Series) -> dict:
    """버킷별 연환산 평균(×12)과 top−bottom 델타."""
    b = assign_terciles(cond)
    out = {k: float(vals[b == k].mean()) * 12.0
           for k in ("bottom", "mid", "top")}
    out["delta"] = out["top"] - out["bottom"]
    return out


def half_deltas(vals: pd.Series, cond: pd.Series) -> dict:
    """전표본 버킷 고정, 시간 반분별 top−bottom 델타(×12). 빈 버킷 → NaN."""
    b = assign_terciles(cond)
    half = len(vals) // 2
    res = {}
    for tag, sl in (("H1", slice(None, half)), ("H2", slice(half, None))):
        v, bb = vals.iloc[sl], b.iloc[sl]
        top, bot = v[bb == "top"], v[bb == "bottom"]
        res[tag] = (float(top.mean() - bot.mean()) * 12.0
                    if len(top) and len(bot) else float("nan"))
    return res


def judge_gates(gap_full: dict, gap_halves: dict, spec_full: dict) -> dict:
    d = gap_full["delta"]
    h1, h2 = gap_halves["H1"], gap_halves["H2"]
    g1 = bool(abs(d) >= G1_MIN_ABS_DELTA)
    g2 = bool(np.isfinite(h1) and np.isfinite(h2) and h1 * h2 > 0.0)
    g3 = bool(abs(d) > abs(spec_full["delta"]))
    return {"G1": g1, "G2": g2, "G3": g3,
            "verdict": "PROCEED" if (g1 and g2 and g3) else "SHELVE"}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    per_date = pd.read_csv(
        root / "outputs/s13_12_ic_ir_transmission/per_date_S0.csv",
        parse_dates=["date"]).set_index("date")
    realized_ann = per_date["realized"].mean() * 12.0
    spec_ann = per_date["spec_cap"].mean() * 12.0
    assert abs(realized_ann - ANCHOR_REALIZED) < ANCHOR_ATOL, realized_ann
    assert abs(spec_ann - ANCHOR_SPEC) < ANCHOR_ATOL, spec_ann
    gap = per_date["realized"] - per_date["spec_cap"]
    print(f"per_date_S0: n={len(per_date)} "
          f"{per_date.index[0].date()}~{per_date.index[-1].date()}  "
          f"realized {realized_ann:+.4f}/yr spec {spec_ann:+.4f}/yr "
          f"gap {(realized_ann - spec_ann):+.4f}/yr (§S13.12 앵커 재현)")

    px = pd.read_excel(PipelineConfig().data_path, sheet_name="Factor_PX_LAST",
                       index_col=0, engine="openpyxl")
    px.index = pd.DatetimeIndex(px.index)
    feats = eps_features(px)
    cond_all = feats.reindex(per_date.index, method="ffill")
    exact = feats.index.intersection(per_date.index)
    print(f"feature alignment: {len(exact)}/{len(per_date)} exact, "
          f"나머지 ffill\n")

    rows = []
    for name in EPS_SPREADS:
        cond = cond_all[name]
        gap_full = tercile_delta(gap, cond)
        spec_full = tercile_delta(per_date["spec_cap"], cond)
        halves = half_deltas(gap, cond)
        rows.append({"spread": name, "primary": name == PRIMARY,
                     **{f"gap_{k}": gap_full[k]
                        for k in ("bottom", "mid", "top", "delta")},
                     "spec_delta": spec_full["delta"],
                     "gap_delta_H1": halves["H1"],
                     "gap_delta_H2": halves["H2"]})
    table = pd.DataFrame(rows).set_index("spread")
    print(table.round(4).to_string())

    cond = cond_all[PRIMARY]
    gates = judge_gates(tercile_delta(gap, cond), half_deltas(gap, cond),
                        tercile_delta(per_date["spec_cap"], cond))
    prim = table.loc[PRIMARY]
    print(f"\n==== §S13.19 gates (1차 = {PRIMARY}) ====")
    print(f"G1 |Δgap| {abs(prim['gap_delta']):.4f} >= {G1_MIN_ABS_DELTA}"
          f"  -> {'PASS' if gates['G1'] else 'FAIL'}")
    print(f"G2 halves Δgap: H1 {prim['gap_delta_H1']:+.4f} / "
          f"H2 {prim['gap_delta_H2']:+.4f}  "
          f"-> {'PASS' if gates['G2'] else 'FAIL'}")
    print(f"G3 |Δgap| {abs(prim['gap_delta']):.4f} > "
          f"|Δspec| {abs(prim['spec_delta']):.4f}  "
          f"-> {'PASS' if gates['G3'] else 'FAIL'}")
    print(f"\n§S13.19 verdict: {gates['verdict']}"
          f"{' (캐리-조건화 arm 설계 착수)' if gates['verdict'] == 'PROCEED' else ' (arm 미설계)'}")

    table.to_csv(root / "outputs/s13_19_carry_eps_preflight.csv",
                 encoding="utf-8-sig")
    print("saved: outputs/s13_19_carry_eps_preflight.csv")
    return 0 if gates["verdict"] == "PROCEED" else 1


if __name__ == "__main__":
    sys.exit(main())
