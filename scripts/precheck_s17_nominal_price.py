# -*- coding: utf-8 -*-
"""§S17.3 G1-01b(M1) 명목가 분모 — 데이터 게이트 사전점검 (read-only, 백테스트 0회).

결정 로그 §S17.3 사전등록(2026-09-03)의 데이터 게이트 A~D 를 09-03 빈티지 워크북
(PX_LAST_UNADJ 시트 포함)에서 확인한다. UniverseData 를 production overrides +
nominal_price_source=PX_LAST_UNADJ 로 1회 로드하고, sellside 피처를 OFF/ON 두 config 로
만들어 비교한다(OFF 는 local_prices, ON 은 local_prices_nominal 을 분모로 쓴다).

  A 무배당 5종(TSLA·AMZN·ADBE·NFLX·ISRG): 원시 PX_LAST_UNADJ == PX_LAST 전 구간 값 동일
  B MO 2014-06-30 원시 UNADJ ≈ 41.94 (|Δ| ≤ 0.01)
  C 고배당 12종(2014-06-30 PX_LAST/UNADJ 비 최저 12) 2014 median(TG/PX):
      OFF 관측(보고서 2.02) → ON ∈ [1.00, 1.25]
  D 같은 군 tg_upside_z 격차 = median z(2014~2019) − median z(2025~2026):
      OFF 관측(보고서 +1.56) → ON |격차| < 0.5
관측: 내재 주식수(mktcap/px) MO 2014 OFF/ON 비, MO 2014-06-30 PE OFF/ON, 로더 진단 키.

출력: outputs/s17_prechecks/nominal_price_gates.json
"""

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AI_PORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_PORT))

OUT = AI_PORT / "outputs" / "s17_prechecks" / "nominal_price_gates.json"
VARIANT = AI_PORT / "variants" / "codex_causal_rank_65.yaml"

NO_DIVIDEND = ["TSLA", "AMZN", "ADBE", "NFLX", "ISRG"]
MO_DATE = pd.Timestamp("2014-06-30")
MO_NOMINAL = 41.94
MO_TOL = 0.01
TG_PX_ON_BAND = (1.00, 1.25)
Z_GAP_ON_MAX = 0.5
N_HIGH_DIV = 12


def evaluate_gates(no_div_identical: bool, mo_ok: bool, tg_px_on: float, z_gap_on: float) -> dict:
    return {
        "A_no_dividend_identical": bool(no_div_identical),
        "B_mo_2014_06_30_nominal": bool(mo_ok),
        "C_high_div_tg_px_2014_on_in_band": bool(
            np.isfinite(tg_px_on) and TG_PX_ON_BAND[0] <= tg_px_on <= TG_PX_ON_BAND[1]),
        "D_high_div_z_gap_on_below_half": bool(np.isfinite(z_gap_on) and abs(z_gap_on) < Z_GAP_ON_MAX),
    }


def _group_median_by_year(panel: pd.DataFrame, tickers, years) -> float:
    """연도 구간 내 일별 그룹 중앙값의 중앙값 (패널에 없는 종목은 무시, 없으면 NaN)."""
    cols = [t for t in tickers if t in panel.columns]
    if not cols:
        return float("nan")
    sub = panel[cols]
    sub = sub[(sub.index.year >= years[0]) & (sub.index.year <= years[1])]
    daily = sub.median(axis=1)
    return float(daily.median()) if daily.notna().any() else float("nan")


def _dated(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "date" in frame.columns:
        frame = frame.set_index(pd.to_datetime(frame["date"])).drop(columns=["date"])
    frame.index = pd.to_datetime(frame.index)
    return frame


def main() -> None:
    import yaml

    from src.data_loader import UniverseData
    from src.features.sellside import build_sellside_features
    from src.harness import build_override_config, inject_config

    overrides = dict(yaml.safe_load(VARIANT.read_text(encoding="utf-8"))["overrides"])
    cfg_off = build_override_config(overrides)
    cfg_on = build_override_config({**overrides, "nominal_price_source": "PX_LAST_UNADJ"})
    inject_config(cfg_on)

    data = UniverseData(cfg_on.data_path, config=cfg_on)
    raw_adj = _dated(data.raw["PX_LAST"])
    raw_un = _dated(data.raw["PX_LAST_UNADJ"])

    # ---------------------------------------------------------------- A
    a_detail = {}
    for t in NO_DIVIDEND:
        if t not in raw_adj.columns or t not in raw_un.columns:
            a_detail[t] = {"present": False}
            continue
        a = pd.to_numeric(raw_adj[t], errors="coerce")
        u = pd.to_numeric(raw_un[t], errors="coerce").reindex(a.index)
        both = a.notna() & u.notna()
        a_detail[t] = {
            "present": True, "n_both": int(both.sum()),
            "identical": bool(both.any() and np.array_equal(a[both].to_numpy(), u[both].to_numpy())),
            "max_abs_diff": float((a[both] - u[both]).abs().max()) if both.any() else None,
            "nan_pattern_equal": bool(a.notna().equals(u.notna())),
        }
    no_div_identical = all(v.get("present") and v.get("identical") for v in a_detail.values())

    # ---------------------------------------------------------------- B
    def _at(frame, t):
        return float(frame.loc[MO_DATE, t]) if (t in frame.columns and MO_DATE in frame.index) else float("nan")

    mo_un, mo_adj = _at(raw_un, "MO"), _at(raw_adj, "MO")
    mo_ok = bool(np.isfinite(mo_un) and abs(mo_un - MO_NOMINAL) <= MO_TOL)

    # ---------------------------------------------------------------- 고배당 12종
    if MO_DATE in raw_adj.index and MO_DATE in raw_un.index:
        ratio_2014 = (pd.to_numeric(raw_adj.loc[MO_DATE], errors="coerce")
                      / pd.to_numeric(raw_un.loc[MO_DATE], errors="coerce").reindex(raw_adj.columns))
    else:
        ratio_2014 = pd.Series(dtype=float)
    ratio_2014 = ratio_2014[np.isfinite(ratio_2014) & (ratio_2014 > 0)]
    high_div = [str(t) for t in ratio_2014.sort_values().index[:N_HIGH_DIV]]
    high_div_ratio = {t: round(float(ratio_2014[t]), 4) for t in high_div}

    # ---------------------------------------------------------------- C (TG/PX) · D (tg_upside_z)
    tg = data.get_sheet("Factset_TG_Price")
    local = data.local_prices.replace(0, np.nan).reindex_like(tg)
    nominal = data.local_prices_nominal.replace(0, np.nan).reindex_like(tg)
    tg_px_off = _group_median_by_year(tg / local, high_div, (2014, 2014))
    tg_px_on = _group_median_by_year(tg / nominal, high_div, (2014, 2014))
    tg_px_by_year = {
        str(y): {"off": round(_group_median_by_year(tg / local, high_div, (y, y)), 3),
                 "on": round(_group_median_by_year(tg / nominal, high_div, (y, y)), 3)}
        for y in range(2014, 2027)
    }

    feats = build_sellside_features(data, config=cfg_off)
    z_off = feats["tg_upside_z"]
    del feats
    gc.collect()
    feats = build_sellside_features(data, config=cfg_on)
    z_on, up_on = feats["tg_upside_z"], feats["tg_upside"]
    del feats
    gc.collect()

    def _gap(z, group):
        return _group_median_by_year(z, group, (2014, 2019)) - _group_median_by_year(z, group, (2025, 2026))

    z_gap_off, z_gap_on = _gap(z_off, high_div), _gap(z_on, high_div)

    # ---------------------------------------------------------------- 관측
    mcap = data.market_cap
    shares_off = mcap / data.prices.replace(0, np.nan)
    shares_on = mcap / data.prices_nominal.replace(0, np.nan)
    mo_shares = {}
    if "MO" in shares_off.columns:
        off_mn = float(shares_off.loc[shares_off.index.year == 2014, "MO"].median())
        on_mn = float(shares_on.loc[shares_on.index.year == 2014, "MO"].median())
        mo_shares = {"off_2014_median": off_mn, "on_2014_median": on_mn,
                     "off_over_on": round(off_mn / on_mn, 3) if on_mn else None}
    pe_on = data.sheets["BEST_PE_RATIO"]
    pe_mo = {"date": str(MO_DATE.date())}
    if "MO" in pe_on.columns and MO_DATE in pe_on.index and np.isfinite(mo_un) and mo_un:
        pe_mo["on"] = float(pe_on.loc[MO_DATE, "MO"])
        pe_mo["off"] = float(pe_on.loc[MO_DATE, "MO"] * (mo_adj / mo_un))
    plus5_on = int((up_on.abs() >= 4.99).sum().sum())

    gates = evaluate_gates(no_div_identical, mo_ok, tg_px_on, z_gap_on)
    out = {
        "preregistration": "decision log §S17.3 (2026-09-03) G1-01b / M1",
        "vintage": {
            "data_path": str(cfg_on.data_path),
            "data_mtime": pd.Timestamp(Path(cfg_on.data_path).stat().st_mtime, unit="s").round("s").isoformat(),
        },
        "loader_diagnostic": data.data_quality["currency"].get("nominal_price"),
        "gates": gates,
        "gates_pass": bool(all(gates.values())),
        "A_no_dividend": a_detail,
        "B_mo": {"date": str(MO_DATE.date()), "px_last": mo_adj, "px_last_unadj": mo_un,
                 "target": MO_NOMINAL, "tolerance": MO_TOL},
        "high_dividend_12": high_div,
        "high_dividend_12_ratio_2014_06_30": high_div_ratio,
        "C_tg_px_2014_median": {"off": tg_px_off, "on": tg_px_on, "on_band": list(TG_PX_ON_BAND),
                                "by_year": tg_px_by_year},
        "D_tg_upside_z_gap": {
            "off": z_gap_off, "on": z_gap_on, "on_max_abs": Z_GAP_ON_MAX,
            "train_2014_2019": {"off": _group_median_by_year(z_off, high_div, (2014, 2019)),
                                "on": _group_median_by_year(z_on, high_div, (2014, 2019))},
            "live_2025_2026": {"off": _group_median_by_year(z_off, high_div, (2025, 2026)),
                               "on": _group_median_by_year(z_on, high_div, (2025, 2026))},
            "no_dividend_group_gap": {"off": _gap(z_off, NO_DIVIDEND), "on": _gap(z_on, NO_DIVIDEND)},
        },
        "observations": {"mo_implied_shares_2014": mo_shares, "mo_pe_2014_06_30": pe_mo,
                         "tg_upside_on_abs_ge_4_99_cells": plus5_on},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    print(f"\nGATES: {gates} -> {'PASS' if out['gates_pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
