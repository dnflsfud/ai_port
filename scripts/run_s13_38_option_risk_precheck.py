# -*- coding: utf-8 -*-
"""§S13.38 사전점검 (read-only): 옵션 리스크 표준화 피처 IC·중복 진단.

§S13.37 워크북 시트(iv30_z / downside_skew_z / downside_skew_chg_5d /
iv_term_structure_z, 참고용 vol_risk_premium_z)의 21d/63d CS Spearman IC를
§S13.35 관용구(5영업일 샘플링, 보수적 t = naive/√(h/5))로 실측하고,
vol 축(VOLATILITY_30D 레벨)·피처 상호 CS 중복 ρ, days_to_earnings 커버리지를
보고한다. 게이트(해석용 — 사용자 지시분이라 arm은 결과 무관 수행):
  P1: |t_cons| ≥ 2 & 서브기간(3분할) 부호 일관
  P2: vol 축 중복 mean |ρ| < 0.6

출력: outputs/s13_38_option_risk_precheck/summary.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

WORKBOOK = Path(
    r"C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine"
    r"\re_study\ai_signal_data.xlsx"
)
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "s13_38_option_risk_precheck"

CANDIDATE_SHEETS = [
    "iv30_z",
    "downside_skew_z",
    "downside_skew_chg_5d",
    "iv_term_structure_z",
    "vol_risk_premium_z",  # 참고용 — arm 사전등록에서 제외(사용자 지시)
]
HORIZONS = (21, 63)
SAMPLE_STEP = 5
MIN_PAIRS = 30


def simple(col: str) -> str:
    return col.split(" ")[0]


def load_panel(xl: pd.ExcelFile, sheet: str, universe: list[str]) -> pd.DataFrame:
    df = pd.read_excel(xl, sheet_name=sheet)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    df = df.drop(columns=[c for c in df.columns if c == "SPX Index"])
    df = df.rename(columns={c: simple(c) for c in df.columns})
    keep = [t for t in universe if t in df.columns]
    return df[keep].apply(pd.to_numeric, errors="coerce")


def cs_spearman(a: pd.Series, b: pd.Series) -> float:
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < MIN_PAIRS:
        return float("nan")
    return float(pair.iloc[:, 0].rank().corr(pair.iloc[:, 1].rank()))


def ic_series(feature: pd.DataFrame, fwd: pd.DataFrame, dates) -> pd.Series:
    out = {}
    for d in dates:
        if d in feature.index and d in fwd.index:
            out[d] = cs_spearman(feature.loc[d], fwd.loc[d])
    return pd.Series(out).dropna()


def summarize_ic(ic: pd.Series, horizon: int) -> dict:
    n = len(ic)
    naive_t = float(ic.mean() / ic.std(ddof=1) * np.sqrt(n)) if n > 2 else float("nan")
    cons_t = naive_t / np.sqrt(horizon / SAMPLE_STEP)
    thirds = np.array_split(ic, 3)
    return {
        "mean_ic": round(float(ic.mean()), 4),
        "t_naive": round(naive_t, 2),
        "t_conservative": round(cons_t, 2),
        "n_dates": n,
        "subperiod_mean_ic": [round(float(t.mean()), 4) for t in thirds],
        "sign_consistent": bool(
            all(np.sign(t.mean()) == np.sign(ic.mean()) for t in thirds)
        ),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mtime = pd.Timestamp(WORKBOOK.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")
    print(f"workbook mtime: {mtime}")

    xl = pd.ExcelFile(WORKBOOK)
    meta = pd.read_excel(xl, sheet_name="Universe_Meta")
    universe = [simple(t) for t in meta["Ticker"].dropna().astype(str)]
    print(f"universe: {len(universe)}")

    returns = load_panel(xl, "Daily_Returns", universe)
    rv = load_panel(xl, "VOLATILITY_30D", universe)
    panels = {s: load_panel(xl, s, universe) for s in CANDIDATE_SHEETS}
    days_earn = load_panel(xl, "days_to_earnings", universe)

    fwd = {h: returns.rolling(h).sum().shift(-h) for h in HORIZONS}
    sample_dates = returns.index[::SAMPLE_STEP]

    results: dict[str, dict] = {}
    for name, panel in panels.items():
        entry: dict = {}
        cover = panel.reindex(index=returns.index).notna().mean(axis=1)
        entry["coverage_recent_252d"] = round(float(cover.tail(252).mean()), 3)
        for h in HORIZONS:
            ic = ic_series(panel, fwd[h], sample_dates)
            entry[f"ic_{h}d"] = summarize_ic(ic, h)
        rho_vol = pd.Series(
            {d: cs_spearman(panel.loc[d], rv.loc[d])
             for d in sample_dates if d in panel.index and d in rv.index}
        ).dropna()
        entry["mean_rho_vs_realized_vol"] = round(float(rho_vol.mean()), 3)
        results[name] = entry
        print(f"\n[{name}] {json.dumps(entry, ensure_ascii=False)}")

    # 피처 상호 CS 중복 (최근 504영업일의 샘플 날짜, arm 후보 4종만)
    arm = CANDIDATE_SHEETS[:4]
    cross: dict[str, float] = {}
    recent = [d for d in sample_dates if d >= returns.index[-504]]
    for i, a in enumerate(arm):
        for b in arm[i + 1:]:
            rho = pd.Series(
                {d: cs_spearman(panels[a].loc[d], panels[b].loc[d])
                 for d in recent if d in panels[a].index and d in panels[b].index}
            ).dropna()
            cross[f"{a}~{b}"] = round(float(rho.mean()), 3)
    print(f"\ncross-feature rho: {json.dumps(cross, ensure_ascii=False)}")

    de_cover = days_earn.reindex(index=returns.index).notna().mean(axis=1)
    de = {
        "coverage_full": round(float(de_cover.mean()), 3),
        "coverage_recent_252d": round(float(de_cover.tail(252).mean()), 3),
        "coverage_last_21d": round(float(de_cover.tail(21).mean()), 3),
    }
    print(f"days_to_earnings coverage: {json.dumps(de)}")

    summary = {
        "workbook_mtime": mtime,
        "sample_step_days": SAMPLE_STEP,
        "features": results,
        "cross_feature_rho_recent504": cross,
        "days_to_earnings": de,
    }
    out = OUT_DIR / "summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
