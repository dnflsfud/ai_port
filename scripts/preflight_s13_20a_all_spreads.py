# -*- coding: utf-8 -*-
"""§S13.20a: EPS-감응도 사전점검을 4스프레드 (innovation, state) 쌍으로 확장.

사용자 지시(2026-07-29) 확장 — §S13.20과 동일한 창(252d)·min_obs(200)·게이트를
스프레드별 쌍에 적용한다. tech_lead63 쌍은 §S13.20 원판의 재현이다.
**탐색적 4-way**: 통과 쌍의 arm 채택은 별도 사전등록에 4-way 선택 사실
명시가 필요하다(결정 로그 §S13.20a 선언).

실행: <PY> scripts/preflight_s13_20a_all_spreads.py  (WD=ai_port, PYTHONPATH=.)
산출물: outputs/s13_20a_all_spreads.csv
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.preflight_s13_19_carry_eps_covariation import eps_features  # noqa: E402
from scripts.preflight_s13_20_eps_loading import (  # noqa: E402
    MIN_OBS, WINDOW, judge_gates, rank_autocorr, rank_ic, window_loadings)
from src.config import PipelineConfig  # noqa: E402
from src.data_loader import UniverseData, mask_pre_listing  # noqa: E402

SPREAD_CONFIGS = (
    {"state": "fac_eps_g63", "innovation": "fac_eps_g63"},
    {"state": "fac_eps_us_lead63", "innovation": "fac_eps_us_lead63"},
    {"state": "fac_eps_us_lead252", "innovation": "fac_eps_us_lead252"},
    {"state": "fac_eps_tech_lead63", "innovation": "fac_eps_tech_lead63"},
)


def spread_innovation(px: pd.DataFrame, kind: str) -> pd.Series:
    """스프레드별 일간 innovation (63/252는 state 지평 차이 — innovation 동일)."""
    d = {k: np.log(px[f"{k}_FWD_EPS"]).diff() for k in ("SPX", "NDX", "MXWD")}
    if kind == "fac_eps_g63":
        return d["MXWD"]
    if kind in ("fac_eps_us_lead63", "fac_eps_us_lead252"):
        return d["SPX"] - d["MXWD"]
    if kind == "fac_eps_tech_lead63":
        return d["NDX"] - d["SPX"]
    raise ValueError(kind)


def run_config(excess: pd.DataFrame, x: pd.Series, state_at: pd.Series,
               targets: pd.DataFrame, per_date: pd.DataFrame) -> dict:
    load_rows, kept, stats = [], [], []
    for t_date in per_date.index:
        pos = excess.index.searchsorted(t_date)  # strictly-before 창
        if pos < WINDOW or t_date not in targets.index:
            continue
        win = slice(pos - WINDOW, pos)
        beta, tstat = window_loadings(excess.iloc[win], x.iloc[win], MIN_OBS)
        n_valid = int(tstat.notna().sum())
        share = (float((tstat.abs() > 2.0).sum()) / n_valid
                 if n_valid else float("nan"))
        ic = rank_ic(beta * float(state_at.loc[t_date]), targets.loc[t_date])
        kept.append(t_date)
        load_rows.append(beta)
        stats.append({"sub": per_date.loc[t_date, "sub"],
                      "share_sig": share, "ic": ic})
    stat = pd.DataFrame(stats, index=kept)
    share_sig = float(stat["share_sig"].mean())
    autocorr = rank_autocorr(pd.DataFrame(load_rows, index=kept))
    mean_ic = float(stat["ic"].mean())
    sub_ics = {s: float(stat.loc[stat["sub"] == s, "ic"].mean())
               for s in ("P1", "P2", "P3")}
    return {"share_sig": share_sig, "autocorr": autocorr, "mean_ic": mean_ic,
            **{f"ic_{k}": v for k, v in sub_ics.items()},
            **judge_gates(share_sig, autocorr, mean_ic, sub_ics)}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cfg = PipelineConfig()
    per_date = pd.read_csv(
        root / "outputs/s13_12_ic_ir_transmission/per_date_S0.csv",
        parse_dates=["date"]).set_index("date")

    px = pd.read_excel(cfg.data_path, sheet_name="Factor_PX_LAST",
                       index_col=0, engine="openpyxl")
    px.index = pd.DatetimeIndex(px.index)
    feats = eps_features(px)

    print("loading UniverseData (USD returns + listing mask)...")
    data = UniverseData(cfg.data_path, cfg)
    returns = mask_pre_listing(data.sheets["Daily_Returns"],
                               data.listing_dates, inclusive=True)
    with open(root / "outputs/codex_causal_rank_65/backtest_result.pkl",
              "rb") as f:
        targets = pickle.load(f).targets
    tickers = [t for t in returns.columns if t in targets.columns]
    returns = returns[tickers]
    excess = returns.sub(returns.mean(axis=1), axis=0)

    rows = []
    for conf in SPREAD_CONFIGS:
        x = spread_innovation(px, conf["innovation"]).reindex(returns.index)
        state_at = feats[conf["state"]].reindex(per_date.index, method="ffill")
        res = run_config(excess, x, state_at, targets, per_date)
        rows.append({"spread": conf["state"], **res})
        print(f"{conf['state']}: share {res['share_sig']:.3f} "
              f"ac {res['autocorr']:.3f} IC {res['mean_ic']:+.4f} "
              f"-> {res['verdict']}")
    out = pd.DataFrame(rows).set_index("spread")
    print("\n==== §S13.20a 감응도 게이트 — 4스프레드 전수 (탐색적) ====")
    print(out.round(4).to_string())
    out.to_csv(root / "outputs/s13_20a_all_spreads.csv",
               encoding="utf-8-sig")
    print("\nsaved: outputs/s13_20a_all_spreads.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
