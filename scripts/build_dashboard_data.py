"""Precompute lightweight dashboard data from a backtest_result.pkl.

Why: backtest_result.pkl is ~65MB (panel = 49MB, models = 4MB). The
mobile dashboard only needs aggregate views, not the raw panel. By
pre-baking IC tables, feature importance, group PnL, and per-rebalance
OW score breakdowns, we cut the runtime payload to ~5MB so it fits
comfortably in a GitHub repo for Streamlit Cloud deployment.

Run:
    python scripts/build_dashboard_data.py \
        --run outputs/baseline_v4 \
        --data data/ai_signal_data.xlsx \
        --out outputs/baseline_v4/dashboard_data.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

# The backtest_result.pkl pickled BacktestResult references src.* modules;
# make them importable when this script runs from any cwd.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

GROUPS_DICT = {
    "Growth": [
        "best_eps_chg_252d","best_sales_chg_252d","best_sales_accel",
        "best_gross_margin_chg_63d","best_gross_margin_chg_252d",
        "oper_margin_chg_63d","oper_margin_chg_252d","oper_margin_accel",
        "fin_roe_chg_63d","fin_roe_chg_252d","fin_eps_chg_63d",
        "fin_sales_chg_63d","fac_F_Growth_mom_63d",
    ],
    "Quality": [
        "best_roe_level_z","fin_roe_level_z","fac_F_Quality_mom_63d",
        "earnings_quality_252d","cash_conversion_z","op_leverage_63d",
        "best_calculated_fcf_level_z","best_capex_level_z","capex_intensity_z",
    ],
    "Value": [
        "best_peg_ratio_level_z","best_px_bps_ratio_level_z",
        "best_ev_to_best_ebitda_level_z","fin_pb_level_z","fin_pe_level_z",
        "fin_pe_chg_63d","fin_pb_chg_63d","fac_F_Value_mom_63d",
        "fac_value_growth_63d","fin_roe_pb_gap","fin_roe_pe_gap",
    ],
    "Revision": [
        "eps_rev","eps_rev_ma_63d","eps_rev_trend","sales_rev_ma_63d",
        "analyst_rec_level","analyst_rec_stability","tg_upside","tg_mom_63d",
    ],
    "Momentum": [
        "momentum_252d","risk_adj_mom_252d","ma_cross_21_50","ma_cross_50_200",
        "dist_52w_high","max_ret_63d","min_ret_63d","mom_accel_63_252",
    ],
    "Low-vol": ["realized_vol_21d","realized_vol_126d","beta_63d","idio_vol_63d"],
    "Macro": [
        "cal_is_Q1","regime_mkt_ret_21d","mc_rate_x_eps_rev","mc_slope_x_eps_rev",
        "mc_vix_x_mom252","mc_vol_x_mom63","mc_dxy_x_eps_rev","fac_yield_slope",
    ],
}


def feature_to_bucket(f: str) -> str:
    for name, feats in GROUPS_DICT.items():
        if f in feats:
            return name
    return "Other"


def build(run_dir: Path, data_file: Path, out: Path) -> None:
    pkl = run_dir / "backtest_result.pkl"
    print(f"[build] loading {pkl} ...")
    with pkl.open("rb") as f:
        r = pickle.load(f)

    feat_names = list(r.feature_names)
    panel = r.panel  # MultiIndex (date, ticker)
    targets_df: pd.DataFrame = r.targets
    models = r.models
    PW = pd.DataFrame(r.portfolio_weights).T.sort_index()
    W_daily = pd.DataFrame(r.daily_weights).T.sort_index()
    predictions = r.predictions

    # ---- 1) Feature importance (gain) aggregated over walk-forward ------
    print("[build] feature importance ...")
    agg = np.zeros(len(feat_names))
    used = 0
    for d, m in models.items():
        imp = m.booster_.feature_importance(importance_type="gain")
        if len(imp) == len(feat_names):
            agg += imp
            used += 1
    print(f"  used {used}/{len(models)} models")
    fi = pd.Series(agg, index=feat_names)
    fi_pct = fi / fi.sum() * 100

    # ---- 2) IC per feature (cross-sectional Spearman, daily mean) -------
    print("[build] IC table ...")
    y = targets_df.stack()
    y.index.names = ["date", "ticker"]
    ic_rows = []
    for f in feat_names:
        if f not in panel.columns:
            continue
        x = panel[f]
        df = pd.concat({"x": x, "y": y}, axis=1, join="inner").dropna()
        if df.empty:
            continue
        ic = (
            df.groupby(level="date")
              .apply(lambda g: g["x"].corr(g["y"], method="spearman") if len(g) > 5 else np.nan)
              .dropna()
        )
        if len(ic) == 0:
            continue
        ic_rows.append({
            "feature": f,
            "bucket": feature_to_bucket(f),
            "IC_mean": ic.mean(),
            "IC_std": ic.std(),
            "IC_IR": ic.mean() / ic.std() if ic.std() > 0 else np.nan,
            "n_days": int(len(ic)),
        })
    ic_df = pd.DataFrame(ic_rows).sort_values("IC_mean", ascending=False).reset_index(drop=True)

    # ---- 3) Group long-short PnL ---------------------------------------
    print("[build] group PnL ...")
    group_rows = {}
    for gname, feats in GROUPS_DICT.items():
        cols = [f for f in feats if f in panel.columns]
        if not cols:
            continue
        grp = panel[cols].mean(axis=1)
        df = pd.concat({"g": grp, "y": y}, axis=1, join="inner").dropna()
        rets = []
        for d, gd in df.groupby(level="date"):
            if len(gd) < 10:
                continue
            rk = gd["g"].rank(pct=True)
            long_r = gd.loc[rk > 0.7, "y"].mean()
            short_r = gd.loc[rk < 0.3, "y"].mean()
            rets.append(long_r - short_r)
        if not rets:
            continue
        rs = pd.Series(rets)
        group_rows[gname] = {
            "AnnRet_%": rs.mean() * 252 * 100,
            "AnnVol_%": rs.std() * np.sqrt(252) * 100,
            "Sharpe": (rs.mean() * 252) / (rs.std() * np.sqrt(252)) if rs.std() > 0 else np.nan,
            "n_days": int(len(rs)),
        }
    group_pnl = pd.DataFrame(group_rows).T.sort_values("Sharpe", ascending=False)

    # ---- 4) Per-rebalance OW score breakdown ---------------------------
    print("[build] per-rebalance score breakdown ...")
    score_breakdowns: dict[pd.Timestamp, pd.DataFrame] = {}
    rebal_predictions: dict[pd.Timestamp, pd.Series] = {}
    for d in PW.index:
        try:
            cs = panel.xs(d, level="date")
        except KeyError:
            continue
        gz = {}
        for gname, feats in GROUPS_DICT.items():
            cols = [f for f in feats if f in cs.columns]
            if cols:
                gz[gname] = cs[cols].mean(axis=1)
        score_breakdowns[d] = pd.DataFrame(gz)
        if d in predictions.index:
            rebal_predictions[d] = predictions.loc[d].dropna()

    # ---- 5) Benchmark weights (cap-weighted from CUR_MKT_CAP) ----------
    print("[build] benchmark weights ...")
    cap = pd.read_excel(data_file, sheet_name="CUR_MKT_CAP", index_col=0)
    cap.index = pd.to_datetime(cap.index)
    cap_a = cap.reindex(W_daily.index).ffill()
    tickers = W_daily.columns.tolist()
    cap_sub = cap_a[tickers]
    bm_weights = cap_sub.div(cap_sub.sum(axis=1), axis=0)

    # ---- 6) Compose package -------------------------------------------
    metrics = r.compute_metrics()
    package = {
        "schema_version": 1,
        "label": run_dir.name,
        "period": {
            "start": str(r.portfolio_returns.dropna().index[0].date()),
            "end": str(r.portfolio_returns.dropna().index[-1].date()),
        },
        "metrics": metrics,
        "portfolio_returns": r.portfolio_returns,
        "benchmark_returns": r.benchmark_returns,
        "daily_weights": W_daily,
        "portfolio_weights": PW,
        "bm_weights_at_rebalances": bm_weights.loc[PW.index],
        "bm_weights_daily": bm_weights,  # for 6M avg view
        "feature_importance_pct": fi_pct,
        "ic_table": ic_df,
        "group_pnl": group_pnl,
        "score_breakdowns": score_breakdowns,
        "rebal_predictions": rebal_predictions,
        "feature_groups": GROUPS_DICT,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(package, f, protocol=4)
    size_mb = out.stat().st_size / 1024**2
    print(f"\n[build] wrote {out}  ({size_mb:.2f} MB)")
    print(f"  rebalances : {len(PW)}")
    print(f"  features   : {len(feat_names)}")
    print(f"  IC entries : {len(ic_df)}")
    print(f"  groups     : {list(group_rows.keys())}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=Path("outputs/baseline_v4"))
    p.add_argument("--data", type=Path, default=Path(r"C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\ai_signal_data.xlsx"))
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    out = args.out or (args.run / "dashboard_data.pkl")
    build(args.run, args.data, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
