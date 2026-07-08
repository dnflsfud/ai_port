#!/usr/bin/env python
"""Read-only Streamlit dashboard for the Pictet portfolio adoption run.

Two views are merged here:

* Six operating tabs (Overview / Performance / Contribution / Risk / Turnover /
  Signals & Gates) driven by ``outputs/operating/*.json`` + ``returns.csv``
  (produced by ``scripts/export_operating_data.py``).
* A ``Backtest Runs`` tab: a run selector over ``outputs/*/metrics.json`` that
  shows headline metrics and — when a ``backtest_result.pkl`` is present —
  cumulative / active / IC / turnover curves for the chosen run.

The pure helpers ``list_runs`` / ``load_metrics`` / ``load_result`` are
module-level and Streamlit-free, so ``import streamlit_app`` has NO UI side
effects. Every Streamlit call lives inside ``main()``, guarded by ``__main__``.
"""
from __future__ import annotations

import itertools
import json
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
PROD_LABEL = "iter15_65tkr_reb21_vtg"

COLOR_PORT = "#183B56"
COLOR_BM = "#829AB1"
COLOR_ACTIVE = "#0B7285"
COLOR_POS = "#147D64"
COLOR_NEG = "#C92A2A"
COLOR_GRID = "#D9E2EC"

CSS = """
<style>
.stApp {
  background: #f5f7fb;
  color: #102a43;
}
.stApp p, .stApp label, .stApp span, .stApp div {
  color: #102a43;
}
.block-container {
  max-width: 1500px;
  padding-top: 1.25rem;
  padding-bottom: 2.5rem;
}
section[data-testid="stSidebar"] {
  background: #ffffff;
  border-right: 1px solid #d9e2ec;
}
section[data-testid="stSidebar"] * {
  color: #102a43 !important;
}
div[data-testid="stMetric"] {
  background: #ffffff;
  border: 1px solid #d9e2ec;
  border-radius: 8px;
  padding: 0.75rem 0.85rem;
  box-shadow: 0 1px 2px rgba(16, 42, 67, 0.05);
}
div[data-testid="stMetricLabel"] *, div[data-testid="stMetricLabel"] p {
  color: #52606d !important;
  font-size: 0.78rem;
}
div[data-testid="stMetricValue"] {
  color: #102a43 !important;
  font-size: 1.45rem;
}
div[data-testid="stTabs"] button p {
  color: #627d98 !important;
  font-size: 0.9rem;
}
div[data-testid="stTabs"] button[aria-selected="true"] p {
  color: #c92a2a !important;
  font-weight: 700;
}
div[data-baseweb="input"] input,
div[data-baseweb="select"] > div {
  background: #ffffff !important;
  color: #102a43 !important;
  border-color: #d9e2ec !important;
}
.status-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  margin: 0.5rem 0 1.0rem 0;
}
.chip {
  background: #ffffff;
  border: 1px solid #d9e2ec;
  border-radius: 999px;
  padding: 0.28rem 0.62rem;
  font-size: 0.78rem;
  color: #102a43;
}
.chip-ok {
  background: #e6fcf5;
  border-color: #96f2d7;
}
.chip-warn {
  background: #fff5f5;
  border-color: #ffc9c9;
}
.note {
  color: #627d98;
  font-size: 0.84rem;
  margin: -0.25rem 0 0.7rem 0;
}
</style>
"""


# ---------------------------------------------------------------------------
# Pure helpers (no Streamlit) — import-safe, unit-tested.
# ---------------------------------------------------------------------------
def list_runs(outputs_dir) -> list:
    """Scan ``<outputs_dir>`` for run folders holding a ``metrics.json``.

    Looks at depth 1 (``<dir>/*/metrics.json``) and depth 2
    (``<dir>/*/*/metrics.json``). Each element is
    ``{"label", "dir", "metrics", "meta"}`` where ``label`` falls back to the
    folder name, ``metrics`` defaults to ``{}`` and ``meta`` carries the
    remaining top-level JSON keys. Production run sorts first, then labels
    alphabetically. Missing/empty dirs return ``[]``; broken JSON is skipped.
    """
    root = Path(outputs_dir)
    if not root.exists():
        return []
    runs = []
    seen = set()
    for mpath in list(root.glob("*/metrics.json")) + list(root.glob("*/*/metrics.json")):
        run_dir = mpath.parent
        if run_dir in seen:
            continue
        try:
            data = json.loads(mpath.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        seen.add(run_dir)
        label = data.get("label") or run_dir.name
        metrics = data.get("metrics", {})
        meta = {k: v for k, v in data.items() if k not in ("label", "metrics")}
        runs.append({"label": label, "dir": run_dir, "metrics": metrics, "meta": meta})
    runs.sort(key=lambda r: (r["label"] != PROD_LABEL, r["label"]))
    return runs


def load_metrics(path) -> dict:
    """Load the full ``metrics.json`` (all top-level keys preserved)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_result(run_dir):
    """Return the unpickled ``backtest_result.pkl`` from ``run_dir`` or None."""
    pkl = Path(run_dir) / "backtest_result.pkl"
    if not pkl.exists():
        return None
    with open(pkl, "rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------------------
# Operating-data loaders (no Streamlit) — read outputs/operating artefacts.
# ---------------------------------------------------------------------------
def load_json(rel: str) -> dict:
    path = OUT / rel
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_error": str(exc)}


def load_csv(rel: str) -> pd.DataFrame:
    path = OUT / rel
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Formatting / chart helpers.
# ---------------------------------------------------------------------------
def pct(value: Any, digits: int = 1, signed: bool = False) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(x):
        return "n/a"
    sign = "+" if signed else ""
    return f"{x:{sign}.{digits}%}"


def num(value: Any, digits: int = 2) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(x):
        return "n/a"
    return f"{x:.{digits}f}"


def rows_df(rows: Any) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    if isinstance(rows, list):
        return pd.DataFrame(rows)
    if isinstance(rows, dict):
        if all(isinstance(v, dict) for v in rows.values()):
            return pd.DataFrame.from_dict(rows, orient="index").reset_index().rename(columns={"index": "name"})
        return pd.Series(rows, name="value").reset_index().rename(columns={"index": "name"})
    return pd.DataFrame(rows)


def apply_theme(fig: go.Figure, height: Optional[int] = None) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"color": COLOR_PORT, "family": "Arial, sans-serif"},
        title_font={"color": COLOR_PORT, "size": 15},
        legend_font={"color": COLOR_PORT},
        margin={"l": 10, "r": 10, "t": 48, "b": 24},
        height=height,
    )
    fig.update_xaxes(
        color=COLOR_PORT,
        tickfont={"color": COLOR_PORT},
        title_font={"color": COLOR_PORT},
        gridcolor=COLOR_GRID,
        zerolinecolor="#BCCCDC",
    )
    fig.update_yaxes(
        color=COLOR_PORT,
        tickfont={"color": COLOR_PORT},
        title_font={"color": COLOR_PORT},
        gridcolor=COLOR_GRID,
        zerolinecolor="#BCCCDC",
    )
    return fig


_chart_seq = itertools.count()


def render_chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, key=f"chart_{next(_chart_seq)}")


def line_fig(df: pd.DataFrame, cols: list, title: str, y_title: str = "") -> go.Figure:
    fig = go.Figure()
    colors = [COLOR_PORT, COLOR_BM, COLOR_ACTIVE, "#7048E8", "#E67700"]
    for i, col in enumerate(cols):
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    mode="lines",
                    name=col,
                    line={"width": 2.2, "color": colors[i % len(colors)]},
                )
            )
    fig.update_layout(title=title, yaxis_title=y_title, xaxis_title="")
    return apply_theme(fig, height=360)


def hbar_fig(df: pd.DataFrame, y: str, x: str, title: str, color: Optional[str] = None) -> go.Figure:
    fig = px.bar(
        df,
        x=x,
        y=y,
        orientation="h",
        color=color,
        color_discrete_sequence=[COLOR_ACTIVE, COLOR_POS, COLOR_NEG, COLOR_BM, "#7048E8"],
        title=title,
        template="plotly_white",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="", yaxis_title="")
    return apply_theme(fig, height=max(320, min(620, 30 * max(len(df), 7))))


def grouped_bar_fig(df: pd.DataFrame, x: str, y: list, title: str) -> go.Figure:
    fig = px.bar(
        df,
        x=x,
        y=y,
        barmode="group",
        color_discrete_map={
            "portfolio": COLOR_PORT,
            "benchmark": COLOR_BM,
            "active": COLOR_ACTIVE,
            "retrains": COLOR_BM,
            "degenerate": COLOR_NEG,
        },
        title=title,
        template="plotly_white",
    )
    return apply_theme(fig, height=360)


def fmt_table(df: pd.DataFrame, percent_cols: list, digits: int = 2):
    formats = {c: "{:.2%}" for c in percent_cols if c in df.columns}
    for col in df.select_dtypes(include="number").columns:
        formats.setdefault(col, f"{{:.{digits}f}}")
    return df.style.format(formats)


# ---------------------------------------------------------------------------
# UI entrypoint — every Streamlit call lives here.
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Pictet Portfolio Monitor",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def load_data() -> dict:
        s0_raw = load_json("iter15_65tkr_reb21_vtg/metrics.json")
        return {
            "summary": load_json("adoption_summary.json"),
            "s0": s0_raw.get("metrics", s0_raw),
            "attr": load_json("alpha_attribution/summary.json"),
            "overlay": load_json("overlay_ablation/summary.json"),
            "factor": load_json("factor_ablation/summary.json"),
            "dq": load_json("data_quality_report.json"),
            "perf": load_json("operating/performance.json"),
            "holdings": load_json("operating/holdings.json"),
            "features": load_json("operating/features.json"),
            "ops": load_json("operating/operations.json"),
            "contribution": load_json("operating/contribution.json"),
            "risk": load_json("operating/risk.json"),
            "monitoring": load_json("operating/monitoring.json"),
            "returns": load_csv("operating/returns.csv"),
        }

    @st.cache_resource(show_spinner=False)
    def cached_result(run_dir_str: str):
        return load_result(run_dir_str)

    data = load_data()
    summary = data["summary"]
    if not summary:
        st.warning("outputs/adoption_summary.json is missing. Run run_pictet_adoption.py first.")
        st.stop()

    perf = data["perf"]
    holdings = data["holdings"]
    ops = data["ops"]
    returns = data["returns"]
    contribution = data["contribution"]
    risk = data["risk"]
    monitoring = data["monitoring"]

    stage0 = summary.get("stage0_baseline", {})
    stage2 = summary.get("stage2_overlay", {})
    stage3 = summary.get("stage3_factor", {})
    overlay_decisions = stage2.get("decisions", {})
    overlay_ok = bool(overlay_decisions) and all(v == "KEEP" for v in overlay_decisions.values())
    factor_ok = not bool(stage3.get("collapsed"))

    st.title("Pictet Portfolio Monitor")
    st.markdown(
        "<div class='note'>Lightweight operating dashboard for performance, contribution, risk, turnover and adoption gates.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='status-row'>"
        f"<span class='chip chip-ok'>S0 IR {num(stage0.get('information_ratio'), 3)}</span>"
        f"<span class='chip chip-ok'>TE {pct(stage0.get('tracking_error'), 2)}</span>"
        f"<span class='chip chip-ok'>Beta {num(stage0.get('realized_beta'), 3)}</span>"
        f"<span class='chip {'chip-ok' if overlay_ok else 'chip-warn'}'>Overlay {'KEEP all' if overlay_ok else 'Review'}</span>"
        f"<span class='chip {'chip-ok' if factor_ok else 'chip-warn'}'>Factor collapsed={stage3.get('collapsed')}</span>"
        "<span class='chip chip-ok'>Production weights unchanged</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Scope")
        st.metric("Performance as of", perf.get("as_of", "n/a"))
        st.metric("Holdings as of", holdings.get("as_of", "n/a"))
        st.metric("Solver", "ECOS")

        date_range = None
        if not returns.empty:
            min_date = returns.index.min().date()
            max_date = returns.index.max().date()
            date_range = st.date_input("Date range", (min_date, max_date), min_value=min_date, max_value=max_date)

        sectors = sorted({r.get("sector", "Unknown") for r in holdings.get("all", [])})
        sector = st.selectbox("Sector", ["All sectors"] + sectors, index=0)

        st.divider()
        if not (OUT / "operating" / "contribution.json").exists():
            st.warning("Run scripts/export_operating_data.py to enable all sections.")

    filtered_returns = returns.copy()
    if date_range and len(date_range) == 2 and not returns.empty:
        start = pd.to_datetime(date_range[0])
        end = pd.to_datetime(date_range[1])
        filtered_returns = returns.loc[(returns.index >= start) & (returns.index <= end)].copy()

    def sector_filter(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or sector == "All sectors" or "sector" not in df.columns:
            return df
        return df[df["sector"] == sector].copy()

    top_cols = st.columns(6)
    top_cols[0].metric("Annual Return", pct(perf.get("annual_return"), 1))
    top_cols[1].metric("Active Return", pct(perf.get("active_return"), 2, signed=True))
    top_cols[2].metric("Information Ratio", num(perf.get("information_ratio"), 3))
    top_cols[3].metric("Tracking Error", pct(perf.get("tracking_error"), 2))
    top_cols[4].metric("Max Drawdown", pct(perf.get("max_drawdown"), 1))
    top_cols[5].metric("Active Share", pct(holdings.get("active_share_one_way"), 2))

    tabs = st.tabs(
        ["Overview", "Performance", "Contribution", "Risk", "Turnover", "Signals & Gates", "Backtest Runs"]
    )

    with tabs[0]:
        left, right = st.columns([1.45, 1.0])
        with left:
            if not filtered_returns.empty:
                render_chart(line_fig(filtered_returns, ["portfolio_cum", "benchmark_cum"], "Cumulative return", "Growth of $1"))
            else:
                st.info("No returns data found.")

        with right:
            c1, c2 = st.columns(2)
            c1.metric("Realized Beta", num(perf.get("realized_beta"), 3))
            c2.metric("Avg IC", num(perf.get("avg_ic"), 3))
            c3, c4 = st.columns(2)
            c3.metric("Fallback Rate", pct(perf.get("optimizer_failure_rate"), 1))
            c4.metric("Latest Turnover", pct(ops.get("turnover_two_way_latest"), 1))
            conc = holdings.get("concentration") or {}
            c5, c6 = st.columns(2)
            c5.metric("Effective Names", num(conc.get("effective_names"), 1))
            c6.metric("Top 5 Active Budget", pct(conc.get("top5_active_budget_share"), 1))

        c = st.columns(4)
        c[0].metric("Beta Gate", stage0.get("P2_beta_neutral_gate", "n/a"))
        c[1].metric("Overlay", "KEEP all" if overlay_ok else "Review")
        c[2].metric("Factor Lever", "OFF default" if factor_ok else "FAIL")
        c[3].metric("Round-trip", num(stage3.get("roundtrip_off_vs_s0_abs_diff"), 1))

        dd = rows_df(monitoring.get("drawdown_events"))
        sec_exp = rows_df(ops.get("sector_exposure"))
        col_a, col_b = st.columns([1.0, 1.15])
        with col_a:
            st.subheader("Largest drawdown events")
            if not dd.empty:
                st.dataframe(fmt_table(dd, ["max_drawdown"]), width="stretch", hide_index=True)
            else:
                st.info("No drawdown event data.")
        with col_b:
            st.subheader("Latest active sector exposure")
            if not sec_exp.empty:
                sec_exp = sec_exp.rename(columns={"name": "sector"}).sort_values("active")
                render_chart(hbar_fig(sec_exp, "sector", "active", "Active sector exposure"))
            else:
                st.info("No sector exposure data.")

    with tabs[1]:
        col_l, col_r = st.columns([1.2, 1.0])
        with col_l:
            if not filtered_returns.empty:
                render_chart(line_fig(filtered_returns, ["portfolio_cum", "benchmark_cum"], "Cumulative return", "Growth of $1"))
                dd_fig = px.area(
                    filtered_returns,
                    x=filtered_returns.index,
                    y="drawdown",
                    title="Portfolio drawdown",
                    color_discrete_sequence=[COLOR_NEG],
                    template="plotly_white",
                )
                render_chart(apply_theme(dd_fig, height=300))
        with col_r:
            by_year = rows_df(perf.get("by_year_returns"))
            if not by_year.empty:
                by_year = by_year.rename(columns={"name": "year"})
                render_chart(grouped_bar_fig(by_year, "year", ["portfolio", "benchmark", "active"], "Calendar-year returns"))

            monthly = rows_df(monitoring.get("monthly_returns"))
            if not monthly.empty:
                monthly["year"] = monthly["month"].str.slice(0, 4)
                monthly["month_num"] = monthly["month"].str.slice(5, 7)
                pivot = monthly.pivot(index="year", columns="month_num", values="active")
                heat = px.imshow(
                    pivot,
                    title="Monthly active return heatmap",
                    color_continuous_scale=["#C92A2A", "#FFFFFF", "#147D64"],
                    color_continuous_midpoint=0,
                    aspect="auto",
                    template="plotly_white",
                )
                render_chart(apply_theme(heat, height=310))

        rolling = rows_df(monitoring.get("rolling"))
        if not rolling.empty:
            rolling["date"] = pd.to_datetime(rolling["date"])
            rolling = rolling.set_index("date").sort_index()
            c1, c2 = st.columns(2)
            with c1:
                render_chart(line_fig(rolling, ["information_ratio_126d", "information_ratio_252d"], "Rolling information ratio", "IR"))
            with c2:
                render_chart(line_fig(rolling, ["tracking_error_126d_ann", "tracking_error_252d_ann"], "Rolling tracking error", "Annualized TE"))

    with tabs[2]:
        st.subheader("Stock and sector contribution")
        st.markdown(
            "<div class='note'>Arithmetic contribution uses entering-day weights. Cost/timing residual is shown separately.</div>",
            unsafe_allow_html=True,
        )
        contrib = sector_filter(rows_df(contribution.get("by_ticker")))
        sector_contrib = sector_filter(rows_df(contribution.get("by_sector")).rename(columns={"name": "sector"}))
        if contrib.empty:
            st.info("No contribution data. Run scripts/export_operating_data.py.")
        else:
            top_n = st.slider("Names shown", 5, 20, 10)
            top = contrib.nlargest(top_n, "active_contribution").sort_values("active_contribution")
            bottom = contrib.nsmallest(top_n, "active_contribution").sort_values("active_contribution", ascending=False)
            c1, c2 = st.columns(2)
            with c1:
                render_chart(hbar_fig(top, "ticker", "active_contribution", "Top active contributors"))
            with c2:
                render_chart(hbar_fig(bottom, "ticker", "active_contribution", "Largest active detractors"))
            if not sector_contrib.empty:
                render_chart(hbar_fig(sector_contrib.sort_values("active_contribution"), "sector", "active_contribution", "Sector active contribution"))

            cols = ["ticker", "sector", "portfolio_contribution", "benchmark_contribution", "active_contribution", "avg_active_weight", "latest_active"]
            st.dataframe(
                fmt_table(contrib[cols].sort_values("active_contribution", ascending=False), cols[2:]),
                width="stretch",
                hide_index=True,
            )
            res = contribution.get("residual") or {}
            r1, r2, r3 = st.columns(3)
            r1.metric("Reconstructed Active", pct(res.get("active_reconstructed_arithmetic"), 2, signed=True))
            r2.metric("Reported Active", pct(res.get("active_reported_arithmetic"), 2, signed=True))
            r3.metric("Cost/Timing Residual", pct(res.get("transaction_cost_and_timing_residual"), 2, signed=True))

    with tabs[3]:
        st.subheader("Risk contribution and holdings")
        risk_df = sector_filter(rows_df(risk.get("by_ticker")))
        sector_risk = sector_filter(rows_df(risk.get("by_sector")).rename(columns={"name": "sector"}))
        if risk.get("error"):
            st.warning(risk["error"])
        elif risk_df.empty:
            st.info("No risk data. Run scripts/export_operating_data.py.")
        else:
            c = st.columns(4)
            c[0].metric("Estimated Vol", pct(risk.get("estimated_portfolio_vol"), 2))
            c[1].metric("Estimated TE", pct(risk.get("estimated_tracking_error"), 2))
            c[2].metric("Lookback", f"{risk.get('cov_lookback_days')}d")
            c[3].metric("Active Share", pct(holdings.get("active_share_one_way"), 2))

            top_risk = risk_df.reindex(risk_df["active_te_contribution"].abs().sort_values(ascending=False).index).head(14)
            c1, c2 = st.columns([1.2, 1.0])
            with c1:
                render_chart(hbar_fig(top_risk.sort_values("active_te_contribution"), "ticker", "active_te_contribution", "Largest active TE contributors"))
            with c2:
                if not sector_risk.empty:
                    render_chart(hbar_fig(sector_risk.sort_values("active_te_contribution"), "sector", "active_te_contribution", "Sector active TE contribution"))

            all_hold = sector_filter(rows_df(holdings.get("all")))
            ow = all_hold.nlargest(12, "active")
            uw = all_hold.nsmallest(12, "active")
            h1, h2 = st.columns(2)
            with h1:
                st.subheader("Top overweight")
                st.dataframe(fmt_table(ow[["ticker", "sector", "weight", "bm_weight", "active"]], ["weight", "bm_weight", "active"]), width="stretch", hide_index=True)
            with h2:
                st.subheader("Top underweight")
                st.dataframe(fmt_table(uw[["ticker", "sector", "weight", "bm_weight", "active"]], ["weight", "bm_weight", "active"]), width="stretch", hide_index=True)

    with tabs[4]:
        st.subheader("Turnover, active share and trade list")
        turnover = rows_df(monitoring.get("turnover"))
        active_share = rows_df(monitoring.get("active_share"))
        c1, c2 = st.columns(2)
        with c1:
            if not turnover.empty:
                turnover["date"] = pd.to_datetime(turnover["date"])
                turnover = turnover.set_index("date").sort_index()
                render_chart(line_fig(turnover, ["turnover_two_way", "rolling6_two_way"], "Turnover by rebalance", "Two-way turnover"))
        with c2:
            if not active_share.empty:
                active_share["date"] = pd.to_datetime(active_share["date"])
                active_share = active_share.set_index("date").sort_index()
                render_chart(line_fig(active_share, ["active_share_one_way", "top5_active_budget_share"], "Active share and concentration", "Share"))

        trades = rows_df(ops.get("trade_list"))
        if not trades.empty:
            sectors_df = rows_df(holdings.get("all"))[["ticker", "sector"]]
            trades = trades.merge(sectors_df, on="ticker", how="left")
            trades = sector_filter(trades)
            trades["abs_delta"] = trades["delta"].abs()
            trades = trades.sort_values("abs_delta", ascending=False)
            t1, t2 = st.columns([1.0, 1.2])
            with t1:
                render_chart(hbar_fig(trades.head(18).sort_values("delta"), "ticker", "delta", "Largest proposed trades"))
            with t2:
                st.dataframe(fmt_table(trades[["ticker", "sector", "prev", "target", "delta"]].head(40), ["prev", "target", "delta"]), width="stretch", hide_index=True)

        k = st.columns(4)
        k[0].metric("Trade Count", ops.get("n_trades", "n/a"))
        k[1].metric("Latest Turnover", pct(ops.get("turnover_two_way_latest"), 1))
        k[2].metric("Rebalance Freq", f"{ops.get('rebalance_freq_days', 'n/a')}d")
        k[3].metric("Fallback Rate", pct(ops.get("optimizer_failure_rate"), 1))

    with tabs[5]:
        st.subheader("Signals and adoption gates")
        features = data["features"]
        fg = rows_df(features.get("group_importance"))
        if not fg.empty:
            fg = fg.rename(columns={"name": "group", "value": "share_pct"})
            render_chart(hbar_fig(fg.sort_values("share_pct"), "group", "share_pct", "Feature group importance"))

        top_features = rows_df(features.get("top_features"))
        if not top_features.empty:
            c1, c2 = st.columns([1.1, 1.0])
            with c1:
                render_chart(hbar_fig(top_features.head(20).sort_values("share_pct"), "feature", "share_pct", "Top features by gain share", color="group"))
            with c2:
                st.dataframe(fmt_table(top_features[["feature", "group", "importance", "share_pct"]], [], digits=2), width="stretch", hide_index=True)

        attr = data["attr"]
        head = (attr.get("legA_B") or {}).get("headline") or {}
        a = st.columns(4)
        a[0].metric("Linear Share", pct(head.get("linear_share"), 1))
        a[1].metric("Nonlinear Upper Bound", pct(head.get("nonlinear_share_upper_bound"), 1))
        a[2].metric("Leg C Active Delta", pct(attr.get("legC_construction_active_delta"), 2, signed=True))
        a[3].metric("Attribution Round-trip", num(attr.get("roundtrip_full_vs_s0_abs_diff"), 2))

        overlay = data["overlay"]
        rows = []
        for arm, vals in overlay.items():
            if not isinstance(vals, dict):
                continue
            sp = vals.get("sub_periods") or {}
            rows.append({
                "arm": arm,
                "IR": vals.get("information_ratio"),
                "active_return": vals.get("active_return"),
                "tracking_error": vals.get("tracking_error"),
                "P1": sp.get("P1_ir"),
                "P2": sp.get("P2_ir"),
                "P3": sp.get("P3_ir"),
            })
        ov = pd.DataFrame(rows)
        if not ov.empty:
            render_chart(hbar_fig(ov.sort_values("IR"), "arm", "IR", "Overlay ablation IR"))
            st.dataframe(fmt_table(ov, ["active_return", "tracking_error"], digits=3), width="stretch", hide_index=True)

        factor = data["factor"]
        drops = rows_df(factor.get("exposure_drop_pct")).rename(columns={"name": "axis", "value": "drop_pct"})
        if not drops.empty:
            render_chart(hbar_fig(drops.sort_values("drop_pct"), "axis", "drop_pct", "Factor exposure reduction"))
        st.info(stage3.get("decision", "No factor verdict"))

        dq = data["dq"]
        cov = dq.get("coverage") or {}
        deg = dq.get("degenerate_models") or {}
        el = cov.get("engine_logged") or {}
        q = st.columns(5)
        q[0].metric("Date Sheets", cov.get("n_date_sheets", "n/a"))
        q[1].metric("Intersection", f"{el.get('intersection_dates', 'n/a')} / {el.get('longest_dates', 'n/a')}")
        q[2].metric("Intersection %", f"{el.get('pct_of_longest', 'n/a')}%")
        q[3].metric("Tail ffill", f"{el.get('tail_ffill_days', 'n/a')}d")
        q[4].metric("Degenerate Folds", f"{deg.get('degenerate_retrains', 'n/a')}/{deg.get('total_retrains', 'n/a')}")

        by_year = rows_df(deg.get("by_year"))
        if not by_year.empty:
            by_year = by_year.rename(columns={"name": "year"})
            render_chart(grouped_bar_fig(by_year, "year", ["retrains", "degenerate"], "Degenerate model folds by year"))

        with st.expander("Raw adoption summary"):
            st.json(summary, expanded=False)

    with tabs[6]:
        st.subheader("Backtest runs")
        st.markdown(
            "<div class='note'>Pick any discovered run to inspect its headline metrics and "
            "backtest curves. ECOS solver.</div>",
            unsafe_allow_html=True,
        )
        runs = list_runs(OUT)
        if not runs:
            st.warning(f"No runs found under {OUT}. Run run_variant.py first.")
        else:
            labels = [r["label"] for r in runs]
            choice = st.selectbox("Select run", labels, index=0, key="bt_run_select")
            st.caption(f"{len(runs)} run(s) discovered")
            run = next(r for r in runs if r["label"] == choice)
            rm = run["metrics"]

            m = st.columns(7)
            m[0].metric("Information Ratio", num(rm.get("information_ratio"), 3))
            m[1].metric("Active Return", pct(rm.get("active_return"), 2, signed=True))
            m[2].metric("Tracking Error", pct(rm.get("tracking_error"), 2))
            m[3].metric("Ann. Turnover", num(rm.get("avg_annual_turnover"), 2))
            m[4].metric("Realized Beta", num(rm.get("realized_beta"), 3))
            m[5].metric("Max Drawdown", pct(rm.get("max_drawdown"), 1))
            m[6].metric("Avg IC", num(rm.get("avg_ic"), 3))

            sp = rm.get("sub_periods") or {}
            sp_df = pd.DataFrame(
                [
                    {"period": "P1", "IR": sp.get("P1_ir")},
                    {"period": "P2", "IR": sp.get("P2_ir")},
                    {"period": "P3", "IR": sp.get("P3_ir")},
                ]
            )
            st.subheader("Sub-period information ratio")
            st.dataframe(sp_df, width="stretch", hide_index=True)

            result = cached_result(str(run["dir"]))
            if result is None:
                st.info(
                    "No backtest_result.pkl for this run — showing metrics only. "
                    "Re-run run_variant.py to regenerate the pickle for chart views."
                )
            else:
                cum = pd.DataFrame(
                    {
                        "portfolio": result.cumulative_returns,
                        "benchmark": result.cumulative_benchmark,
                    }
                )
                active = (result.cumulative_returns - result.cumulative_benchmark).rename("active")
                b1, b2 = st.columns(2)
                with b1:
                    render_chart(line_fig(cum, ["portfolio", "benchmark"], "Cumulative return", "Growth of $1"))
                with b2:
                    render_chart(line_fig(active.to_frame(), ["active"], "Cumulative active return", "Active growth"))

                b3, b4 = st.columns(2)
                with b3:
                    render_chart(line_fig(result.ic_series.rename("IC").to_frame(), ["IC"], "Information coefficient (per rebalance)", "IC"))
                with b4:
                    render_chart(line_fig(result.turnover.rename("turnover").to_frame(), ["turnover"], "Two-way turnover per rebalance", "Turnover"))

            with st.expander("Raw metrics.json"):
                st.json(load_metrics(run["dir"] / "metrics.json"), expanded=False)


if __name__ == "__main__":
    main()
