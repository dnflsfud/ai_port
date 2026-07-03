#!/usr/bin/env python
"""Read-only Streamlit results dashboard for ai_port backtest runs.

The pure helpers (``list_runs`` / ``load_metrics`` / ``load_result``) are
module-level and Streamlit-free, so ``import streamlit_app`` has no UI side
effects. All Streamlit UI lives inside ``main()``, guarded by ``__main__``.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
PROD_LABEL = "iter15_65tkr_reb21_vtg"

COLOR_PORT = "#183B56"
COLOR_BM = "#829AB1"
COLOR_ACTIVE = "#0B7285"
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
div[data-baseweb="input"] input,
div[data-baseweb="select"] > div {
  background: #ffffff !important;
  color: #102a43 !important;
  border-color: #d9e2ec !important;
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
    return apply_theme(fig, height=340)


def render_chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", theme=None)


# ---------------------------------------------------------------------------
# UI entrypoint.
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Pictet Portfolio Monitor",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    @st.cache_resource(show_spinner=False)
    def cached_result(run_dir_str: str):
        return load_result(run_dir_str)

    runs = list_runs(OUT)

    st.title("Pictet Portfolio Monitor")
    st.markdown(
        "<div class='note'>Read-only results dashboard: pick a run to inspect "
        "its metrics and backtest curves. ECOS solver.</div>",
        unsafe_allow_html=True,
    )

    if not runs:
        st.warning(f"No runs found under {OUT}. Run run_variant.py first.")
        st.stop()

    labels = [r["label"] for r in runs]
    with st.sidebar:
        st.subheader("Run")
        choice = st.selectbox("Select run", labels, index=0)
        st.metric("Solver", "ECOS")
        st.caption(f"{len(runs)} run(s) discovered")

    run = next(r for r in runs if r["label"] == choice)
    metrics = run["metrics"]

    cols = st.columns(7)
    cols[0].metric("Information Ratio", num(metrics.get("information_ratio"), 3))
    cols[1].metric("Active Return", pct(metrics.get("active_return"), 2, signed=True))
    cols[2].metric("Tracking Error", pct(metrics.get("tracking_error"), 2))
    cols[3].metric("Ann. Turnover", num(metrics.get("avg_annual_turnover"), 2))
    cols[4].metric("Realized Beta", num(metrics.get("realized_beta"), 3))
    cols[5].metric("Max Drawdown", pct(metrics.get("max_drawdown"), 1))
    cols[6].metric("Avg IC", num(metrics.get("avg_ic"), 3))

    sp = metrics.get("sub_periods") or {}
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
        c1, c2 = st.columns(2)
        with c1:
            render_chart(line_fig(cum, ["portfolio", "benchmark"], "Cumulative return", "Growth of $1"))
        with c2:
            render_chart(line_fig(active.to_frame(), ["active"], "Cumulative active return", "Active growth"))

        c3, c4 = st.columns(2)
        with c3:
            render_chart(line_fig(result.ic_series.rename("IC").to_frame(), ["IC"], "Information coefficient (per rebalance)", "IC"))
        with c4:
            render_chart(line_fig(result.turnover.rename("turnover").to_frame(), ["turnover"], "Two-way turnover per rebalance", "Turnover"))

    with st.expander("Raw metrics.json"):
        st.json(load_metrics(run["dir"] / "metrics.json"), expanded=False)


if __name__ == "__main__":
    main()
