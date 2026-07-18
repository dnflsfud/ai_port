#!/usr/bin/env python
"""Read-only Streamlit dashboard for the Pictet portfolio adoption run.

Two views are merged here:

* Six operating tabs (Overview / Performance / Contribution / Risk / Trading /
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
from datetime import datetime, timezone
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

# Validated light-mode palette (dataviz skill references/palette.md).
# Categorical slots keep their fixed CVD-safe order; benchmark is neutral grey
# (never a categorical hue); pos/neg are status tokens (never themed); axis/ink
# use text tokens, not series colors.
COLOR_PORT = "#2a78d6"        # categorical slot 1 (blue) — portfolio
COLOR_BM = "#898781"          # neutral grey — benchmark
COLOR_ACTIVE = "#1baf7a"      # categorical slot 2 (aqua) — active
COLOR_CAT3 = "#eda100"        # categorical slot 3 (yellow)
COLOR_CAT4 = "#008300"        # categorical slot 4 (green)
COLOR_CAT5 = "#4a3aa7"        # categorical slot 5 (violet)
COLOR_POS = "#0ca30c"         # status good
COLOR_NEG = "#d03b3b"         # status critical
COLOR_DIVERGE_MID = "#f0efec" # diverging neutral gray midpoint
COLOR_INK = "#0b0b0b"         # primary ink
COLOR_INK_SECONDARY = "#52514e"  # secondary ink
COLOR_INK_MUTED = "#898781"   # muted ink (axis/labels)
COLOR_GRID = "#e1e0d9"        # gridline hairline
COLOR_AXIS = "#c3c2b7"        # baseline / axis (one step darker than grid)
COLOR_SURFACE = "#fcfcfb"     # chart surface

CSS = """
<style>
:root {
  --ink: #0b0b0b;
  --ink-2: #52514e;
  --ink-muted: #898781;
  --surface: #fcfcfb;
  --page: #f9f9f7;
  --border: rgba(11, 11, 11, 0.10);
}
.stApp {
  background: var(--page);
  color: var(--ink);
}
.block-container {
  max-width: 1500px;
  padding-top: 1.25rem;
  padding-bottom: 2.5rem;
}
section[data-testid="stSidebar"] {
  background: var(--surface);
  border-right: 1px solid var(--border);
}
div[data-testid="stMetric"] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 0.75rem 0.9rem;
  box-shadow: 0 1px 2px rgba(11, 11, 11, 0.04);
}
div[data-testid="stMetricLabel"] *, div[data-testid="stMetricLabel"] p {
  color: var(--ink-2);
  font-size: 0.78rem;
}
div[data-testid="stMetricValue"] {
  color: var(--ink);
  font-size: 1.45rem;
}
div[data-testid="stTabs"] button p {
  color: var(--ink-muted);
  font-size: 0.9rem;
}
div[data-testid="stTabs"] button[aria-selected="true"] p {
  color: var(--ink);
  font-weight: 700;
}
.status-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  margin: 0.5rem 0 1.0rem 0;
}
.chip {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 0.28rem 0.62rem;
  font-size: 0.78rem;
  color: var(--ink-2);
}
.chip-ok {
  background: rgba(12, 163, 12, 0.10);
  border-color: rgba(12, 163, 12, 0.35);
}
.chip-warn {
  background: rgba(208, 59, 59, 0.10);
  border-color: rgba(208, 59, 59, 0.35);
}
.note {
  color: var(--ink-muted);
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


def load_portfolio_registry(outputs_dir=OUT) -> dict:
    """Load the validated registry or return a legacy-only compatibility view."""
    outputs_dir = Path(outputs_dir)
    path = outputs_dir / "portfolio_registry.json"
    if path.exists():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and value.get("portfolios"):
                return value
        except Exception:
            pass
    return {
        "schema_version": 0,
        "generated_at_utc": None,
        "stale_after_hours": 96,
        "portfolios": [{
            "id": PROD_LABEL,
            "display_name": "Production S0",
            "portfolio_role": "production",
            "operating_dir": "outputs/operating",
            "run_dir": f"outputs/{PROD_LABEL}",
            "status": "PRODUCTION",
        }],
    }


def load_operating_bundle(meta: dict, project_root=HERE) -> dict:
    """Read one registry-described operating bundle with a stable schema."""
    root = Path(project_root)
    operating = Path(meta.get("operating_dir", "outputs/operating"))
    operating = operating if operating.is_absolute() else root / operating

    def read_json(name: str) -> dict:
        path = operating / name
        if not path.exists():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except Exception as exc:
            return {"_error": str(exc)}

    returns_path = operating / "returns.csv"
    try:
        returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    except Exception:
        returns = pd.DataFrame()
    return {
        "meta": {**meta, **read_json("portfolio.json")},
        "perf": read_json("performance.json"),
        "holdings": read_json("holdings.json"),
        "features": read_json("features.json"),
        "ops": read_json("operations.json"),
        "contribution": read_json("contribution.json"),
        "risk": read_json("risk.json"),
        "monitoring": read_json("monitoring.json"),
        "currency": read_json("currency.json"),
        "feature_attribution": read_json("feature_attribution.json"),
        "returns": returns,
    }


def build_comparison_returns(
    production: pd.DataFrame,
    challenger: pd.DataFrame,
    prod_name: str = "Production",
    chal_name: str = "Challenger",
) -> pd.DataFrame:
    """Align the two portfolio curves and retain one common benchmark curve."""
    if production.empty:
        return pd.DataFrame()
    out = production[[c for c in ("portfolio_cum", "benchmark_cum") if c in production]].rename(
        columns={"portfolio_cum": prod_name, "benchmark_cum": "Benchmark"}
    )
    if not challenger.empty and "portfolio_cum" in challenger:
        out = out.join(
            challenger[["portfolio_cum"]].rename(columns={"portfolio_cum": chal_name}),
            how="inner",
        )
    return out.sort_index()


def currency_daily_frame(currency: dict) -> pd.DataFrame:
    """Return the auditable daily local/FX decomposition indexed by date."""
    daily = rows_df((currency or {}).get("daily"))
    if daily.empty or "date" not in daily.columns:
        return pd.DataFrame()
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily = daily.dropna(subset=["date"]).set_index("date").sort_index()
    for column in daily.columns:
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
    return daily


def summarize_currency_period(currency: dict, start=None, end=None) -> dict:
    """Sum arithmetic local/FX effects over a selected dashboard window."""
    daily = currency_daily_frame(currency)
    if daily.empty:
        summary = dict((currency or {}).get("summary") or {})
        summary["observations"] = 0
        return summary
    if start is not None:
        daily = daily.loc[daily.index >= pd.Timestamp(start)]
    if end is not None:
        daily = daily.loc[daily.index <= pd.Timestamp(end)]
    numeric = daily.select_dtypes(include=[np.number])
    result = {column: float(value) for column, value in numeric.sum().items()}
    if "active_fx_effect" not in result:
        result["active_fx_effect"] = (
            result.get("portfolio_fx_effect", 0.0)
            - result.get("benchmark_fx_effect", 0.0)
        )
    result["observations"] = int(len(daily))
    return result


def collect_operating_alerts(
    monitoring: dict, currency: dict | None = None, operations: dict | None = None
) -> list[dict]:
    """Collect actionable risk, model, data, and FX breaches for one panel."""
    guardrails = (monitoring or {}).get("guardrails") or {}
    labels = {
        "estimated_te_breached": "Estimated tracking-error limit",
        "te_constraint_breached": "Historical rebalance TE constraint",
        "top_name_active_risk_breached": "Single-name active-risk concentration",
        "top_sector_active_risk_breached": "Sector active-risk concentration",
        "model_degenerate_rate_breached": "Degenerate model-fold rate",
        "tail_ffill_breached": "Market-data tail fill",
    }
    value_keys = {
        "estimated_te_breached": "latest_estimated_te",
        "te_constraint_breached": "max_rebalance_estimated_te",
        "top_name_active_risk_breached": "top_name_active_risk_share",
        "top_sector_active_risk_breached": "top_sector_active_risk_share",
        "model_degenerate_rate_breached": "model_degenerate_rate",
        "tail_ffill_breached": "tail_ffill_days",
    }
    alerts = []
    for key, label in labels.items():
        if bool(guardrails.get(key)):
            alerts.append({"key": key, "label": label, "value": guardrails.get(value_keys[key])})

    coverage = (currency or {}).get("coverage") or {}
    missing = (
        coverage.get("missing_tickers")
        or coverage.get("missing")
        or coverage.get("missing_currency")
        or []
    )
    missing_fx = coverage.get("missing_fx_tickers") or coverage.get("missing_fx") or []
    stale = coverage.get("stale_tickers") or coverage.get("stale") or coverage.get("stale_fx") or []
    if missing:
        alerts.append({"key": "currency_mapping_missing", "label": "Missing currency mapping", "value": missing})
    if missing_fx:
        alerts.append({"key": "fx_series_missing", "label": "Missing FX series", "value": missing_fx})
    if stale:
        alerts.append({"key": "fx_series_stale", "label": "Stale FX series", "value": stale})

    sector_active = (operations or {}).get("sector_active") or []
    binding_sectors = [row.get("sector") for row in sector_active if row.get("binding")]
    if binding_sectors:
        alerts.append({
            "key": "sector_deviation_binding",
            "label": "Sector deviation at limit",
            "value": binding_sectors,
        })
    return alerts


def prepare_stock_drivers(attr, ticker=None):
    """Shape the feature_attribution JSON for the Stock Drivers tab (no Streamlit).

    ``attr`` is the parsed ``operating/feature_attribution.json`` (``{}`` when the
    file is absent). Returns ``None`` when there is nothing to show, else
    ``{"options", "default", "selected", "top_features", "metrics"}`` where
    ``options`` is [(ticker, active), ...] active-DESC, ``default`` is the max-OW
    ticker, ``top_features`` is [(feat, group, shap), ...] |shap|-DESC capped at 12
    for the selected (or default) stock, and ``metrics`` is that stock's
    weight/bm_weight/active/mu.
    """
    if not attr:
        return None
    tickers = attr.get("tickers") or {}
    if not tickers:
        return None
    groups = attr.get("feature_groups") or {}
    options = sorted(
        ((t, float(rec.get("active", 0.0))) for t, rec in tickers.items()),
        key=lambda ta: ta[1], reverse=True,
    )
    default = options[0][0]
    selected = ticker if ticker in tickers else default
    rec = tickers[selected]
    shap = rec.get("shap") or {}
    top = sorted(shap.items(), key=lambda kv: abs(kv[1]), reverse=True)[:12]
    top_features = [(f, groups.get(f, "?"), float(v)) for f, v in top]
    metrics = {k: float(rec.get(k, 0.0)) for k in ("weight", "bm_weight", "active", "mu")}
    return {"options": options, "default": default, "selected": selected,
            "top_features": top_features, "metrics": metrics}


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
        paper_bgcolor=COLOR_SURFACE,
        plot_bgcolor=COLOR_SURFACE,
        font={"color": COLOR_INK_SECONDARY, "family": "system-ui, -apple-system, 'Segoe UI', sans-serif"},
        title_font={"color": COLOR_INK, "size": 15},
        legend_font={"color": COLOR_INK_SECONDARY},
        margin={"l": 10, "r": 10, "t": 48, "b": 24},
        height=height,
    )
    fig.update_xaxes(
        color=COLOR_INK_SECONDARY,
        tickfont={"color": COLOR_INK_MUTED},
        title_font={"color": COLOR_INK_SECONDARY},
        gridcolor=COLOR_GRID,
        zerolinecolor=COLOR_AXIS,
    )
    fig.update_yaxes(
        color=COLOR_INK_SECONDARY,
        tickfont={"color": COLOR_INK_MUTED},
        title_font={"color": COLOR_INK_SECONDARY},
        gridcolor=COLOR_GRID,
        zerolinecolor=COLOR_AXIS,
    )
    return fig


_chart_seq = itertools.count()


def render_chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, key=f"chart_{next(_chart_seq)}")


def line_fig(df: pd.DataFrame, cols: list, title: str, y_title: str = "") -> go.Figure:
    fig = go.Figure()
    # Fixed-order slots (never cycled): portfolio, benchmark, active, cat3, cat4.
    colors = [COLOR_PORT, COLOR_BM, COLOR_ACTIVE, COLOR_CAT3, COLOR_CAT4]
    assert len(cols) <= len(colors), "line_fig: more series than fixed palette slots"
    for i, col in enumerate(cols):
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    mode="lines",
                    name=col,
                    line={"width": 2, "color": colors[i]},
                )
            )
    fig.update_layout(title=title, yaxis_title=y_title, xaxis_title="", hovermode="x unified")
    return apply_theme(fig, height=360)


def hbar_fig(df: pd.DataFrame, y: str, x: str, title: str, color: Optional[str] = None) -> go.Figure:
    fig = px.bar(
        df,
        x=x,
        y=y,
        orientation="h",
        color=color,
        color_discrete_sequence=[COLOR_ACTIVE, COLOR_CAT3, COLOR_CAT4, COLOR_CAT5, COLOR_PORT],
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


def fx_decomposition_fig(daily: pd.DataFrame) -> go.Figure:
    """Monthly arithmetic USD-return decomposition into local and FX effects."""
    columns = ["portfolio_local_return", "portfolio_fx_effect"]
    available = [column for column in columns if column in daily.columns]
    if daily.empty or not available:
        return apply_theme(go.Figure(), height=360)
    monthly = daily[available].groupby(daily.index.to_period("M")).sum(min_count=1)
    monthly.index = monthly.index.astype(str)
    labels = {
        "portfolio_local_return": "Local-market return",
        "portfolio_fx_effect": "FX effect",
    }
    colors = {
        "portfolio_local_return": COLOR_PORT,
        "portfolio_fx_effect": COLOR_CAT3,
    }
    fig = go.Figure()
    for column in available:
        fig.add_bar(
            x=monthly.index,
            y=monthly[column],
            name=labels[column],
            marker_color=colors[column],
            hovertemplate="%{x}<br>%{y:.2%}<extra>" + labels[column] + "</extra>",
        )
    fig.update_layout(
        title="Monthly portfolio return decomposition (USD)",
        barmode="relative",
        xaxis_title="",
        yaxis_title="Arithmetic contribution",
        hovermode="x unified",
    )
    fig.update_yaxes(tickformat=".1%")
    return apply_theme(fig, height=380)


def currency_exposure_fig(by_currency: pd.DataFrame) -> go.Figure:
    """Latest portfolio and benchmark currency exposures."""
    fig = go.Figure()
    if by_currency.empty or "currency" not in by_currency.columns:
        return apply_theme(fig, height=340)
    for column, label, color in (
        ("target_weight", "Portfolio", COLOR_PORT),
        ("benchmark_weight", "Benchmark", COLOR_BM),
    ):
        if column in by_currency.columns:
            fig.add_bar(
                x=by_currency["currency"],
                y=by_currency[column],
                name=label,
                marker_color=color,
            )
    fig.update_layout(
        title="Latest currency exposure",
        barmode="group",
        xaxis_title="",
        yaxis_title="Weight",
        hovermode="x unified",
    )
    fig.update_yaxes(tickformat=".1%")
    return apply_theme(fig, height=340)


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
        registry = load_portfolio_registry(OUT)
        bundles = {
            meta["id"]: load_operating_bundle(meta, HERE)
            for meta in registry.get("portfolios", [])
        }
        production = next(
            (b for b in bundles.values() if b["meta"].get("portfolio_role") == "production"),
            load_operating_bundle({"id": PROD_LABEL, "operating_dir": "outputs/operating"}, HERE),
        )
        challenger = next(
            (b for b in bundles.values() if b["meta"].get("portfolio_role") == "challenger"),
            None,
        )
        s0_raw = load_json("iter15_65tkr_reb21_vtg/metrics.json")
        return {
            "registry": registry,
            "production": production,
            "challenger": challenger,
            "summary": load_json("adoption_summary.json"),
            "s0": s0_raw.get("metrics", s0_raw),
            "attr": load_json("alpha_attribution/summary.json"),
            "overlay": load_json("overlay_ablation/summary.json"),
            "factor": load_json("factor_ablation/summary.json"),
            "dq": load_json("data_quality_report.json"),
            "perf": production["perf"],
            "holdings": production["holdings"],
            "features": production["features"],
            "ops": production["ops"],
            "contribution": production["contribution"],
            "risk": production["risk"],
            "monitoring": production["monitoring"],
            "currency": production["currency"],
            "feature_attribution": production["feature_attribution"],
            "returns": production["returns"],
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
    currency = data["currency"]
    challenger = data.get("challenger") or {}
    challenger_meta = challenger.get("meta") or {}
    prod_name = (data["production"]["meta"] or {}).get("display_name") or "Production"
    chal_name = challenger_meta.get("display_name") or "Challenger"
    challenger_perf = challenger.get("perf") or {}
    challenger_holdings = challenger.get("holdings") or {}
    challenger_returns = challenger.get("returns")
    if not isinstance(challenger_returns, pd.DataFrame):
        challenger_returns = pd.DataFrame()
    comparison_returns = build_comparison_returns(returns, challenger_returns, prod_name, chal_name)

    stage0 = summary.get("stage0_baseline", {})
    stage2 = summary.get("stage2_overlay", {})
    stage3 = summary.get("stage3_factor", {})
    overlay_decisions = stage2.get("decisions", {})
    overlay_ok = bool(overlay_decisions) and all(v == "KEEP" for v in overlay_decisions.values())
    factor_ok = not bool(stage3.get("collapsed"))

    production_meta = data["production"].get("meta") or {}
    fx_coverage = (currency or {}).get("coverage") or {}
    universe_size = int(production_meta.get("universe_size") or fx_coverage.get("total") or 0)
    fx_mapped = int(fx_coverage.get("mapped") or 0)
    fx_gaps = []
    for coverage_key in ("missing", "missing_fx", "stale"):
        coverage_value = fx_coverage.get(coverage_key)
        if isinstance(coverage_value, (list, tuple, set)):
            fx_gaps.extend(coverage_value)
        elif coverage_value:
            fx_gaps.append(coverage_value)
    fx_coverage_ok = bool(universe_size and fx_mapped == universe_size and not fx_gaps)
    currency_summary = (currency or {}).get("summary") or {}

    production_gate = data["registry"].get("production_gate")
    if production_gate:
        _pg_status = production_gate.get("status")
        production_gate_chip = (
            f"<span class='chip {'chip-ok' if _pg_status == 'PRODUCTION' else 'chip-warn'}'>"
            f"Prod gate {_pg_status}</span>"
        )
    else:
        production_gate_chip = ""

    st.title("Pictet Portfolio Monitor — USD")
    st.markdown(
        "<div class='note'>Operating dashboard for the 100-name universe, unhedged USD performance, FX attribution, risk and rebalance controls.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='status-row'>"
        f"<span class='chip chip-ok'>Portfolio IR {num(perf.get('information_ratio'), 3)}</span>"
        f"<span class='chip chip-ok'>TE {pct(perf.get('tracking_error'), 2)}</span>"
        f"<span class='chip chip-ok'>Beta {num(perf.get('realized_beta'), 3)}</span>"
        f"<span class='chip {'chip-ok' if overlay_ok else 'chip-warn'}'>Overlay {'KEEP all' if overlay_ok else 'Review'}</span>"
        f"<span class='chip {'chip-ok' if factor_ok else 'chip-warn'}'>Factor collapsed={stage3.get('collapsed')}</span>"
        f"<span class='chip chip-ok'>Base currency {(currency or {}).get('base_currency', 'USD')}</span>"
        f"<span class='chip {'chip-ok' if universe_size == 100 else 'chip-warn'}'>Universe {universe_size or 'n/a'}</span>"
        f"<span class='chip {'chip-ok' if fx_coverage_ok else 'chip-warn'}'>FX mapped {fx_mapped or 'n/a'}/{universe_size or 'n/a'}</span>"
        f"<span class='chip chip-ok'>Non-USD {pct(currency_summary.get('non_usd_target_weight'), 1)}</span>"
        + (f"<span class='chip {'chip-ok' if challenger_meta.get('status') == 'PASS' else 'chip-warn'}'>"
           f"{chal_name} {challenger_meta.get('status', 'not built')}</span>" if challenger else "")
        + production_gate_chip +
        "</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Scope")
        st.metric("Market data as of", production_meta.get("data_as_of", perf.get("as_of", "n/a")))
        st.metric("FX data as of", (currency or {}).get("fx_data_as_of", "n/a"))
        st.metric("Holdings as of", holdings.get("as_of", "n/a"))
        st.metric("Next rebalance", ops.get("next_expected_rebalance_date", "n/a"))
        st.caption(
            f"{ops.get('rows_until_next_rebalance', 'n/a')} weekday rows remaining | "
            f"last rebalance {ops.get('last_rebalance_date', ops.get('as_of', 'n/a'))}"
        )
        if challenger:
            st.metric(f"{chal_name} data as of", challenger_perf.get("as_of", "n/a"))
            st.metric(f"{chal_name} holdings as of", challenger_holdings.get("as_of", "n/a"))
        st.metric("Solver", ops.get("solver_protocol", "ECOS"))

        generated = data["registry"].get("generated_at_utc")
        if generated:
            try:
                generated_dt = datetime.fromisoformat(str(generated).replace("Z", "+00:00"))
                age_hours = (datetime.now(timezone.utc) - generated_dt).total_seconds() / 3600.0
                st.caption(f"Bundle generated: {generated_dt.astimezone().strftime('%Y-%m-%d %H:%M')}")
                if age_hours > float(data["registry"].get("stale_after_hours", 96)):
                    st.warning(f"Portfolio bundle is stale ({age_hours / 24:.1f} days old).")
            except Exception:
                st.warning("Portfolio registry timestamp is invalid.")

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
    filtered_challenger_returns = challenger_returns.copy()
    if date_range and len(date_range) == 2 and not returns.empty:
        start = pd.to_datetime(date_range[0])
        end = pd.to_datetime(date_range[1])
        filtered_returns = returns.loc[(returns.index >= start) & (returns.index <= end)].copy()
        if not challenger_returns.empty:
            filtered_challenger_returns = challenger_returns.loc[
                (challenger_returns.index >= start) & (challenger_returns.index <= end)
            ].copy()
    comparison_returns = build_comparison_returns(filtered_returns, filtered_challenger_returns, prod_name, chal_name)
    period_start = filtered_returns.index.min() if not filtered_returns.empty else None
    period_end = filtered_returns.index.max() if not filtered_returns.empty else None
    filtered_currency_daily = currency_daily_frame(currency)
    if period_start is not None and not filtered_currency_daily.empty:
        filtered_currency_daily = filtered_currency_daily.loc[
            (filtered_currency_daily.index >= period_start)
            & (filtered_currency_daily.index <= period_end)
        ]
    fx_period = summarize_currency_period(currency, period_start, period_end)

    def sector_filter(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or sector == "All sectors" or "sector" not in df.columns:
            return df
        return df[df["sector"] == sector].copy()

    top_cols = st.columns(6)
    top_cols[0].metric("Annual Return (USD)", pct(perf.get("annual_return"), 1))
    top_cols[1].metric("Active Return", pct(perf.get("active_return"), 2, signed=True))
    top_cols[2].metric("Information Ratio", num(perf.get("information_ratio"), 3))
    top_cols[3].metric("Tracking Error", pct(perf.get("tracking_error"), 2))
    top_cols[4].metric("Max Drawdown", pct(perf.get("max_drawdown"), 1))
    top_cols[5].metric("Active Share", pct(holdings.get("active_share_one_way"), 2))

    if challenger:
        st.caption(f"{chal_name} challenger")
        challenger_cols = st.columns(6)
        challenger_cols[0].metric("Annual Return (USD)", pct(challenger_perf.get("annual_return"), 1))
        challenger_cols[1].metric("Active Return", pct(challenger_perf.get("active_return"), 2, signed=True))
        challenger_cols[2].metric("Information Ratio", num(challenger_perf.get("information_ratio"), 3))
        challenger_cols[3].metric("Tracking Error", pct(challenger_perf.get("tracking_error"), 2))
        challenger_cols[4].metric("Max Drawdown", pct(challenger_perf.get("max_drawdown"), 1))
        challenger_cols[5].metric("Active Share", pct(challenger_holdings.get("active_share_one_way"), 2))

    tabs = st.tabs(
        ["Overview", "Performance", "Contribution", "Risk", "Trading & Operations", "Signals & Gates", "Backtest Runs", "Stock Drivers", "Comparison"]
    )

    with tabs[0]:
        left, right = st.columns([1.45, 1.0])
        with left:
            if not comparison_returns.empty:
                render_chart(line_fig(comparison_returns, list(comparison_returns.columns), "Cumulative return", "Growth of $1"))
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
            if not comparison_returns.empty:
                render_chart(line_fig(comparison_returns, list(comparison_returns.columns), "Cumulative return", "Growth of $1"))
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
                    color_continuous_scale=[COLOR_NEG, COLOR_DIVERGE_MID, COLOR_PORT],
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

        st.divider()
        st.subheader("USD return and FX effect")
        st.markdown(
            "<div class='note'>Unhedged USD returns are decomposed daily using entering-day weights. "
            "Local return + FX effect equals gross USD return; period figures are arithmetic contributions.</div>",
            unsafe_allow_html=True,
        )
        if not filtered_currency_daily.empty:
            fx_cards = st.columns(4)
            fx_cards[0].metric(
                "Portfolio FX effect",
                pct(fx_period.get("portfolio_fx_effect"), 2, signed=True),
            )
            fx_cards[1].metric(
                "Benchmark FX effect",
                pct(fx_period.get("benchmark_fx_effect"), 2, signed=True),
            )
            fx_cards[2].metric(
                "Active FX effect",
                pct(fx_period.get("active_fx_effect"), 2, signed=True),
            )
            fx_cards[3].metric(
                "Latest non-USD weight",
                pct(currency_summary.get("non_usd_target_weight"), 1),
            )
            render_chart(fx_decomposition_fig(filtered_currency_daily))
            reconciliation = (currency or {}).get("reconciliation") or {}
            max_error = max(
                abs(float(reconciliation.get("max_daily_portfolio_error") or 0.0)),
                abs(float(reconciliation.get("max_daily_benchmark_error") or 0.0)),
            )
            st.caption(
                f"{fx_period.get('observations', 0):,} daily observations | "
                f"maximum daily reconciliation error {max_error:.2e}"
            )
        else:
            st.info("No FX attribution is available. Re-export the operating bundle after the USD backtest.")

    with tabs[2]:
        st.subheader("Stock and sector contribution")
        st.markdown(
            "<div class='note'>Arithmetic contribution uses entering-day weights. Cost/timing residual is shown separately.</div>",
            unsafe_allow_html=True,
        )
        if challenger:
            st.caption("Latest active contribution leaders — production vs challenger")
            pc1, pc2 = st.columns(2)
            prod_leaders = rows_df(contribution.get("by_ticker"))
            causal_leaders = rows_df((challenger.get("contribution") or {}).get("by_ticker"))
            with pc1:
                st.markdown(f"**{prod_name}**")
                if not prod_leaders.empty:
                    st.dataframe(
                        fmt_table(prod_leaders.nlargest(8, "active_contribution")[["ticker", "sector", "active_contribution"]], ["active_contribution"]),
                        width="stretch", hide_index=True,
                    )
            with pc2:
                st.markdown(f"**{chal_name}**")
                if not causal_leaders.empty:
                    st.dataframe(
                        fmt_table(causal_leaders.nlargest(8, "active_contribution")[["ticker", "sector", "active_contribution"]], ["active_contribution"]),
                        width="stretch", hide_index=True,
                    )
            st.divider()
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
        if challenger:
            st.markdown(
                "<div class='note'>The same sector filter is applied to both funds. "
                "Position weights and risk contributions are shown from each fund's latest rebalance.</div>",
                unsafe_allow_html=True,
            )
            pr1, pr2 = st.columns(2)
            for col, label, bundle in (
                (pr1, prod_name, data["production"]),
                (pr2, chal_name, challenger),
            ):
                with col:
                    rmeta = bundle.get("risk") or {}
                    hmeta = bundle.get("holdings") or {}
                    fund_risk = sector_filter(rows_df(rmeta.get("by_ticker")))
                    fund_sector_risk = sector_filter(
                        rows_df(rmeta.get("by_sector")).rename(columns={"name": "sector"})
                    )
                    st.markdown(f"### {label}")
                    st.caption(f"Risk/positions as of {rmeta.get('as_of', hmeta.get('as_of', 'n/a'))}")
                    q1, q2, q3, q4, q5, q6 = st.columns(6)
                    q1.metric("Est. Vol", pct(rmeta.get("estimated_portfolio_vol"), 2))
                    q2.metric("Est. TE", pct(rmeta.get("estimated_tracking_error"), 2))
                    q3.metric("TE Limit", pct(rmeta.get("max_tracking_error_annual"), 2))
                    q4.metric(
                        "TE Headroom",
                        pct((rmeta.get("guardrails") or {}).get("estimated_te_headroom"), 2),
                    )
                    q5.metric("Active Share", pct(hmeta.get("active_share_one_way"), 2))
                    q6.metric("Holdings", hmeta.get("n_holdings", "n/a"))
                    if not fund_risk.empty:
                        fund_risk = fund_risk.copy()
                        fund_risk["abs_active_risk"] = fund_risk["active_te_contribution"].abs()
                        top_fund_risk = fund_risk.nlargest(12, "abs_active_risk").sort_values("active_te_contribution")
                        render_chart(hbar_fig(
                            top_fund_risk, "ticker", "active_te_contribution",
                            f"{label} — active TE contributors",
                        ))
                        position_cols = [
                            "ticker", "sector", "weight", "bm_weight", "active_weight",
                            "active_te_contribution", "active_risk_pct", "total_vol_contribution",
                        ]
                        st.markdown("**Position weights and risk decomposition**")
                        st.dataframe(
                            fmt_table(
                                fund_risk.sort_values("abs_active_risk", ascending=False)[position_cols],
                                [
                                    "weight", "bm_weight", "active_weight",
                                    "active_te_contribution", "active_risk_pct", "total_vol_contribution",
                                ],
                            ),
                            width="stretch", hide_index=True, height=420,
                        )
                    else:
                        st.info(f"No {label} risk detail is available.")
                    if not fund_sector_risk.empty:
                        render_chart(hbar_fig(
                            fund_sector_risk.sort_values("active_te_contribution"),
                            "sector", "active_te_contribution",
                            f"{label} — sector active TE",
                        ))
            st.divider()
            st.subheader(f"{prod_name} legacy detail")
        risk_df = sector_filter(rows_df(risk.get("by_ticker")))
        sector_risk = sector_filter(rows_df(risk.get("by_sector")).rename(columns={"name": "sector"}))
        if risk.get("error"):
            st.warning(risk["error"])
        elif risk_df.empty:
            st.info("No risk data. Run scripts/export_operating_data.py.")
        else:
            c = st.columns(5)
            c[0].metric("Estimated Vol", pct(risk.get("estimated_portfolio_vol"), 2))
            c[1].metric("Estimated TE", pct(risk.get("estimated_tracking_error"), 2))
            c[2].metric("TE Limit", pct(risk.get("max_tracking_error_annual"), 2))
            c[3].metric(
                "TE Headroom",
                pct((risk.get("guardrails") or {}).get("estimated_te_headroom"), 2),
            )
            c[4].metric("Active Share", pct(holdings.get("active_share_one_way"), 2))
            st.caption(
                "TE limit is an ex-ante rebalance constraint; realized rolling TE can differ. "
                f"Covariance lookback: {risk.get('cov_lookback_days')}d."
            )

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

        sector_active_rows = rows_df(ops.get("sector_active"))
        if not sector_active_rows.empty:
            st.divider()
            st.subheader("Sector deviation vs cap band")
            if "binding" in sector_active_rows.columns:
                binding_now = sector_active_rows[sector_active_rows["binding"] == True]
                if not binding_now.empty:
                    st.error("At sector cap: " + ", ".join(binding_now["sector"].astype(str)))
            sector_active_cols = [
                c for c in ("sector", "port", "bm", "active", "limit", "binding")
                if c in sector_active_rows.columns
            ]
            st.dataframe(
                fmt_table(sector_active_rows[sector_active_cols],
                          [c for c in ("port", "bm", "active", "limit") if c in sector_active_cols]),
                width="stretch", hide_index=True,
            )

        currency_risk = rows_df((currency or {}).get("by_currency"))
        if not currency_risk.empty:
            currency_risk = currency_risk.rename(columns={"name": "currency"})
            st.divider()
            st.subheader("Currency exposure and FX contribution")
            st.markdown(
                "<div class='note'>Latest unhedged weights are shown by listing currency. The stress column "
                "approximates portfolio P&L if that currency strengthens 1% against USD, holding local prices fixed.</div>",
                unsafe_allow_html=True,
            )
            if "target_weight" in currency_risk.columns:
                currency_risk = currency_risk.sort_values("target_weight", ascending=False)
                currency_risk["pnl_if_currency_up_1pct"] = np.where(
                    currency_risk["currency"].eq("USD"),
                    0.0,
                    pd.to_numeric(currency_risk["target_weight"], errors="coerce") * 0.01,
                )
            ccy_left, ccy_right = st.columns([1.05, 1.15])
            with ccy_left:
                render_chart(currency_exposure_fig(currency_risk))
            with ccy_right:
                currency_columns = [
                    column for column in (
                        "currency", "target_weight", "benchmark_weight", "active_weight",
                        "fx_move_1d", "fx_move_21d", "fx_source_date", "fx_stale",
                        "portfolio_fx_contribution", "benchmark_fx_contribution",
                        "active_fx_contribution", "pnl_if_currency_up_1pct",
                    ) if column in currency_risk.columns
                ]
                percent_columns = [
                    column for column in currency_columns
                    if column not in {"currency", "fx_source_date", "fx_stale"}
                ]
                st.dataframe(
                    fmt_table(currency_risk[currency_columns], percent_columns),
                    width="stretch",
                    hide_index=True,
                    height=340,
                )
            fx_stress = (currency or {}).get("stress") or {}
            plus_stress = fx_stress.get("plus_1pct") or {}
            minus_stress = fx_stress.get("minus_1pct") or {}
            if plus_stress or minus_stress:
                stress_cards = st.columns(3)
                stress_cards[0].metric(
                    "All non-USD currencies +1%",
                    pct(plus_stress.get("portfolio"), 2, signed=True),
                )
                stress_cards[1].metric(
                    "Active P&L at +1%",
                    pct(plus_stress.get("active"), 2, signed=True),
                )
                stress_cards[2].metric(
                    "All non-USD currencies -1%",
                    pct(minus_stress.get("portfolio"), 2, signed=True),
                )

    with tabs[4]:
        st.subheader("Trading and operating controls")
        operating_alerts = collect_operating_alerts(monitoring, currency, operations=ops)
        if operating_alerts:
            st.error(
                "Action required: "
                + "; ".join(
                    f"{alert['label']} ({alert.get('value', 'n/a')})"
                    for alert in operating_alerts
                )
            )
        else:
            st.success("No configured model, concentration, data-tail, or FX coverage breach.")

        drift = ops.get("current_drift")
        st.divider()
        st.subheader("Drift since last rebalance")
        if drift:
            band = drift.get("no_trade_band")
            d = st.columns(4)
            d[0].metric("Drift L1", pct(drift.get("drift_l1"), 2))
            d[1].metric("Max single drift", pct(drift.get("max_single_drift"), 2))
            d[2].metric(f"Names outside band ({pct(band, 1)})", drift.get("names_outside_band", "n/a"))
            d[3].metric("Days since rebalance", drift.get("days_since_rebalance", "n/a"))
            drift_rows = rows_df(drift.get("by_ticker"))
            if not drift_rows.empty:
                drift_cols = [c for c in ("ticker", "target", "current", "drift") if c in drift_rows.columns]
                st.dataframe(
                    fmt_table(drift_rows.head(12)[drift_cols],
                              [c for c in ("target", "current", "drift") if c in drift_cols]),
                    width="stretch", hide_index=True,
                )
        else:
            st.info("Re-export the operating bundle to see drift monitoring.")

        tc = monitoring.get("transaction_costs")
        st.divider()
        st.subheader("Transaction costs")
        if tc:
            n_reb = tc.get("n_rebalances") or 0
            cum_cost = tc.get("cumulative_transaction_cost")
            avg_cost = (cum_cost / n_reb) if (cum_cost is not None and n_reb) else None
            cc = st.columns(3)
            cc[0].metric("Cumulative cost", pct(cum_cost, 3))
            cc[1].metric("Annualized drag", f"{num((tc.get('annualized_cost_drag') or 0.0) * 1e4, 1)} bps")
            cc[2].metric("Cost per rebalance", pct(avg_cost, 3))
            cost_series = rows_df(tc.get("series"))
            if not cost_series.empty:
                cost_series["date"] = pd.to_datetime(cost_series["date"])
                cost_series = cost_series.set_index("date").sort_index()
                render_chart(line_fig(cost_series, ["cumulative_cost"], "Cumulative transaction cost", "Fraction of AUM"))
        else:
            st.info("Re-export the operating bundle to see transaction costs.")

        schedule = st.columns(4)
        schedule[0].metric("Market data as of", production_meta.get("data_as_of", perf.get("as_of", "n/a")))
        schedule[1].metric("FX data as of", (currency or {}).get("fx_data_as_of", "n/a"))
        schedule[2].metric("Next rebalance", ops.get("next_expected_rebalance_date", "n/a"))
        schedule[3].metric("Weekday rows to go", ops.get("rows_until_next_rebalance", "n/a"))

        funnel = st.columns(4)
        funnel[0].metric("Source universe", universe_size or "n/a")
        funnel[1].metric("Currency mapped", f"{fx_mapped}/{universe_size}" if universe_size else "n/a")
        funnel[2].metric("Current holdings", holdings.get("n_holdings", "n/a"))
        funnel[3].metric("Orders", ops.get("n_trades", "n/a"))

        if challenger:
            prod_turn = rows_df(monitoring.get("turnover"))
            causal_turn = rows_df((challenger.get("monitoring") or {}).get("turnover"))
            if not prod_turn.empty and not causal_turn.empty:
                prod_turn["date"] = pd.to_datetime(prod_turn["date"])
                causal_turn["date"] = pd.to_datetime(causal_turn["date"])
                turn_compare = (
                    prod_turn.set_index("date")[["turnover_two_way"]]
                    .rename(columns={"turnover_two_way": prod_name})
                    .join(
                        causal_turn.set_index("date")[["turnover_two_way"]]
                        .rename(columns={"turnover_two_way": chal_name}),
                        how="outer",
                    )
                    .sort_index()
                )
                render_chart(line_fig(turn_compare, list(turn_compare.columns), "Rebalance turnover comparison", "Two-way turnover"))
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

        aum_usd_m = st.number_input(
            "Portfolio AUM (USD millions)",
            min_value=0.0,
            value=100.0,
            step=10.0,
            help="Scales target-weight changes into indicative USD order sizes; it does not alter the backtest.",
            key="operations_aum_usd_m",
        )
        trades = rows_df(ops.get("trade_list"))
        gross_order_usd_m = None
        if not trades.empty:
            holdings_rows = rows_df(holdings.get("all"))
            if {"ticker", "sector"}.issubset(holdings_rows.columns):
                trades = trades.merge(holdings_rows[["ticker", "sector"]], on="ticker", how="left")
            trades = sector_filter(trades)
            pre_trade_column = next(
                (column for column in ("pre_trade", "drifted_pre_trade", "prev") if column in trades.columns),
                None,
            )
            if pre_trade_column:
                trades["pre_trade"] = pd.to_numeric(trades[pre_trade_column], errors="coerce")
            elif "target" in trades.columns and "delta" in trades.columns:
                trades["pre_trade"] = trades["target"] - trades["delta"]
            trades["abs_delta"] = trades["delta"].abs()
            trades["order_usd_m"] = trades["delta"] * float(aum_usd_m)
            gross_order_usd_m = float(trades["order_usd_m"].abs().sum())
            trades = trades.sort_values("abs_delta", ascending=False)
            t1, t2 = st.columns([1.0, 1.2])
            with t1:
                render_chart(hbar_fig(trades.head(18).sort_values("delta"), "ticker", "delta", "Largest rebalance orders"))
            with t2:
                order_columns = [
                    column for column in ("ticker", "sector", "currency", "pre_trade", "target", "delta", "order_usd_m")
                    if column in trades.columns
                ]
                st.dataframe(
                    fmt_table(
                        trades[order_columns].head(40),
                        [column for column in ("pre_trade", "target", "delta") if column in order_columns],
                    ),
                    width="stretch",
                    hide_index=True,
                )

        k = st.columns(6)
        k[0].metric("Trade Count", ops.get("n_trades", "n/a"))
        k[1].metric("Latest Turnover", pct(ops.get("turnover_two_way_latest"), 1))
        k[2].metric("Gross Orders", f"${gross_order_usd_m:,.2f}m" if gross_order_usd_m is not None else "n/a")
        k[3].metric("Rebalance Freq", f"{ops.get('rebalance_freq_days', 'n/a')}d")
        k[4].metric("Fallback Rate", pct(ops.get("optimizer_failure_rate"), 1))
        k[5].metric("Expected Cost", pct(ops.get("expected_transaction_cost"), 3))

    with tabs[5]:
        st.subheader("Signals and adoption gates")
        if challenger:
            gate = data["registry"].get("comparison_gate") or {}
            cmq = challenger_perf.get("model_quality") or {}
            gc = st.columns(5)
            gc[0].metric("Challenger", gate.get("status", challenger_meta.get("status", "n/a")))
            gc[1].metric("Model", challenger_meta.get("model_type", "n/a"))
            gc[2].metric("Causal Split", "PASS" if challenger_meta.get("causal_validation_ok") else "FAIL")
            gc[3].metric("Signal Lag", f"{challenger_meta.get('execution_signal_lag_days', 'n/a')}d")
            gc[4].metric("Degenerate Folds", f"{cmq.get('degenerate_retrains', 'n/a')}/{cmq.get('total_retrains', 'n/a')}")
            checks = gate.get("checks") or {}
            if checks:
                gate_rows = pd.DataFrame(
                    [{"criterion": k.replace("_", " "), "passed": bool(v)} for k, v in checks.items()]
                )
                st.dataframe(gate_rows, width="stretch", hide_index=True)
            st.divider()
            st.subheader("Feature score decomposition by fund")
            st.markdown(
                "<div class='note'>Gain importance is normalized within each walk-forward model set. "
                "Compare shares within a fund; raw gain scales are not cross-model return forecasts.</div>",
                unsafe_allow_html=True,
            )
            feature_cols = st.columns(2)
            for feature_col, feature_label, feature_bundle in (
                (feature_cols[0], prod_name, data["production"]),
                (feature_cols[1], chal_name, challenger),
            ):
                with feature_col:
                    fund_features = feature_bundle.get("features") or {}
                    fund_groups = rows_df(fund_features.get("group_importance"))
                    fund_top = rows_df(fund_features.get("top_features"))
                    st.markdown(f"### {feature_label}")
                    st.caption(
                        f"{fund_features.get('n_models', 'n/a')} walk-forward models | "
                        f"{len(fund_top)} scored features shown"
                    )
                    if not fund_groups.empty:
                        fund_groups = fund_groups.rename(columns={"name": "group", "value": "share_pct"})
                        render_chart(hbar_fig(
                            fund_groups.sort_values("share_pct"), "group", "share_pct",
                            f"{feature_label} — feature group share (%)",
                        ))
                    if not fund_top.empty:
                        display_top = fund_top.head(20).copy()
                        render_chart(hbar_fig(
                            display_top.sort_values("share_pct"), "feature", "share_pct",
                            f"{feature_label} — top feature score share (%)", color="group",
                        ))
                        st.dataframe(
                            fmt_table(
                                display_top[["feature", "group", "importance", "share_pct"]],
                                [], digits=2,
                            ),
                            width="stretch", hide_index=True, height=420,
                        )
                    else:
                        st.info(f"No {feature_label} feature scores are available.")
            st.divider()
            st.subheader(f"{prod_name} adoption diagnostics")
            st.caption("The attribution, overlay and factor-ablation diagnostics below remain S0-specific research artifacts.")
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

    with tabs[7]:
        st.subheader("Stock drivers")
        attr = data["feature_attribution"]
        challenger_attr = (challenger.get("feature_attribution") or {}) if challenger else {}
        shared_tickers = sorted(
            set((attr.get("tickers") or {}).keys())
            & set((challenger_attr.get("tickers") or {}).keys())
        )
        if shared_tickers:
            st.markdown(
                "<div class='note'>Select one stock to compare its portfolio weight, active weight, "
                "model score and local SHAP drivers in both funds. SHAP scales are model-specific.</div>",
                unsafe_allow_html=True,
            )
            challenger_default = prepare_stock_drivers(challenger_attr, None)["default"]
            shared_default_idx = (
                shared_tickers.index(challenger_default)
                if challenger_default in shared_tickers else 0
            )
            shared_selected = st.selectbox(
                "Stock", shared_tickers, index=shared_default_idx,
                key="shared_stock_drivers_select",
            )
            pair_cols = st.columns(2)
            for pair_col, pair_label, pair_attr in (
                (pair_cols[0], prod_name, attr),
                (pair_cols[1], chal_name, challenger_attr),
            ):
                pair_view = prepare_stock_drivers(pair_attr, shared_selected)
                with pair_col:
                    st.markdown(f"### {pair_label}")
                    st.caption(
                        f"Position as of {pair_attr.get('as_of', 'n/a')} | "
                        f"model {pair_attr.get('model_date', 'n/a')}"
                    )
                    pm = pair_view["metrics"]
                    pmetrics = st.columns(4)
                    pmetrics[0].metric("Weight", pct(pm["weight"], 2))
                    pmetrics[1].metric("BM Weight", pct(pm["bm_weight"], 2))
                    pmetrics[2].metric("Active", pct(pm["active"], 2, signed=True))
                    pmetrics[3].metric("Model score", num(pm["mu"], 4))
                    pdrv = pd.DataFrame({
                        "feature": [f"{f} ({g})" for f, g, _v in pair_view["top_features"]],
                        "shap": [v for _f, _g, v in pair_view["top_features"]],
                    })
                    pdrv["direction"] = np.where(pdrv["shap"] >= 0, "pos", "neg")
                    pfig = px.bar(
                        pdrv, x="shap", y="feature", orientation="h",
                        color="direction", color_discrete_map={"pos": COLOR_POS, "neg": COLOR_NEG},
                        title=f"{pair_label} — {shared_selected} drivers", template="plotly_white",
                    )
                    pfig.update_layout(
                        yaxis={"categoryorder": "total ascending"},
                        xaxis_title="SHAP contribution", yaxis_title="", showlegend=False,
                    )
                    render_chart(apply_theme(
                        pfig, height=max(360, min(620, 30 * max(len(pdrv), 7)))
                    ))
                    st.dataframe(
                        fmt_table(pdrv[["feature", "shap"]], [], digits=5),
                        width="stretch", hide_index=True, height=380,
                    )
            st.divider()
            st.subheader(f"{prod_name} legacy detail")
        base = prepare_stock_drivers(attr, None)
        if base is None:
            st.info("Run export_operating_data.py to generate feature attribution.")
        else:
            st.markdown(
                "<div class='note'>Per-stock SHAP attribution of the model return "
                "forecast (mu) at the latest rebalance.</div>",
                unsafe_allow_html=True,
            )
            tickers_map = attr.get("tickers") or {}
            ow_rows = []
            for t, a in base["options"]:
                if a <= 0:
                    continue
                shap_t = (tickers_map.get(t) or {}).get("shap") or {}
                top_drv = max(shap_t, key=lambda k: abs(shap_t[k])) if shap_t else "n/a"
                ow_rows.append({"ticker": t, "active": a, "top_driver": top_drv})
            st.subheader("Top overweights and their lead driver")
            if ow_rows:
                st.dataframe(fmt_table(pd.DataFrame(ow_rows[:10]), ["active"]),
                             width="stretch", hide_index=True)
            else:
                st.info("No overweight names at the latest rebalance.")

            options = base["options"]
            labels = [f"{t} · active {a:+.1%}" for t, a in options]
            default_idx = next((i for i, (t, _a) in enumerate(options) if t == base["default"]), 0)
            choice = st.selectbox("Stock", labels, index=default_idx, key="stock_drivers_select")
            selected = options[labels.index(choice)][0]
            view = prepare_stock_drivers(attr, selected)

            m = view["metrics"]
            mc = st.columns(4)
            mc[0].metric("Weight", pct(m["weight"], 2))
            mc[1].metric("BM Weight", pct(m["bm_weight"], 2))
            mc[2].metric("Active", pct(m["active"], 2, signed=True))
            mc[3].metric("Model mu", num(m["mu"], 4))

            top = view["top_features"]
            drv = pd.DataFrame({
                "feature": [f"{f} ({g})" for f, g, _v in top],
                "shap": [v for _f, _g, v in top],
            })
            drv["direction"] = np.where(drv["shap"] >= 0, "pos", "neg")
            fig = px.bar(
                drv, x="shap", y="feature", orientation="h",
                color="direction", color_discrete_map={"pos": COLOR_POS, "neg": COLOR_NEG},
                title="Top feature drivers of model forecast", template="plotly_white",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"},
                              xaxis_title="SHAP", yaxis_title="", showlegend=False)
            render_chart(apply_theme(fig, height=max(320, min(620, 30 * max(len(drv), 7)))))

            st.caption(
                "SHAP는 모델 수익률 예측(mu)에 대한 귀속이다. 실제 OW/UW는 mu에 더해 "
                "리스크 모델·제약(TE·캡·섹터)을 거친 최적화 결과다."
            )
    with tabs[8]:
        st.subheader("Portfolio comparison")
        if not challenger:
            st.info(f"{chal_name} bundle is not available yet. The dashboard is in legacy compatibility mode.")
        else:
            metric_defs = [
                ("Annual Return", "annual_return"),
                ("Active Return", "active_return"),
                ("Information Ratio", "information_ratio"),
                ("Tracking Error", "tracking_error"),
                ("Realized Beta", "realized_beta"),
                ("Annual Turnover", "avg_annual_turnover"),
                ("Max Drawdown", "max_drawdown"),
                ("Average IC", "avg_ic"),
            ]
            comparison_rows = []
            for label, key in metric_defs:
                base_value = perf.get(key)
                challenger_value = challenger_perf.get(key)
                comparison_rows.append({
                    "metric": label,
                    prod_name: base_value,
                    chal_name: challenger_value,
                    "delta": (
                        float(challenger_value) - float(base_value)
                        if base_value is not None and challenger_value is not None else None
                    ),
                })
            st.dataframe(pd.DataFrame(comparison_rows), width="stretch", hide_index=True)
            if not comparison_returns.empty:
                render_chart(line_fig(
                    comparison_returns, list(comparison_returns.columns),
                    "Cumulative wealth comparison", "Growth of $1",
                ))

            prod_roll = rows_df(monitoring.get("rolling"))
            causal_roll = rows_df((challenger.get("monitoring") or {}).get("rolling"))
            if not prod_roll.empty and not causal_roll.empty:
                prod_roll["date"] = pd.to_datetime(prod_roll["date"])
                causal_roll["date"] = pd.to_datetime(causal_roll["date"])
                rolling_ir = (
                    prod_roll.set_index("date")[["information_ratio_252d"]]
                    .rename(columns={"information_ratio_252d": prod_name})
                    .join(
                        causal_roll.set_index("date")[["information_ratio_252d"]]
                        .rename(columns={"information_ratio_252d": chal_name}),
                        how="inner",
                    )
                )
                render_chart(line_fig(
                    rolling_ir, list(rolling_ir.columns),
                    "Rolling 252-day information ratio", "IR",
                ))

            prod_hold = rows_df(holdings.get("all"))
            causal_hold = rows_df(challenger_holdings.get("all"))
            if not prod_hold.empty and not causal_hold.empty:
                weight_diff = prod_hold[["ticker", "sector", "weight", "active"]].merge(
                    causal_hold[["ticker", "weight", "active"]],
                    on="ticker", suffixes=("_s0", "_causal"),
                )
                weight_diff["weight_delta"] = weight_diff["weight_causal"] - weight_diff["weight_s0"]
                weight_diff["active_delta"] = weight_diff["active_causal"] - weight_diff["active_s0"]
                weight_diff = weight_diff.reindex(
                    weight_diff["weight_delta"].abs().sort_values(ascending=False).index
                )
                st.subheader("Largest holding differences")
                st.dataframe(
                    fmt_table(
                        weight_diff.head(20),
                        ["weight_s0", "weight_causal", "active_s0", "active_causal", "weight_delta", "active_delta"],
                    ),
                    width="stretch", hide_index=True,
                )


if __name__ == "__main__":
    main()
