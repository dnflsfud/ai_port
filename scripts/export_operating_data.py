#!/usr/bin/env python
"""Export operating data for the dashboard — production backtest -> JSON/CSV.

Runs the production variant ONCE and extracts everything an operator needs that
the scalar metrics.json does not carry:

  returns.csv      daily portfolio vs benchmark returns + cumulative
  performance.json annual/active return, vol, sharpe, IR, TE, beta, max DD,
                   per-year returns, sub-period IR
  holdings.json    latest-rebalance weights vs benchmark -> per-name active,
                   sector, top OW / top UW, concentration
  contribution.json arithmetic stock/sector contribution to portfolio and
                   active return, plus transaction-cost residual
  risk.json        latest absolute/active risk contribution by stock and sector
  monitoring.json  turnover, active-share, rolling IR/TE/beta, monthly active,
                   drawdown events and guardrail diagnostics
  features.json    gain-based feature importance averaged across the walk-forward
                   models (which features actually drove the book) + group rollup
  operations.json  latest realized rebalance weights/trades, expected next
                   rebalance metadata, sector exposure and fallback rate
  currency.json    exact USD/local/FX arithmetic attribution, currency
                   exposure, source freshness, and FX stress diagnostics

Run FROM the project root (ai_port), engine vendored under ./src:
    PYTHONPATH=. <PY> scripts/export_operating_data.py
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_VARIANT = ROOT / "variants" / "iter15_65tkr_reb21_vtg.yaml"
DEFAULT_OPERATING_DIR = ROOT / "outputs" / "operating"

# Code contract version for the operating bundle; bumped when the exported
# meta/schema meaning changes materially.
PORTFOLIO_VERSION = "universe100-usd-v1"

# (display_name, portfolio_role) defaults used when the variant yaml is silent
# (e.g. the argument-free export path). Unknown labels stay challenger so an
# unrecognized run can never seize the single production slot.
_LABEL_DEFAULTS = {
    "iter15_65tkr_reb21_vtg": ("Legacy S0 (100)", "challenger"),
    "codex_causal_rank_65": ("Causal Rank 100", "production"),
}


def _rooted(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_provenance_meta(run_dir: Path) -> dict:
    """Provenance fields for the operating bundle meta.

    Records the code version and the git identity of the *backtest run* — the
    git_hash/git_dirty are copied from run_dir/experiment_manifest.json (NOT
    recomputed here) so the bundle points at the commit that produced it. Also
    checksums that manifest. Missing/unparseable manifest -> None fields.
    """
    manifest_path = run_dir / "experiment_manifest.json"
    git_hash = None
    git_dirty = None
    if manifest_path.exists():
        try:
            run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            git_hash = run_manifest.get("git_hash")
            git_dirty = run_manifest.get("git_dirty")
        except Exception:
            pass
    return {
        "portfolio_version": PORTFOLIO_VERSION,
        "source_manifest_sha256": _sha256(manifest_path),
        "git_hash": git_hash,
        "git_dirty": git_dirty,
    }


def _safe_float(x, ndigits: Optional[int] = 6):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return round(v, ndigits) if ndigits is not None else v


def build_rebalance_metadata(rebalance_dates, portfolio_dates, rebalance_freq: int) -> dict:
    """Return auditable latest/next rebalance metadata on a weekday-row calendar."""
    dates = pd.DatetimeIndex(portfolio_dates).sort_values().unique()
    rebalances = pd.DatetimeIndex(rebalance_dates).sort_values().unique()
    freq = int(rebalance_freq)
    if len(dates) == 0 or len(rebalances) == 0:
        raise ValueError("rebalance metadata requires non-empty date indexes")
    if freq <= 0:
        raise ValueError("rebalance_freq must be positive")
    if bool((dates.dayofweek >= 5).any()):
        raise ValueError("portfolio calendar contains weekend dates")

    last = pd.Timestamp(rebalances[-1])
    previous = pd.Timestamp(rebalances[-2] if len(rebalances) >= 2 else rebalances[-1])
    last_pos = int(dates.get_indexer([last])[0])
    if last_pos < 0:
        raise ValueError("latest rebalance is absent from portfolio calendar")
    rows_since = (len(dates) - 1) - last_pos
    if rows_since < 0 or rows_since >= freq:
        raise ValueError(
            f"latest rebalance is inconsistent with calendar: rows_since={rows_since}, freq={freq}"
        )
    rows_until = freq - rows_since
    next_pos = last_pos + freq
    if next_pos < len(dates):
        next_expected = pd.Timestamp(dates[next_pos])
    else:
        next_expected = pd.Timestamp(dates[-1]) + pd.offsets.BDay(rows_until)

    return {
        "last_rebalance_date": last.strftime("%Y-%m-%d"),
        "previous_rebalance_date": previous.strftime("%Y-%m-%d"),
        "next_expected_rebalance_date": next_expected.strftime("%Y-%m-%d"),
        "rebalance_freq_days": freq,
        "rebalance_calendar": "weekday_index",
        "rows_since_last_rebalance": int(rows_since),
        "rows_until_next_rebalance": int(rows_until),
        "is_rebalance_data_as_of": bool(pd.Timestamp(dates[-1]) == last),
        "next_rebalance_is_estimate": bool(next_pos >= len(dates)),
    }


def _drift_weights(weights: np.ndarray, daily_ret: np.ndarray) -> np.ndarray:
    vals = np.asarray(weights, dtype=float) * (1.0 + np.asarray(daily_ret, dtype=float))
    total = float(np.nansum(vals))
    if not np.isfinite(total) or total <= 0:
        return np.asarray(weights, dtype=float)
    return vals / total


def _normalise_currency(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().upper()
    return text or None


def _attribution_frame(frame, dates, tickers, name: str) -> pd.DataFrame:
    """Return a finite date x ticker frame or fail before publishing bad FX math."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame")
    out = (
        frame.reindex(index=pd.DatetimeIndex(dates), columns=list(tickers))
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    if out.isna().any().any():
        bad = out.isna().stack()
        sample = [f"{d.date()}:{t}" for (d, t), flag in bad.items() if flag][:5]
        raise ValueError(f"{name} is incomplete for FX attribution: {sample}")
    return out.astype(float)


def _quality_currency_list(data_quality: dict, key: str) -> set[str]:
    fx_quality = data_quality.get("fx") if isinstance(data_quality.get("fx"), dict) else {}
    values = fx_quality.get(key)
    if values is None:
        values = data_quality.get(f"fx_{key}")
    if isinstance(values, dict):
        values = [k for k, value in values.items() if value]
    if not isinstance(values, (list, tuple, set, pd.Index, np.ndarray)):
        values = []
    return {c for c in (_normalise_currency(v) for v in values) if c}


def build_currency_attribution(
    *,
    tickers,
    dates,
    usd_returns: pd.DataFrame,
    local_returns: pd.DataFrame,
    fx_returns: pd.DataFrame,
    fx_rates_usd_per_local: pd.DataFrame,
    currency_map: dict,
    portfolio_entering_weights: pd.DataFrame,
    benchmark_entering_weights: pd.DataFrame,
    latest_portfolio_weights: pd.Series,
    latest_benchmark_weights: pd.Series,
    portfolio_reported_returns: Optional[pd.Series] = None,
    benchmark_reported_returns: Optional[pd.Series] = None,
    data_quality: Optional[dict] = None,
) -> dict:
    """Build exact arithmetic local/FX attribution for a USD-base portfolio.

    ``fx_effect`` is deliberately defined as ``r_usd - r_local``.  It therefore
    includes the local-return/FX interaction and guarantees, name by name and
    day by day, that ``local + fx_effect == USD gross return``.
    """
    tickers = [str(t) for t in tickers]
    dates = pd.DatetimeIndex(dates).sort_values().unique()
    if not tickers or len(dates) == 0:
        raise ValueError("currency attribution requires tickers and dates")

    usd = _attribution_frame(usd_returns, dates, tickers, "USD returns")
    local = _attribution_frame(local_returns, dates, tickers, "local returns")
    port_w = _attribution_frame(
        portfolio_entering_weights, dates, tickers, "portfolio entering weights"
    )
    bm_w = _attribution_frame(
        benchmark_entering_weights, dates, tickers, "benchmark entering weights"
    )
    latest_port = pd.to_numeric(
        pd.Series(latest_portfolio_weights).reindex(tickers), errors="coerce"
    )
    latest_bm = pd.to_numeric(
        pd.Series(latest_benchmark_weights).reindex(tickers), errors="coerce"
    )
    if latest_port.isna().any() or latest_bm.isna().any():
        raise ValueError("latest portfolio/benchmark weights are incomplete")

    currency_by_ticker = {
        ticker: _normalise_currency((currency_map or {}).get(ticker))
        for ticker in tickers
    }
    missing_tickers = [t for t in tickers if currency_by_ticker[t] is None]
    display_currency = {t: (currency_by_ticker[t] or "UNKNOWN") for t in tickers}

    fx_ret = fx_returns if isinstance(fx_returns, pd.DataFrame) else pd.DataFrame()
    fx_rate = (
        fx_rates_usd_per_local
        if isinstance(fx_rates_usd_per_local, pd.DataFrame)
        else pd.DataFrame()
    )
    data_quality = data_quality if isinstance(data_quality, dict) else {}
    missing_currencies = _quality_currency_list(data_quality, "missing_currencies")
    stale_currencies = _quality_currency_list(data_quality, "stale_currencies")
    non_usd = [t for t in tickers if display_currency[t] not in {"USD", "UNKNOWN"}]
    missing_fx_tickers = []
    for ticker in non_usd:
        currency = display_currency[ticker]
        ret_ok = ticker in fx_ret.columns and fx_ret.reindex(dates)[ticker].notna().any()
        rate_ok = ticker in fx_rate.columns and fx_rate.loc[: dates[-1], ticker].notna().any()
        if currency in missing_currencies or not (ret_ok and rate_ok):
            missing_fx_tickers.append(ticker)

    fx_quality = data_quality.get("fx") if isinstance(data_quality.get("fx"), dict) else {}
    latest_source_by_currency = fx_quality.get("latest_source_date_by_currency") or {}
    max_staleness_by_currency = fx_quality.get("max_staleness_days_by_currency") or {}
    freshness_quality = (
        data_quality.get("data_freshness")
        if isinstance(data_quality.get("data_freshness"), dict)
        else {}
    )
    try:
        allowed_staleness_days = int(
            freshness_quality.get("max_fx_staleness_days", 7)
        )
    except (TypeError, ValueError):
        allowed_staleness_days = 7
    allowed_staleness_days = max(allowed_staleness_days, 0)
    freshness_by_currency = []
    currency_last_source: dict[str, Optional[pd.Timestamp]] = {}
    currency_stale: dict[str, bool] = {}
    for currency in sorted(set(display_currency.values())):
        names = [t for t in tickers if display_currency[t] == currency]
        if currency == "USD":
            last_source = pd.Timestamp(dates[-1])
        else:
            raw_date = latest_source_by_currency.get(currency)
            try:
                last_source = pd.Timestamp(raw_date) if raw_date is not None else None
            except (TypeError, ValueError):
                last_source = None
            if last_source is None or pd.isna(last_source):
                candidates = []
                for ticker in names:
                    if ticker in fx_rate.columns:
                        valid = fx_rate.loc[: dates[-1], ticker].dropna()
                        if not valid.empty:
                            candidates.append(pd.Timestamp(valid.index.max()))
                last_source = min(candidates) if candidates else None
        lag_days = (
            (pd.Timestamp(dates[-1]) - last_source).days
            if last_source is not None
            else None
        )
        try:
            historical_max_lag = int(max_staleness_by_currency.get(currency))
        except (TypeError, ValueError):
            historical_max_lag = None
        is_stale = currency in stale_currencies or (
            currency not in {"USD", "UNKNOWN"}
            and (lag_days is None or lag_days > allowed_staleness_days)
        )
        currency_last_source[currency] = last_source
        currency_stale[currency] = bool(is_stale)
        freshness_by_currency.append({
            "currency": currency,
            "ticker_count": len(names),
            "last_source_date": str(last_source)[:10] if last_source is not None else None,
            "staleness_days": lag_days,
            "historical_max_staleness_days": historical_max_lag,
            "allowed_staleness_days": allowed_staleness_days,
            "stale": bool(is_stale),
        })

    stale_tickers = [
        ticker for ticker in non_usd if currency_stale.get(display_currency[ticker], False)
    ]
    fx_data_as_of_raw = data_quality.get("fx_data_as_of")
    if fx_data_as_of_raw is not None:
        try:
            fx_data_as_of = pd.Timestamp(fx_data_as_of_raw)
        except (TypeError, ValueError):
            fx_data_as_of = None
    else:
        required_dates = [
            currency_last_source[display_currency[t]]
            for t in non_usd
            if currency_last_source.get(display_currency[t]) is not None
        ]
        fx_data_as_of = min(required_dates) if required_dates else pd.Timestamp(dates[-1])

    # Exact effect, interaction included. Do not substitute raw r_fx here.
    fx_effect = usd - local
    port_local_daily = (port_w * local).sum(axis=1)
    port_fx_daily = (port_w * fx_effect).sum(axis=1)
    port_usd_daily = (port_w * usd).sum(axis=1)
    bm_local_daily = (bm_w * local).sum(axis=1)
    bm_fx_daily = (bm_w * fx_effect).sum(axis=1)
    bm_usd_daily = (bm_w * usd).sum(axis=1)
    active_local_daily = port_local_daily - bm_local_daily
    active_fx_daily = port_fx_daily - bm_fx_daily
    active_usd_daily = port_usd_daily - bm_usd_daily

    daily_frame = pd.DataFrame({
        "portfolio_local_return": port_local_daily,
        "portfolio_fx_effect": port_fx_daily,
        "portfolio_usd_gross_return": port_usd_daily,
        "benchmark_local_return": bm_local_daily,
        "benchmark_fx_effect": bm_fx_daily,
        "benchmark_usd_gross_return": bm_usd_daily,
        "active_local_return": active_local_daily,
        "active_fx_effect": active_fx_daily,
        "active_usd_gross_return": active_usd_daily,
    }, index=dates)
    daily_frame.index.name = "date"

    ticker_rows = []
    for ticker in tickers:
        p_local = float((port_w[ticker] * local[ticker]).sum())
        p_fx = float((port_w[ticker] * fx_effect[ticker]).sum())
        p_usd = float((port_w[ticker] * usd[ticker]).sum())
        b_local = float((bm_w[ticker] * local[ticker]).sum())
        b_fx = float((bm_w[ticker] * fx_effect[ticker]).sum())
        b_usd = float((bm_w[ticker] * usd[ticker]).sum())
        ticker_rows.append({
            "ticker": ticker,
            "currency": display_currency[ticker],
            "target_weight": _safe_float(latest_port[ticker], 10),
            "benchmark_weight": _safe_float(latest_bm[ticker], 10),
            "active_weight": _safe_float(latest_port[ticker] - latest_bm[ticker], 10),
            "portfolio_local_contribution": _safe_float(p_local, 10),
            "portfolio_fx_contribution": _safe_float(p_fx, 10),
            "portfolio_usd_contribution": _safe_float(p_usd, 10),
            "benchmark_local_contribution": _safe_float(b_local, 10),
            "benchmark_fx_contribution": _safe_float(b_fx, 10),
            "benchmark_usd_contribution": _safe_float(b_usd, 10),
            "active_local_contribution": _safe_float(p_local - b_local, 10),
            "active_fx_contribution": _safe_float(p_fx - b_fx, 10),
            "active_usd_contribution": _safe_float(p_usd - b_usd, 10),
            "fx_source_date": (
                str(currency_last_source.get(display_currency[ticker]))[:10]
                if currency_last_source.get(display_currency[ticker]) is not None else None
            ),
            "fx_stale": bool(currency_stale.get(display_currency[ticker], False)),
        })

    numeric_currency_fields = (
        "target_weight", "benchmark_weight", "active_weight",
        "portfolio_local_contribution", "portfolio_fx_contribution",
        "portfolio_usd_contribution", "benchmark_local_contribution",
        "benchmark_fx_contribution", "benchmark_usd_contribution",
        "active_local_contribution", "active_fx_contribution",
        "active_usd_contribution",
    )
    currency_rows = []
    for currency in sorted(set(display_currency.values())):
        names = [row for row in ticker_rows if row["currency"] == currency]
        row = {"currency": currency, "ticker_count": len(names)}
        for field in numeric_currency_fields:
            row[field] = _safe_float(sum(float(item[field] or 0.0) for item in names), 10)
        rate_series = pd.Series(dtype=float)
        currency_tickers = [t for t in tickers if display_currency[t] == currency]
        if currency == "USD":
            move_1d = move_21d = 0.0
        else:
            for ticker in currency_tickers:
                if ticker in fx_rate.columns:
                    candidate = pd.to_numeric(
                        fx_rate.loc[: dates[-1], ticker], errors="coerce"
                    ).replace([np.inf, -np.inf], np.nan).dropna()
                    if not candidate.empty:
                        rate_series = candidate
                        break
            move_1d = (
                float(rate_series.iloc[-1] / rate_series.iloc[-2] - 1.0)
                if len(rate_series) >= 2 and float(rate_series.iloc[-2]) != 0.0 else None
            )
            move_21d = (
                float(rate_series.iloc[-1] / rate_series.iloc[-22] - 1.0)
                if len(rate_series) >= 22 and float(rate_series.iloc[-22]) != 0.0 else None
            )
        row.update({
            "fx_move_1d": _safe_float(move_1d, 10),
            "fx_move_21d": _safe_float(move_21d, 10),
            "fx_source_date": (
                str(currency_last_source.get(currency))[:10]
                if currency_last_source.get(currency) is not None else None
            ),
            "fx_stale": bool(currency_stale.get(currency, False)),
        })
        currency_rows.append(row)
    currency_rows.sort(key=lambda row: float(row["target_weight"] or 0.0), reverse=True)

    p_err = port_local_daily + port_fx_daily - port_usd_daily
    b_err = bm_local_daily + bm_fx_daily - bm_usd_daily
    a_err = active_local_daily + active_fx_daily - active_usd_daily
    tolerance = 1e-10
    reconciliation = {
        "tolerance": tolerance,
        "max_daily_portfolio_error": _safe_float(p_err.abs().max(), 12),
        "max_daily_benchmark_error": _safe_float(b_err.abs().max(), 12),
        "max_daily_active_error": _safe_float(a_err.abs().max(), 12),
        "period_portfolio_error": _safe_float(p_err.sum(), 12),
        "period_benchmark_error": _safe_float(b_err.sum(), 12),
        "period_active_error": _safe_float(a_err.sum(), 12),
        "portfolio_gross_vs_reported_arithmetic_residual": (
            _safe_float(
                port_usd_daily.sum()
                - pd.Series(portfolio_reported_returns).reindex(dates).sum(),
                10,
            )
            if portfolio_reported_returns is not None else None
        ),
        "benchmark_gross_vs_reported_arithmetic_residual": (
            _safe_float(
                bm_usd_daily.sum()
                - pd.Series(benchmark_reported_returns).reindex(dates).sum(),
                10,
            )
            if benchmark_reported_returns is not None else None
        ),
    }
    reconciliation["passed"] = all(
        abs(float(reconciliation[key] or 0.0)) <= tolerance
        for key in (
            "max_daily_portfolio_error", "max_daily_benchmark_error",
            "max_daily_active_error", "period_portfolio_error",
            "period_benchmark_error", "period_active_error",
        )
    )

    non_usd_target = sum(float(latest_port[t]) for t in non_usd)
    non_usd_benchmark = sum(float(latest_bm[t]) for t in non_usd)
    summary = {
        "non_usd_target_weight": _safe_float(non_usd_target, 10),
        "non_usd_benchmark_weight": _safe_float(non_usd_benchmark, 10),
        "non_usd_active_weight": _safe_float(non_usd_target - non_usd_benchmark, 10),
        "portfolio_local_arithmetic_contribution": _safe_float(port_local_daily.sum(), 10),
        "portfolio_fx_arithmetic_contribution": _safe_float(port_fx_daily.sum(), 10),
        "portfolio_usd_gross_arithmetic_return": _safe_float(port_usd_daily.sum(), 10),
        "benchmark_local_arithmetic_contribution": _safe_float(bm_local_daily.sum(), 10),
        "benchmark_fx_arithmetic_contribution": _safe_float(bm_fx_daily.sum(), 10),
        "benchmark_usd_gross_arithmetic_return": _safe_float(bm_usd_daily.sum(), 10),
        "active_local_arithmetic_contribution": _safe_float(active_local_daily.sum(), 10),
        "active_fx_arithmetic_contribution": _safe_float(active_fx_daily.sum(), 10),
        "active_usd_gross_arithmetic_return": _safe_float(active_usd_daily.sum(), 10),
    }
    shock = 0.01
    stress_plus_port = non_usd_target * shock
    stress_plus_bm = non_usd_benchmark * shock

    return {
        "schema_version": 1,
        "base_currency": "USD",
        "as_of": str(dates[-1])[:10],
        "fx_data_as_of": str(fx_data_as_of)[:10] if fx_data_as_of is not None else None,
        "period": {"start": str(dates[0])[:10], "end": str(dates[-1])[:10]},
        "method": (
            "Arithmetic attribution with entering-day weights. Exact fx_effect = "
            "r_usd - r_local, including the local/FX interaction; transaction costs "
            "remain a separate portfolio residual."
        ),
        "coverage": {
            "total": len(tickers),
            "mapped": len(tickers) - len(missing_tickers),
            "missing": len(missing_tickers),
            "missing_fx": len(missing_fx_tickers),
            "stale": len(stale_tickers),
            "missing_tickers": missing_tickers,
            "missing_fx_tickers": missing_fx_tickers,
            "stale_tickers": stale_tickers,
        },
        "summary": summary,
        "daily": _serialize_records(daily_frame, date_col="date", ndigits=10),
        "by_currency": currency_rows,
        "by_ticker": ticker_rows,
        "freshness_by_currency": freshness_by_currency,
        "stress": {
            "shock": "USD-per-local-currency rate +/-1%, local prices unchanged",
            "plus_1pct": {
                "portfolio": _safe_float(stress_plus_port, 10),
                "benchmark": _safe_float(stress_plus_bm, 10),
                "active": _safe_float(stress_plus_port - stress_plus_bm, 10),
            },
            "minus_1pct": {
                "portfolio": _safe_float(-stress_plus_port, 10),
                "benchmark": _safe_float(-stress_plus_bm, 10),
                "active": _safe_float(-stress_plus_port + stress_plus_bm, 10),
            },
        },
        "reconciliation": reconciliation,
    }


def build_latest_trade_plan(
    *,
    target_weights: pd.Series,
    entering_weights: pd.Series,
    latest_returns: pd.Series,
    reported_turnover: float,
    one_way_transaction_cost: float,
    as_of,
    returns_data_as_of,
) -> dict:
    """Reconstruct latest close pre-trade weights and reconcile exact turnover."""
    target = pd.to_numeric(pd.Series(target_weights), errors="coerce").astype(float)
    tickers = list(target.index)
    entering = pd.to_numeric(pd.Series(entering_weights).reindex(tickers), errors="coerce")
    latest_ret = pd.to_numeric(pd.Series(latest_returns).reindex(tickers), errors="coerce")
    if target.isna().any() or entering.isna().any() or latest_ret.isna().any():
        raise ValueError("latest trade reconstruction contains missing weights or returns")
    pre_trade = pd.Series(
        _drift_weights(entering.to_numpy(dtype=float), latest_ret.to_numpy(dtype=float)),
        index=tickers,
    )
    delta = target - pre_trade
    calculated_turnover = float(delta.abs().sum())
    reported_turnover = float(reported_turnover)
    error = calculated_turnover - reported_turnover
    tolerance = max(1e-10, 1e-8 * max(1.0, abs(reported_turnover)))
    if not np.isfinite(reported_turnover) or abs(error) > tolerance:
        raise ValueError(
            "latest drifted trade turnover does not match backtest turnover: "
            f"calculated={calculated_turnover:.12f}, reported={reported_turnover:.12f}"
        )
    as_of = pd.Timestamp(as_of)
    returns_data_as_of = pd.Timestamp(returns_data_as_of)
    data_valid = bool(as_of <= returns_data_as_of and np.isfinite(latest_ret).all())
    trades = []
    for ticker in tickers:
        change = float(delta[ticker])
        if abs(change) <= 1e-12:
            continue
        before = float(pre_trade[ticker])
        trades.append({
            "ticker": ticker,
            "pre_trade": _safe_float(before, 10),
            "prev": _safe_float(before, 10),  # compatibility alias for the dashboard
            "target": _safe_float(target[ticker], 10),
            "delta": _safe_float(change, 10),
        })
    trades.sort(key=lambda row: abs(float(row["delta"])), reverse=True)
    cost_rate = float(one_way_transaction_cost)
    return {
        "n_trades": len(trades),
        "trade_list": trades,
        "turnover_two_way_latest": _safe_float(reported_turnover, 10),
        "turnover_two_way_reconstructed": _safe_float(calculated_turnover, 10),
        "turnover_reconciliation_error": _safe_float(error, 12),
        "turnover_reconciled": True,
        "one_way_transaction_cost_rate": _safe_float(cost_rate, 10),
        "one_way_transaction_cost_bps": _safe_float(cost_rate * 10000.0, 4),
        "expected_transaction_cost": _safe_float(reported_turnover * cost_rate, 10),
        "trade_data_as_of": str(as_of)[:10],
        "returns_data_as_of": str(returns_data_as_of)[:10],
        "trade_data_as_of_valid": data_valid,
        "pre_trade_weight_method": "entering weights drifted by latest USD close-to-close returns",
    }


def build_current_drift(
    target_weights: pd.Series,
    post_rebalance_returns: pd.DataFrame,
    no_trade_band: float,
    as_of,
    last_rebalance_date,
) -> dict:
    """Intra-rebalance drift of the latest target book under post-rebalance USD returns.

    Applies the same sequential ``w <- w*(1+r)/Σ(w*(1+r))`` renormalization as
    ``_drift_weights``; an empty returns frame leaves the (renormalized) target book
    in place, i.e. zero drift.  Numeric fields keep full precision so the reported
    ``max_single_drift`` reconciles exactly against a strict ``>`` no-trade band.
    """
    target = pd.to_numeric(pd.Series(target_weights), errors="coerce").astype(float)
    tickers = list(target.index)
    target_vals = target.to_numpy(dtype=float)
    total = float(np.nansum(target_vals))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("current drift requires a positive, finite target weight sum")
    current = target_vals / total
    returns = pd.DataFrame(post_rebalance_returns)
    n_days = int(len(returns))
    if n_days:
        aligned = (
            returns.reindex(columns=tickers)
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        for _, row in aligned.iterrows():
            current = _drift_weights(current, row.to_numpy(dtype=float))
    drift = current - target_vals
    band = float(no_trade_band)
    rows = [
        {
            "ticker": tickers[i],
            "target": _safe_float(target_vals[i], None),
            "current": _safe_float(current[i], None),
            "drift": _safe_float(drift[i], None),
        }
        for i in range(len(tickers))
    ]
    rows.sort(key=lambda r: abs(float(r["drift"])), reverse=True)
    return {
        "as_of": str(as_of)[:10],
        "last_rebalance_date": str(last_rebalance_date)[:10],
        "days_since_rebalance": n_days,
        "drift_l1": _safe_float(float(np.abs(drift).sum()), None),
        "max_single_drift": _safe_float(float(np.abs(drift).max()), None),
        "no_trade_band": _safe_float(band, None),
        "names_outside_band": int((np.abs(drift) > band).sum()),
        "weight_sum": _safe_float(float(current.sum()), None),
        "by_ticker": rows,
    }


def build_transaction_cost_history(
    turnover: pd.Series,
    one_way_tc: float,
    n_return_days: int,
    annualization_days: int = 252,
) -> dict:
    """Whole-period cumulative two-way turnover cost history.

    Per-rebalance cost = ``turnover * one_way_tc`` — the same definition as the
    trade plan's single-rebalance ``expected_transaction_cost``, but aggregated
    across every rebalance.  ``annualized_cost_drag`` scales the cumulative cost by
    ``annualization_days / max(n_return_days, 1)`` where ``n_return_days`` is the
    portfolio return observation count supplied by the caller.
    """
    rate = float(one_way_tc)
    series = pd.Series(turnover, dtype=float)
    costs = series * rate
    cumulative = costs.cumsum()
    cum_cost = float(costs.sum())
    cum_turnover = float(series.sum())
    drag = cum_cost * float(annualization_days) / max(int(n_return_days), 1)
    records = [
        {
            "date": str(idx)[:10],
            "turnover": _safe_float(float(series.iloc[i]), None),
            "cost": _safe_float(float(costs.iloc[i]), None),
            "cumulative_cost": _safe_float(float(cumulative.iloc[i]), None),
        }
        for i, idx in enumerate(series.index)
    ]
    return {
        "one_way_transaction_cost_rate": _safe_float(rate, None),
        "one_way_transaction_cost_bps": _safe_float(rate * 1e4, None),
        "n_rebalances": int(len(series)),
        "cumulative_two_way_turnover": _safe_float(cum_turnover, None),
        "cumulative_transaction_cost": _safe_float(cum_cost, None),
        "annualized_cost_drag": _safe_float(drag, None),
        "series": records,
    }


def build_operating_quality_fields(data, tickers) -> dict:
    """Expose the loader's universe funnel and source freshness without coupling."""
    quality = getattr(data, "data_quality", None)
    quality = quality if isinstance(quality, dict) else {}
    universe = quality.get("universe") if isinstance(quality.get("universe"), dict) else {}
    if not universe:
        full_universe = list(getattr(data, "full_universe", tickers) or tickers)
        missing = list(getattr(data, "missing_tickers", []) or [])
        universe = {
            "full_universe_count": len(full_universe),
            "loaded_ticker_count": len(tickers),
            "missing_tickers": missing,
        }
    fx_quality = quality.get("fx") if isinstance(quality.get("fx"), dict) else {}
    freshness = {
        "data_as_of": str(pd.DatetimeIndex(getattr(data, "dates", [])).max())[:10]
        if len(getattr(data, "dates", [])) else None,
        "fx_data_as_of": quality.get("fx_data_as_of"),
        "fx_missing_currencies": quality.get(
            "fx_missing_currencies", fx_quality.get("missing_currencies", [])
        ),
        "fx_stale_currencies": quality.get(
            "fx_stale_currencies", fx_quality.get("stale_currencies", [])
        ),
        "tail_ffill_days": quality.get("tail_ffill_days"),
        "max_tail_ffill_days": quality.get("max_tail_ffill_days"),
    }
    return {"universe_funnel": universe, "data_freshness": freshness}


def validate_cached_result_compatibility(
    result,
    data_tickers,
    data_returns: pd.DataFrame,
    *,
    variant_path: Optional[Path] = None,
) -> None:
    """Fail fast when a stale/65-name result is paired with current 100-name data."""
    expected = list(data_tickers)
    problems = []
    portfolio_weights = getattr(result, "portfolio_weights", None) or {}
    daily_weights = getattr(result, "daily_weights", None) or {}
    if not portfolio_weights:
        problems.append("cached result has no portfolio_weights")
    else:
        latest_rebalance = max(portfolio_weights)
        actual = list(pd.Series(portfolio_weights[latest_rebalance]).index)
        if actual != expected:
            problems.append(
                f"latest portfolio universe/order is {len(actual)} names, expected {len(expected)}"
            )
    if not daily_weights:
        problems.append("cached result has no daily_weights")
    else:
        latest_daily = max(daily_weights)
        actual_daily = list(pd.Series(daily_weights[latest_daily]).index)
        if actual_daily != expected:
            problems.append(
                f"latest daily-weight universe/order is {len(actual_daily)} names, expected {len(expected)}"
            )

    port_returns = getattr(result, "portfolio_returns", None)
    if port_returns is None or len(port_returns) == 0:
        problems.append("cached result has no portfolio return dates")
    elif not isinstance(data_returns, pd.DataFrame) or data_returns.empty:
        problems.append("current UniverseData has no return dates")
    else:
        result_as_of = pd.Timestamp(pd.Series(port_returns).dropna().index.max())
        data_as_of = pd.Timestamp(data_returns.index.max())
        if result_as_of != data_as_of:
            problems.append(
                f"cached result as_of={result_as_of.date()} but current data as_of={data_as_of.date()}"
            )

    if problems:
        variant_arg = str(variant_path) if variant_path is not None else "<variant.yaml>"
        raise ValueError(
            "cached backtest_result.pkl is incompatible with current universe/data: "
            + "; ".join(problems)
            + f". Re-run: python run_variant.py --variant {variant_arg} --no-cache"
        )


def _drawdown_events(cum: pd.Series, dd: pd.Series, max_events: int = 5) -> list[dict]:
    """Contiguous underwater periods, sorted by worst trough."""
    if cum.empty or dd.empty:
        return []
    events = []
    idx = list(dd.index)
    i = 0
    while i < len(idx):
        if dd.iloc[i] >= 0:
            i += 1
            continue
        start_i = i
        while i < len(idx) and dd.iloc[i] < 0:
            i += 1
        end_i = i
        seg = dd.iloc[start_i:end_i]
        trough = seg.idxmin()
        peak_hist = cum.loc[:idx[start_i]]
        peak = peak_hist.idxmax() if not peak_hist.empty else idx[start_i]
        recovery = idx[end_i] if end_i < len(idx) else None
        events.append({
            "peak": str(peak)[:10],
            "start": str(idx[start_i])[:10],
            "trough": str(trough)[:10],
            "recovery": str(recovery)[:10] if recovery is not None else None,
            "max_drawdown": _safe_float(seg.min(), 6),
            "days_underwater": int(end_i - start_i),
        })
    events.sort(key=lambda r: r["max_drawdown"] if r["max_drawdown"] is not None else 0)
    return events[:max_events]


def _serialize_records(
    df: pd.DataFrame, date_col: str = "date", ndigits: int = 6
) -> list[dict]:
    if df is None or df.empty:
        return []
    out = df.copy()
    if out.index.name or isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index().rename(columns={out.index.name or "index": date_col})
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d")
    rows = []
    for rec in out.to_dict(orient="records"):
        rows.append({
            k: (
                _safe_float(v, ndigits)
                if isinstance(v, (int, float, np.integer, np.floating)) else v
            )
            for k, v in rec.items()
        })
    return rows


def build_feature_attribution(result) -> dict:
    """Per-stock SHAP feature attribution at the latest rebalance (§2 schema).

    Pure builder over a duck-typed ``result`` carrying ``models`` (each with
    ``_active_features``), ``panel`` (MultiIndex[date, ticker]), ``portfolio_weights``,
    ``feature_groups`` (group -> [feat]), ``sector_map`` and ``bm_weights``. File
    I/O / pkl load stay in ``main``. SHAP comes from
    ``src.attribution.compute_shap_values``; ``base_value`` is the TreeExplainer
    expected value, so ``base_value + sum(shap) == mu`` is a genuine local-accuracy
    check (not a tautology). Missing model columns raise (no silent reindex-NaN).
    """
    import shap
    from src.attribution import compute_shap_values

    as_of = max(result.portfolio_weights.keys())
    model_dates = [d for d in result.models if d <= as_of]
    if not model_dates:
        raise ValueError(f"no model retrain on/before as_of {as_of}")
    model_date = max(model_dates)
    model = result.models[model_date]
    feats = list(model._active_features)

    at = result.panel.xs(as_of, level="date")
    missing = [f for f in feats if f not in at.columns]
    if missing:
        raise KeyError(f"panel@{as_of} missing model features: {missing}")
    X = at[feats]
    tickers = list(X.index)

    mu = np.asarray(model.predict(X), dtype=float)
    shap_matrix = np.asarray(compute_shap_values(model, X.to_numpy(dtype=float), feats), dtype=float)
    base_value = float(np.ravel(shap.TreeExplainer(model).expected_value)[0])

    feat_to_group = {}
    for grp, group_feats in (result.feature_groups or {}).items():
        for fn in group_feats:
            feat_to_group[fn] = grp

    weights = result.portfolio_weights[as_of]
    additivity_ok = True
    out_tickers = {}
    for i, t in enumerate(tickers):
        shap_i = {f: float(shap_matrix[i, j]) for j, f in enumerate(feats)}
        mu_i = float(mu[i])
        if abs(base_value + sum(shap_i.values()) - mu_i) > 1e-3 * abs(mu_i) + 1e-9:
            additivity_ok = False
        w = float(weights.get(t, 0.0))
        bm = float(result.bm_weights.get(t, 0.0))
        out_tickers[t] = {
            "weight": w, "bm_weight": bm, "active": w - bm,
            "sector": result.sector_map.get(t, "Unknown"),
            "mu": mu_i, "base_value": base_value, "shap": shap_i,
        }
    if not additivity_ok:
        print("[export] WARNING: SHAP additivity gate failed (base+sum(shap) != mu)",
              file=sys.stderr)

    return {
        "as_of": str(as_of)[:10],
        "model_date": str(model_date)[:10],
        "feature_groups": feat_to_group,
        "additivity_ok": additivity_ok,
        "tickers": out_tickers,
    }


def main(argv=None) -> int:
    import yaml
    from src.harness import build_override_config, inject_config, sub_period_irs
    from src.backtest import run_backtest, get_benchmark_fn, get_sector_map
    from src.data_loader import UniverseData
    from src.attribution import compute_feature_importance
    from src.portfolio_optimizer import estimate_covariance

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", type=Path, default=DEFAULT_VARIANT)
    parser.add_argument("--operating-dir", type=Path, default=DEFAULT_OPERATING_DIR)
    args = parser.parse_args(argv)

    variant = _rooted(args.variant).resolve()
    out = _rooted(args.operating_dir).resolve()
    manifest = yaml.safe_load(variant.read_text(encoding="utf-8")) or {}
    overrides = manifest.get("overrides", {})
    cfg = build_override_config(dict(overrides))
    inject_config(cfg)

    # Prefer the cached full result pkl (run_variant saves it) — loading it is
    # ~1GB vs a fresh walk-forward backtest's ~3-4GB peak, so the export still
    # runs under a tight commit limit. Falls back to a fresh backtest if absent.
    t0 = time.time()
    label = str(manifest.get("label") or variant.stem)
    run_dir = _rooted(Path(manifest.get("out_dir") or f"outputs/{label}")).resolve()
    pkl = run_dir / "backtest_result.pkl"
    metrics_path = run_dir / "metrics.json"
    data = UniverseData(cfg.data_path, config=cfg)   # needed for sector map + benchmark weights
    if pkl.exists():
        import pickle
        print(f"[export] loading cached backtest_result.pkl ({pkl.stat().st_size//1_000_000}MB)…")
        with open(pkl, "rb") as fh:
            res = pickle.load(fh)
        try:
            variant_hint = variant.relative_to(ROOT)
        except ValueError:
            variant_hint = variant
        validate_cached_result_compatibility(
            res,
            data.tickers,
            data.returns,
            variant_path=variant_hint,
        )
        print(f"[export] loaded in {time.time()-t0:.0f}s")
    else:
        print("[export] no cached result — running production backtest (single-thread BLAS)…")
        res = run_backtest(data, config=cfg)
        print(f"[export] backtest done in {time.time()-t0:.0f}s")

    out.mkdir(parents=True, exist_ok=True)
    tickers = list(data.tickers)
    sector_map = get_sector_map(data)
    bm_fn = get_benchmark_fn(data, tickers, config=cfg)
    operating_quality = build_operating_quality_fields(data, tickers)

    # ---- returns.csv + performance.json -----------------------------------
    port = res.portfolio_returns.dropna()
    bm = res.benchmark_returns.reindex(port.index).ffill().fillna(0.0)
    cum_p = (1 + port).cumprod()
    cum_b = (1 + bm).cumprod()
    dd = cum_p / cum_p.cummax() - 1.0
    ret_df = pd.DataFrame({
        "portfolio_ret": port, "benchmark_ret": bm,
        "portfolio_cum": cum_p, "benchmark_cum": cum_b, "drawdown": dd,
    })
    ret_df.index.name = "date"

    m = res.compute_metrics()
    by_year = {}
    for yr, g in port.groupby(port.index.year):
        b_g = bm.reindex(g.index)
        p_r = float((1 + g).prod() - 1); b_r = float((1 + b_g).prod() - 1)
        by_year[int(yr)] = {"portfolio": round(p_r, 4), "benchmark": round(b_r, 4),
                            "active": round(p_r - b_r, 4)}
    # The cached pkl snapshots the loader's data_quality at backtest time; the
    # current loader's FX diagnostics are authoritative, so override the fx
    # block from fresh data.data_quality (same design as build_operating_quality_fields).
    merged_data_quality = getattr(res, "data_quality", None) or getattr(data, "data_quality", None)
    fresh_data_quality = getattr(data, "data_quality", None)
    if isinstance(merged_data_quality, dict):
        merged_data_quality = dict(merged_data_quality)
        if isinstance(fresh_data_quality, dict):
            for _fx_key in (
                "fx", "fx_data_as_of", "fx_missing_currencies",
                "fx_stale_currencies", "data_freshness",
            ):
                if _fx_key in fresh_data_quality:
                    merged_data_quality[_fx_key] = fresh_data_quality[_fx_key]
    perf = {
        "as_of": str(port.index.max())[:10],
        "annual_return": m.get("annual_return"), "annual_vol": m.get("annual_vol"),
        "sharpe_ratio": m.get("sharpe_ratio"), "active_return": m.get("active_return"),
        "tracking_error": m.get("tracking_error"), "information_ratio": m.get("information_ratio"),
        "realized_beta": m.get("realized_beta"), "avg_annual_turnover": m.get("avg_annual_turnover"),
        "avg_ic": m.get("avg_ic"),
        "total_return": float(cum_p.iloc[-1] - 1.0), "bm_total_return": float(cum_b.iloc[-1] - 1.0),
        "max_drawdown": float(dd.min()),
        "sub_period_ir": sub_period_irs(port, bm),
        "by_year_returns": by_year,
        "optimizer_failure_rate": getattr(res, "optimizer_failure_rate", None),
        "optimizer_solver_counts": getattr(res, "optimizer_solver_counts", {}),
        "optimizer_solver_fallback_rate": getattr(res, "optimizer_solver_fallback_rate", None),
        "model_quality": getattr(res, "model_quality", None),
        "data_quality": merged_data_quality,
        "base_currency": "USD",
        **operating_quality,
    }
    json.dump(perf, open(out / "performance.json", "w", encoding="utf-8"), indent=2, default=str)

    # ---- holdings.json (latest rebalance OW/UW) ---------------------------
    reb_dates = sorted(res.portfolio_weights.keys())
    last = reb_dates[-1]
    rebalance_meta = build_rebalance_metadata(
        reb_dates,
        port.index,
        getattr(cfg, "rebalance_freq", 21),
    )
    w = res.portfolio_weights[last].reindex(tickers).fillna(0.0)
    bm_w = pd.Series(np.asarray(bm_fn(last, tickers, len(tickers)), dtype=float), index=tickers)
    active = w - bm_w
    rows = [{"ticker": t, "weight": round(float(w[t]), 5), "bm_weight": round(float(bm_w[t]), 5),
             "active": round(float(active[t]), 5), "sector": sector_map.get(t, "Unknown")}
            for t in tickers]
    rows.sort(key=lambda r: r["active"], reverse=True)
    active_share_l1 = float(active.abs().sum())
    abs_active = active.abs().sort_values(ascending=False)
    active_budget = float(abs_active.sum())
    concentration = {
        "top5_weight": _safe_float(w.sort_values(ascending=False).head(5).sum(), 6),
        "top10_weight": _safe_float(w.sort_values(ascending=False).head(10).sum(), 6),
        "effective_names": _safe_float(1.0 / float((w.pow(2)).sum()), 2)
        if float((w.pow(2)).sum()) > 0 else None,
        "max_name_weight": _safe_float(w.max(), 6),
        "max_name_active_abs": _safe_float(abs_active.iloc[0], 6) if len(abs_active) else None,
        "top5_active_budget_share": _safe_float(abs_active.head(5).sum() / active_budget, 6)
        if active_budget > 0 else None,
        "top10_active_budget_share": _safe_float(abs_active.head(10).sum() / active_budget, 6)
        if active_budget > 0 else None,
    }
    holdings = {
        "as_of": str(last)[:10], "n_holdings": int((w > 1e-6).sum()),
        "active_share_l1": round(active_share_l1, 4),
        "active_share_one_way": round(active_share_l1 / 2, 4),
        "concentration": concentration,
        "top_ow": rows[:12], "top_uw": rows[-12:][::-1], "all": rows,
    }
    json.dump(holdings, open(out / "holdings.json", "w", encoding="utf-8"), indent=2, default=str)

    # ---- contribution.json (name/sector contribution) ---------------------
    # Entering-day weights are previous day's recorded end-of-day weights.
    # This reconciles to gross portfolio return before transaction costs; the
    # cost/residual line keeps the displayed active contribution auditable.
    dates = port.index
    raw_stock_rets = data.returns.reindex(dates)[tickers]
    stock_rets = (
        raw_stock_rets
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(float)
    )
    w_daily = pd.DataFrame(res.daily_weights).T.sort_index().reindex(dates).ffill()
    init_bm = pd.Series(np.asarray(bm_fn(dates[0], tickers, len(tickers)), dtype=float), index=tickers)
    w_enter = w_daily.shift(1)
    if not w_enter.empty:
        w_enter.iloc[0] = init_bm
    w_enter = w_enter.fillna(0.0)[tickers]

    rebal_set = set(pd.Timestamp(d) for d in reb_dates)
    bm_enter_rows = []
    bm_curr = init_bm.values.copy()
    for d in dates:
        bm_enter_rows.append(pd.Series(bm_curr.copy(), index=tickers, name=d))
        bm_curr = _drift_weights(bm_curr, stock_rets.loc[d].values)
        if pd.Timestamp(d) in rebal_set:
            bm_curr = np.asarray(bm_fn(d, tickers, len(tickers)), dtype=float)
    bm_enter = pd.DataFrame(bm_enter_rows).reindex(dates)[tickers]

    currency = build_currency_attribution(
        tickers=tickers,
        dates=dates,
        usd_returns=stock_rets,
        local_returns=data.local_returns,
        fx_returns=data.fx_returns,
        fx_rates_usd_per_local=data.fx_rates_usd_per_local,
        currency_map=data.currency_map,
        portfolio_entering_weights=w_enter,
        benchmark_entering_weights=bm_enter,
        latest_portfolio_weights=w,
        latest_benchmark_weights=bm_w,
        portfolio_reported_returns=port,
        benchmark_reported_returns=bm,
        data_quality=getattr(data, "data_quality", None),
    )
    if currency["reconciliation"].get("passed") is not True:
        raise ValueError("currency attribution failed exact local + FX = USD reconciliation")
    currency_daily = pd.DataFrame(currency["daily"])
    currency_daily["date"] = pd.to_datetime(currency_daily["date"])
    currency_daily = currency_daily.set_index("date")
    ret_df = ret_df.join(pd.DataFrame({
        "portfolio_local_gross_ret": currency_daily["portfolio_local_return"],
        "portfolio_fx_effect": currency_daily["portfolio_fx_effect"],
        "portfolio_usd_gross_ret": currency_daily["portfolio_usd_gross_return"],
        "benchmark_local_ret": currency_daily["benchmark_local_return"],
        "benchmark_fx_effect": currency_daily["benchmark_fx_effect"],
        "benchmark_usd_gross_ret": currency_daily["benchmark_usd_gross_return"],
        "active_local_gross_ret": currency_daily["active_local_return"],
        "active_fx_effect": currency_daily["active_fx_effect"],
        "active_usd_gross_ret": currency_daily["active_usd_gross_return"],
    }))
    ret_df.to_csv(out / "returns.csv")

    port_name_contrib = (w_enter * stock_rets).sum(axis=0)
    bm_name_contrib = (bm_enter * stock_rets).sum(axis=0)
    active_name_contrib = port_name_contrib - bm_name_contrib
    gross_port = (w_enter * stock_rets).sum(axis=1)
    gross_bm = (bm_enter * stock_rets).sum(axis=1)
    contribution_residual = {
        "portfolio_gross_arithmetic": _safe_float(gross_port.sum(), 6),
        "portfolio_reported_arithmetic": _safe_float(port.sum(), 6),
        "transaction_cost_and_timing_residual": _safe_float(gross_port.sum() - port.sum(), 6),
        "benchmark_reconstructed_arithmetic": _safe_float(gross_bm.sum(), 6),
        "benchmark_reported_arithmetic": _safe_float(bm.sum(), 6),
        "active_reconstructed_arithmetic": _safe_float((gross_port - gross_bm).sum(), 6),
        "active_reported_arithmetic": _safe_float((port - bm).sum(), 6),
    }
    currency_by_ticker = {row["ticker"]: row for row in currency["by_ticker"]}
    contrib_rows = []
    latest_active = w - bm_w
    for t in tickers:
        fx_row = currency_by_ticker[t]
        contrib_rows.append({
            "ticker": t,
            "sector": sector_map.get(t, "Unknown"),
            "currency": fx_row["currency"],
            "portfolio_contribution": _safe_float(port_name_contrib.get(t), 6),
            "benchmark_contribution": _safe_float(bm_name_contrib.get(t), 6),
            "active_contribution": _safe_float(active_name_contrib.get(t), 6),
            "portfolio_local_contribution": fx_row["portfolio_local_contribution"],
            "portfolio_fx_contribution": fx_row["portfolio_fx_contribution"],
            "portfolio_usd_contribution": fx_row["portfolio_usd_contribution"],
            "benchmark_local_contribution": fx_row["benchmark_local_contribution"],
            "benchmark_fx_contribution": fx_row["benchmark_fx_contribution"],
            "benchmark_usd_contribution": fx_row["benchmark_usd_contribution"],
            "active_local_contribution": fx_row["active_local_contribution"],
            "active_fx_contribution": fx_row["active_fx_contribution"],
            "active_usd_contribution": fx_row["active_usd_contribution"],
            "avg_weight": _safe_float(w_enter[t].mean(), 6),
            "avg_bm_weight": _safe_float(bm_enter[t].mean(), 6),
            "avg_active_weight": _safe_float((w_enter[t] - bm_enter[t]).mean(), 6),
            "latest_weight": _safe_float(w.get(t), 6),
            "latest_bm_weight": _safe_float(bm_w.get(t), 6),
            "latest_active": _safe_float(latest_active.get(t), 6),
        })
    contrib_rows.sort(key=lambda r: r["active_contribution"] or 0, reverse=True)
    contrib_df = pd.DataFrame(contrib_rows)
    sector_contrib = (
        contrib_df.groupby("sector")[["portfolio_contribution", "benchmark_contribution", "active_contribution"]]
        .sum()
        .sort_values("active_contribution", ascending=False)
    )
    contribution = {
        "period": {"start": str(dates.min())[:10], "end": str(dates.max())[:10]},
        "method": "Arithmetic contribution using entering-day weights; excludes compounding. Residual captures transaction costs/timing.",
        "residual": contribution_residual,
        "top_active_contributors": contrib_rows[:15],
        "bottom_active_contributors": sorted(contrib_rows, key=lambda r: r["active_contribution"] or 0)[:15],
        "by_ticker": contrib_rows,
        "by_sector": _serialize_records(sector_contrib, date_col="sector"),
        "currency_attribution_file": "currency.json",
    }
    json.dump(contribution, open(out / "contribution.json", "w", encoding="utf-8"), indent=2, default=str)
    json.dump(currency, open(out / "currency.json", "w", encoding="utf-8"), indent=2, default=str)

    # ---- risk.json (latest total/active risk contribution) ----------------
    te_limit = float(getattr(cfg, "max_te_annual", 0.045))
    risk = {
        "as_of": str(last)[:10],
        "method": "Latest rebalance, Ledoit-Wolf covariance, annualized.",
        "max_tracking_error_annual": _safe_float(te_limit, 6),
    }
    risk_guardrails = {}
    try:
        cov_lookback = int(getattr(cfg, "cov_lookback", 126))
        hist_returns = data.returns[tickers].loc[data.returns.index < last].tail(cov_lookback)
        cov = np.asarray(estimate_covariance(hist_returns, bm_weights=bm_w.values, config=cfg), dtype=float)
        wv = w.values.astype(float)
        av = (w - bm_w).values.astype(float)
        port_var = float(wv @ cov @ wv)
        active_var = float(av @ cov @ av)
        port_vol = float(np.sqrt(max(port_var, 0.0)) * np.sqrt(252.0))
        active_te = float(np.sqrt(max(active_var, 0.0)) * np.sqrt(252.0))
        total_marginal = cov @ wv
        active_marginal = cov @ av
        total_component = wv * total_marginal
        active_component = av * active_marginal
        risk_rows = []
        for i, t in enumerate(tickers):
            risk_rows.append({
                "ticker": t,
                "sector": sector_map.get(t, "Unknown"),
                "weight": _safe_float(wv[i], 6),
                "bm_weight": _safe_float(bm_w.iloc[i], 6),
                "active_weight": _safe_float(av[i], 6),
                "total_risk_pct": _safe_float(total_component[i] / port_var, 6)
                if port_var > 0 else None,
                "total_vol_contribution": _safe_float(total_component[i] / np.sqrt(port_var) * np.sqrt(252.0), 6)
                if port_var > 0 else None,
                "active_risk_pct": _safe_float(active_component[i] / active_var, 6)
                if active_var > 0 else None,
                "active_te_contribution": _safe_float(active_component[i] / np.sqrt(active_var) * np.sqrt(252.0), 6)
                if active_var > 0 else None,
            })
        risk_rows.sort(key=lambda r: abs(r["active_te_contribution"] or 0), reverse=True)
        risk_df = pd.DataFrame(risk_rows)
        sector_risk = (
            risk_df.groupby("sector")[["total_vol_contribution", "active_te_contribution"]]
            .sum()
            .sort_values("active_te_contribution", key=lambda s: s.abs(), ascending=False)
        )
        top_name = risk_rows[0] if risk_rows else {}
        top_name_active_risk_share = None
        top_sector_active_risk_share = None
        top_sector_name = None
        if active_te > 0 and top_name.get("active_te_contribution") is not None:
            top_name_active_risk_share = abs(float(top_name["active_te_contribution"])) / active_te
        if active_te > 0 and not sector_risk.empty:
            top_sector_name = str(sector_risk.index[0])
            top_sector_active_risk_share = (
                abs(float(sector_risk.iloc[0]["active_te_contribution"])) / active_te
            )
        max_name_risk = float(getattr(cfg, "max_name_active_risk_share", 0.35))
        max_sector_risk = float(getattr(cfg, "max_sector_active_risk_share", 0.75))
        risk_guardrails = {
            "max_tracking_error_annual": _safe_float(te_limit, 6),
            "estimated_te_headroom": _safe_float(te_limit - active_te, 6),
            "estimated_te_breached": bool(active_te > te_limit + 1e-6),
            "max_name_active_risk_share": _safe_float(max_name_risk, 6),
            "top_name": top_name.get("ticker"),
            "top_name_active_risk_share": _safe_float(top_name_active_risk_share, 6),
            "top_name_active_risk_breached": (
                top_name_active_risk_share is not None
                and top_name_active_risk_share > max_name_risk
            ),
            "max_sector_active_risk_share": _safe_float(max_sector_risk, 6),
            "top_sector": top_sector_name,
            "top_sector_active_risk_share": _safe_float(top_sector_active_risk_share, 6),
            "top_sector_active_risk_breached": (
                top_sector_active_risk_share is not None
                and top_sector_active_risk_share > max_sector_risk
            ),
        }
        risk.update({
            "cov_lookback_days": cov_lookback,
            "estimated_portfolio_vol": _safe_float(port_vol, 6),
            "estimated_tracking_error": _safe_float(active_te, 6),
            "guardrails": risk_guardrails,
            "by_ticker": risk_rows,
            "by_sector": _serialize_records(sector_risk, date_col="sector"),
        })
    except Exception as exc:
        risk["error"] = str(exc)
    json.dump(risk, open(out / "risk.json", "w", encoding="utf-8"), indent=2, default=str)

    # ---- monitoring.json (trend/guardrail diagnostics) --------------------
    turnover = res.turnover.sort_index()
    turnover_df = pd.DataFrame({
        "turnover_two_way": turnover,
        "turnover_one_way": turnover * 0.5,
        "rolling6_two_way": turnover.rolling(6, min_periods=1).mean(),
        "rolling6_one_way": turnover.rolling(6, min_periods=1).mean() * 0.5,
    })
    active_share_rows = []
    for d in reb_dates:
        wd = res.portfolio_weights[d].reindex(tickers).fillna(0.0)
        bmd = pd.Series(np.asarray(bm_fn(d, tickers, len(tickers)), dtype=float), index=tickers)
        ad = wd - bmd
        budget = float(ad.abs().sum())
        active_share_rows.append({
            "date": str(d)[:10],
            "active_share_one_way": _safe_float(0.5 * budget, 6),
            "max_name_active_abs": _safe_float(ad.abs().max(), 6),
            "top5_active_budget_share": _safe_float(ad.abs().sort_values(ascending=False).head(5).sum() / budget, 6)
            if budget > 0 else None,
            "effective_names": _safe_float(1.0 / float(wd.pow(2).sum()), 2)
            if float(wd.pow(2).sum()) > 0 else None,
        })

    # Recompute the optimizer's ex-ante TE on every realized rebalance using
    # the same trailing covariance window and cap-weighted benchmark.  This is
    # an operating audit of the hard constraint, not realized return TE.
    te_constraint_rows = []
    cov_lookback = int(getattr(cfg, "cov_lookback", 126))
    for d in reb_dates:
        date = pd.Timestamp(d)
        date_pos = int(data.returns.index.get_indexer([date])[0])
        if date_pos < 0:
            continue
        hist = data.returns[tickers].iloc[max(0, date_pos - cov_lookback):date_pos]
        if hist.empty:
            continue
        cov_d = np.asarray(
            estimate_covariance(
                hist,
                bm_weights=np.asarray(bm_fn(date, tickers, len(tickers)), dtype=float),
                config=cfg,
            ),
            dtype=float,
        )
        weights_d = res.portfolio_weights[date].reindex(tickers).fillna(0.0).values
        benchmark_d = np.asarray(bm_fn(date, tickers, len(tickers)), dtype=float)
        active_d = weights_d - benchmark_d
        estimated_te_d = float(
            np.sqrt(max(float(active_d @ cov_d @ active_d), 0.0)) * np.sqrt(252.0)
        )
        te_constraint_rows.append({
            "date": str(date)[:10],
            "estimated_te": _safe_float(estimated_te_d, 6),
            "limit": _safe_float(te_limit, 6),
            "headroom": _safe_float(te_limit - estimated_te_d, 6),
            "breached": bool(estimated_te_d > te_limit + 1e-6),
        })
    te_breach_count = sum(bool(row["breached"]) for row in te_constraint_rows)
    max_rebalance_te = max(
        (float(row["estimated_te"]) for row in te_constraint_rows),
        default=None,
    )

    active_ret = port - bm
    rolling_df = pd.DataFrame(index=port.index)
    for window in (63, 126, 252):
        ar = active_ret.rolling(window).mean() * 252.0
        te = active_ret.rolling(window).std() * np.sqrt(252.0)
        beta = port.rolling(window).cov(bm) / bm.rolling(window).var()
        rolling_df[f"active_return_{window}d_ann"] = ar
        rolling_df[f"tracking_error_{window}d_ann"] = te
        rolling_df[f"information_ratio_{window}d"] = ar / te.replace(0.0, np.nan)
        rolling_df[f"beta_{window}d"] = beta
    rolling_df = rolling_df.dropna(how="all")

    monthly = (1.0 + ret_df[["portfolio_ret", "benchmark_ret"]]).resample("ME").prod() - 1.0
    monthly["active"] = monthly["portfolio_ret"] - monthly["benchmark_ret"]
    monthly.index = monthly.index.to_period("M").astype(str)
    monthly.index.name = "month"

    model_quality = getattr(res, "model_quality", None) or {}
    data_quality = getattr(res, "data_quality", None) or getattr(data, "data_quality", None) or {}
    tail_ffill_days = data_quality.get("tail_ffill_days")
    max_tail_ffill_days = data_quality.get(
        "max_tail_ffill_days", getattr(cfg, "max_tail_ffill_days", None)
    )
    monitoring = {
        "period": {"start": str(port.index.min())[:10], "end": str(port.index.max())[:10]},
        "turnover": _serialize_records(turnover_df, date_col="date"),
        "active_share": active_share_rows,
        "tracking_error_constraint": te_constraint_rows,
        "rolling": _serialize_records(rolling_df, date_col="date"),
        "monthly_returns": _serialize_records(monthly, date_col="month"),
        "drawdown_events": _drawdown_events(cum_p, dd),
        "guardrails": {
            "optimizer_failure_rate": _safe_float(getattr(res, "optimizer_failure_rate", None), 6),
            "optimizer_solver_fallback_rate": _safe_float(
                getattr(res, "optimizer_solver_fallback_rate", None), 6
            ),
            "latest_active_share_one_way": holdings["active_share_one_way"],
            "latest_turnover_two_way": _safe_float(turnover.iloc[-1], 6) if len(turnover) else None,
            "latest_estimated_te": risk.get("estimated_tracking_error"),
            "max_tracking_error_annual": _safe_float(te_limit, 6),
            "estimated_te_headroom": risk_guardrails.get("estimated_te_headroom"),
            "estimated_te_breached": risk_guardrails.get("estimated_te_breached"),
            "max_rebalance_estimated_te": _safe_float(max_rebalance_te, 6),
            "te_constraint_breach_count": int(te_breach_count),
            "te_constraint_breached": bool(te_breach_count),
            "top5_active_budget_share": concentration["top5_active_budget_share"],
            "top_name_active_risk_share": risk_guardrails.get("top_name_active_risk_share"),
            "top_name_active_risk_breached": risk_guardrails.get("top_name_active_risk_breached"),
            "top_sector_active_risk_share": risk_guardrails.get("top_sector_active_risk_share"),
            "top_sector_active_risk_breached": risk_guardrails.get("top_sector_active_risk_breached"),
            "model_degenerate_rate": _safe_float(model_quality.get("degenerate_rate"), 6),
            "model_degenerate_rate_breached": (
                model_quality.get("degenerate_rate") is not None
                and model_quality.get("max_degenerate_model_rate") is not None
                and model_quality.get("degenerate_rate") > model_quality.get("max_degenerate_model_rate")
            ),
            "tail_ffill_days": tail_ffill_days,
            "max_tail_ffill_days": max_tail_ffill_days,
            "tail_ffill_breached": (
                tail_ffill_days is not None
                and max_tail_ffill_days is not None
                and tail_ffill_days > max_tail_ffill_days
            ),
        },
    }
    monitoring["transaction_costs"] = build_transaction_cost_history(
        turnover, float(getattr(cfg, "one_way_tc", 0.001)), n_return_days=len(port)
    )
    json.dump(monitoring, open(out / "monitoring.json", "w", encoding="utf-8"), indent=2, default=str)

    # ---- features.json (gain importance across walk-forward models) --------
    feat_to_group = {}
    for grp, feats in (res.feature_groups or {}).items():
        for fn in feats:
            feat_to_group[fn] = grp
    imps = []
    for d, model in res.models.items():
        feats = getattr(model, "_active_features", None) or res.feature_names
        try:
            imps.append(compute_feature_importance(model, feats))
        except Exception:
            continue
    features = {"n_models": len(imps), "top_features": [], "group_importance": {}}
    if imps:
        avg = pd.concat(imps, axis=1).fillna(0.0).mean(axis=1).sort_values(ascending=False)
        total = float(avg.sum()) or 1.0
        features["top_features"] = [
            {"feature": f, "importance": round(float(v), 1),
             "share_pct": round(float(v) / total * 100, 2),
             "group": feat_to_group.get(f, "?")}
            for f, v in avg.head(30).items()]
        grp = avg.groupby(avg.index.map(lambda f: feat_to_group.get(f, "?"))).sum().sort_values(ascending=False)
        features["group_importance"] = {g: round(float(v) / total * 100, 2) for g, v in grp.items()}
    json.dump(features, open(out / "features.json", "w", encoding="utf-8"), indent=2, default=str)

    # ---- operations.json (next target + trade list + sector exposure) -----
    prev = reb_dates[-2] if len(reb_dates) >= 2 else last
    turnover_index = pd.DatetimeIndex(turnover.index)
    if pd.Timestamp(last) not in turnover_index:
        raise ValueError(f"latest rebalance {last} is absent from res.turnover")
    latest_turnover = float(turnover.iloc[turnover_index.get_loc(pd.Timestamp(last))])
    trade_plan = build_latest_trade_plan(
        target_weights=w,
        entering_weights=w_enter.loc[last],
        latest_returns=raw_stock_rets.loc[last],
        reported_turnover=latest_turnover,
        one_way_transaction_cost=float(getattr(cfg, "one_way_tc", 0.001)),
        as_of=last,
        returns_data_as_of=dates.max(),
    )
    sec_port, sec_bm = {}, {}
    for t in tickers:
        s = sector_map.get(t, "Unknown")
        sec_port[s] = sec_port.get(s, 0.0) + float(w[t])
        sec_bm[s] = sec_bm.get(s, 0.0) + float(bm_w[t])
    sector_exposure = {s: {"portfolio": round(sec_port[s], 4), "benchmark": round(sec_bm.get(s, 0.0), 4),
                           "active": round(sec_port[s] - sec_bm.get(s, 0.0), 4)}
                       for s in sorted(sec_port, key=lambda x: -sec_port[x])}
    sector_dev_limit = float(getattr(cfg, "sector_deviation", 0.10))
    sector_active = []
    for s in sector_exposure:
        active_s = round(sec_port[s] - sec_bm.get(s, 0.0), 4)
        sector_active.append({
            "sector": s,
            "port": round(sec_port[s], 4),
            "bm": round(sec_bm.get(s, 0.0), 4),
            "active": active_s,
            "limit": sector_dev_limit,
            "binding": bool(abs(active_s) >= sector_dev_limit - 1e-9),
        })
    sector_active.sort(key=lambda r: abs(r["active"]), reverse=True)
    current_drift = build_current_drift(
        target_weights=w,
        post_rebalance_returns=stock_rets.loc[stock_rets.index > pd.Timestamp(last)],
        no_trade_band=float(getattr(cfg, "no_trade_band", 0.003)),
        as_of=dates.max(),
        last_rebalance_date=last,
    )
    ops = {
        "as_of": str(last)[:10], "prev_rebalance": str(prev)[:10],
        "data_as_of": perf["as_of"],
        "base_currency": "USD",
        "rebalance_freq_days": getattr(cfg, "rebalance_freq", None),
        "sector_exposure": sector_exposure,
        "sector_deviation_limit": sector_dev_limit,
        "sector_active": sector_active,
        "current_drift": current_drift,
        "optimizer_failure_rate": getattr(res, "optimizer_failure_rate", None),
        "optimizer_failures": getattr(res, "optimizer_failures", None),
        "solver_protocol": (
            "ECOS-only"
            if not getattr(cfg, "allow_scs_on_ecos_exception", False)
            else "ECOS with SCS on ECOS exception"
        ),
        "optimizer_solver_counts": getattr(res, "optimizer_solver_counts", {}),
        "optimizer_solver_fallbacks": getattr(res, "optimizer_solver_fallbacks", None),
        "optimizer_solver_fallback_rate": getattr(res, "optimizer_solver_fallback_rate", None),
        **operating_quality,
    }
    ops.update(trade_plan)
    ops.update(rebalance_meta)
    json.dump(ops, open(out / "operations.json", "w", encoding="utf-8"), indent=2, default=str)

    # ---- feature_attribution.json (per-stock SHAP drivers, latest rebalance) --
    from types import SimpleNamespace
    fa = build_feature_attribution(SimpleNamespace(
        models=res.models,
        panel=res.panel,
        portfolio_weights=res.portfolio_weights,
        feature_groups=res.feature_groups,
        sector_map=sector_map,
        bm_weights={t: float(bm_w[t]) for t in tickers},
    ))
    json.dump(fa, open(out / "feature_attribution.json", "w", encoding="utf-8"), indent=2, default=str)

    universe_hash = hashlib.sha256("\n".join(tickers).encode("utf-8")).hexdigest()
    model_quality = getattr(res, "model_quality", None) or {}
    portfolio_meta = {
        "schema_version": 1,
        "id": label,
        "display_name": manifest.get("display_name", _LABEL_DEFAULTS.get(label, (label, "challenger"))[0]),
        "portfolio_role": manifest.get("portfolio_role", _LABEL_DEFAULTS.get(label, (label, "challenger"))[1]),
        "model_type": getattr(cfg, "model_objective", "regression"),
        "variant_path": str(variant.relative_to(ROOT)).replace("\\", "/"),
        "run_dir": str(run_dir.relative_to(ROOT)).replace("\\", "/"),
        "operating_dir": str(out.relative_to(ROOT)).replace("\\", "/"),
        "benchmark_type": getattr(cfg, "benchmark_type", "cap_weighted"),
        "base_currency": "USD",
        "max_tracking_error_annual": _safe_float(te_limit, 6),
        "universe_size": len(tickers),
        "universe": tickers,
        "universe_hash": universe_hash,
        "data_as_of": perf["as_of"],
        **rebalance_meta,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_metrics_sha256": _sha256(metrics_path),
        "source_metrics_mtime_utc": (
            datetime.fromtimestamp(metrics_path.stat().st_mtime, timezone.utc).isoformat()
            if metrics_path.exists() else None
        ),
        **build_provenance_meta(run_dir),
        "causal_validation_enabled": bool(getattr(cfg, "causal_validation_enabled", False)),
        "causal_validation_ok": model_quality.get("causal_validation_ok"),
        "execution_signal_lag_days": int(getattr(cfg, "execution_signal_lag_days", 0)),
        **operating_quality,
    }
    json.dump(portfolio_meta, open(out / "portfolio.json", "w", encoding="utf-8"), indent=2, default=str)

    print(f"\n=== export summary ===")
    print(f"  perf: annual {perf['annual_return']*100:.1f}% | active {perf['active_return']*100:.1f}% | "
          f"IR {perf['information_ratio']:.3f} | maxDD {perf['max_drawdown']*100:.1f}%")
    print(f"  holdings: {holdings['n_holdings']} names, active_share L1 {holdings['active_share_l1']:.3f} "
          f"(one-way {holdings['active_share_one_way']:.3f}), as_of {holdings['as_of']}")
    print(f"  top OW: " + ", ".join(f"{r['ticker']}{r['active']*100:+.1f}%" for r in holdings['top_ow'][:5]))
    print(f"  top UW: " + ", ".join(f"{r['ticker']}{r['active']*100:+.1f}%" for r in holdings['top_uw'][:5]))
    print(f"  contribution: active arithmetic {contribution_residual['active_reconstructed_arithmetic']:+.3f} "
          f"| residual {contribution_residual['transaction_cost_and_timing_residual']:+.3f}")
    if "error" in risk:
        print(f"  risk: skipped ({risk['error']})")
    else:
        print(f"  risk: est vol {risk['estimated_portfolio_vol']*100:.1f}% | "
              f"est TE {risk['estimated_tracking_error']*100:.2f}%")
    print(f"  monitoring: turnover points {len(turnover)} | rolling rows {len(rolling_df)}")
    print(f"  features({features['n_models']} models) top: " +
          ", ".join(f"{f['feature']}" for f in features['top_features'][:5]))
    print(f"  ops: {ops['n_trades']} latest-rebalance trades, wrote {out}")
    print(
        f"  rebalance: last {rebalance_meta['last_rebalance_date']} | "
        f"next expected {rebalance_meta['next_expected_rebalance_date']} | "
        f"rows remaining {rebalance_meta['rows_until_next_rebalance']}"
    )
    print(f"  feature_attribution: {len(fa['tickers'])} names, additivity_ok={fa['additivity_ok']}, "
          f"as_of {fa['as_of']}, model_date {fa['model_date']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
