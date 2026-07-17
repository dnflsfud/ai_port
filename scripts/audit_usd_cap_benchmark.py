#!/usr/bin/env python
"""Audit the production benchmark as a 100-name USD cap-weighted index."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest import get_benchmark_fn
from src.data_loader import UniverseData
from src.harness import build_override_config


def _drift(weights: np.ndarray, returns: np.ndarray) -> np.ndarray:
    gross = np.asarray(weights, dtype=float) * (1.0 + np.asarray(returns, dtype=float))
    total = float(gross.sum())
    return gross / total if np.isfinite(total) and total > 0 else np.asarray(weights, dtype=float)


def _median_abs(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.abs().median()) if len(values) else None


def audit(variant_path: Path, result_path: Path | None = None) -> dict:
    manifest = yaml.safe_load(variant_path.read_text(encoding="utf-8")) or {}
    cfg = build_override_config(dict(manifest.get("overrides") or {}))
    data = UniverseData(cfg.data_path, config=cfg)
    tickers = list(data.tickers)
    bm_fn = get_benchmark_fn(data, tickers, config=cfg)

    if len(tickers) != 100 or len(set(tickers)) != 100:
        raise ValueError(f"expected 100 unique benchmark members, got {len(tickers)}")
    if cfg.base_currency != "USD" or not cfg.convert_returns_to_usd:
        raise ValueError("portfolio return accounting is not configured in USD")
    if cfg.benchmark_type != "cap_weighted":
        raise ValueError(f"benchmark_type is {cfg.benchmark_type!r}, not cap_weighted")

    # Independent direct-cap calculation on the first/middle/latest dates.
    check_dates = [data.dates[0], data.dates[len(data.dates) // 2], data.dates[-1]]
    max_weight_error = 0.0
    for date in check_dates:
        caps = data.market_cap.reindex(columns=tickers).ffill().loc[date].to_numpy(dtype=float)
        caps = np.where(np.isfinite(caps) & (caps > 0), caps, 0.0)
        direct = caps / caps.sum()
        implemented = np.asarray(bm_fn(date, tickers, len(tickers)), dtype=float)
        max_weight_error = max(max_weight_error, float(np.max(np.abs(direct - implemented))))

    latest = pd.Timestamp(data.dates[-1])
    latest_caps = data.market_cap.loc[latest, tickers].astype(float)
    latest_weights = latest_caps.clip(lower=0.0) / latest_caps.clip(lower=0.0).sum()
    non_usd = [ticker for ticker in tickers if data.currency_map[ticker] != "USD"]

    # Unit evidence: if cap is USD while price is local, d(cap/local_price)
    # follows d(USD-per-local FX).  If cap were local currency, that ratio
    # would instead be flat apart from share-count changes.
    cap_to_local_price = data.market_cap[tickers] / data.local_prices[tickers]
    ratio_change = cap_to_local_price.pct_change(fill_method=None)
    fx_change = data.fx_rates_usd_per_local[tickers].pct_change(fill_method=None)
    currency_evidence = []
    for ticker in non_usd:
        frame = pd.concat(
            [ratio_change[ticker].rename("ratio"), fx_change[ticker].rename("fx")],
            axis=1,
        ).replace([np.inf, -np.inf], np.nan).dropna()
        # Remove corporate-action/share-count jumps; FX moves in this sample
        # are far inside this broad 10% daily robustness filter.
        frame = frame.loc[(frame["ratio"].abs() <= 0.10) & (frame["fx"].abs() <= 0.10)]
        usd_error = _median_abs(frame["ratio"] - frame["fx"])
        local_error = _median_abs(frame["ratio"])
        currency_evidence.append({
            "ticker": ticker,
            "currency": data.currency_map[ticker],
            "observations": int(len(frame)),
            "median_error_usd_cap_hypothesis": usd_error,
            "median_error_local_cap_hypothesis": local_error,
            "supports_usd_cap": bool(
                usd_error is not None and local_error is not None and usd_error < local_error
            ),
        })
    usd_cap_support_count = sum(row["supports_usd_cap"] for row in currency_evidence)

    benchmark_return_error = None
    if result_path is not None and result_path.exists():
        with result_path.open("rb") as fh:
            result = pickle.load(fh)
        reported = result.benchmark_returns.dropna().sort_index()
        rebalance_dates = {pd.Timestamp(date) for date in result.portfolio_weights}
        current = np.asarray(bm_fn(reported.index[0], tickers, len(tickers)), dtype=float)
        reconstructed = []
        for date in reported.index:
            returns = (
                data.returns.loc[date, tickers]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            reconstructed.append(float(current @ returns))
            current = _drift(current, returns)
            if pd.Timestamp(date) in rebalance_dates:
                current = np.asarray(bm_fn(date, tickers, len(tickers)), dtype=float)
        benchmark_return_error = float(
            np.max(np.abs(np.asarray(reconstructed) - reported.to_numpy(dtype=float)))
        )

    passed = bool(
        max_weight_error <= 1e-12
        and usd_cap_support_count == len(non_usd)
        and (benchmark_return_error is None or benchmark_return_error <= 1e-12)
    )
    return {
        "passed": passed,
        "variant": str(variant_path),
        "data_path": str(cfg.data_path),
        "data_as_of": str(latest)[:10],
        "base_currency": cfg.base_currency,
        "benchmark_type": cfg.benchmark_type,
        "universe_size": len(tickers),
        "unique_tickers": len(set(tickers)),
        "non_usd_ticker_count": len(non_usd),
        "direct_weight_max_abs_error": max_weight_error,
        "benchmark_return_max_abs_error": benchmark_return_error,
        "usd_cap_support_count": usd_cap_support_count,
        "currency_evidence": currency_evidence,
        "latest_weight_sum": float(latest_weights.sum()),
        "latest_non_usd_benchmark_weight": float(latest_weights[non_usd].sum()),
        "latest_top_weights": {
            ticker: float(weight)
            for ticker, weight in latest_weights.sort_values(ascending=False).head(10).items()
        },
        "latest_cap_usd_millions_sample": {
            ticker: float(latest_caps[ticker])
            for ticker in ("AAPL", "000660", "ASML", "NESN", "RR/", "NOVOB")
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        type=Path,
        default=ROOT / "variants" / "codex_causal_rank_65.yaml",
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=ROOT / "outputs" / "codex_causal_rank_65" / "backtest_result.pkl",
    )
    args = parser.parse_args(argv)
    output = audit(args.variant.resolve(), args.result.resolve())
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
