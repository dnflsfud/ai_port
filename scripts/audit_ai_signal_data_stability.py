"""Read-only stability audit for ai_signal_data.xlsx.

The audit intentionally treats pre-listing history as out of scope. Eligibility
dates are resolved with the same rules used by ``UniverseData``; all completeness
and continuity checks start on or after that date.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_CONFIG
from src.data_loader import (
    BLOOMBERG_EQUITY_SHEETS,
    SENT_TREND_SHEETS,
    _filter_tickers,
    _rename_bloomberg_equity_columns,
    _rename_sent_trend_columns,
    _standardize_columns,
    _standardize_index,
    build_company_to_ticker,
    load_all_sheets,
    load_universe_meta,
    resolve_listing_dates,
)


ESSENTIAL_SHEETS = {
    "PX_LAST",
    "Daily_Returns",
    "CUR_MKT_CAP",
    "BEST_EPS",
    "BEST_SALES",
    "BEST_PE_RATIO",
    "OPER_MARGIN",
    "BEST_ROE",
    "NEWS_SENTIMENT_DAILY_AVG",
    "EQY_REC_CONS",
    "Factset_EPS_Revision",
    "Factset_Sales_Revision",
    "Factset_TG_Price",
}

OPTIONAL_SHEETS = {
    "BEST_CALCULATED_FCF",
    "BEST_CAPEX",
    "BEST_EV_TO_BEST_EBITDA",
    "BEST_GROSS_MARGIN",
    "BEST_PEG_RATIO",
    "BEST_PX_BPS_RATIO",
}

CONTINUOUS_SHEETS = {"PX_LAST", "Daily_Returns", "CUR_MKT_CAP"}
NON_TICKER_SHEETS = {
    "Universe_Meta",
    "BusinessDays",
    "Summary_Stats",
    "Factor_PX_LAST",
    "Factor_Returns",
    "Factor_Meta",
}

SEVERITY_RANK = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def _business_index(raw: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    frame = raw["BusinessDays"]
    values: Iterable[Any]
    if isinstance(frame.index, pd.DatetimeIndex):
        values = frame.index
    elif len(frame.columns):
        values = frame.iloc[:, 0]
    else:
        values = frame.index
    idx = pd.DatetimeIndex(pd.to_datetime(list(values), errors="coerce")).dropna()
    return idx.drop_duplicates().sort_values()


def _normalize_ticker_sheet(
    sheet_name: str,
    frame: pd.DataFrame,
    tickers: list[str],
    company_to_ticker: dict[str, str],
) -> tuple[pd.DataFrame, int]:
    work = _standardize_columns(frame.copy())
    if sheet_name in SENT_TREND_SHEETS:
        work = _rename_sent_trend_columns(
            work,
            tickers=tickers,
            company_to_ticker=company_to_ticker,
        )
    if sheet_name in BLOOMBERG_EQUITY_SHEETS:
        work = _rename_bloomberg_equity_columns(work)
    work = _standardize_index(work)
    work = _filter_tickers(work, tickers=tickers)
    numeric = work.apply(pd.to_numeric, errors="coerce")
    coercion_loss = int((work.notna() & numeric.isna()).sum().sum())
    return numeric.sort_index(), coercion_loss


def _longest_true_run(mask: pd.Series) -> int:
    if mask.empty or not bool(mask.any()):
        return 0
    groups = mask.ne(mask.shift(fill_value=False)).cumsum()
    return int(mask.groupby(groups).sum().max())


def _trailing_true_run(mask: pd.Series) -> int:
    if mask.empty or not bool(mask.iloc[-1]):
        return 0
    return int(mask.iloc[::-1].cumprod().sum())


def _trailing_constant_run(series: pd.Series) -> int:
    valid = series.dropna()
    if valid.empty:
        return 0
    last = valid.iloc[-1]
    same = np.isclose(valid.to_numpy(dtype=float), last, rtol=1e-10, atol=1e-12)
    return int(np.cumprod(same[::-1]).sum())


def _add_issue(
    issues: list[dict[str, Any]],
    severity: str,
    check: str,
    evidence: str,
    sheet: str | None = None,
    ticker: str | None = None,
    confidence: str = "High",
) -> None:
    issues.append(
        {
            "severity": severity,
            "check": check,
            "sheet": sheet,
            "ticker": ticker,
            "evidence": evidence,
            "confidence": confidence,
        }
    )


def _audit_ticker_sheet(
    sheet_name: str,
    frame: pd.DataFrame,
    tickers: list[str],
    eligibility: dict[str, str],
    business_days: pd.DatetimeIndex,
    coercion_loss: int,
    issues: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    duplicate_dates = int(frame.index.duplicated().sum())
    non_monotonic = not frame.index.is_monotonic_increasing
    future_dates = int((frame.index > pd.Timestamp.today().normalize()).sum())
    missing_columns = [ticker for ticker in tickers if ticker not in frame.columns]
    extra_columns = [str(column) for column in frame.columns if column not in tickers]

    if duplicate_dates:
        _add_issue(
            issues,
            "High",
            "duplicate_dates",
            f"{duplicate_dates} duplicate date row(s)",
            sheet=sheet_name,
        )
    if non_monotonic:
        _add_issue(
            issues,
            "High",
            "date_order",
            "Date index is not monotonically increasing",
            sheet=sheet_name,
        )
    if future_dates:
        _add_issue(
            issues,
            "High",
            "future_dates",
            f"{future_dates} future date row(s)",
            sheet=sheet_name,
        )
    if coercion_loss:
        _add_issue(
            issues,
            "High",
            "non_numeric_values",
            f"{coercion_loss} populated cell(s) cannot be parsed as numeric",
            sheet=sheet_name,
        )
    if missing_columns:
        severity = "Critical" if sheet_name in ESSENTIAL_SHEETS else "Low"
        _add_issue(
            issues,
            severity,
            "missing_ticker_columns",
            f"{len(missing_columns)} Universe_Meta ticker(s) absent: {', '.join(missing_columns[:12])}",
            sheet=sheet_name,
        )

    date_index = frame.index.intersection(business_days)
    business = frame.reindex(date_index)
    metrics: list[dict[str, Any]] = []
    for ticker in tickers:
        if ticker not in business.columns or business.empty:
            continue
        start = pd.Timestamp(eligibility.get(ticker, business.index.min()))
        if sheet_name == "Daily_Returns":
            eligible = business.loc[business.index > start, ticker]
        else:
            eligible = business.loc[business.index >= max(start, business.index.min()), ticker]
        if eligible.empty:
            continue
        missing = eligible.isna()
        valid = eligible.dropna()
        coverage = float(valid.size / eligible.size)
        longest_gap = _longest_true_run(missing)
        trailing_gap = _trailing_true_run(missing)
        metric = {
            "sheet": sheet_name,
            "ticker": ticker,
            "eligibility_date": start.strftime("%Y-%m-%d"),
            "eligible_business_rows": int(eligible.size),
            "valid_rows": int(valid.size),
            "coverage": coverage,
            "longest_missing_run": longest_gap,
            "trailing_missing_run": trailing_gap,
            "first_valid": valid.index.min().strftime("%Y-%m-%d") if len(valid) else None,
            "last_valid": valid.index.max().strftime("%Y-%m-%d") if len(valid) else None,
            "inf_count": int(np.isinf(eligible.to_numpy(dtype=float, na_value=np.nan)).sum()),
        }
        metrics.append(metric)

        if sheet_name in CONTINUOUS_SHEETS:
            if coverage < 0.99 or longest_gap > 5 or trailing_gap > 3:
                severity = "High" if coverage < 0.95 or longest_gap > 20 or trailing_gap > 10 else "Medium"
                _add_issue(
                    issues,
                    severity,
                    "post_listing_continuity",
                    f"coverage={coverage:.2%}, longest gap={longest_gap}, trailing gap={trailing_gap} business day(s)",
                    sheet=sheet_name,
                    ticker=ticker,
                )
        elif sheet_name in ESSENTIAL_SHEETS:
            if coverage < 0.90 or trailing_gap > 20:
                severity = "High" if coverage < 0.70 or trailing_gap > 60 else "Medium"
                _add_issue(
                    issues,
                    severity,
                    "post_listing_essential_coverage",
                    f"coverage={coverage:.2%}, longest gap={longest_gap}, trailing gap={trailing_gap} business day(s)",
                    sheet=sheet_name,
                    ticker=ticker,
                )

        if metric["inf_count"]:
            _add_issue(
                issues,
                "High",
                "infinite_values",
                f"{metric['inf_count']} infinite value(s) after eligibility",
                sheet=sheet_name,
                ticker=ticker,
            )

        if sheet_name == "PX_LAST" and len(valid):
            non_positive = int((valid <= 0).sum())
            flat_tail = _trailing_constant_run(valid)
            metric["non_positive_count"] = non_positive
            metric["trailing_constant_run"] = flat_tail
            if non_positive:
                _add_issue(
                    issues,
                    "Critical",
                    "non_positive_price",
                    f"{non_positive} non-positive price(s)",
                    sheet=sheet_name,
                    ticker=ticker,
                )
            if flat_tail >= 6:
                _add_issue(
                    issues,
                    "High" if flat_tail >= 11 else "Medium",
                    "stale_price_tail",
                    f"Last price is unchanged for {flat_tail} business observations",
                    sheet=sheet_name,
                    ticker=ticker,
                )
        elif sheet_name == "CUR_MKT_CAP" and len(valid):
            non_positive = int((valid <= 0).sum())
            metric["non_positive_count"] = non_positive
            if non_positive:
                _add_issue(
                    issues,
                    "Critical",
                    "non_positive_market_cap",
                    f"{non_positive} non-positive market-cap value(s)",
                    sheet=sheet_name,
                    ticker=ticker,
                )
        elif sheet_name == "Daily_Returns" and len(valid):
            extreme = int((valid.abs() > 0.50).sum())
            metric["absolute_return_gt_50pct"] = extreme
            if extreme:
                _add_issue(
                    issues,
                    "Medium",
                    "extreme_daily_return",
                    f"{extreme} daily return(s) exceed 50% in absolute value",
                    sheet=sheet_name,
                    ticker=ticker,
                )

    coverages = [row["coverage"] for row in metrics]
    summary = {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "first_date": frame.index.min().strftime("%Y-%m-%d") if len(frame) else None,
        "last_date": frame.index.max().strftime("%Y-%m-%d") if len(frame) else None,
        "business_rows": int(len(date_index)),
        "duplicate_dates": duplicate_dates,
        "future_dates": future_dates,
        "non_monotonic": non_monotonic,
        "missing_universe_columns": missing_columns,
        "extra_columns": extra_columns,
        "coercion_loss": coercion_loss,
        "coverage_median": float(np.median(coverages)) if coverages else None,
        "coverage_min": float(np.min(coverages)) if coverages else None,
    }
    return summary, metrics


def _audit_price_return_pair(
    price: pd.DataFrame,
    returns: pd.DataFrame,
    tickers: list[str],
    eligibility: dict[str, str],
    business_days: pd.DatetimeIndex,
    issues: list[dict[str, Any]],
    pair_name: str,
) -> list[dict[str, Any]]:
    common_dates = price.index.intersection(returns.index).intersection(business_days)
    expected = price.sort_index().pct_change(fill_method=None).reindex(common_dates)
    actual = returns.reindex(common_dates)
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        if ticker not in expected or ticker not in actual:
            continue
        start = pd.Timestamp(eligibility.get(ticker, common_dates.min()))
        pair = pd.concat(
            [expected.loc[expected.index > start, ticker], actual.loc[actual.index > start, ticker]],
            axis=1,
            keys=["expected", "actual"],
        ).dropna()
        if pair.empty:
            continue
        diff = (pair["actual"] - pair["expected"]).abs()
        corr = float(pair["actual"].corr(pair["expected"])) if len(pair) > 2 else None
        row = {
            "pair": pair_name,
            "ticker": ticker,
            "observations": int(len(pair)),
            "correlation": corr,
            "median_absolute_error": float(diff.median()),
            "p99_absolute_error": float(diff.quantile(0.99)),
            "mismatch_gt_1bp": int((diff > 0.0001).sum()),
            "mismatch_gt_100bp": int((diff > 0.01).sum()),
        }
        rows.append(row)
        mismatch_share = row["mismatch_gt_1bp"] / row["observations"]
        if (corr is not None and corr < 0.99) or mismatch_share > 0.01:
            severity = "High" if (corr is not None and corr < 0.90) or mismatch_share > 0.10 else "Medium"
            _add_issue(
                issues,
                severity,
                "price_return_mismatch",
                f"corr={corr:.4f}, >1bp mismatch={mismatch_share:.2%}, p99 error={row['p99_absolute_error']:.4%}",
                sheet=pair_name,
                ticker=ticker,
            )
    return rows


def _audit_factor_sheets(
    raw: dict[str, pd.DataFrame],
    business_days: pd.DatetimeIndex,
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    normalized: dict[str, pd.DataFrame] = {}
    for name in ("Factor_PX_LAST", "Factor_Returns"):
        frame = _standardize_index(_standardize_columns(raw[name].copy()))
        frame = frame.apply(pd.to_numeric, errors="coerce").sort_index()
        normalized[name] = frame
        active = frame.reindex(frame.index.intersection(business_days))
        inf_count = int(np.isinf(active.to_numpy(dtype=float, na_value=np.nan)).sum())
        result[name] = {
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "first_date": frame.index.min().strftime("%Y-%m-%d"),
            "last_date": frame.index.max().strftime("%Y-%m-%d"),
            "duplicate_dates": int(frame.index.duplicated().sum()),
            "missing_cells": int(active.isna().sum().sum()),
            "infinite_cells": inf_count,
        }
        if result[name]["duplicate_dates"] or inf_count:
            _add_issue(
                issues,
                "High",
                "factor_structure",
                f"duplicate dates={result[name]['duplicate_dates']}, infinite cells={inf_count}",
                sheet=name,
            )
    factor_meta = raw["Factor_Meta"].copy()
    short_name_column = next(
        (column for column in factor_meta.columns if str(column).strip().casefold() == "short_name"),
        None,
    )
    category_column = next(
        (column for column in factor_meta.columns if str(column).strip().casefold() == "category"),
        None,
    )
    status_column = next(
        (column for column in factor_meta.columns if str(column).strip().casefold() == "status"),
        None,
    )
    factor_columns = set(normalized["Factor_PX_LAST"].columns).intersection(
        normalized["Factor_Returns"].columns
    )
    factor_tickers = sorted(factor_columns)
    price_like_categories = {
        "Market Index",
        "FX",
        "Commodity",
        "Factor ETF",
        "GS Thematic",
        "GICS1 Sector",
    }
    price_like_tickers = factor_tickers
    if short_name_column is not None and category_column is not None:
        meta_short_names = factor_meta[short_name_column].astype(str).str.strip()
        duplicates = meta_short_names[meta_short_names.duplicated()].tolist()
        missing_from_data = sorted(set(meta_short_names).difference(factor_columns))
        extra_in_data = sorted(factor_columns.difference(meta_short_names))
        result["Factor_Meta"] = {
            "rows": int(len(factor_meta)),
            "duplicate_short_names": duplicates,
            "missing_from_factor_sheets": missing_from_data,
            "extra_in_factor_sheets": extra_in_data,
        }
        if duplicates or missing_from_data or extra_in_data:
            _add_issue(
                issues,
                "High",
                "factor_metadata_alignment",
                f"duplicates={duplicates}, missing={missing_from_data}, extra={extra_in_data}",
                sheet="Factor_Meta",
            )
        category_by_ticker = dict(zip(meta_short_names, factor_meta[category_column].astype(str).str.strip()))
        price_like_tickers = [
            ticker for ticker in factor_tickers
            if category_by_ticker.get(ticker) in price_like_categories
        ]
    if status_column is not None:
        unavailable = factor_meta.loc[
            factor_meta[status_column].astype(str).str.strip().ne("Available")
        ].index.astype(str).tolist()
        if unavailable:
            _add_issue(
                issues,
                "High",
                "factor_unavailable_status",
                f"{len(unavailable)} factor(s) are not Available",
                sheet="Factor_Meta",
            )
    pair_rows = _audit_price_return_pair(
        normalized["Factor_PX_LAST"],
        normalized["Factor_Returns"],
        price_like_tickers,
        {},
        business_days,
        issues,
        "Factor_PX_LAST vs Factor_Returns",
    )
    result["price_return_consistency"] = pair_rows
    return result


def _audit_summary_stats(
    raw_summary: pd.DataFrame,
    prices: pd.DataFrame,
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    stored = raw_summary.copy()
    stored.index = stored.index.map(lambda value: str(value).strip())
    expected = prices.describe().T
    expected["missing_pct"] = prices.isna().mean() * 100.0
    shared_columns = [
        column for column in ("count", "mean", "std", "min", "25%", "50%", "75%", "max", "missing_pct")
        if column in stored.columns and column in expected.columns
    ]
    shared_tickers = stored.index.intersection(expected.index)
    diffs = (
        stored.loc[shared_tickers, shared_columns].apply(pd.to_numeric, errors="coerce")
        - expected.loc[shared_tickers, shared_columns]
    ).abs()
    mismatched_cells = int((diffs > 1e-8).sum().sum())
    max_abs_difference = float(diffs.max().max()) if diffs.size else None
    missing_tickers = sorted(set(expected.index).difference(stored.index))
    extra_tickers = sorted(set(stored.index).difference(expected.index))
    if mismatched_cells or missing_tickers or extra_tickers:
        _add_issue(
            issues,
            "High",
            "summary_stats_mismatch",
            f"mismatched cells={mismatched_cells}, missing tickers={missing_tickers}, extra tickers={extra_tickers}",
            sheet="Summary_Stats",
        )
    return {
        "rows": int(len(stored)),
        "columns_compared": shared_columns,
        "tickers_compared": int(len(shared_tickers)),
        "mismatched_cells": mismatched_cells,
        "max_absolute_difference": max_abs_difference,
        "missing_tickers": missing_tickers,
        "extra_tickers": extra_tickers,
    }


def _write_markdown(
    path: Path,
    workbook: Path,
    summary: dict[str, Any],
    issues: list[dict[str, Any]],
    sheet_summary: dict[str, Any],
) -> None:
    counts = summary["issue_counts"]
    lines = [
        "# ai_signal_data stability audit",
        "",
        f"- Source: `{workbook}`",
        f"- Universe: {summary['universe_count']} tickers",
        f"- Sheets: {summary['sheet_count']}",
        f"- Business-date range: {summary['business_date_range'][0]} to {summary['business_date_range'][1]}",
        f"- Eligibility dates resolved: {summary['eligibility_resolved_count']}/{summary['universe_count']}",
        f"- Issues: Critical {counts.get('Critical', 0)}, High {counts.get('High', 0)}, Medium {counts.get('Medium', 0)}, Low {counts.get('Low', 0)}",
        "",
        "## Severity interpretation",
        "",
        "Pre-listing cells are excluded. Weekend forward-fills are excluded by evaluating continuity only on BusinessDays.",
        "Optional fundamental-sheet gaps are reported as Low unless another validity rule fails.",
        "",
        "## Top issues",
        "",
        "| Severity | Check | Sheet | Ticker | Evidence |",
        "|---|---|---|---|---|",
    ]
    for issue in issues[:100]:
        evidence = str(issue["evidence"]).replace("|", "/")
        lines.append(
            f"| {issue['severity']} | {issue['check']} | {issue.get('sheet') or ''} | "
            f"{issue.get('ticker') or ''} | {evidence} |"
        )
    lines.extend(
        [
            "",
            "## Sheet coverage",
            "",
            "| Sheet | Rows | Columns | Last date | Median coverage | Minimum coverage |",
            "|---|---:|---:|---|---:|---:|",
        ]
    )
    for name, row in sheet_summary.items():
        median = row.get("coverage_median")
        minimum = row.get("coverage_min")
        median_text = f"{median:.2%}" if isinstance(median, (int, float)) else ""
        minimum_text = f"{minimum:.2%}" if isinstance(minimum, (int, float)) else ""
        lines.append(
            f"| {name} | {row.get('rows', '')} | {row.get('columns', '')} | "
            f"{row.get('last_date', '')} | {median_text} | {minimum_text} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(workbook: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_all_sheets(str(workbook))
    meta = load_universe_meta(raw)
    tickers = list(meta.index)
    company_to_ticker = build_company_to_ticker(meta, tickers=tickers)
    eligibility, eligibility_sources = resolve_listing_dates(meta, raw, DEFAULT_CONFIG)
    business_days = _business_index(raw)
    issues: list[dict[str, Any]] = []

    business_duplicate_dates = int(business_days.duplicated().sum())
    business_future_dates = int((business_days > pd.Timestamp.today().normalize()).sum())
    if business_duplicate_dates or business_future_dates:
        _add_issue(
            issues,
            "High",
            "business_calendar_integrity",
            f"duplicate dates={business_duplicate_dates}, future dates={business_future_dates}",
            "BusinessDays",
        )

    if meta.index.duplicated().any():
        _add_issue(issues, "Critical", "duplicate_ticker", "Universe_Meta has duplicate tickers", "Universe_Meta")
    raw_meta = raw["Universe_Meta"]
    required_meta = {"Name", "Sector", "Status"}
    missing_meta = sorted(required_meta.difference(raw_meta.columns))
    if missing_meta:
        _add_issue(
            issues,
            "Critical",
            "missing_metadata_columns",
            f"Missing: {', '.join(missing_meta)}",
            "Universe_Meta",
        )
    unavailable = []
    if "Status" in raw_meta.columns:
        unavailable = raw_meta.index[raw_meta["Status"].astype(str).str.strip().ne("Available")].astype(str).tolist()
        if unavailable:
            _add_issue(
                issues,
                "High",
                "unavailable_status",
                f"{len(unavailable)} ticker(s) are not Available",
                "Universe_Meta",
            )

    missing_essential_sheets = sorted(ESSENTIAL_SHEETS.difference(raw))
    if missing_essential_sheets:
        _add_issue(
            issues,
            "Critical",
            "missing_essential_sheets",
            f"Missing: {', '.join(missing_essential_sheets)}",
        )

    sheet_summary: dict[str, Any] = {}
    ticker_metrics: list[dict[str, Any]] = []
    normalized: dict[str, pd.DataFrame] = {}
    for sheet_name, frame in raw.items():
        if sheet_name in NON_TICKER_SHEETS:
            continue
        try:
            clean, coercion_loss = _normalize_ticker_sheet(
                sheet_name,
                frame,
                tickers,
                company_to_ticker,
            )
        except Exception as exc:  # Keep auditing other sheets and surface the failure.
            _add_issue(
                issues,
                "Critical",
                "sheet_normalization_failed",
                f"{type(exc).__name__}: {exc}",
                sheet_name,
            )
            continue
        normalized[sheet_name] = clean
        summary, metrics = _audit_ticker_sheet(
            sheet_name,
            clean,
            tickers,
            eligibility,
            business_days,
            coercion_loss,
            issues,
        )
        sheet_summary[sheet_name] = summary
        ticker_metrics.extend(metrics)

    price_return = []
    if "PX_LAST" in normalized and "Daily_Returns" in normalized:
        price_return = _audit_price_return_pair(
            normalized["PX_LAST"],
            normalized["Daily_Returns"],
            tickers,
            eligibility,
            business_days,
            issues,
            "PX_LAST vs Daily_Returns",
        )

    summary_stats_audit: dict[str, Any] = {}
    if "Summary_Stats" in raw and "PX_LAST" in normalized:
        summary_stats_audit = _audit_summary_stats(
            raw["Summary_Stats"],
            normalized["PX_LAST"],
            issues,
        )

    factor_audit = _audit_factor_sheets(raw, business_days, issues)

    unresolved = sorted(set(tickers).difference(eligibility))
    if unresolved:
        _add_issue(
            issues,
            "High",
            "unresolved_eligibility",
            f"No listing/eligibility date for {len(unresolved)} ticker(s): {', '.join(unresolved[:20])}",
        )

    issues.sort(
        key=lambda row: (
            SEVERITY_RANK[row["severity"]],
            str(row.get("sheet") or ""),
            str(row.get("ticker") or ""),
            row["check"],
        )
    )
    counts = pd.Series([row["severity"] for row in issues], dtype=object).value_counts().to_dict()
    sector_column = "sector" if "sector" in meta.columns else "Sector" if "Sector" in meta.columns else None
    metric_frame = pd.DataFrame(ticker_metrics)
    newest_50 = tickers[-50:]
    newest_50_issue_count = sum(
        1 for issue in issues if issue.get("ticker") in set(newest_50)
    )
    summary = {
        "workbook": str(workbook),
        "workbook_size_bytes": workbook.stat().st_size,
        "workbook_modified": pd.Timestamp(workbook.stat().st_mtime, unit="s").isoformat(),
        "sheet_count": len(raw),
        "sheet_names": list(raw),
        "universe_count": len(tickers),
        "newest_50_tickers": newest_50,
        "newest_50_ticker_issue_count": newest_50_issue_count,
        "sector_counts": meta[sector_column].value_counts().to_dict() if sector_column else {},
        "status_non_available": unavailable,
        "business_date_range": [
            business_days.min().strftime("%Y-%m-%d"),
            business_days.max().strftime("%Y-%m-%d"),
        ],
        "business_day_count": len(business_days),
        "business_duplicate_dates": business_duplicate_dates,
        "business_future_dates": business_future_dates,
        "eligibility_resolved_count": len(eligibility),
        "eligibility_unresolved": unresolved,
        "eligibility_source_counts": pd.Series(list(eligibility_sources.values()), dtype=object).value_counts().to_dict(),
        "issue_counts": {severity: int(counts.get(severity, 0)) for severity in SEVERITY_RANK},
    }
    result = {
        "summary": summary,
        "listing_eligibility": eligibility,
        "listing_eligibility_sources": eligibility_sources,
        "sheet_summary": sheet_summary,
        "issues": issues,
        "summary_stats_audit": summary_stats_audit,
        "factor_audit": factor_audit,
    }

    metric_frame.to_csv(output_dir / "ticker_sheet_metrics.csv", index=False)
    pd.DataFrame(issues).to_csv(output_dir / "issues.csv", index=False)
    pd.DataFrame(price_return).to_csv(output_dir / "price_return_consistency.csv", index=False)
    (output_dir / "audit_results.json").write_text(
        json.dumps(_json_value(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown(
        output_dir / "data_quality_report.md",
        workbook,
        summary,
        issues,
        sheet_summary,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, default=Path(DEFAULT_CONFIG.data_path))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ai_signal_data_stability"),
    )
    args = parser.parse_args()
    result = run_audit(args.workbook, args.output_dir)
    print(json.dumps(_json_value(result["summary"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
