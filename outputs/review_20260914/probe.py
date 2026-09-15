"""Read-only structural audit; writes evidence only under this directory."""
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from run_variant import annotate_tg_ratio_jump
from src.config import DEFAULT_CONFIG
from src.data_loader import FACTOR_CATEGORIES, UniverseData, restrict_to_business_days
from scripts.export_operating_data import build_rebalance_metadata, validate_cached_result_compatibility
from scripts.validate_portfolio_bundles import evaluate_production

OUT = Path(__file__).resolve().parent
results = {}

dates = pd.to_datetime(["2026-07-02", "2026-07-03", "2026-07-06"])
prices = pd.DataFrame({"AAA": [100.0, 110.0, 121.0]}, index=dates)
factor_prices = prices.rename(columns={"AAA": "NKY"})
raw = {
    "BusinessDays": pd.DataFrame(index=dates[[0, 2]]),
    "PX_LAST": prices,
    "Daily_Returns": prices.pct_change(fill_method=None),
    "Factor_PX_LAST": factor_prices,
    "Factor_Returns": factor_prices.pct_change(fill_method=None),
    "Earnings_Timeline": pd.DataFrame({"AAA": [0, 1, 0]}, index=dates),
}
filtered, _ = restrict_to_business_days(raw)
shell = UniverseData.__new__(UniverseData)
shell.raw = filtered
shell.full_universe = ["AAA"]
shell.dates = filtered["PX_LAST"].index
earnings = shell._load_earnings_timeline()
results["synthetic_calendar"] = {
    "stock_ret_after_holiday": float(filtered["Daily_Returns"].iloc[-1, 0]),
    "factor_ret_after_holiday": float(filtered["Factor_Returns"].iloc[-1, 0]),
    "factor_ret_from_kept_prices": float(filtered["Factor_PX_LAST"].pct_change(fill_method=None).iloc[-1, 0]),
    "earnings_before": int(raw["Earnings_Timeline"].to_numpy().sum()),
    "earnings_after": int(earnings.to_numpy().sum()),
}
results["synthetic_schedule"] = build_rebalance_metadata(dates[:1], dates[:1], 1)
results["synthetic_schedule"]["next_date_in_supplied_business_calendar"] = "2026-07-06"

# Reproduce two consecutive run_variant writes with one unchanged suspect vintage.
scratch = OUT / "jump_scratch"
scratch.mkdir(exist_ok=True)
prior = {"data_quality": {"currency": {"tg_px_ratio_median": {"AAA": 1.0}}}}
(scratch / "metrics.json").write_text(json.dumps(prior), encoding="utf-8")
record = {
    "_risk_guardrails": {
        "estimated_te_breached": False,
        "top_name_active_risk_breached": False,
        "top_sector_active_risk_breached": False,
    },
    "_model_quality": {"split_audit": [{"prediction_date": "2026-07-06"}], "events": []},
    "performance": {"tracking_error": 0.03},
}
checks = []
for _ in range(2):
    quality = {
        "currency": {"tg_px_ratio_median": {"AAA": 1.5}, "tg_px_ratio_suspect": {}},
        "tail_ffill_days": 0,
        "max_tail_ffill_days": 10,
    }
    jumps = annotate_tg_ratio_jump(scratch, quality)
    record["performance"]["data_quality"] = quality
    gate = evaluate_production(record)
    checks.append({"jumps": jumps, "status": gate["status"], "checks": gate["checks"]})
    (scratch / "metrics.json").write_text(json.dumps({"data_quality": quality}), encoding="utf-8")
results["repeated_vintage_gate"] = checks

# The cache contract accepts the same names and end date even when the
# history/calendar has changed. A fundamentals-only refresh is not an input
# to this validator at all; the exporter also does not compare config/vintage.
cached = SimpleNamespace(
    portfolio_weights={dates[-1]: pd.Series({"AAA": 1.0})},
    daily_weights={dates[-1]: pd.Series({"AAA": 1.0})},
    portfolio_returns=pd.Series([0.0, 0.10, 0.10], index=dates),
)
validate_cached_result_compatibility(cached, ["AAA"], filtered["Daily_Returns"])
results["cache_compatibility"] = {
    "accepted": True,
    "cached_dates": [str(d)[:10] for d in dates],
    "current_dates": [str(d)[:10] for d in filtered["Daily_Returns"].index],
    "cached_last_return": 0.10,
    "current_last_return": float(filtered["Daily_Returns"].iloc[-1, 0]),
    "scope": "Compatibility guard only; full export may reject changed P&L during later reconciliation.",
}

print("SYNTHETIC", json.dumps(results, default=str), flush=True)

# Inspect only the source sheets needed for calendar/event evidence.
path = Path(DEFAULT_CONFIG.data_path)
stat_before = path.stat()
with pd.ExcelFile(path, engine="openpyxl") as book:
    wanted = ["BusinessDays", "Factor_PX_LAST", "Factor_Returns"]
    event_key = next((k for k in ("Earnings_Timeline", "Earnings_Date") if k in book.sheet_names), None)
    if event_key:
        wanted.append(event_key)
    sheets = pd.read_excel(book, sheet_name=wanted, index_col=0)
bd = sheets["BusinessDays"]
business_days = pd.DatetimeIndex(pd.to_datetime(bd["BusinessDay"] if "BusinessDay" in bd else bd.index)).normalize()
if event_key:
    events = sheets[event_key]
    events.index = pd.DatetimeIndex(pd.to_datetime(events.index)).normalize()
    events = events.loc[(events.index >= business_days.min()) & (events.index <= business_days.max())]
    holiday_events = events.loc[(events.index.dayofweek < 5) & ~events.index.isin(business_days)]
    counts = holiday_events.eq(1).sum()
    pairs = holiday_events.eq(1).stack()
    pairs = pairs[pairs]
    results["actual_earnings"] = {
        "source_sheet": event_key,
        "weekday_holiday_events_lost": int(counts.sum()),
        "names_affected": int((counts > 0).sum()),
        "by_name": {str(k): int(v) for k, v in counts[counts > 0].sort_values(ascending=False).items()},
        "recent_examples": [{"date": str(d)[:10], "ticker": str(t)} for d, t in pairs.index[-20:]],
    }
    performance_start = pd.read_csv("outputs/operating_codex_causal_rank_65/returns.csv", nrows=1)["date"].iloc[0]
    live_pairs = [(d, t) for d, t in pairs.index if d >= pd.Timestamp(performance_start)]
    results["actual_earnings"].update({
        "backtest_start": performance_start,
        "events_lost_in_backtest_period": len(live_pairs),
        "names_in_backtest_period": sorted({str(t) for d, t in live_pairs}),
    })
fpx = sheets["Factor_PX_LAST"]
fret = sheets["Factor_Returns"]
for frame in (fpx, fret):
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).normalize()
    frame.columns = frame.columns.astype(str).str.strip()
kept_dates = fpx.index.intersection(business_days).sort_values()
fpx = fpx.apply(pd.to_numeric, errors="coerce")
fret = fret.apply(pd.to_numeric, errors="coerce")
price_returns = fpx.reindex(kept_dates).pct_change(fill_method=None)
actual_ret = fret.reindex(index=kept_dates, columns=price_returns.columns)
raw_step_ret = fpx.pct_change(fill_method=None).reindex(kept_dates)
delta = (price_returns - actual_ret).abs()
# Factor_Returns includes LEVEL CHANGES for volatility/rate series; comparing
# those to percentage returns is invalid. Audit percentage-return columns only.
return_columns = [c for group in ("Market_Index", "FX", "Commodity", "Factor_ETF", "GS_Thematic")
                  for c in FACTOR_CATEGORIES[group] if c in delta]
delta = delta[return_columns]
results["actual_factors"] = {
    "scope": "Percentage-return factor columns only; level-change columns excluded.",
    "ret_minus_price_ret_cells_gt_1bp": int((delta > 0.0001).sum().sum()),
    "by_factor": {str(k): int(v) for k, v in (delta > 0.0001).sum().items() if v > 0},
    "max_abs_error": {str(k): float(v) for k, v in delta.max().items() if np.isfinite(v) and v > 0.0001},
    "raw_ret_vs_raw_price_max_error": float((raw_step_ret - actual_ret)[return_columns].abs().max().max()),
}
stat_after = path.stat()
results["source"] = {
    "path": str(path), "size": stat_before.st_size, "mtime_ns": stat_before.st_mtime_ns,
    "unchanged_during_read": (stat_before.st_size, stat_before.st_mtime_ns) == (stat_after.st_size, stat_after.st_mtime_ns),
    "calendar_start": str(business_days.min())[:10], "calendar_end": str(business_days.max())[:10],
}

# Compare the exporter's forecasts made at historical rebalances to the next
# observed rebalance under the exact current production result.
with Path("outputs/codex_causal_rank_65/backtest_result.pkl").open("rb") as handle:
    cached_production = pickle.load(handle)
rebalances = sorted(cached_production.portfolio_weights)
portfolio_dates = pd.DatetimeIndex(cached_production.portfolio_returns.index)
schedule_mismatches = []
for previous, following in zip(rebalances, rebalances[1:]):
    observed_dates = portfolio_dates[portfolio_dates <= previous]
    forecast = build_rebalance_metadata([previous], observed_dates, 21)["next_expected_rebalance_date"]
    if forecast != str(following)[:10]:
        schedule_mismatches.append({"as_of": str(previous)[:10], "forecast": forecast, "actual_next": str(following)[:10]})
results["actual_schedule"] = {
    "intervals": len(rebalances) - 1,
    "mismatches": len(schedule_mismatches),
    "examples": schedule_mismatches[-10:],
}
(OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print("ACTUAL", json.dumps({k: v for k, v in results.items() if k.startswith("actual") or k == "source"}, ensure_ascii=False, default=str), flush=True)
