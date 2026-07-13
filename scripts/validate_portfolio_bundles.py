#!/usr/bin/env python
"""Validate operating bundles and publish the dashboard portfolio registry."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "portfolio.json",
    "performance.json",
    "holdings.json",
    "features.json",
    "operations.json",
    "contribution.json",
    "risk.json",
    "monitoring.json",
    "feature_attribution.json",
    "returns.csv",
)


def _rooted(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        value = json.load(fh)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_bundle(bundle_dir: Path) -> dict:
    """Validate one bundle and return registry-ready metadata."""
    bundle_dir = _rooted(bundle_dir).resolve()
    missing = [name for name in REQUIRED_FILES if not (bundle_dir / name).exists()]
    if missing:
        raise ValueError(f"{bundle_dir}: missing required files: {missing}")

    meta = _read_json(bundle_dir / "portfolio.json")
    perf = _read_json(bundle_dir / "performance.json")
    holdings = _read_json(bundle_dir / "holdings.json")
    operations = _read_json(bundle_dir / "operations.json")
    for name in REQUIRED_FILES:
        if name.endswith(".json"):
            _read_json(bundle_dir / name)

    returns = pd.read_csv(bundle_dir / "returns.csv", parse_dates=["date"])
    if returns.empty or returns["date"].isna().all():
        raise ValueError(f"{bundle_dir}: returns.csv has no dated observations")
    if returns["date"].duplicated().any():
        raise ValueError(f"{bundle_dir}: returns.csv has duplicate dates")
    weekend_rows = returns.loc[returns["date"].dt.dayofweek >= 5, "date"]
    if not weekend_rows.empty:
        sample = weekend_rows.dt.strftime("%Y-%m-%d").head(5).tolist()
        raise ValueError(f"{bundle_dir}: returns.csv contains weekend dates: {sample}")
    returns_as_of = returns["date"].max().strftime("%Y-%m-%d")
    if perf.get("as_of") != returns_as_of or meta.get("data_as_of") != returns_as_of:
        raise ValueError(
            f"{bundle_dir}: as_of mismatch "
            f"performance={perf.get('as_of')} meta={meta.get('data_as_of')} "
            f"returns={returns_as_of}"
        )
    if pd.Timestamp(holdings.get("as_of")) > pd.Timestamp(returns_as_of):
        raise ValueError(f"{bundle_dir}: holdings date is after performance date")

    data_quality = perf.get("data_quality") or {}
    tail_days = data_quality.get("tail_ffill_days")
    max_tail_days = data_quality.get("max_tail_ffill_days")
    if tail_days is not None and max_tail_days is not None and int(tail_days) > int(max_tail_days):
        fail_on = bool(data_quality.get("fail_on_stale_tail_ffill", False))
        if fail_on:
            raise ValueError(
                f"{bundle_dir}: stale tail exceeds limit: {tail_days} > {max_tail_days}"
            )
        print(
            f"[bundle-validator] WARN: {bundle_dir}: stale tail {tail_days} > {max_tail_days} "
            f"(fail_on_stale_tail_ffill=false)",
            file=sys.stderr,
        )

    rebalance_fields = (
        "last_rebalance_date",
        "previous_rebalance_date",
        "next_expected_rebalance_date",
        "rebalance_freq_days",
        "rebalance_calendar",
        "rows_since_last_rebalance",
        "rows_until_next_rebalance",
        "is_rebalance_data_as_of",
        "next_rebalance_is_estimate",
    )
    missing_rebalance = [field for field in rebalance_fields if field not in meta]
    if missing_rebalance:
        raise ValueError(f"{bundle_dir}: missing rebalance metadata: {missing_rebalance}")
    for field in rebalance_fields:
        if operations.get(field) != meta.get(field):
            raise ValueError(f"{bundle_dir}: operations/portfolio {field} mismatch")
    last_rebalance = pd.Timestamp(meta["last_rebalance_date"])
    next_rebalance = pd.Timestamp(meta["next_expected_rebalance_date"])
    if (
        holdings.get("as_of") != meta["last_rebalance_date"]
        or operations.get("as_of") != meta["last_rebalance_date"]
    ):
        raise ValueError(f"{bundle_dir}: latest holdings/operations rebalance date mismatch")
    if next_rebalance <= last_rebalance or next_rebalance.dayofweek >= 5:
        raise ValueError(f"{bundle_dir}: invalid next expected rebalance date")
    freq = int(meta["rebalance_freq_days"])
    rows_since = int(meta["rows_since_last_rebalance"])
    rows_until = int(meta["rows_until_next_rebalance"])
    if freq <= 0 or rows_since < 0 or rows_until <= 0 or rows_since + rows_until != freq:
        raise ValueError(f"{bundle_dir}: inconsistent rebalance row counters")
    expected_is_rebalance = returns_as_of == meta["last_rebalance_date"]
    if bool(meta["is_rebalance_data_as_of"]) != expected_is_rebalance:
        raise ValueError(f"{bundle_dir}: is_rebalance_data_as_of mismatch")
    if int(meta.get("universe_size", 0)) != len(meta.get("universe") or []):
        raise ValueError(f"{bundle_dir}: universe_size does not match universe list")
    expected_universe_hash = hashlib.sha256(
        "\n".join(meta.get("universe") or []).encode("utf-8")
    ).hexdigest()
    if meta.get("universe_hash") != expected_universe_hash:
        raise ValueError(f"{bundle_dir}: universe hash mismatch")

    run_dir = _rooted(Path(meta.get("run_dir", ""))).resolve()
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise ValueError(f"{bundle_dir}: source metrics missing: {metrics_path}")
    if meta.get("source_metrics_sha256") != _sha256(metrics_path):
        raise ValueError(f"{bundle_dir}: source metrics hash mismatch")
    exported_at = datetime.fromisoformat(str(meta["exported_at_utc"]).replace("Z", "+00:00"))
    metrics_mtime = datetime.fromtimestamp(metrics_path.stat().st_mtime, timezone.utc)
    if exported_at < metrics_mtime:
        raise ValueError(f"{bundle_dir}: bundle predates its source metrics")

    try:
        expected_operating = str(bundle_dir.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        expected_operating = str(bundle_dir).replace("\\", "/")
    if meta.get("operating_dir") != expected_operating:
        raise ValueError(
            f"{bundle_dir}: operating_dir metadata is {meta.get('operating_dir')!r}, "
            f"expected {expected_operating!r}"
        )
    if meta.get("causal_validation_enabled") and meta.get("causal_validation_ok") is not True:
        raise ValueError(f"{bundle_dir}: causal validation audit failed")

    return {**meta, "performance": perf}


def evaluate_challenger(production: dict, challenger: dict) -> dict:
    """Apply the pre-registered, non-blocking investment performance gate."""
    base = production["performance"]
    arm = challenger["performance"]
    base_sub = base.get("sub_period_ir") or {}
    arm_sub = arm.get("sub_period_ir") or {}
    sub_wins = sum(
        float(arm_sub.get(k, float("-inf"))) >= float(base_sub.get(k, float("inf")))
        for k in ("P1_ir", "P2_ir", "P3_ir")
    )
    checks = {
        "information_ratio_improved": float(arm.get("information_ratio")) > float(base.get("information_ratio")),
        "active_return_improved": float(arm.get("active_return")) > float(base.get("active_return")),
        "tracking_error_within_4_5pct": float(arm.get("tracking_error")) <= 0.045,
        "beta_between_0_95_and_1_05": 0.95 <= float(arm.get("realized_beta")) <= 1.05,
        "turnover_within_1_25x": float(arm.get("avg_annual_turnover")) <= 1.25 * float(base.get("avg_annual_turnover")),
        "max_drawdown_not_worse_by_3pp": float(arm.get("max_drawdown")) >= float(base.get("max_drawdown")) - 0.03,
        "subperiod_ir_wins_at_least_2": sub_wins >= 2,
    }
    passed = all(checks.values())
    return {
        "status": "PASS" if passed else "RESEARCH/FAIL",
        "passed": passed,
        "checks": checks,
        "subperiod_ir_wins": sub_wins,
    }


def build_registry(bundle_dirs: list[Path]) -> dict:
    records = [validate_bundle(path) for path in bundle_dirs]
    ids = [row["id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate portfolio ids: {ids}")

    common_fields = (
        "data_as_of",
        "universe_hash",
        "benchmark_type",
        "last_rebalance_date",
        "previous_rebalance_date",
        "next_expected_rebalance_date",
        "rebalance_freq_days",
        "rebalance_calendar",
        "rows_since_last_rebalance",
        "rows_until_next_rebalance",
        "is_rebalance_data_as_of",
    )
    for field in common_fields:
        values = {row.get(field) for row in records}
        if len(values) != 1:
            raise ValueError(f"portfolio {field} mismatch: {sorted(values)}")

    production = next((r for r in records if r.get("portfolio_role") == "production"), None)
    if production is None:
        raise ValueError("exactly one production portfolio is required")
    challengers = [r for r in records if r.get("portfolio_role") == "challenger"]
    if len(challengers) != 1:
        raise ValueError("exactly one challenger portfolio is required")
    gate = evaluate_challenger(production, challengers[0])

    ordered = sorted(records, key=lambda r: (r.get("portfolio_role") != "production", r["id"]))
    entries = []
    for row in ordered:
        item = {k: v for k, v in row.items() if k != "performance"}
        item["status"] = "PRODUCTION" if row is production else gate["status"]
        entries.append(item)
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stale_after_hours": 96,
        "data_as_of": production["data_as_of"],
        "comparison_gate": gate,
        "portfolios": entries,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", action="append", type=Path, required=True)
    parser.add_argument(
        "--registry", type=Path, default=ROOT / "outputs" / "portfolio_registry.json"
    )
    args = parser.parse_args(argv)
    try:
        registry = build_registry(args.bundle)
        target = _rooted(args.registry).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, target)
    except Exception as exc:
        print(f"[bundle-validator] ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"[bundle-validator] OK: {len(registry['portfolios'])} portfolios, "
        f"as_of={registry['data_as_of']}, gate={registry['comparison_gate']['status']}"
    )
    print(f"[bundle-validator] wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
