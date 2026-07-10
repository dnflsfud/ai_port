import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from scripts.validate_portfolio_bundles import build_registry, validate_bundle


def _write_bundle(root: Path, name: str, role: str, *, universe=("AAA", "BBB"), as_of="2026-06-11"):
    run_dir = root / f"run_{name}"
    bundle = root / f"bundle_{name}"
    run_dir.mkdir()
    bundle.mkdir()
    metrics = run_dir / "metrics.json"
    metrics.write_text(json.dumps({"label": name, "metrics": {"information_ratio": 1.0}}), encoding="utf-8")
    metrics_hash = hashlib.sha256(metrics.read_bytes()).hexdigest()
    universe_hash = hashlib.sha256("\n".join(universe).encode()).hexdigest()
    perf = {
        "as_of": as_of, "annual_return": 0.2, "active_return": 0.03,
        "information_ratio": 1.0, "tracking_error": 0.03, "realized_beta": 1.0,
        "avg_annual_turnover": 1.0, "max_drawdown": -0.2, "avg_ic": 0.04,
        "sub_period_ir": {"P1_ir": 1.0, "P2_ir": 1.0, "P3_ir": 1.0},
    }
    meta = {
        "schema_version": 1, "id": name, "display_name": name,
        "portfolio_role": role, "model_type": "regression",
        "run_dir": str(run_dir).replace("\\", "/"),
        "operating_dir": str(bundle).replace("\\", "/"),
        "benchmark_type": "cap_weighted", "universe_size": len(universe),
        "universe": list(universe), "universe_hash": universe_hash,
        "data_as_of": as_of, "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_metrics_sha256": metrics_hash, "causal_validation_enabled": role == "challenger",
        "causal_validation_ok": True if role == "challenger" else None,
        "execution_signal_lag_days": 1 if role == "challenger" else 0,
    }
    (bundle / "portfolio.json").write_text(json.dumps(meta), encoding="utf-8")
    (bundle / "performance.json").write_text(json.dumps(perf), encoding="utf-8")
    (bundle / "holdings.json").write_text(json.dumps({"as_of": as_of}), encoding="utf-8")
    for fname in ("features.json", "operations.json", "contribution.json", "risk.json", "monitoring.json", "feature_attribution.json"):
        (bundle / fname).write_text("{}", encoding="utf-8")
    pd.DataFrame({"date": [as_of], "portfolio_ret": [0.0]}).to_csv(bundle / "returns.csv", index=False)
    return bundle


def test_registry_requires_matching_source_and_common_contract(tmp_path):
    production = _write_bundle(tmp_path, "prod", "production")
    challenger = _write_bundle(tmp_path, "causal", "challenger")
    registry = build_registry([production, challenger])
    assert [p["portfolio_role"] for p in registry["portfolios"]] == ["production", "challenger"]
    assert registry["data_as_of"] == "2026-06-11"
    assert registry["comparison_gate"]["status"] in {"PASS", "RESEARCH/FAIL"}


def test_registry_rejects_universe_mismatch(tmp_path):
    production = _write_bundle(tmp_path, "prod", "production")
    challenger = _write_bundle(tmp_path, "causal", "challenger", universe=("AAA", "CCC"))
    with pytest.raises(ValueError, match="universe_hash mismatch"):
        build_registry([production, challenger])


def test_bundle_rejects_missing_or_tampered_files(tmp_path):
    bundle = _write_bundle(tmp_path, "prod", "production")
    (bundle / "risk.json").unlink()
    with pytest.raises(ValueError, match="missing required files"):
        validate_bundle(bundle)
