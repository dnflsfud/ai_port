"""Bind a saved backtest to the exact inputs and computation that produced it."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _file_identity(path, *, required=False):
    path = Path(path).resolve()
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return {"path": str(path), "sha256": None, "size": None, "mtime_ns": None}
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Input changed while fingerprinting: {path}")
    return {"path": str(path), "sha256": digest.hexdigest(),
            "size": before.st_size, "mtime_ns": before.st_mtime_ns}


def code_fingerprint():
    paths = sorted((ROOT / "src").rglob("*.py")) + [ROOT / "run_variant.py"]
    return _digest({p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in paths})


def capture_run_inputs(config):
    """Capture BEFORE loading; recheck after loading and before publishing."""
    return {"data": _file_identity(config.data_path, required=True),
            "fx": _file_identity(config.fx_source_path), "code_sha256": code_fingerprint()}


def verify_run_inputs(config, captured):
    if capture_run_inputs(config) != captured:
        raise ValueError("Source data/code changed during this run; do not publish mixed inputs. Re-run --no-cache.")


def vintage_from_inputs(captured):
    def stamp(identity):
        ns = identity["mtime_ns"]
        return None if ns is None else datetime.fromtimestamp(ns / 1e9, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"data_path": captured["data"]["path"], "data_mtime_utc": stamp(captured["data"]),
            "data_size_bytes": captured["data"]["size"], "fx_source_path": captured["fx"]["path"],
            "fx_mtime_utc": stamp(captured["fx"]), "fx_size_bytes": captured["fx"]["size"]}


def build_run_contract(config, data, captured):
    config_values = asdict(config)
    config_values.pop("output_dir", None)  # output location does not change the calculation
    # Dates are text so pandas datetime precision (us/ns) cannot alter equality.
    return {"schema_version": 1, "inputs": captured,
            "config_sha256": _digest(config_values),
            "data_dates": [d.isoformat() for d in data.returns.index],
            "tickers": list(data.tickers)}


def validate_run_contract(saved, current):
    required = {"schema_version", "inputs", "config_sha256", "data_dates", "tickers"}
    if (not isinstance(saved, dict) or not isinstance(current, dict)
            or not required.issubset(saved) or not required.issubset(current)
            or saved["schema_version"] != 1 or current["schema_version"] != 1):
        raise ValueError("cached result is missing verified run provenance; re-run --no-cache")
    changed = [key for key in sorted(required) if saved[key] != current[key]]
    if changed:
        raise ValueError(f"cached result run provenance mismatch ({', '.join(changed)}); re-run --no-cache")
