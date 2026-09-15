"""Persist target-price anomalies independently of the latest run's metrics."""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path


def _positive(value):
    try:
        value = float(value)
        return value if math.isfinite(value) and value > 0 else None
    except (TypeError, ValueError):
        return None


def _read(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("expected an object")
        return value
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read target-price guard state {path}: {exc}") from exc


def _save(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name, suffix=".tmp", delete=False) as fh:
        tmp = Path(fh.name)
        json.dump(state, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.flush()
        os.fsync(fh.fileno())
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def annotate_basis_guard(out_dir, data_quality, threshold=0.25):
    """Keep a pending anomaly until its original normal basis is recovered.

    First use migrates the prior metrics, including a legacy jump's ORIGINAL
    reference. A clean first-ever observation establishes a baseline. Invalid
    state is fatal; invalid/missing observations never clear pending names.
    No automatic acknowledgement of a persistent new basis is performed.
    """
    if not isinstance(data_quality, dict) or not isinstance(data_quality.get("currency"), dict):
        return {}
    currency = data_quality["currency"]
    path = Path(out_dir) / "tg_basis_state.json"
    if path.exists():
        state = _read(path)
    else:
        previous_path = Path(out_dir) / "metrics.json"
        previous = _read(previous_path) if previous_path.exists() else {}
        previous_currency = (previous.get("data_quality") or {}).get("currency") or {}
        # Metrics carry a recovery copy so deleting a sidecar cannot accept a
        # previously rejected vintage merely by running again.
        state = previous_currency.get("tg_basis_guard_state")
        if state is None:
            baseline = {
                str(k): v for k, value in (previous_currency.get("tg_px_ratio_median") or {}).items()
                if (v := _positive(value)) is not None
                and k not in (previous_currency.get("tg_px_ratio_suspect") or {})
            }
            pending = dict(previous_currency.get("tg_px_ratio_jump_vs_prev") or {})
            for ticker, jump in pending.items():
                reference = _positive(jump.get("previous"))
                if reference is not None:
                    baseline[ticker] = reference
            state = {"schema_version": 1, "baseline": baseline, "pending": pending}
    if (state.get("schema_version") != 1 or not isinstance(state.get("baseline"), dict)
            or not isinstance(state.get("pending"), dict)):
        raise ValueError(f"Invalid target-price guard state: {path}")
    baseline, pending = dict(state["baseline"]), dict(state["pending"])
    if any(_positive(v) is None for v in baseline.values()):
        raise ValueError(f"Invalid target-price guard baseline: {path}")
    if any(not isinstance(v, dict) for v in pending.values()):
        raise ValueError(f"Invalid target-price guard pending events: {path}")
    now = currency.get("tg_px_ratio_median") or {}
    suspect = currency.get("tg_px_ratio_suspect") or {}
    valid = bool(now)
    for ticker, value in now.items():
        value = _positive(value)
        old = baseline.get(ticker)
        if value is None:
            valid = False
            continue
        changed = old is not None and abs(math.log(value / old)) > threshold
        if changed or ticker in suspect:
            pending[ticker] = {"previous": old, "now": value}
        else:
            baseline[ticker] = value
            pending.pop(ticker, None)
    state = {"schema_version": 1, "baseline": baseline, "pending": pending}
    _save(path, state)
    currency["tg_px_ratio_jump_vs_prev"] = pending
    currency["tg_basis_guard_state"] = state
    currency["tg_basis_guard_ok"] = bool(valid and not pending and not suspect)
    return pending
