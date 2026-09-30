"""Persist target-price anomalies independently of the latest run's metrics."""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import pandas as pd


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
    §S22 A-04: a pending name absent from a non-empty observation set (left the
    universe, no TG/price overlap) cannot reach the book and is released; its
    baseline is kept, so a return on the same abnormal basis is flagged again.
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
    if now:
        pending = {ticker: event for ticker, event in pending.items() if ticker in now}
    state = {"schema_version": 1, "baseline": baseline, "pending": pending}
    _save(path, state)
    currency["tg_px_ratio_jump_vs_prev"] = pending
    currency["tg_basis_guard_state"] = state
    currency["tg_basis_guard_ok"] = bool(valid and not pending and not suspect)
    return pending


# §S24.4 (b) — registered-event consistency (decision log §S24.4). A registered
# tg_basis_events factor is only right while the VENDOR's raw TG/price ratio
# still shows the matching step around the event date. When the vendor moves to
# the other basis (RTX/T on the 2026-09-29 pull, §S23.5/§S23.6) the step
# disappears and the factor silently double-corrects; the reverse (a step that
# re-appears for a de-registered name) is the §S18.1 suspect/jump guard's job.
EVENT_WINDOW = 60   # business rows on each side of the event
EVENT_GAP = 5       # rows skipped next to the event (announcement-day noise)
EVENT_MIN_OBS = 20  # minimum valid rows per side, else "insufficient" -> None
EVENT_TOL_LOG = 0.20  # |log(step * factor)| bar (DELL/DHR measure 0.013/0.018; RTX/T post-switch 0.37/0.48)


def event_consistency(ratio, events, window=EVENT_WINDOW, gap=EVENT_GAP,
                      min_obs=EVENT_MIN_OBS, tol_log=EVENT_TOL_LOG):
    """Compare each registered (ticker, date, factor) with the raw TG/price ratio step.

    ratio: DataFrame (date x ticker) of RAW target price / nominal price (events NOT applied).
    Returns {"params", "events": [...], "n_events", "n_inconsistent", "n_insufficient", "ok"} where
    ok is True (all registered events consistent, or none registered), False (any inconsistent),
    None (any event that could not be judged: insufficient rows, invalid date/factor).
    """
    rows = []
    if not isinstance(events, dict):
        events = {}
    for ticker in sorted(events):
        per_date = events[ticker]
        if not isinstance(per_date, dict):
            rows.append({"ticker": str(ticker), "date": None, "factor": None, "status": "invalid_date"})
            continue
        for date_str in sorted(per_date, key=str):
            rec = {"ticker": str(ticker), "date": str(date_str), "factor": _positive(per_date[date_str])}
            if rec["factor"] is None:
                rec["status"] = "invalid_factor"
                rows.append(rec)
                continue
            try:
                event = pd.Timestamp(date_str)
                if pd.isna(event):
                    raise ValueError(date_str)
            except (TypeError, ValueError):
                rec["status"] = "invalid_date"
                rows.append(rec)
                continue
            if ratio is None or str(ticker) not in getattr(ratio, "columns", []):
                rec["status"] = "not_in_universe"
                rows.append(rec)
                continue
            series = ratio[str(ticker)]
            pos = int(series.index.searchsorted(event))
            before = series.iloc[max(0, pos - gap - window): max(0, pos - gap)].dropna()
            after = series.iloc[pos + gap: pos + gap + window].dropna()
            rec["n_before"], rec["n_after"] = int(len(before)), int(len(after))
            if len(before) < min_obs or len(after) < min_obs:
                rec["status"] = "insufficient"
                rows.append(rec)
                continue
            med_b, med_a = _positive(before.median()), _positive(after.median())
            if med_b is None or med_a is None:
                rec["status"] = "insufficient"
                rows.append(rec)
                continue
            step = med_b / med_a
            resid = math.log(step * rec["factor"])
            rec.update({"median_before": round(med_b, 6), "median_after": round(med_a, 6),
                        "raw_step": round(step, 6), "implied_factor": round(1.0 / step, 6),
                        "log_residual": round(resid, 6),
                        "status": "consistent" if abs(resid) <= tol_log else "inconsistent"})
            rows.append(rec)
    n_inconsistent = sum(1 for r in rows if r["status"] == "inconsistent")
    n_unjudged = sum(1 for r in rows if r["status"] in ("insufficient", "invalid_date", "invalid_factor"))
    ok = None if n_unjudged else n_inconsistent == 0
    return {"params": {"window": window, "gap": gap, "min_obs": min_obs, "tol_log": tol_log},
            "events": rows, "n_events": len(rows), "n_inconsistent": n_inconsistent,
            "n_insufficient": n_unjudged, "ok": ok}
