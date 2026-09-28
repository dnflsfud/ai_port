"""A-04: a pending TG-basis anomaly whose ticker no longer appears in the current
observations (universe swap / column dropped) is never cleared, keeps
tg_basis_guard_ok False (-> HOLD), and cannot be acknowledged through the
sanctioned script (it requires a current median equal to the pending value).

Real functions: src.tg_basis_guard.annotate_basis_guard and
scripts/acknowledge_tg_basis.acknowledge (imported by path).
Run from the repo dir:  PYTHONPATH=. <PY> <this file>
"""
import importlib.util
import json
import tempfile
from pathlib import Path

from src.tg_basis_guard import annotate_basis_guard

spec = importlib.util.spec_from_file_location("ack", "scripts/acknowledge_tg_basis.py")
ack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ack)

with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as d:
    Path(d, "tg_basis_state.json").write_text(json.dumps({
        "schema_version": 1,
        "baseline": {"AAA": 1.10},
        "pending": {"OLD": {"previous": 1.10, "now": 1.60}},
    }))
    for run in (1, 2):
        dq = {"currency": {"tg_px_ratio_median": {"AAA": 1.10, "NEW": 1.20},
                           "tg_px_ratio_suspect": {}}}
        pending = annotate_basis_guard(d, dq, 0.25)
        print(f"run {run}: pending={sorted(pending)} tg_basis_guard_ok={dq['currency']['tg_basis_guard_ok']}")
        Path(d, "metrics.json").write_text(json.dumps({"data_quality": dq}))
    try:
        ack.acknowledge(d, "OLD", 1.60, "ticker removed from the universe")
        print("acknowledge -> cleared")
    except ValueError as exc:
        print("acknowledge -> ValueError:", exc)
