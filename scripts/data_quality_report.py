#!/usr/bin/env python
"""Data-quality / coverage report (review F + G) — read-only, no backtest.

Quantifies the two residual-risk warnings codex flagged so performance numbers
can be trusted on a documented coverage basis:

  F  per-sheet date coverage + the all-sheets date intersection vs the longest
     sheet (the "intersection is 66% of longest" warning) + the PX_LAST tail
     ffill span. Loads the raw sheets only (seconds); runs NO walk-forward.
  G  per-period (yearly) frequency of degenerate-model fold reuse, parsed from a
     walk-forward log (default logs/stage0.log): each "재훈련 @ DATE (... trees:N)"
     line and whether it was followed by a "Degenerate model" reuse warning.

Run FROM the project root (ai_port), engine vendored under ./src:
    PYTHONPATH=. <PY> scripts/data_quality_report.py            # uses logs/stage0.log
    PYTHONPATH=. <PY> scripts/data_quality_report.py logs/stage3.log
Writes outputs/data_quality_report.json and prints a summary.
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import re
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


def _coverage() -> dict:
    """F: per-sheet date coverage + DATE-indexed intersection vs longest sheet.

    Only date-indexed data sheets count toward the intersection — meta sheets
    (Factor_Meta/Universe_Meta/Summary_Stats: factor- or ticker-indexed) are
    listed separately, matching the engine's align_dates (which intersects the
    processed date sheets only).
    """
    from src.config import DEFAULT_CONFIG
    from src.data_loader import load_all_sheets

    raw = load_all_sheets(DEFAULT_CONFIG.data_path)
    per_sheet, non_date, common_idx, max_len = {}, [], None, 0
    for name, df in raw.items():
        dt = pd.to_datetime(pd.Index(df.index), errors="coerce")
        if len(dt) < 100 or float(dt.notna().mean()) < 0.9:   # not a date sheet
            non_date.append({"sheet": name, "n_rows": int(len(df.index)),
                             "n_cols": int(df.shape[1])})
            continue
        didx = pd.DatetimeIndex(dt[dt.notna()]).unique()
        per_sheet[name] = {
            "first": str(didx.min())[:10], "last": str(didx.max())[:10],
            "n_dates": int(len(didx)), "n_cols": int(df.shape[1]),
            "nan_frac": round(float(df.isna().to_numpy().mean()), 4) if df.size else None,
        }
        common_idx = didx if common_idx is None else common_idx.intersection(didx)
        max_len = max(max_len, len(didx))
    n_common = int(len(common_idx)) if common_idx is not None else 0
    # Date sheets whose history is short enough to drive the intersection down.
    shortest = sorted(per_sheet.items(), key=lambda kv: kv[1]["n_dates"])[:6]
    return {
        "n_date_sheets": len(per_sheet),
        "n_non_date_sheets": len(non_date),
        "intersection_dates": n_common,
        "longest_sheet_dates": int(max_len),
        "intersection_pct_of_longest": round(n_common / max_len * 100, 1) if max_len else None,
        "shortest_date_sheets": [{"sheet": k, **v} for k, v in shortest],
        "non_date_sheets": non_date,
        "per_sheet": per_sheet,
    }


_COV = re.compile(r"Date intersection \((\d+)\) is (\d+)% of longest sheet \((\d+)\)")
_TAIL = re.compile(r"Extending (\d+) tail dates beyond intersection \((\S+) -> (\S+)\)")


def _engine_logged_coverage(log_path: Path) -> dict:
    """The engine's own authoritative coverage line, parsed from a stage log."""
    if not log_path.exists():
        return {}
    txt = log_path.read_text(encoding="utf-8", errors="replace")
    out = {}
    m = _COV.search(txt)
    if m:
        out.update(intersection_dates=int(m.group(1)), pct_of_longest=int(m.group(2)),
                   longest_dates=int(m.group(3)))
    t = _TAIL.search(txt)
    if t:
        out.update(tail_ffill_days=int(t.group(1)), tail_from=t.group(2), tail_to=t.group(3))
    return out


_RETRAIN = re.compile(r"재훈련 @ (\d{4})-(\d{2})-(\d{2}).*trees:\s*(\d+)")
_DEGEN = re.compile(r"Degenerate model \((\d+) trees\)")


def _degenerate(log_path: Path) -> dict:
    """G: per-year frequency of degenerate-model fold reuse, parsed from a log."""
    if not log_path.exists():
        return {"status": f"log not found: {log_path}"}
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    total_by_year, degen_by_year, tree_counts = Counter(), Counter(), Counter()
    last_year = None
    for ln in lines:
        m = _RETRAIN.search(ln)
        if m:
            last_year = m.group(1)
            total_by_year[last_year] += 1
            continue
        d = _DEGEN.search(ln)
        if d and last_year is not None:
            degen_by_year[last_year] += 1
            tree_counts[int(d.group(1))] += 1
    total, degen = sum(total_by_year.values()), sum(degen_by_year.values())
    years = sorted(total_by_year)
    return {
        "log": str(log_path.name),
        "total_retrains": total,
        "degenerate_retrains": degen,
        "degenerate_rate_pct": round(degen / total * 100, 1) if total else None,
        "by_year": {y: {"retrains": total_by_year[y], "degenerate": degen_by_year.get(y, 0)}
                    for y in years},
        "degenerate_tree_count_dist": dict(sorted(tree_counts.items())),
    }


def main() -> int:
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "logs" / "stage0.log"
    print("[dq] loading raw sheets for coverage (no backtest)…")
    cov = _coverage()
    cov["engine_logged"] = _engine_logged_coverage(log_path)
    deg = _degenerate(log_path)
    report = {"coverage": cov, "degenerate_models": deg}

    out_dir = Path("outputs"); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data_quality_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")

    el = cov.get("engine_logged") or {}
    print(f"\n=== F. coverage ===")
    print(f"  date sheets={cov['n_date_sheets']} (+{cov['n_non_date_sheets']} non-date/meta)")
    print(f"  recomputed intersection={cov['intersection_dates']} "
          f"({cov['intersection_pct_of_longest']}% of longest {cov['longest_sheet_dates']})")
    if el:
        print(f"  engine-logged: intersection={el.get('intersection_dates')} "
              f"({el.get('pct_of_longest')}% of {el.get('longest_dates')}); "
              f"tail ffill {el.get('tail_ffill_days')}d ({el.get('tail_from')} -> {el.get('tail_to')})")
    print(f"  shortest date sheets (drive intersection down):")
    for s in cov["shortest_date_sheets"]:
        print(f"    {s['sheet'][:28]:28} {s['first']}..{s['last']}  n={s['n_dates']} nan={s['nan_frac']}")
    print(f"\n=== G. degenerate models ({deg.get('log')}) ===")
    if "status" in deg:
        print(f"  {deg['status']}")
    else:
        print(f"  total_retrains={deg['total_retrains']}  degenerate={deg['degenerate_retrains']} "
              f"({deg['degenerate_rate_pct']}%)")
        print(f"  by year: " + "  ".join(
            f"{y}:{v['degenerate']}/{v['retrains']}" for y, v in deg["by_year"].items()))
        print(f"  degenerate tree-count dist: {deg['degenerate_tree_count_dist']}")
    print(f"\n[dq] wrote {out_dir / 'data_quality_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
