"""Record an explicitly reviewed vendor basis change for the NEXT run.

Example:
    python scripts/acknowledge_tg_basis.py --run-dir outputs/codex_causal_rank_65 \
        --ticker AAA --expected-ratio 1.5 --reason "Verified vendor adjustment"

This never changes historical metrics or makes an existing bundle PRODUCTION.
The next complete run must still pass every normal operating check.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tg_basis_guard import _read, _save


def acknowledge(run_dir, ticker, expected_ratio, reason):
    if not reason.strip() or not math.isfinite(expected_ratio) or expected_ratio <= 0:
        raise ValueError("A review reason and a positive expected ratio are required")
    run_dir = Path(run_dir)
    state_path = run_dir / "tg_basis_state.json"
    state = _read(state_path)
    metrics = _read(run_dir / "metrics.json")
    currency = (metrics.get("data_quality") or {}).get("currency") or {}
    pending = state.get("pending", {}).get(ticker)
    actual = (currency.get("tg_px_ratio_median") or {}).get(ticker)
    if not isinstance(pending, dict) or actual != expected_ratio or pending.get("now") != expected_ratio:
        raise ValueError("Expected ratio must match this ticker's latest unresolved anomaly")
    if ticker in (currency.get("tg_px_ratio_suspect") or {}):
        raise ValueError("The absolute ratio guard is still breached; correct the source data first")
    history_path = run_dir / "tg_basis_acknowledgements.json"
    history = _read(history_path) if history_path.exists() else {"reviews": []}
    history["reviews"].append({
        "ticker": ticker, "previous": pending.get("previous"), "accepted": expected_ratio,
        "reason": reason.strip(), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    })
    state["baseline"][ticker] = expected_ratio
    del state["pending"][ticker]
    _save(history_path, history)
    _save(state_path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--expected-ratio", type=float, required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    acknowledge(args.run_dir, args.ticker, args.expected_ratio, args.reason)
    print("Review recorded. Re-run the portfolio to evaluate the new basis; existing bundle is unchanged.")


if __name__ == "__main__":
    main()
