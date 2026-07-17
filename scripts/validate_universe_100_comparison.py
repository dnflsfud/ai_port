"""Integrity checks for the universe proposal comparison deliverables."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "universe_100_comparison"

results = json.loads((OUT / "universe_100_comparison_results.json").read_text(encoding="utf-8"))
notebook = json.loads((OUT / "universe_100_comparison.ipynb").read_text(encoding="utf-8"))
artifact_text = (OUT / "artifact.json").read_text(encoding="utf-8")
artifact = json.loads(artifact_text)

assert all(results["checks"].values())
assert len(results["consensus"]) == 18
assert len(results["hybrid_candidates"]) == 35
assert {row["ticker"] for row in results["hybrid_candidates"]} >= {"285A JP", "SNDK US"}
assert sum(row["final_names"] for row in results["region_comparison"] if row["proposal"] == "Hybrid recommendation") == 100
assert next(row for row in results["proposal_summaries"] if row["proposal"] == "Fable proposal")["final_us_share"] == 0.78
assert next(row for row in results["proposal_summaries"] if row["proposal"] == "Hybrid recommendation")["final_us_share"] == 0.73
assert len(results["history_gates"]) == 4
assert not any(
    output.get("output_type") == "error"
    for cell in notebook["cells"]
    for output in cell.get("outputs", [])
)

assert artifact["surface"] == "report"
assert artifact["manifest"]["title"] == "100종목 유니버스: 두 제안 비교와 하이브리드 추천"
assert artifact["snapshot"]["status"] == "ready"
assert len(artifact["snapshot"]["datasets"]["hybrid_candidates"]) == 35
assert len(artifact["manifest"]["charts"]) == 2
assert len(artifact["manifest"]["tables"]) == 6
assert artifact["manifest"]["blocks"][0]["body"].startswith("# 100종목 유니버스")
assert "\ufffd" not in artifact_text

with sqlite3.connect(OUT / "universe_100_comparison.sqlite") as connection:
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    expected = {
        "headline",
        "proposal_summaries",
        "region_comparison",
        "sector_comparison",
        "consensus",
        "hybrid_candidates",
        "candidate_comparison",
        "conflict_decisions",
        "selection_framework",
        "history_gates",
    }
    assert expected <= tables
    assert connection.execute("SELECT COUNT(*) FROM hybrid_candidates").fetchone()[0] == 35
    assert connection.execute("SELECT COUNT(*) FROM consensus").fetchone()[0] == 18
    assert connection.execute("SELECT COUNT(*) FROM history_gates").fetchone()[0] == 4

print("checks_all=True")
print(f"consensus={len(results['consensus'])}")
print(f"hybrid_candidates={len(results['hybrid_candidates'])}")
print(f"proposal_summaries={results['proposal_summaries']}")
print(f"history_gates={[row['ticker'] for row in results['history_gates']]}")
print(f"notebook_cells={len(notebook['cells'])}")
print(f"sqlite_tables={sorted(tables)}")
print(f"artifact_blocks={len(artifact['manifest']['blocks'])}")
