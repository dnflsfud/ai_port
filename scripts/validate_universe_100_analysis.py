"""Compact integrity checks for the generated global universe artifacts."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "universe_100_recommendation"

results = json.loads((OUT / "universe_100_results.json").read_text(encoding="utf-8"))
notebook = json.loads((OUT / "universe_100_analysis.ipynb").read_text(encoding="utf-8"))
artifact_text = (OUT / "artifact.json").read_text(encoding="utf-8")
artifact = json.loads(artifact_text)
report_text = (OUT / "universe_100_report.html").read_text(encoding="utf-8")

assert all(results["checks"].values())
assert len(results["candidates"]) == 35
assert sum(row["final_names"] for row in results["proposed_country"]) == 100
assert {row["ticker"] for row in results["candidates"]} >= {"285A JP", "SNDK US"}
assert not any(
    output.get("output_type") == "error"
    for cell in notebook["cells"]
    for output in cell.get("outputs", [])
)
assert artifact["surface"] == "report"
assert artifact["snapshot"]["status"] == "ready"
assert len(artifact["snapshot"]["datasets"]["candidates"]) == 35
assert len(artifact["snapshot"]["datasets"]["region_mix_long"]) == 8
assert {block["id"] for block in artifact["manifest"]["blocks"]} >= {
    "title",
    "executive-summary",
    "current-story",
    "current-chart",
    "proposed-story",
    "proposed-chart",
    "geo-story",
    "geo-chart",
    "country-table",
    "candidate-story",
    "candidate-table",
    "history-gate",
    "next-steps",
    "questions",
    "caveats",
}
assert {chart["id"] for chart in artifact["manifest"]["charts"]} >= {
    "chart-current",
    "chart-proposed",
    "chart-geo",
}
assert {table["id"] for table in artifact["manifest"]["tables"]} >= {
    "table-candidates",
    "table-country",
}
assert artifact["snapshot"]["datasets"]["summary"][0]["final_non_us_share"] == 0.31
assert {row["ticker"] for row in artifact["snapshot"]["datasets"]["candidates"]} == {
    row["ticker"] for row in results["candidates"]
}
assert "country" in artifact["snapshot"]["datasets"]["candidates"][0]
assert "Kioxia Holdings" in report_text
assert "ASML Holding" in report_text
assert "LSEG LN" in report_text
assert len(report_text) > len(artifact_text)
assert "\ufffd" not in artifact_text + report_text

with sqlite3.connect(OUT / "universe_100_analysis.sqlite") as connection:
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert {
        "summary",
        "current_sector_long",
        "proposed_sector_long",
        "proposed_country",
        "region_mix_long",
        "candidates",
    } <= tables
    candidate_count = connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    region_rows = connection.execute(
        "SELECT region, metric, value, current_names, final_names "
        "FROM region_mix_long ORDER BY region, metric"
    ).fetchall()
    assert candidate_count == 35

print("checks_all=True")
print(f"candidate_count={len(results['candidates'])}")
print(f"summary={results['summary']}")
print(f"region_mix={results['region_mix']}")
print(f"countries={[(row['country'], row['final_names']) for row in results['proposed_country']]}")
print(f"notebook_cells={len(notebook['cells'])}")
print(f"sqlite_tables={sorted(tables)}")
print(f"region_rows={region_rows}")
print(f"artifact_blocks={len(artifact['manifest']['blocks'])}")
print(f"artifact_charts={len(artifact['manifest']['charts'])}")
print(f"artifact_tables={len(artifact['manifest']['tables'])}")
print(f"report_bytes={len(report_text.encode('utf-8'))}")
