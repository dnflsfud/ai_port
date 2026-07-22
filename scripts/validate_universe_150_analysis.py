"""Independent integrity checks for the 100-to-150 recommendation artifacts."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "universe_150_recommendation"
SOURCE_XLSX = ROOT.parents[1] / "ai_signal_data.xlsx"

results = json.loads((OUT / "universe_150_results.json").read_text(encoding="utf-8"))
artifact = json.loads((OUT / "artifact.json").read_text(encoding="utf-8"))
notebook = json.loads((OUT / "universe_150_analysis.ipynb").read_text(encoding="utf-8"))
meta = pd.read_excel(SOURCE_XLSX, sheet_name="Universe_Meta")

candidates = pd.DataFrame(results["candidates"])
sector_mix = pd.DataFrame(results["sector_mix"])
region_mix = pd.DataFrame(results["region_mix"])

assert all(results["checks"].values())
assert len(meta) == 100 and meta["Status"].eq("Available").all()
assert len(candidates) == 50
assert candidates["simple_ticker"].nunique() == 50
assert not set(meta["Ticker"].str.split().str[0]) & set(candidates["simple_ticker"])
assert set(candidates["market"]) <= {"US", "GR", "FP", "JP", "SW", "LN"}
assert set(candidates["currency"]) == {"USD", "EUR", "JPY", "GBP", "CHF"}
assert int(sector_mix["final_names"].sum()) == 150
assert int(region_mix["final_names"].sum()) == 150
assert int(candidates["country"].eq("United States").sum()) == 30
assert int((~candidates["country"].eq("United States")).sum()) == 20
assert abs(results["summary"]["final_tech_share"] - 43 / 150) < 1e-12
assert abs(results["summary"]["final_non_us_share"] - 46 / 150) < 1e-12
assert {"EA US", "GE US", "IBE SM", "SAN FP / SAN SM"} == {
    row["ticker"] for row in results["excluded_current_refresh"]
}

assert artifact["surface"] == "report"
assert artifact["snapshot"]["status"] == "ready"
assert len(artifact["snapshot"]["datasets"]["candidates"]) == 50
assert len(artifact["snapshot"]["datasets"]["sector_mix_long"]) == 22
assert len(artifact["snapshot"]["datasets"]["region_mix_long"]) == 14
assert artifact["manifest"]["blocks"][0]["body"] == "# 유니버스 150종목 확장 추천"
assert any(block.get("type") == "chart" for block in artifact["manifest"]["blocks"])
assert {table["id"] for table in artifact["manifest"]["tables"]} == {
    "table-candidates",
    "table-currency",
}

assert notebook["nbformat"] == 4
assert not any(
    output.get("output_type") == "error"
    for cell in notebook["cells"]
    for output in cell.get("outputs", [])
)

with sqlite3.connect(OUT / "universe_150_analysis.sqlite") as connection:
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert {
        "summary",
        "sector_mix",
        "sector_mix_long",
        "region_mix",
        "region_mix_long",
        "country_mix",
        "currency_mix",
        "candidates",
    } <= tables
    assert connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 50
    assert connection.execute("SELECT SUM(final_names) FROM sector_mix").fetchone()[0] == 150

print("confidence=share_with_caveats")
print("methodology_checks=passed")
print("calculation_checks=passed")
print("ticker_collision_checks=passed")
print("artifact_structure_checks=passed")
print("notebook_execution_checks=passed")
print("remaining_gap=new_candidate_workbook_coverage_not_loaded")
print(f"candidate_count={len(candidates)}")
print(f"final_tech_share={results['summary']['final_tech_share']:.4f}")
print(f"final_non_us_share={results['summary']['final_non_us_share']:.4f}")
