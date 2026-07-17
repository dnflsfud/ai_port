"""Build the reproducible comparison of two 35-name universe expansion proposals.

The analysis uses the previously reviewed 65-name universe summary and prior
35-name recommendation as its local source of truth.  It compares that proposal
with the user-provided Fable proposal, constructs a 35-name hybrid, and writes
reviewable JSON, SQLite, and an executed nbformat-v4 notebook.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sqlite3
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIOR_DIR = ROOT / "outputs" / "universe_100_recommendation"
OUT_DIR = ROOT / "outputs" / "universe_100_comparison"
RESULTS_PATH = OUT_DIR / "universe_100_comparison_results.json"
SQLITE_PATH = OUT_DIR / "universe_100_comparison.sqlite"
NOTEBOOK_PATH = OUT_DIR / "universe_100_comparison.ipynb"

REGION_ORDER = ["United States", "United Kingdom", "Europe ex-UK", "Asia"]
SECTOR_ORDER = [
    "Technology",
    "Industrials",
    "Financials",
    "Healthcare",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Utilities",
    "Real Estate",
    "Communication Services",
    "Materials",
]


def candidate(
    ticker: str,
    name: str,
    sector: str,
    country: str,
    region: str,
    subindustry: str,
    rationale: str,
    listing_date: str | None = None,
    history_gate: bool = False,
) -> dict:
    return {
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "country": country,
        "region": region,
        "subindustry": subindustry,
        "rationale": rationale,
        "listing_date": listing_date,
        "history_gate": history_gate,
    }


FABLE_CANDIDATES = [
    candidate("285A JP", "Kioxia Holdings", "Technology", "Japan", "Asia", "NAND flash / SSD", "Mandatory memory-cycle exposure", "2024-12-18", True),
    candidate("SNDK US", "Sandisk", "Technology", "United States", "United States", "Flash memory / SSD", "Mandatory memory-cycle exposure", "2025-02-24", True),
    candidate("ASML NA", "ASML Holding", "Technology", "Netherlands", "Europe ex-UK", "Lithography equipment", "Critical semiconductor equipment exposure"),
    candidate("SAP GR", "SAP", "Technology", "Germany", "Europe ex-UK", "Enterprise software", "European enterprise software anchor"),
    candidate("ARM US", "Arm Holdings", "Technology", "United Kingdom", "United Kingdom", "CPU architecture IP", "Adds CPU-IP economics and UK domicile", "2023-09-14", True),
    candidate("ANET US", "Arista Networks", "Technology", "United States", "United States", "Data-center networking", "AI Ethernet peer exposure"),
    candidate("KLAC US", "KLA", "Technology", "United States", "United States", "Process control equipment", "Inspection and metrology coverage"),
    candidate("NOW US", "ServiceNow", "Technology", "United States", "United States", "Enterprise workflow software", "Enterprise AI software exposure"),
    candidate("STX US", "Seagate Technology", "Technology", "United States", "United States", "Hard-disk drives", "Capacity-storage cycle exposure"),
    candidate("6857 JP", "Advantest", "Technology", "Japan", "Asia", "Semiconductor test systems", "AI accelerator test peer for Teradyne"),
    candidate("SU FP", "Schneider Electric", "Industrials", "France", "Europe ex-UK", "Electrification / automation", "Data-center power and automation peer"),
    candidate("SIE GR", "Siemens", "Industrials", "Germany", "Europe ex-UK", "Industrial automation", "Industrial AI, automation and grid exposure"),
    candidate("RHM GR", "Rheinmetall", "Industrials", "Germany", "Europe ex-UK", "Defense systems", "European rearmament exposure"),
    candidate("RR/ LN", "Rolls-Royce Holdings", "Industrials", "United Kingdom", "United Kingdom", "Aerospace engines", "Civil aerospace aftermarket and SMR optionality"),
    candidate("PWR US", "Quanta Services", "Industrials", "United States", "United States", "Grid infrastructure", "Grid and data-center construction exposure"),
    candidate("BX US", "Blackstone", "Financials", "United States", "United States", "Alternative asset management", "Private-markets exposure"),
    candidate("MS US", "Morgan Stanley", "Financials", "United States", "United States", "Investment banking / wealth", "Capital-markets and wealth exposure"),
    candidate("PGR US", "Progressive", "Financials", "United States", "United States", "Property and casualty insurance", "Fills insurance model gap"),
    candidate("HSBA LN", "HSBC Holdings", "Financials", "United Kingdom", "United Kingdom", "Global banking", "Asia and UK banking exposure"),
    candidate("NOVOB DC", "Novo Nordisk", "Healthcare", "Denmark", "Europe ex-UK", "Diabetes / obesity pharmaceuticals", "GLP-1 peer for Eli Lilly"),
    candidate("AZN LN", "AstraZeneca", "Healthcare", "United Kingdom", "United Kingdom", "Pharmaceuticals", "Oncology-led global pharma exposure"),
    candidate("TMO US", "Thermo Fisher Scientific", "Healthcare", "United States", "United States", "Life-science tools", "Life-science tools exposure"),
    candidate("BSX US", "Boston Scientific", "Healthcare", "United States", "United States", "Medical devices", "Adds medical-device growth exposure"),
    candidate("BKNG US", "Booking Holdings", "Consumer Discretionary", "United States", "United States", "Online travel", "Global travel platform exposure"),
    candidate("MC FP", "LVMH", "Consumer Discretionary", "France", "Europe ex-UK", "Luxury goods", "European luxury anchor"),
    candidate("TJX US", "TJX Companies", "Consumer Discretionary", "United States", "United States", "Off-price retail", "Defensive discretionary exposure"),
    candidate("PM US", "Philip Morris International", "Consumer Staples", "United States", "United States", "Tobacco / smoke-free products", "Distinct smoke-free transition exposure"),
    candidate("NESN SW", "Nestle", "Consumer Staples", "Switzerland", "Europe ex-UK", "Packaged food / beverages", "European staples anchor"),
    candidate("SHEL LN", "Shell", "Energy", "United Kingdom", "United Kingdom", "Integrated energy / LNG", "Global LNG and integrated energy exposure"),
    candidate("WMB US", "Williams Companies", "Energy", "United States", "United States", "Natural-gas midstream", "Gas infrastructure for power-demand growth"),
    candidate("CEG US", "Constellation Energy", "Utilities", "United States", "United States", "Nuclear generation", "Nuclear power exposure for data-center demand", "2022-02-02", True),
    candidate("VST US", "Vistra", "Utilities", "United States", "United States", "Competitive power generation", "Merchant-power and data-center PPA exposure"),
    candidate("DLR US", "Digital Realty", "Real Estate", "United States", "United States", "Data-center REIT", "Global data-center property exposure"),
    candidate("SPOT US", "Spotify Technology", "Communication Services", "Sweden", "Europe ex-UK", "Audio streaming", "Global subscription audio platform"),
    candidate("RIO LN", "Rio Tinto", "Materials", "United Kingdom", "United Kingdom", "Diversified mining", "Copper and grid-material exposure"),
]


HYBRID_TICKERS = [
    "285A JP", "SNDK US", "ASML NA", "SAP GR", "KLAC US", "ARM US", "ANET US", "CDNS US", "STM FP", "6857 JP",
    "SU FP", "SIE GR", "RR/ LN", "PWR US",
    "HSBA LN", "ALV GR", "PGR US", "LSEG LN",
    "NOVOB DC", "AZN LN", "TMO US", "BSX US", "ROG SW",
    "BKNG US", "MC FP", "TJX US",
    "PM US", "NESN SW",
    "SHEL LN", "WMB US",
    "CEG US", "IBE SM",
    "VNA GR", "SPOT US", "RIO LN",
]


SELECTION_NOTES = {
    "ARM US": "Choose over another mature semiconductor name: CPU-IP economics and UK domicile add a different factor, subject to a history gate.",
    "ANET US": "Choose as the AI-networking systems peer; more differentiated from the semiconductor-heavy current universe than another chip vendor.",
    "6857 JP": "Choose for a clean cross-sectional test-equipment pair with existing Teradyne and for a second Japanese addition.",
    "CDNS US": "Keep because EDA is a bottleneck business and less duplicative than adding another enterprise workflow vendor.",
    "STM FP": "Keep as a European analog/auto semiconductor diversifier; avoid a third memory/storage-cycle name after Kioxia and Sandisk.",
    "PGR US": "Choose to fill the current insurance gap rather than add another bank or asset manager.",
    "ALV GR": "Keep a European insurance and asset-management compounder to diversify US financial factors.",
    "LSEG LN": "Keep market-data and exchange infrastructure, which is less correlated with bank balance-sheet exposure.",
    "BSX US": "Choose medical devices to broaden healthcare beyond pharmaceuticals and life-science tools.",
    "ROG SW": "Keep pharma plus diagnostics; use one Swiss defensive anchor instead of adding two large European drug makers.",
    "TJX US": "Choose off-price retail over a second luxury name because LVMH already covers luxury demand.",
    "PM US": "Choose for a differentiated smoke-free transition, with an explicit regulatory-risk flag.",
    "WMB US": "Choose midstream gas infrastructure over another integrated oil major already represented by current energy holdings.",
    "CEG US": "Choose one US nuclear/merchant-power beneficiary, subject to a full-history gate.",
    "IBE SM": "Pair CEG with a regulated European networks/renewables utility instead of doubling US merchant-power exposure.",
    "VNA GR": "Choose European residential real estate and rate sensitivity; existing Equinix already covers data-center real estate.",
    "SPOT US": "Choose a global subscription platform; avoid Deutsche Telekom look-through duplication with existing T-Mobile US.",
}


CONFLICT_DECISIONS = [
    {"area": "Technology – software", "selected": "CDNS", "not_selected": "NOW", "decision": "EDA adds a scarcer supply-chain bottleneck; NOW overlaps current enterprise software exposure."},
    {"area": "Technology – storage", "selected": "STM", "not_selected": "STX", "decision": "Kioxia and Sandisk already create a large memory/storage cluster; STM adds European analog/auto exposure."},
    {"area": "Technology – peers", "selected": "ARM, ANET, Advantest", "not_selected": "QCOM, MRVL, DELL", "decision": "The selected trio improves CPU-IP, networking-system and test-equipment peer coverage with less semiconductor/OEM repetition."},
    {"area": "Industrials", "selected": "Schneider, Siemens, Rolls-Royce, Quanta", "not_selected": "Rheinmetall", "decision": "Current Lockheed already supplies defense beta; keep four industrial slots and use the released slot in healthcare."},
    {"area": "Financials", "selected": "HSBC, Allianz, PGR, LSEG", "not_selected": "BX, MS, UBS", "decision": "Bank, two insurance-related models and market infrastructure diversify better than more asset-management and investment-bank exposure."},
    {"area": "Healthcare", "selected": "NVO, AZN, TMO, BSX, Roche", "not_selected": "Novartis", "decision": "Five slots cover obesity, oncology, tools, devices and diagnostics; a second Swiss pharma is redundant."},
    {"area": "Consumer discretionary", "selected": "TJX", "not_selected": "Ferrari", "decision": "TJX adds defensive retail while LVMH already represents the luxury factor."},
    {"area": "Consumer staples", "selected": "PM", "not_selected": "Unilever", "decision": "PM adds a distinct business-model transition; carry a regulatory-risk flag rather than add another broad staples conglomerate."},
    {"area": "Energy", "selected": "WMB", "not_selected": "TotalEnergies", "decision": "WMB adds midstream gas infrastructure; the current universe already has integrated energy and LNG exposure."},
    {"area": "Utilities", "selected": "CEG + Iberdrola", "not_selected": "CEG + VST", "decision": "One US nuclear beneficiary plus one European regulated utility is more balanced than two US merchant-power trades."},
    {"area": "Real estate", "selected": "Vonovia", "not_selected": "Digital Realty", "decision": "Existing Equinix already covers data-center REITs; Vonovia contributes European residential and rates exposure."},
    {"area": "Communication services", "selected": "Spotify", "not_selected": "Deutsche Telekom, Vodafone", "decision": "Spotify adds subscription media; Deutsche Telekom duplicates existing T-Mobile US at the controlling-shareholder level."},
]


SELECTION_FRAMEWORK = [
    {"criterion": "Marginal diversification vs. current 65", "weight": 0.30, "test": "Country, sector, business model and current look-through overlap"},
    {"criterion": "Data readiness", "weight": 0.20, "test": "Price/fundamental coverage, FX, calendar and 1,260-session train window"},
    {"criterion": "Cross-sectional peer usefulness", "weight": 0.20, "test": "Creates a comparable pair or fills a model peer-group gap"},
    {"criterion": "Liquidity and quality", "weight": 0.15, "test": "Tradability, accounting continuity and corporate-action cleanliness"},
    {"criterion": "Theme / ownership overlap penalty", "weight": 0.15, "test": "Penalize repeated AI, memory, merchant-power or controlled-subsidiary exposure"},
]


def _round(value: float) -> float:
    return round(value, 10)


def _proposal_summary(name: str, candidates: list[dict], current_regions: dict[str, int]) -> tuple[dict, list[dict]]:
    additions = Counter(row["region"] for row in candidates)
    final = {region: current_regions.get(region, 0) + additions.get(region, 0) for region in REGION_ORDER}
    summary = {
        "proposal": name,
        "add_names": len(candidates),
        "us_add_names": additions["United States"],
        "non_us_add_names": len(candidates) - additions["United States"],
        "final_us_share": _round(final["United States"] / 100),
        "final_non_us_share": _round(1 - final["United States"] / 100),
        "final_uk_share": _round(final["United Kingdom"] / 100),
        "final_europe_ex_uk_share": _round(final["Europe ex-UK"] / 100),
        "final_asia_share": _round(final["Asia"] / 100),
    }
    rows = []
    for region in REGION_ORDER:
        rows.append({
            "proposal": name,
            "region": region,
            "current_names": current_regions.get(region, 0),
            "add_names": additions.get(region, 0),
            "final_names": final[region],
            "final_share": _round(final[region] / 100),
        })
    return summary, rows


def _write_table(connection: sqlite3.Connection, table_name: str, rows: list[dict]) -> None:
    connection.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    if not rows:
        connection.execute(f'CREATE TABLE "{table_name}" (empty TEXT)')
        return
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    types = {}
    for column in columns:
        values = [row.get(column) for row in rows if row.get(column) is not None]
        if values and all(isinstance(value, (bool, int)) for value in values):
            types[column] = "INTEGER"
        elif values and all(isinstance(value, (bool, int, float)) for value in values):
            types[column] = "REAL"
        else:
            types[column] = "TEXT"
    ddl = ", ".join(f'"{column}" {types[column]}' for column in columns)
    connection.execute(f'CREATE TABLE "{table_name}" ({ddl})')
    placeholders = ", ".join("?" for _ in columns)
    quoted = ", ".join(f'"{column}"' for column in columns)
    payload = []
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, bool):
                value = int(value)
            elif isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            values.append(value)
        payload.append(values)
    connection.executemany(f'INSERT INTO "{table_name}" ({quoted}) VALUES ({placeholders})', payload)


def build_results(write_outputs: bool = True) -> dict:
    prior = json.loads((PRIOR_DIR / "universe_100_results.json").read_text(encoding="utf-8"))
    prior_candidates = prior["candidates"]
    prior_map = {row["ticker"]: dict(row) for row in prior_candidates}
    fable_map = {row["ticker"]: dict(row) for row in FABLE_CANDIDATES}
    combined = {**prior_map, **fable_map}

    prior_set = set(prior_map)
    fable_set = set(fable_map)
    consensus_tickers = sorted(prior_set & fable_set)
    prior_only_tickers = sorted(prior_set - fable_set)
    fable_only_tickers = sorted(fable_set - prior_set)

    hybrid = []
    for order, ticker in enumerate(HYBRID_TICKERS, start=1):
        row = dict(combined[ticker])
        row["selection_order"] = order
        if ticker in prior_set and ticker in fable_set:
            row["source_status"] = "Consensus"
            row["selection_note"] = "Retain as common core in both proposals."
        elif ticker in fable_set:
            row["source_status"] = "Fable-only"
            row["selection_note"] = SELECTION_NOTES[ticker]
        else:
            row["source_status"] = "Prior-only"
            row["selection_note"] = SELECTION_NOTES[ticker]
        hybrid.append(row)

    current_regions = {row["region"]: row["current_names"] for row in prior["region_mix"]}
    proposal_summaries = []
    region_comparison = []
    for name, proposal in (("Prior proposal", prior_candidates), ("Fable proposal", FABLE_CANDIDATES), ("Hybrid recommendation", hybrid)):
        summary, region_rows = _proposal_summary(name, proposal, current_regions)
        proposal_summaries.append(summary)
        region_comparison.extend(region_rows)

    current_sector_counts = {row["Sector"]: row["current_names"] for row in prior["current_sector"]}
    sector_comparison = []
    for proposal_name, proposal in (("Prior proposal", prior_candidates), ("Fable proposal", FABLE_CANDIDATES), ("Hybrid recommendation", hybrid)):
        additions = Counter(row["sector"] for row in proposal)
        for sector in SECTOR_ORDER:
            sector_comparison.append({
                "proposal": proposal_name,
                "sector": sector,
                "current_names": current_sector_counts.get(sector, 0),
                "add_names": additions.get(sector, 0),
                "final_names": current_sector_counts.get(sector, 0) + additions.get(sector, 0),
            })

    consensus = []
    for ticker in consensus_tickers:
        row = combined[ticker]
        consensus.append({"ticker": ticker, "name": row["name"], "sector": row["sector"], "region": row["region"]})

    candidate_comparison = []
    for proposal_name, proposal in (("Prior proposal", prior_candidates), ("Fable proposal", FABLE_CANDIDATES), ("Hybrid recommendation", hybrid)):
        for row in proposal:
            candidate_comparison.append({
                "proposal": proposal_name,
                "ticker": row["ticker"],
                "name": row["name"],
                "sector": row["sector"],
                "country": row["country"],
                "region": row["region"],
                "history_gate": bool(row.get("history_gate", False)),
            })

    history_gates = [
        {
            "ticker": row["ticker"],
            "name": row["name"],
            "listing_date": row["listing_date"],
            "reason": "Less than the configured 1,260-session training window as of 2026-07-15; mask before listing and keep trade-ineligible until the minimum-history rule passes.",
        }
        for row in hybrid
        if row.get("history_gate")
    ]

    checks = {
        "prior_has_35": len(prior_candidates) == 35,
        "fable_has_35": len(FABLE_CANDIDATES) == 35,
        "hybrid_has_35": len(hybrid) == 35,
        "hybrid_unique": len(set(HYBRID_TICKERS)) == 35,
        "mandatory_names_present": {"285A JP", "SNDK US"} <= set(HYBRID_TICKERS),
        "consensus_is_18": len(consensus_tickers) == 18,
        "hybrid_final_is_100": sum(row["final_names"] for row in sector_comparison if row["proposal"] == "Hybrid recommendation") == 100,
        "hybrid_region_is_100": sum(row["final_names"] for row in region_comparison if row["proposal"] == "Hybrid recommendation") == 100,
        "hybrid_us_is_73": next(row for row in proposal_summaries if row["proposal"] == "Hybrid recommendation")["final_us_share"] == 0.73,
        "history_gates_are_4": len(history_gates) == 4,
    }

    headline = {
        "consensus_names": len(consensus_tickers),
        "prior_final_us_share": 0.69,
        "fable_final_us_share": 0.78,
        "hybrid_final_us_share": 0.73,
        "hybrid_non_us_share": 0.27,
        "hybrid_history_gates": len(history_gates),
    }

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_as_of": "2026-07-15",
        "source": {
            "prior_results": str(PRIOR_DIR / "universe_100_results.json"),
            "current_universe_names": 65,
            "current_region_counts": current_regions,
            "model_train_window_sessions": 1260,
        },
        "headline": headline,
        "proposal_summaries": proposal_summaries,
        "region_comparison": region_comparison,
        "sector_comparison": sector_comparison,
        "consensus": consensus,
        "prior_only": [{"ticker": ticker, "name": prior_map[ticker]["name"], "sector": prior_map[ticker]["sector"]} for ticker in prior_only_tickers],
        "fable_only": [{"ticker": ticker, "name": fable_map[ticker]["name"], "sector": fable_map[ticker]["sector"]} for ticker in fable_only_tickers],
        "hybrid_candidates": hybrid,
        "candidate_comparison": candidate_comparison,
        "conflict_decisions": CONFLICT_DECISIONS,
        "selection_framework": SELECTION_FRAMEWORK,
        "history_gates": history_gates,
        "checks": checks,
        "validation_status": "Share with caveats",
        "validation_caveat": "The comparison validates structure, overlap, listing history and implementation gates. It does not rank expected returns because current valuation, liquidity and point-in-time fundamental panels for all 35 candidates were not ingested in this analysis.",
    }

    if write_outputs:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        with sqlite3.connect(SQLITE_PATH) as connection:
            _write_table(connection, "headline", [headline])
            _write_table(connection, "proposal_summaries", proposal_summaries)
            _write_table(connection, "region_comparison", region_comparison)
            _write_table(connection, "sector_comparison", sector_comparison)
            _write_table(connection, "consensus", consensus)
            _write_table(connection, "hybrid_candidates", hybrid)
            _write_table(connection, "candidate_comparison", candidate_comparison)
            _write_table(connection, "conflict_decisions", CONFLICT_DECISIONS)
            _write_table(connection, "selection_framework", SELECTION_FRAMEWORK)
            _write_table(connection, "history_gates", history_gates)
            connection.commit()
    return results


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code_cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


def make_notebook() -> dict:
    cells = [
        markdown_cell(
            """# 100종목 유니버스 제안 비교와 하이브리드 추천

## tl;dr

- 기존 제안과 Fable 제안은 35개 중 18개가 일치한다.
- Fable안은 AI 인프라와 상대가치 페어가 강하지만 최종 미국 기업 비중이 78%다.
- 기존안은 최종 미국 비중 69%로 지역 분산이 좋지만 일부 실질 중복과 저성장 후보가 있다.
- 하이브리드안은 공통 18개에 차별화된 17개를 더해 최종 미국 73%, 영국 8%, 유럽(영국 제외) 13%, 아시아 6%로 구성한다.
"""
        ),
        markdown_cell(
            """## Context & Methods

현재 65종목과 기존 35종목 제안은 앞선 검증 결과 JSON을 사용한다. Fable 35종목은 사용자 제공 목록을 표준 티커와 경제적 본거지 기준으로 정규화했다. 두 목록의 교집합, 최종 지역·섹터 구성, 현재 종목과의 사업/지배구조 중복, 1,260거래일 학습창을 비교한다.
"""
        ),
        code_cell(
            f"""from pathlib import Path
import importlib.util

ROOT = Path(r'{ROOT}')
SCRIPT = ROOT / 'scripts' / 'build_universe_100_comparison.py'
spec = importlib.util.spec_from_file_location('universe_comparison', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
results = module.build_results(write_outputs=True)
print('checks_all=', all(results['checks'].values()))
print('headline=', results['headline'])
"""
        ),
        markdown_cell("""## Results

### Proposal comparison
"""),
        code_cell(
            """for row in results['proposal_summaries']:
    print(row)
print('consensus_count=', len(results['consensus']))
print('consensus=', ', '.join(row['ticker'] for row in results['consensus']))
"""
        ),
        markdown_cell("""### Hybrid recommendation and sector design
"""),
        code_cell(
            """for sector in module.SECTOR_ORDER:
    rows = [row for row in results['hybrid_candidates'] if row['sector'] == sector]
    print(f"{sector} ({len(rows)}): " + ', '.join(row['ticker'] for row in rows))

print('\\nFinal sector counts:')
for row in results['sector_comparison']:
    if row['proposal'] == 'Hybrid recommendation':
        print(row['sector'], row['final_names'])
"""
        ),
        markdown_cell("""## Decision Implications

하이브리드안은 종목 스토리보다 기존 65종목 대비 한계 분산효과를 우선한다. 실제 편입 전에는 모든 후보에 동일한 데이터 커버리지·유동성·비용·환율·거래일 달력 검증을 적용하고, 상장 이력이 짧은 네 종목은 마스킹과 거래 자격 게이트를 통과해야 한다.
"""),
        code_cell(
            """print('history_gates:')
for row in results['history_gates']:
    print(row['ticker'], row['listing_date'])
print('validation_status=', results['validation_status'])
print('validation_caveat=', results['validation_caveat'])
"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def execute_notebook(notebook: dict) -> dict:
    namespace = {"__name__": "__notebook__"}
    execution_count = 1
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                exec(compile(source, str(NOTEBOOK_PATH), "exec"), namespace)
            output_text = stdout.getvalue()
            if output_text:
                cell["outputs"] = [{"name": "stdout", "output_type": "stream", "text": output_text.splitlines(keepends=True)}]
        except Exception as exc:  # pragma: no cover - surfaced in notebook output
            cell["outputs"] = [{
                "ename": type(exc).__name__,
                "evalue": str(exc),
                "output_type": "error",
                "traceback": traceback.format_exc().splitlines(),
            }]
            raise
        cell["execution_count"] = execution_count
        execution_count += 1
    return notebook


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    notebook = execute_notebook(make_notebook())
    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    print(f"results={RESULTS_PATH}")
    print(f"sqlite={SQLITE_PATH}")
    print(f"notebook={NOTEBOOK_PATH}")
    print(f"checks_all={all(results['checks'].values())}")


if __name__ == "__main__":
    main()
