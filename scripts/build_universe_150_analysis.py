"""Build a reproducible 100-to-150 structural universe recommendation.

The source workbook is read-only.  The script writes reviewed analysis tables,
an executed nbformat-v4 notebook, SQLite evidence, and an MCP report payload.
"""

from __future__ import annotations

import contextlib
import io
import json
import sqlite3
import traceback
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_XLSX = ROOT.parents[1] / "ai_signal_data.xlsx"
PRIOR_FRAMEWORK = ROOT / "outputs" / "universe_100_comparison" / "universe_100_comparison_results.json"
OUTPUT_DIR = ROOT / "outputs" / "universe_150_recommendation"
RESULTS_JSON = OUTPUT_DIR / "universe_150_results.json"
SQLITE_PATH = OUTPUT_DIR / "universe_150_analysis.sqlite"
NOTEBOOK_PATH = OUTPUT_DIR / "universe_150_analysis.ipynb"
ARTIFACT_PATH = OUTPUT_DIR / "artifact.json"

ANALYSIS_AS_OF = "2026-07-18"
SUPPORTED_MARKETS = {
    "US": "USD",
    "KS": "KRW",
    "JP": "JPY",
    "FP": "EUR",
    "GR": "EUR",
    "NA": "EUR",
    "SW": "CHF",
    "LN": "GBP",
    "DC": "DKK",
}


def candidate(
    ticker: str,
    name: str,
    sector: str,
    subindustry: str,
    country: str,
    region: str,
    peer_anchor: str,
    rationale: str,
    scores: tuple[int, int, int, int, int],
) -> dict:
    simple, market, _ = ticker.split()
    diversification, data_readiness, peer_usefulness, liquidity_quality, independence = scores
    structural_score = (
        diversification * 6
        + data_readiness * 4
        + peer_usefulness * 4
        + liquidity_quality * 3
        + independence * 3
    )
    if structural_score >= 97:
        priority = "A"
    elif structural_score >= 94:
        priority = "B"
    else:
        priority = "C"
    return {
        "ticker": ticker.replace(" Equity", ""),
        "bloomberg_ticker": ticker,
        "simple_ticker": simple,
        "market": market,
        "currency": SUPPORTED_MARKETS[market],
        "name": name,
        "sector": sector,
        "subindustry": subindustry,
        "country": country,
        "region": region,
        "peer_anchor": peer_anchor,
        "rationale": rationale,
        "diversification_score": diversification,
        "data_readiness_score": data_readiness,
        "peer_usefulness_score": peer_usefulness,
        "liquidity_quality_score": liquidity_quality,
        "independence_score": independence,
        "structural_score": structural_score,
        "priority": priority,
        "data_gate": "D1 미적재",
    }


CANDIDATES = [
    # Technology: add peers, but cap the increase at eight names.
    candidate("QCOM US Equity", "Qualcomm", "Technology", "Wireless / edge semiconductors", "United States", "United States", "AVGO, AMD", "모바일·엣지·연결성 반도체 비교군", (4, 5, 5, 5, 3)),
    candidate("NXPI US Equity", "NXP Semiconductors", "Technology", "Automotive / industrial semiconductors", "Netherlands", "Europe ex-UK", "TSLA, TER", "자동차·산업용 반도체로 AI 서버 편중 완화", (5, 4, 5, 5, 4)),
    candidate("SNPS US Equity", "Synopsys", "Technology", "EDA software", "United States", "United States", "CDNS", "Cadence와 직접 비교 가능한 EDA 동종사", (4, 5, 5, 5, 3)),
    candidate("NOW US Equity", "ServiceNow", "Technology", "Workflow software", "United States", "United States", "CRM, SAP, ORCL", "기업용 워크플로 소프트웨어 비교군", (4, 5, 4, 5, 4)),
    candidate("CRWD US Equity", "CrowdStrike", "Technology", "Cybersecurity", "United States", "United States", "PANW", "사이버보안 단일 비교군을 두 종목으로 확장", (4, 5, 5, 5, 3)),
    candidate("IBM US Equity", "IBM", "Technology", "Hybrid cloud / IT services", "United States", "United States", "ORCL, SAP, CSCO", "메인프레임·IT서비스로 반도체와 다른 기술 수익원", (5, 5, 4, 5, 5)),
    candidate("IFX GR Equity", "Infineon Technologies", "Technology", "Power / automotive semiconductors", "Germany", "Europe ex-UK", "ETN, TSLA", "유럽 전력·자동차 반도체 비교군", (5, 5, 5, 5, 4)),
    candidate("8035 JP Equity", "Tokyo Electron", "Technology", "Semiconductor equipment", "Japan", "Japan", "LRCX, AMAT, KLAC", "일본 전공정 장비 동종사로 장비 신호 밀도 강화", (5, 5, 5, 5, 4)),
    # Industrials: diversify away from the existing electrification cluster.
    candidate("TDG US Equity", "TransDigm Group", "Industrials", "Aerospace components / aftermarket", "United States", "United States", "RR/, LMT", "항공 부품·애프터마켓이라는 독립 수익원", (5, 5, 5, 4, 5)),
    candidate("RTX US Equity", "RTX", "Industrials", "Aerospace / defense systems", "United States", "United States", "LMT, RHM", "방산과 민항 시스템의 혼합 비교군", (4, 5, 5, 5, 3)),
    candidate("URI US Equity", "United Rentals", "Industrials", "Equipment rental", "United States", "United States", "CAT, DE", "장비 판매와 다른 렌털·설비투자 사이클", (5, 5, 4, 5, 5)),
    candidate("UPS US Equity", "UPS", "Industrials", "Parcel logistics", "United States", "United States", "UNP", "철도 외 육상 물류 비교군", (5, 5, 4, 5, 5)),
    candidate("ROK US Equity", "Rockwell Automation", "Industrials", "Industrial automation", "United States", "United States", "HON, SIE, SU", "공장 자동화의 직접 동종사", (5, 5, 5, 4, 4)),
    candidate("7011 JP Equity", "Mitsubishi Heavy Industries", "Industrials", "Heavy industry / aerospace / power", "Japan", "Japan", "GEV, RHM, RR/", "일본 중공업·발전·항공의 복합 비교군", (5, 5, 4, 5, 4)),
    candidate("AIR FP Equity", "Airbus", "Industrials", "Commercial aerospace", "France", "Europe ex-UK", "RR/, LMT", "민항 OEM을 추가해 엔진·방산 편중 완화", (5, 5, 5, 5, 4)),
    # Financials: current universe is bank/card heavy; add insurance and market infrastructure.
    candidate("PGR US Equity", "Progressive", "Financials", "Property & casualty insurance", "United States", "United States", "ALV", "미국 자동차보험 언더라이팅 비교군", (5, 5, 5, 5, 5)),
    candidate("CB US Equity", "Chubb", "Financials", "Global property & casualty insurance", "Switzerland", "Europe ex-UK", "ALV, PGR", "글로벌 손해보험 비교군", (5, 4, 5, 5, 5)),
    candidate("CME US Equity", "CME Group", "Financials", "Derivatives exchange", "United States", "United States", "SPGI", "은행이 아닌 거래소·파생상품 인프라", (5, 5, 5, 5, 4)),
    candidate("LSEG LN Equity", "London Stock Exchange Group", "Financials", "Market infrastructure / data", "United Kingdom", "United Kingdom", "SPGI", "거래소·지수·데이터 수익의 직접 비교군", (5, 5, 5, 5, 5)),
    candidate("MUV2 GR Equity", "Munich Re", "Financials", "Reinsurance", "Germany", "Europe ex-UK", "ALV", "재보험 가격 사이클이라는 독립 신호", (5, 5, 5, 5, 5)),
    candidate("BNP FP Equity", "BNP Paribas", "Financials", "European banking", "France", "Europe ex-UK", "JPM, HSBA", "유럽 대형은행 비교군; SAN 티커 충돌 회피", (4, 5, 4, 5, 5)),
    # Healthcare: add life-science tools, devices, diagnostics, and mature biopharma peers.
    candidate("MRK US Equity", "Merck", "Healthcare", "Pharmaceuticals", "United States", "United States", "LLY, ABBV, REGN", "대형 제약 비교군의 폭 확대", (4, 5, 5, 5, 4)),
    candidate("AMGN US Equity", "Amgen", "Healthcare", "Biotechnology", "United States", "United States", "REGN, LLY", "성숙 바이오의 현금흐름·파이프라인 비교군", (4, 5, 5, 5, 4)),
    candidate("DHR US Equity", "Danaher", "Healthcare", "Life-science tools", "United States", "United States", "TMO", "Thermo Fisher와 직접 비교 가능한 도구 기업", (5, 5, 5, 5, 5)),
    candidate("MDT US Equity", "Medtronic", "Healthcare", "Medical devices", "Ireland", "Europe ex-UK", "BSX, ISRG", "의료기기 비교군을 수술로봇 밖으로 확장", (5, 4, 5, 5, 5)),
    candidate("ROG SW Equity", "Roche Holding", "Healthcare", "Pharma / diagnostics", "Switzerland", "Europe ex-UK", "AZN, TMO", "제약과 진단의 결합 비교군", (5, 5, 5, 5, 5)),
    candidate("GILD US Equity", "Gilead Sciences", "Healthcare", "Biopharma", "United States", "United States", "REGN, ABBV", "항바이러스·종양 중심의 방어적 바이오", (5, 5, 4, 5, 5)),
    # Consumer Discretionary.
    candidate("TJX US Equity", "TJX Companies", "Consumer Discretionary", "Off-price retail", "United States", "United States", "WMT, COST", "가격민감 소비와 재고 정상화 비교군", (5, 5, 5, 5, 5)),
    candidate("NKE US Equity", "Nike", "Consumer Discretionary", "Global apparel / footwear", "United States", "United States", "MC, RACE", "대중 브랜드·재고·중국 소비 신호", (5, 5, 4, 5, 5)),
    candidate("MAR US Equity", "Marriott International", "Consumer Discretionary", "Lodging", "United States", "United States", "BKNG", "온라인 여행과 호텔 운영을 연결하는 비교군", (5, 5, 5, 5, 5)),
    candidate("7203 JP Equity", "Toyota Motor", "Consumer Discretionary", "Automobiles", "Japan", "Japan", "TSLA, RACE", "대중 내연·하이브리드 자동차 비교군", (5, 5, 5, 5, 5)),
    candidate("9983 JP Equity", "Fast Retailing", "Consumer Discretionary", "Apparel retail", "Japan", "Japan", "MC, WMT", "일본·아시아 의류 소매 신호", (5, 5, 4, 5, 5)),
    # Consumer Staples.
    candidate("PEP US Equity", "PepsiCo", "Consumer Staples", "Beverages / snacks", "United States", "United States", "PG, NESN", "음료와 스낵의 방어적 현금흐름", (4, 5, 5, 5, 4)),
    candidate("MDLZ US Equity", "Mondelez International", "Consumer Staples", "Packaged food", "United States", "United States", "NESN", "글로벌 스낵·가격전가 비교군", (5, 5, 5, 5, 5)),
    candidate("OR FP Equity", "L'Oreal", "Consumer Staples", "Beauty / personal care", "France", "Europe ex-UK", "PG, MC", "뷰티 소비와 브랜드 파워 비교군", (5, 5, 4, 5, 4)),
    candidate("DGE LN Equity", "Diageo", "Consumer Staples", "Spirits", "United Kingdom", "United Kingdom", "PM, NESN", "주류·프리미엄 소비의 독립 비교군", (5, 5, 4, 5, 5)),
    # Energy.
    candidate("SLB US Equity", "SLB", "Energy", "Oilfield services", "United States", "United States", "XOM, SHEL, MPC", "원유 생산자가 아닌 유전 서비스 사이클", (5, 5, 5, 5, 5)),
    candidate("CNQ US Equity", "Canadian Natural Resources", "Energy", "Upstream oil & gas", "Canada", "Canada", "XOM, SHEL", "캐나다 장수명 자산의 상류 비교군", (5, 4, 5, 5, 4)),
    candidate("ENB US Equity", "Enbridge", "Energy", "Midstream / pipelines", "Canada", "Canada", "WMB", "북미 파이프라인·규제형 현금흐름 비교군", (5, 4, 5, 5, 4)),
    # Utilities.
    candidate("DUK US Equity", "Duke Energy", "Utilities", "Regulated electric utility", "United States", "United States", "NEE, CEG, VST", "규제형 전력의 방어적 비교군", (5, 5, 5, 5, 5)),
    candidate("SO US Equity", "Southern Company", "Utilities", "Regulated electric / gas utility", "United States", "United States", "NEE, CEG, VST", "미국 남동부 규제형 유틸리티", (5, 5, 5, 5, 5)),
    candidate("RWE GR Equity", "RWE", "Utilities", "European power / renewables", "Germany", "Europe ex-UK", "NEE, CEG, VST", "지원되는 EUR 접미사로 유럽 전력 비교군 확보", (5, 5, 5, 5, 5)),
    # Real Estate.
    candidate("WELL US Equity", "Welltower", "Real Estate", "Healthcare REIT", "United States", "United States", "AMT, EQIX, PLD, DLR", "데이터센터·물류 밖 헬스케어 부동산", (5, 5, 5, 5, 5)),
    candidate("AVB US Equity", "AvalonBay Communities", "Real Estate", "Residential REIT", "United States", "United States", "PLD, DLR", "주거 임대와 금리 민감도 비교군", (5, 5, 5, 5, 5)),
    # Communication Services.
    candidate("DIS US Equity", "Walt Disney", "Communication Services", "Media / entertainment / parks", "United States", "United States", "NFLX", "스트리밍 외 콘텐츠·파크 수익원", (5, 5, 5, 5, 4)),
    candidate("TTWO US Equity", "Take-Two Interactive", "Communication Services", "Interactive entertainment", "United States", "United States", "NFLX, SPOT", "게임 콘텐츠라는 새로운 비교군; EA 기업행사 회피", (5, 5, 5, 5, 5)),
    candidate("9432 JP Equity", "NTT", "Communication Services", "Telecommunications", "Japan", "Japan", "TMUS", "일본 통신·데이터 인프라 비교군", (5, 5, 5, 5, 5)),
    # Materials.
    candidate("NEM US Equity", "Newmont", "Materials", "Gold mining", "United States", "United States", "FCX, RIO", "금 가격 민감도로 구리·철광석 편중 완화", (5, 5, 5, 5, 5)),
    candidate("APD US Equity", "Air Products and Chemicals", "Materials", "Industrial gases", "United States", "United States", "LIN", "Linde의 직접 산업가스 비교군", (4, 5, 5, 5, 4)),
    candidate("BAS GR Equity", "BASF", "Materials", "Diversified chemicals", "Germany", "Europe ex-UK", "LIN, RIO", "유럽 범용·특수화학 사이클 비교군", (5, 5, 5, 5, 5)),
]


CURRENT_COUNTRY_OVERRIDES = {
    "000660": "South Korea",
    "005930": "South Korea",
    "ETN": "Ireland",
    "LIN": "United Kingdom",
    "FN": "Thailand",
    "TSM": "Taiwan",
    "STX": "Ireland",
    "ARM": "United Kingdom",
    "SPOT": "Sweden",
    "RACE": "Italy",
    "285A": "Japan",
    "6857": "Japan",
    "SU": "France",
    "SIE": "Germany",
    "RHM": "Germany",
    "ALV": "Germany",
    "MC": "France",
    "NESN": "Switzerland",
    "RR/": "United Kingdom",
    "SAP": "Germany",
    "ASML": "Netherlands",
    "AZN": "United Kingdom",
    "SHEL": "United Kingdom",
    "HSBA": "United Kingdom",
    "NOVOB": "Denmark",
    "RIO": "United Kingdom",
}


def region_for_country(country: str) -> str:
    if country == "United States":
        return "United States"
    if country == "United Kingdom":
        return "United Kingdom"
    if country == "Japan":
        return "Japan"
    if country == "South Korea":
        return "South Korea"
    if country == "Canada":
        return "Canada"
    if country in {"Taiwan", "Thailand"}:
        return "Asia ex-Japan/Korea"
    return "Europe ex-UK"


def build_analysis() -> tuple[dict, dict]:
    meta = pd.read_excel(SOURCE_XLSX, sheet_name="Universe_Meta")
    candidate_df = pd.DataFrame(CANDIDATES)
    candidate_df.insert(0, "selection_order", range(1, len(candidate_df) + 1))

    current = meta.copy()
    current["simple_ticker"] = current["Ticker"].str.split().str[0]
    current["country"] = current["simple_ticker"].map(CURRENT_COUNTRY_OVERRIDES).fillna("United States")
    current["region"] = current["country"].map(region_for_country)

    current_sector = current.groupby("Sector").size().rename("current_names")
    add_sector = candidate_df.groupby("sector").size().rename("add_names")
    sector_mix = pd.concat([current_sector, add_sector], axis=1).fillna(0).astype(int)
    sector_mix["final_names"] = sector_mix["current_names"] + sector_mix["add_names"]
    sector_mix["current_share"] = sector_mix["current_names"] / len(current)
    sector_mix["final_share"] = sector_mix["final_names"] / (len(current) + len(candidate_df))
    sector_mix = sector_mix.reset_index().rename(columns={"index": "sector", "Sector": "sector"})
    sector_mix = sector_mix.sort_values(["final_names", "sector"], ascending=[False, True])

    current_region = current.groupby("region").size().rename("current_names")
    add_region = candidate_df.groupby("region").size().rename("add_names")
    region_order = [
        "United States",
        "Europe ex-UK",
        "United Kingdom",
        "Japan",
        "South Korea",
        "Canada",
        "Asia ex-Japan/Korea",
    ]
    region_mix = pd.concat([current_region, add_region], axis=1).fillna(0).astype(int).reindex(region_order, fill_value=0)
    region_mix["final_names"] = region_mix["current_names"] + region_mix["add_names"]
    region_mix["current_share"] = region_mix["current_names"] / len(current)
    region_mix["final_share"] = region_mix["final_names"] / (len(current) + len(candidate_df))
    region_mix = region_mix.reset_index().rename(columns={"index": "region"})

    current_country = current.groupby("country").size().rename("current_names")
    add_country = candidate_df.groupby("country").size().rename("add_names")
    country_mix = pd.concat([current_country, add_country], axis=1).fillna(0).astype(int)
    country_mix["final_names"] = country_mix["current_names"] + country_mix["add_names"]
    country_mix["final_share"] = country_mix["final_names"] / (len(current) + len(candidate_df))
    country_mix = country_mix.reset_index().sort_values(["final_names", "country"], ascending=[False, True])

    currency_mix = candidate_df.groupby(["currency", "market"]).size().rename("add_names").reset_index()
    currency_mix = currency_mix.sort_values(["add_names", "currency"], ascending=[False, True])

    sector_long = pd.DataFrame(
        [
            {
                "sector": row.sector,
                "metric": metric,
                "value": getattr(row, field),
                "current_names": row.current_names,
                "add_names": row.add_names,
                "final_names": row.final_names,
                "current_share": row.current_share,
                "final_share": row.final_share,
            }
            for row in sector_mix.itertuples(index=False)
            for metric, field in (("현재 100", "current_names"), ("제안 150", "final_names"))
        ]
    )
    region_long = pd.DataFrame(
        [
            {
                "region": row.region,
                "metric": metric,
                "value": getattr(row, field),
                "current_names": row.current_names,
                "add_names": row.add_names,
                "final_names": row.final_names,
                "current_share": row.current_share,
                "final_share": row.final_share,
            }
            for row in region_mix.itertuples(index=False)
            for metric, field in (("현재 100", "current_share"), ("제안 150", "final_share"))
        ]
    )

    current_tech_share = float(sector_mix.loc[sector_mix.sector.eq("Technology"), "current_share"].iloc[0])
    final_tech_share = float(sector_mix.loc[sector_mix.sector.eq("Technology"), "final_share"].iloc[0])
    current_us_share = float(region_mix.loc[region_mix.region.eq("United States"), "current_share"].iloc[0])
    final_us_share = float(region_mix.loc[region_mix.region.eq("United States"), "final_share"].iloc[0])
    summary = {
        "current_names": len(current),
        "add_names": len(candidate_df),
        "final_names": len(current) + len(candidate_df),
        "add_us_names": int(candidate_df.country.eq("United States").sum()),
        "add_non_us_names": int((~candidate_df.country.eq("United States")).sum()),
        "current_tech_share": current_tech_share,
        "final_tech_share": final_tech_share,
        "tech_share_change": final_tech_share - current_tech_share,
        "current_non_us_share": 1 - current_us_share,
        "final_non_us_share": 1 - final_us_share,
        "non_us_share_change": current_us_share - final_us_share,
        "supported_candidate_currencies": int(candidate_df.currency.nunique()),
    }

    existing_short = set(current.simple_ticker)
    checks = {
        "current_universe_is_100": len(current) == 100,
        "all_current_status_available": bool(current.Status.eq("Available").all()),
        "candidate_count_is_50": len(candidate_df) == 50,
        "candidate_full_tickers_unique": bool(candidate_df.bloomberg_ticker.is_unique),
        "candidate_short_tickers_unique": bool(candidate_df.simple_ticker.is_unique),
        "no_overlap_with_current_universe": not bool(set(candidate_df.simple_ticker) & existing_short),
        "all_candidate_markets_supported": bool(candidate_df.market.isin(SUPPORTED_MARKETS).all()),
        "all_candidate_currencies_already_supported": set(candidate_df.currency) <= set(SUPPORTED_MARKETS.values()),
        "final_universe_is_150": int(sector_mix.final_names.sum()) == 150,
        "candidate_us_is_30": summary["add_us_names"] == 30,
        "candidate_non_us_is_20": summary["add_non_us_names"] == 20,
        "final_us_share_near_prior_posture": abs(final_us_share - 104 / 150) < 1e-12,
        "technology_share_below_30pct": final_tech_share < 0.30,
        "no_san_collision": "SAN" not in set(candidate_df.simple_ticker),
        "score_bounds_valid": bool(candidate_df.structural_score.between(20, 100).all()),
    }
    assert all(checks.values()), checks

    with sqlite3.connect(SQLITE_PATH) as connection:
        pd.DataFrame([summary]).to_sql("summary", connection, if_exists="replace", index=False)
        sector_mix.to_sql("sector_mix", connection, if_exists="replace", index=False)
        sector_long.to_sql("sector_mix_long", connection, if_exists="replace", index=False)
        region_mix.to_sql("region_mix", connection, if_exists="replace", index=False)
        region_long.to_sql("region_mix_long", connection, if_exists="replace", index=False)
        country_mix.to_sql("country_mix", connection, if_exists="replace", index=False)
        currency_mix.to_sql("currency_mix", connection, if_exists="replace", index=False)
        candidate_df.to_sql("candidates", connection, if_exists="replace", index=False)

    result = {
        "analysis_as_of": ANALYSIS_AS_OF,
        "source": {
            "workbook": SOURCE_XLSX.name,
            "prior_framework": str(PRIOR_FRAMEWORK.relative_to(ROOT)),
            "current_universe_rows": len(current),
            "country_mapping_basis": "economic domicile; manual overrides where workbook has no country field",
        },
        "selection_framework": [
            {"criterion": "Marginal diversification", "weight": 0.30},
            {"criterion": "Data readiness", "weight": 0.20},
            {"criterion": "Cross-sectional peer usefulness", "weight": 0.20},
            {"criterion": "Liquidity / quality", "weight": 0.15},
            {"criterion": "Theme / ownership independence", "weight": 0.15},
        ],
        "summary": summary,
        "sector_mix": sector_mix.to_dict(orient="records"),
        "region_mix": region_mix.to_dict(orient="records"),
        "country_mix": country_mix.to_dict(orient="records"),
        "currency_mix": currency_mix.to_dict(orient="records"),
        "candidates": candidate_df.to_dict(orient="records"),
        "checks": checks,
        "excluded_current_refresh": [
            {"ticker": "EA US", "reason": "2025년 발표된 비상장화 거래가 진행 중이어서 상장 지속성과 데이터 연속성이 불확실"},
            {"ticker": "GE US", "reason": "2024년 분할 이후 GE Aerospace 이전 구간의 사업 연속성이 1,260일 학습창과 맞지 않음"},
            {"ticker": "IBE SM", "reason": "현행 MARKET_TO_CURRENCY에 SM 접미사가 없어 RWE GR로 대체"},
            {"ticker": "SAN FP / SAN SM", "reason": "내부 simple_ticker가 둘 다 SAN으로 축약되는 충돌"},
        ],
        "data_approval_status": "Share with caveats: structural recommendation ready; new-name workbook coverage not yet loaded",
    }
    RESULTS_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    artifact = build_artifact(result, sector_long, region_long, candidate_df, currency_mix)
    ARTIFACT_PATH.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    return result, artifact


def source_query(source_id: str, label: str, sql: str, description: str, table: str) -> dict:
    return {
        "id": source_id,
        "label": label,
        "path": "outputs/universe_150_recommendation/universe_150_analysis.sqlite",
        "query": {
            "engine": "SQLite",
            "language": "sql",
            "sql": sql,
            "description": description,
            "executed_at": f"{ANALYSIS_AS_OF}T00:00:00+09:00",
            "tables_used": [table],
        },
    }


def build_artifact(result: dict, sector_long: pd.DataFrame, region_long: pd.DataFrame, candidate_df: pd.DataFrame, currency_mix: pd.DataFrame) -> dict:
    sources = [
        {"id": "src-analysis", "label": "100-to-150 structural universe analysis", "path": "outputs/universe_150_recommendation/universe_150_results.json"},
        {"id": "src-workbook", "label": "ai_signal_data.xlsx — Universe_Meta"},
        {"id": "src-framework", "label": "Prior 65-to-100 selection framework", "path": "outputs/universe_100_comparison/universe_100_comparison_results.json"},
        {"id": "src-loader", "label": "Supported Bloomberg market suffixes and currencies", "path": "src/data_loader.py"},
        source_query("src-summary-sql", "Universe expansion headline query", "SELECT * FROM summary;", "Headline name-count, sector, and region shares.", "summary"),
        source_query("src-sector-sql", "Sector mix query", "SELECT sector, metric, value, current_names, add_names, final_names, current_share, final_share FROM sector_mix_long ORDER BY final_names DESC, sector, metric;", "Current and proposed name counts by project sector.", "sector_mix_long"),
        source_query("src-region-sql", "Economic-domicile region query", "SELECT region, metric, value, current_names, add_names, final_names, current_share, final_share FROM region_mix_long ORDER BY final_names DESC, region, metric;", "Current and proposed shares by economic domicile.", "region_mix_long"),
        source_query("src-candidates-sql", "Recommended 50-name query", "SELECT selection_order, ticker, name, sector, subindustry, country, region, peer_anchor, priority, structural_score, data_gate, rationale FROM candidates ORDER BY selection_order;", "The reviewed 50-name structural recommendation.", "candidates"),
        source_query("src-currency-sql", "Candidate listing-currency query", "SELECT currency, market, add_names FROM currency_mix ORDER BY add_names DESC, currency;", "Listing currency and Bloomberg suffix counts for the 50 candidates.", "currency_mix"),
        {"id": "src-europe", "label": "STOXX Europe 600 current components", "href": "https://www.stoxx.com/download/indices/components/SXXGR.pdf"},
        {"id": "src-japan", "label": "Nikkei 225 current components", "href": "https://indexes.nikkei.co.jp/en/nkave/index/component"},
        {"id": "src-nasdaq", "label": "Nasdaq-100 companies", "href": "https://www.nasdaq.com/solutions/global-indexes/nasdaq-100/companies"},
        {"id": "src-ea", "label": "EA definitive acquisition agreement", "href": "https://www.ea.com/news/ea-announces-agreement-to-be-acquired"},
    ]
    summary = result["summary"]
    manifest = {
        "version": 1,
        "surface": "report",
        "title": "유니버스 150종목 확장 추천",
        "description": "현행 100종목에 추가할 50종목의 구조적 추천과 데이터 승인 게이트",
        "generatedAt": f"{ANALYSIS_AS_OF}T00:00:00+09:00",
        "sources": sources,
        "cards": [
            {"id": "card-final", "dataset": "summary", "sourceId": "src-summary-sql", "description": "구조 제안의 최종 종목 수", "metrics": [{"label": "제안 종목 수", "field": "final_names", "format": "number"}, {"label": "현재", "field": "current_names", "format": "number"}]},
            {"id": "card-add", "dataset": "summary", "sourceId": "src-summary-sql", "description": "경제적 본거지 기준 미국 30·비미국 20", "metrics": [{"label": "신규 종목", "field": "add_names", "format": "number"}, {"label": "미국", "field": "add_us_names", "format": "number"}, {"label": "비미국", "field": "add_non_us_names", "format": "number"}]},
            {"id": "card-tech", "dataset": "summary", "sourceId": "src-summary-sql", "description": "종목 수 기준 기술주 비중", "metrics": [{"label": "최종 기술주 비중", "field": "final_tech_share", "format": "percent"}, {"label": "현재", "field": "current_tech_share", "format": "percent"}, {"label": "변화", "field": "tech_share_change", "format": "percent", "signed": True}]},
            {"id": "card-global", "dataset": "summary", "sourceId": "src-summary-sql", "description": "경제적 본거지 기준 비미국 종목 비중", "metrics": [{"label": "최종 비미국 비중", "field": "final_non_us_share", "format": "percent"}, {"label": "현재", "field": "current_non_us_share", "format": "percent"}, {"label": "변화", "field": "non_us_share_change", "format": "percent", "signed": True}]},
        ],
        "charts": [
            {
                "id": "chart-sector",
                "title": "현재 및 제안 유니버스의 섹터별 종목 수",
                "subtitle": "기술주는 35개에서 43개로 늘지만 전체 비중은 35.0%에서 28.7%로 하락",
                "intent": "comparison",
                "question": "50개 추가 후 섹터별 종목 수 구조는 어떻게 바뀌는가?",
                "type": "bar",
                "dataset": "sector_mix_long",
                "sourceId": "src-sector-sql",
                "encodings": {
                    "x": {"field": "sector", "type": "nominal", "label": "섹터"},
                    "y": {"field": "value", "type": "quantitative", "format": "number", "label": "종목 수"},
                    "color": {"field": "metric", "type": "nominal", "label": "구분"},
                    "tooltip": [
                        {"field": "add_names", "type": "quantitative", "label": "신규"},
                        {"field": "final_share", "type": "quantitative", "format": "percent", "label": "최종 비중"},
                    ],
                },
                "valueFormat": "number",
                "layout": "full",
                "palette": {"kind": "categorical"},
                "settings": {"orientation": "horizontal", "groupMode": "grouped", "sort": "descending", "categoryLabelPolicy": "wrap", "showValues": True},
                "legend": {"position": "bottom", "title": "구분"},
                "surface": {"surface": "card", "viewMode": "visualization"},
            },
            {
                "id": "chart-region",
                "title": "현재 및 제안 유니버스의 지역별 종목 수 비중",
                "subtitle": "경제적 본거지 기준 미국 비중은 74.0%에서 69.3%로 완화",
                "intent": "comparison",
                "question": "50개 추가 후 경제적 본거지 편중은 어떻게 바뀌는가?",
                "type": "bar",
                "dataset": "region_mix_long",
                "sourceId": "src-region-sql",
                "encodings": {
                    "x": {"field": "region", "type": "nominal", "label": "지역"},
                    "y": {"field": "value", "type": "quantitative", "format": "percent", "label": "종목 수 비중"},
                    "color": {"field": "metric", "type": "nominal", "label": "구분"},
                    "tooltip": [
                        {"field": "current_names", "type": "quantitative", "label": "현재"},
                        {"field": "add_names", "type": "quantitative", "label": "신규"},
                        {"field": "final_names", "type": "quantitative", "label": "최종"},
                    ],
                },
                "valueFormat": "percent",
                "layout": "full",
                "palette": {"kind": "categorical"},
                "settings": {"orientation": "horizontal", "groupMode": "grouped", "sort": "descending", "categoryLabelPolicy": "wrap", "showValues": True},
                "legend": {"position": "bottom", "title": "구분"},
                "surface": {"surface": "card", "viewMode": "visualization"},
            },
        ],
        "tables": [
            {
                "id": "table-candidates",
                "title": "추천 신규 50종목",
                "subtitle": "구조 점수는 기대수익 순위가 아니라 분산·데이터·비교군·유동성·독립성의 구현 우선순위",
                "dataset": "candidates",
                "sourceId": "src-candidates-sql",
                "defaultSort": {"field": "structural_score", "direction": "desc"},
                "density": "dense",
                "layout": "full",
                "columns": [
                    {"field": "selection_order", "label": "순번", "format": "number"},
                    {"field": "ticker", "label": "티커", "type": "text"},
                    {"field": "name", "label": "종목명", "type": "text"},
                    {"field": "sector", "label": "섹터", "type": "text"},
                    {"field": "subindustry", "label": "하위산업", "type": "text"},
                    {"field": "country", "label": "본거지", "type": "text"},
                    {"field": "peer_anchor", "label": "기존 비교군", "type": "text"},
                    {"field": "structural_score", "label": "구조 점수", "format": "number"},
                    {"field": "priority", "label": "적재 우선", "type": "text"},
                    {"field": "data_gate", "label": "데이터 상태", "type": "text"},
                    {"field": "rationale", "label": "선정 이유", "type": "text"},
                ],
            },
            {
                "id": "table-currency",
                "title": "신규 50종목의 상장 통화와 시장 접미사",
                "subtitle": "USD·EUR·JPY·GBP·CHF만 사용하여 새 FX 통화를 추가하지 않음",
                "dataset": "currency_mix",
                "sourceId": "src-currency-sql",
                "defaultSort": {"field": "add_names", "direction": "desc"},
                "density": "spacious",
                "layout": "full",
                "columns": [
                    {"field": "currency", "label": "통화", "type": "text"},
                    {"field": "market", "label": "접미사", "type": "text"},
                    {"field": "add_names", "label": "종목 수", "format": "number"},
                ],
            },
        ],
        "blocks": [
            {"id": "title", "type": "markdown", "body": "# 유니버스 150종목 확장 추천", "layout": "full"},
            {"id": "executive-summary", "type": "markdown", "sourceId": "src-analysis", "body": "## Executive Summary\n\n- **추가 50종목은 확장해도 좋습니다.** 다만 100→150의 목적은 종목 수 자체가 아니라 교차단면 비교군을 넓히고 기술·미국 편중을 낮추는 것입니다.\n- **권고 배분은 기술 8개·비기술 42개, 경제적 본거지 기준 미국 30개·비미국 20개입니다.** 최종 기술주 종목 수 비중은 35.0%에서 28.7%, 미국 비중은 74.0%에서 69.3%가 됩니다.\n- **현행 파이프라인의 통화·접미사 범위 안에서만 골랐습니다.** 신규 50개는 USD·EUR·JPY·GBP·CHF를 사용하며 새 FX 통화를 요구하지 않습니다.\n- **이 명단은 구조 추천이지 즉시 운영 승인안은 아닙니다.** 50개 모두 신규 데이터가 미적재 상태이므로 가격 1,260거래일·필수 시트·기업행사·캘린더 검증을 통과한 뒤 편입해야 합니다."},
            {"id": "metrics", "type": "metric-strip", "cardIds": ["card-final", "card-add", "card-tech", "card-global"], "layout": "full"},
            {"id": "sector-story", "type": "markdown", "sourceId": "src-analysis", "body": "## 기술주 수는 늘리되 기술주 비중은 낮춘다\n\n기술 후보는 QCOM·NXPI·SNPS·NOW·CRWD·IBM·IFX·8035의 8개로 제한했습니다. 기존 AI 공급망을 그대로 복제하기보다 EDA, 사이버보안, 워크플로, IT서비스, 자동차·전력 반도체를 보강합니다. 비기술 42개는 보험, 거래소, 생명과학 도구, 의료기기, 물류, 규제형 유틸리티, 주거·헬스케어 REIT, 금광 등 현재 얕은 비교군에 배분했습니다.\n\n**의미:** 확장이 퇴화를 완화한다면 단순 표본 수보다 섹터 안의 유효 동종사 수가 늘어난 효과로 설명할 수 있습니다."},
            {"id": "sector-chart", "type": "chart", "chartId": "chart-sector", "layout": "full"},
            {"id": "sector-note", "type": "markdown", "body": "### 읽는 법\n\n막대는 실제 자본 비중이 아니라 종목 수입니다. 최종 자본 편중은 150종목 시가총액 벤치마크와 최적화 결과를 다시 계산해야 판단할 수 있습니다."},
            {"id": "candidate-story", "type": "markdown", "sourceId": "src-analysis", "body": "## 추천 50종목은 비교군 공백을 채우는 순서로 골랐다\n\n기존 100종목과의 단순 티커 중복과 내부 축약 티커 충돌을 제거했습니다. A/B/C는 매수 매력도가 아니라 데이터 적재 순서입니다. 구조 점수는 이전 65→100 확장과 같은 가중치인 분산 30%, 데이터 준비도 20%, 동종사 유용성 20%, 유동성·품질 15%, 테마·소유구조 독립성 15%를 사용했습니다."},
            {"id": "candidate-table", "type": "table", "tableId": "table-candidates", "layout": "full"},
            {"id": "region-story", "type": "markdown", "sourceId": "src-region-sql", "body": "## 지역 비중은 이전 확장의 70대30 기조를 유지한다\n\n현행 100종목은 경제적 본거지 기준 미국 74개·비미국 26개로 추정됩니다. 미국 30개와 비미국 20개를 더하면 최종 미국 104개(69.3%)·비미국 46개(30.7%)가 되어 65→100 때 의도한 약 70대30 기조를 유지합니다. 일본은 2개에서 7개, 유럽(영국 제외)은 13개에서 24개로 늘어납니다.\n\n**의미:** 지역 분산을 늘리되 미국 대형주 중심의 데이터 품질과 유동성을 버리지 않는 절충안입니다."},
            {"id": "region-chart", "type": "chart", "chartId": "chart-region", "layout": "full"},
            {"id": "region-note", "type": "markdown", "body": "### 분류 기준\n\n지역은 상장 거래소가 아니라 경제적 본거지 기준입니다. 원본 워크북에는 국가 필드가 없어 기존 100개는 수동 매핑했으며, 운영 편입 전 공급사 국가 분류와 재대사해야 합니다."},
            {"id": "currency-story", "type": "markdown", "sourceId": "src-currency-sql", "body": "## 새 FX 통화 없이 구현할 수 있다\n\n현행 로더가 지원하는 `US`, `GR`, `FP`, `JP`, `SW`, `LN` 접미사만 사용했습니다. 이 때문에 스페인 접미사 `IBE SM`은 제외하고 `RWE GR`로 대체했고, 캐나다 CNQ·ENB는 미국 상장 보통주를 사용합니다."},
            {"id": "currency-table", "type": "table", "tableId": "table-currency", "layout": "full"},
            {"id": "next-steps", "type": "markdown", "body": "## 권장 적용 순서\n\n1. **유니버스만 바꾸는 단일 arm으로 사전등록**합니다. `val_window`, 학습률, 정규화와 동시 변경하지 않습니다.\n2. **D1 데이터 게이트**를 적용합니다: 1,260거래일 가격, 필수 재무·컨센서스 시트, 현지 휴장일, 기업행사, FX 정합성.\n3. **섀도 백테스트**에서 100종목 대 150종목을 같은 기간·비용·하이퍼파라미터로 비교합니다.\n4. **기존 성공 기준을 유지**합니다: 퇴화율 목표와 IR·회전율·추적오차·섹터/국가 위험 가드를 사전에 고정합니다.\n5. **실패 시 해석을 분리**합니다. 데이터 게이트 실패와 모델 성과 실패를 같은 실패로 합치지 않습니다."},
            {"id": "questions", "type": "markdown", "body": "## 추가 확인할 질문\n\n- 50개 모두 필수 시트에서 동일한 날짜 범위와 회계 단위로 공급되는가?\n- CB·MDT·NXPI처럼 미국 상장 해외 기업의 공급사 국가·통화·펀더멘털 매핑은 일관적인가?\n- 휴장일이 다른 종목을 월간 리밸런싱 시점에 어떤 규칙으로 정렬할 것인가?\n- 퇴화율 개선이 특정 섹터나 지역의 표본 증가에만 의존하지 않는가?"},
            {"id": "caveats", "type": "markdown", "sourceId": "src-analysis", "body": "## 가정과 유의사항\n\n- 이번 점수는 구조·구현 점수이며 밸류에이션, 목표가, 단기 모멘텀을 반영한 매수 순위가 아닙니다.\n- 현재 국가 비중은 `Universe_Meta`에 국가 필드가 없어 수동으로 분류했습니다. Linde는 이전 분석과 같이 영국, Fabrinet은 태국 기준을 유지했습니다.\n- 신규 50개의 실제 가격·재무·컨센서스 커버리지는 아직 원본 엑셀에 없으므로 검증할 수 없습니다. 따라서 최종 신뢰도는 **조건부 공유 가능**입니다.\n- 유니버스 확장은 교차단면 표본과 동종사 비교에는 도움이 될 수 있지만, 시간축 검증창 자체가 불안정한 문제를 자동으로 해결하지는 않습니다.\n- EA는 진행 중인 비상장화 거래 때문에 제외했습니다. GE Aerospace는 2024년 분할 전후의 사업 연속성 문제로 제외했습니다."},
        ],
    }
    snapshot = {
        "version": 1,
        "status": "ready",
        "generatedAt": f"{ANALYSIS_AS_OF}T00:00:00+09:00",
        "datasets": {
            "summary": [summary],
            "sector_mix_long": sector_long.to_dict(orient="records"),
            "region_mix_long": region_long.to_dict(orient="records"),
            "candidates": candidate_df.to_dict(orient="records"),
            "currency_mix": currency_mix.to_dict(orient="records"),
        },
    }
    return {"surface": "report", "manifest": manifest, "snapshot": snapshot, "sources": sources}


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code_cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


def execute_cells(cells: list[dict]) -> tuple[list[dict], str | None]:
    namespace: dict = {"__name__": "__main__"}
    execution_count = 0
    failure = None
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        execution_count += 1
        cell["execution_count"] = execution_count
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                exec(compile("".join(cell["source"]), f"<cell-{execution_count}>", "exec"), namespace)
            if stdout.getvalue():
                cell["outputs"] = [{"name": "stdout", "output_type": "stream", "text": stdout.getvalue().splitlines(keepends=True)}]
        except Exception:
            failure = traceback.format_exc()
            cell["outputs"] = [{"ename": "ExecutionError", "evalue": failure.splitlines()[-1], "output_type": "error", "traceback": failure.splitlines()}]
            break
    return cells, failure


def build_notebook(result: dict) -> None:
    cells = [
        markdown_cell("""# 100→150 유니버스 구조 추천\n\n## tl;dr\n\n- 기술 8개·비기술 42개를 추가해 기술주 종목 수 비중을 35.0%에서 28.7%로 낮춘다.\n- 경제적 본거지 기준 미국 30개·비미국 20개를 추가해 최종 미국/비미국 비중을 69.3%/30.7%로 맞춘다.\n- 신규 50개는 기존 로더가 지원하는 USD·EUR·JPY·GBP·CHF 상장만 사용한다.\n- 실제 신규 데이터는 미적재 상태이므로 이 결과는 구조 추천이며 D1 데이터 게이트가 남아 있다.\n"""),
        markdown_cell("""## Context & Methods\n\n### Key Assumptions\n\n- 현재 100종목의 섹터와 상태는 `ai_signal_data.xlsx`의 `Universe_Meta`를 그대로 사용한다.\n- 국가는 경제적 본거지 기준으로 수동 매핑한다.\n- 이전 65→100 확장의 가중치(분산 30%, 데이터 20%, 동종사 20%, 유동성·품질 15%, 독립성 15%)를 유지한다.\n- 구조 점수는 기대수익이나 밸류에이션 점수가 아니다.\n"""),
        code_cell("""import json\nimport sqlite3\nfrom pathlib import Path\nimport pandas as pd\n\nROOT = Path.cwd()\nif not (ROOT / 'outputs').exists():\n    ROOT = ROOT.parents[1]\nOUT = ROOT / 'outputs' / 'universe_150_recommendation'\nRESULTS = json.loads((OUT / 'universe_150_results.json').read_text(encoding='utf-8'))\nDB = OUT / 'universe_150_analysis.sqlite'\nprint('analysis_as_of=', RESULTS['analysis_as_of'])\nprint('data_approval_status=', RESULTS['data_approval_status'])\n"""),
        markdown_cell("## Data\n\n원본 100종목, 제안 50종목, 섹터·지역·통화 집계와 검증 결과를 SQLite 스냅샷에서 다시 읽는다.\n"),
        code_cell("""with sqlite3.connect(DB) as connection:\n    candidates = pd.read_sql_query('SELECT * FROM candidates ORDER BY selection_order', connection)\n    sector_mix = pd.read_sql_query('SELECT * FROM sector_mix ORDER BY final_names DESC, sector', connection)\n    region_mix = pd.read_sql_query('SELECT * FROM region_mix ORDER BY final_names DESC, region', connection)\n    currency_mix = pd.read_sql_query('SELECT * FROM currency_mix ORDER BY add_names DESC, currency', connection)\nprint('candidate_rows=', len(candidates))\nprint('unique_short_tickers=', candidates['simple_ticker'].nunique())\nprint('markets=', sorted(candidates['market'].unique()))\nprint('currencies=', sorted(candidates['currency'].unique()))\n"""),
        markdown_cell("## Results\n\n### 1. 섹터 구성\n"),
        code_cell("""print(sector_mix.to_string(index=False, formatters={'current_share': lambda x: f'{x:.1%}', 'final_share': lambda x: f'{x:.1%}'}))\n"""),
        markdown_cell("### 2. 경제적 본거지 구성\n"),
        code_cell("""print(region_mix.to_string(index=False, formatters={'current_share': lambda x: f'{x:.1%}', 'final_share': lambda x: f'{x:.1%}'}))\n"""),
        markdown_cell("### 3. 추천 50종목\n"),
        code_cell("""cols = ['selection_order','ticker','name','sector','country','peer_anchor','structural_score','priority','data_gate']\nprint(candidates[cols].to_string(index=False))\n"""),
        markdown_cell("### 4. 검증 체크\n"),
        code_cell("""checks = pd.Series(RESULTS['checks'], name='passed')\nprint(checks.to_string())\nassert checks.all()\nassert len(candidates) == 50\nassert candidates['simple_ticker'].nunique() == 50\nassert not candidates['simple_ticker'].eq('SAN').any()\n"""),
        markdown_cell("""## Takeaways\n\n1. 50개 추가안은 이전 확장의 70대30 지역 기조를 유지하면서 기술주 종목 수 비중을 30% 아래로 낮춘다.\n2. 보험·거래소·생명과학 도구·의료기기·물류·규제형 유틸리티·주거/헬스케어 REIT·금광의 동종사 밀도가 개선된다.\n3. 모든 후보가 현행 접미사·통화 범위 안에 있지만 신규 워크북 데이터는 아직 검증되지 않았다.\n4. 유니버스 단일 arm으로 사전등록하고 `val_window` 등 다른 축은 고정해야 효과를 해석할 수 있다.\n"""),
    ]
    executed, failure = execute_cells(cells)
    notebook = {
        "cells": executed,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    reloaded = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert reloaded["nbformat"] == 4 and reloaded["cells"]
    if failure:
        raise RuntimeError(f"Notebook execution failed:\n{failure}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result, _ = build_analysis()
    build_notebook(result)
    print(f"Wrote {RESULTS_JSON}")
    print(f"Wrote {SQLITE_PATH}")
    print(f"Wrote {NOTEBOOK_PATH}")
    print(f"Wrote {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
