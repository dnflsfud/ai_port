"""Patch the full universe report artifact with the global 100-name proposal."""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "universe_100_recommendation"
ARTIFACT_PATH = OUT / "artifact.json"
RESULTS_PATH = OUT / "universe_100_results.json"

artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
manifest = artifact["manifest"]
snapshot = artifact["snapshot"]

old_block_ids = [block["id"] for block in manifest["blocks"]]
old_chart_ids = [chart["id"] for chart in manifest.get("charts", [])]
old_table_ids = [table["id"] for table in manifest.get("tables", [])]
old_card_ids = [card["id"] for card in manifest.get("cards", [])]

generated_at = results["generated_at"]
summary = results["summary"]


def replace_by_id(items: list[dict], item_id: str, replacement: dict) -> None:
    for index, item in enumerate(items):
        if item.get("id") == item_id:
            items[index] = replacement
            return
    raise KeyError(item_id)


def upsert_by_id(items: list[dict], replacement: dict) -> None:
    for index, item in enumerate(items):
        if item.get("id") == replacement["id"]:
            items[index] = replacement
            return
    items.append(replacement)


snapshot["generatedAt"] = generated_at
snapshot["status"] = "ready"
snapshot["datasets"]["summary"] = [summary]
snapshot["datasets"]["current_sector"] = results["current_sector"]
snapshot["datasets"]["proposed_sector"] = results["proposed_sector"]
snapshot["datasets"]["proposed_country"] = results["proposed_country"]

candidate_rows = []
for selection_order, candidate in enumerate(results["candidates"], start=1):
    row = copy.deepcopy(candidate)
    row["selection_order"] = selection_order
    row["role_label"] = {
        "Required": "필수",
        "Core": "핵심",
        "Diversifier": "분산",
    }[row["priority"]]
    row["history_gate_label"] = "필수" if row["history_gate"] else "일반"
    candidate_rows.append(row)
snapshot["datasets"]["candidates"] = candidate_rows
snapshot["datasets"]["history_gates"] = results["history_gates"]

region_mix_long = []
for region in results["region_mix"]:
    for metric, field in (("현재", "current_share"), ("제안", "final_share")):
        region_mix_long.append(
            {
                "region": region["region"],
                "metric": metric,
                "value": region[field],
                "current_names": region["current_names"],
                "add_names": region["add_names"],
                "final_names": region["final_names"],
                "current_share": region["current_share"],
                "final_share": region["final_share"],
            }
        )
snapshot["datasets"]["region_mix_long"] = region_mix_long

replace_by_id(
    manifest["blocks"],
    "executive-summary",
    {
        "id": "executive-summary",
        "type": "markdown",
        "sourceId": "src-analysis",
        "body": """## Executive Summary

- **현재 핵심 문제는 기술주 자본 편중과 미국 기업 편중입니다.** 기술주는 24개로 전체의 36.9%지만 마지막 운영 포트폴리오에서 68.5%를 차지하고, 경제적 본거지 기준 미국 기업은 60개로 92.3%입니다.
- **신규 35종목은 섹터와 지역을 동시에 분산하는 안을 권고합니다.** 최종 기술주 종목 수 비중은 34%로 낮아지고, 지역 구성은 미국 69%, 영국 9%, 유럽(영국 제외) 17%, 아시아 5%가 됩니다.
- **키옥시아(285A JP)와 샌디스크(SNDK US)는 포함했습니다.** 두 종목은 독립 상장 이력이 짧아 상장 전 구간 마스킹과 최소 이력 충족 전 거래 제한을 전제로 합니다.
- **이번 명단은 구조 개선안이며 즉시 운영 승인안은 아닙니다.** 현지 통화 환산, 거래소별 휴장일, 데이터 커버리지, 기업행사 처리와 100종목 재백테스트를 통과한 뒤 적용해야 합니다.""",
    },
)

metrics_block = next(block for block in manifest["blocks"] if block["id"] == "metrics")
metrics_block["cardIds"] = [
    "card-current",
    "card-tech",
    "card-final",
    "card-final-tech",
    "card-global",
]

upsert_by_id(
    manifest["cards"],
    {
        "id": "card-global",
        "dataset": "summary",
        "sourceId": "src-summary-sql",
        "description": "발행기업의 경제적 본거지 기준 비미국 종목 수 비중",
        "metrics": [
            {"label": "최종 비미국 비중", "field": "final_non_us_share", "format": "percent"},
            {"label": "현재 비미국 비중", "field": "current_non_us_share", "format": "percent"},
        ],
    },
)

replace_by_id(
    manifest["blocks"],
    "proposed-story",
    {
        "id": "proposed-story",
        "type": "markdown",
        "sourceId": "src-analysis",
        "body": """## 35개는 기술 10개·비기술 25개로 배분한다

기술주는 키옥시아·샌디스크와 ASML·SAP·STMicroelectronics를 포함해 메모리, 반도체 장비, EDA, 네트워킹, 서버, 소프트웨어를 보강하되 10개로 제한합니다. 나머지 25개는 헬스케어 5개, 산업재·금융 각 4개, 경기소비재 3개, 필수소비재·에너지·커뮤니케이션 각 2개, 부동산·소재·유틸리티 각 1개로 배분합니다.

**의미:** AI 공급망의 깊이는 유지하면서 종목 수 기준 기술주 비중은 36.9%에서 34.0%로 낮아집니다. 동시에 보험, 유럽 산업자동화, 글로벌 제약, 럭셔리, 통신, 광산, 유틸리티 등 현재 빈 영역을 보완합니다.""",
    },
)

geo_blocks = [
    {
        "id": "geo-story",
        "type": "markdown",
        "sourceId": "src-geo-sql",
        "body": """## 미국 92.3%를 69%로 낮추고 영국·유럽을 26% 확보한다

현재 65종목은 경제적 본거지 기준 미국 60개, 한국 2개, 대만·영국·태국 각 1개입니다. 글로벌 후보 35개를 더하면 미국 69개, 영국 9개, 유럽(영국 제외) 17개, 아시아 5개가 됩니다. 비미국 종목 비중은 7.7%에서 31.0%로 높아집니다.

**의미:** 미국 기술주 단일 레짐에 대한 의존을 줄이면서, 유럽의 산업자동화·제약·금융·필수소비재와 영국의 은행·에너지·통신·소재를 독립 신호 원천으로 확보합니다.""",
    },
    {"id": "geo-chart", "type": "chart", "chartId": "chart-geo", "layout": "full"},
    {"id": "country-table", "type": "table", "tableId": "table-country", "layout": "full"},
    {
        "id": "listing-note",
        "type": "markdown",
        "body": """### 현지 상장 티커를 우선한다

대표적으로 ASML은 Euronext Amsterdam이 주 거래시장이고, SAP·Siemens·Allianz·Deutsche Telekom은 각 회사가 공개한 Bloomberg 독일 티커를 사용했습니다. LSEG도 회사 IR 페이지의 Bloomberg 표기인 `LSEG LN`을 따랐습니다. [ASML 주식정보](https://www.asml.com/en/en/investors/shares) · [SAP 기본정보](https://www.sap.com/investors/en/stock/basic-data.html) · [Siemens 주식정보](https://www.siemens.com/de-de/company/investor-relations/share-bonds-rating/basic-data-key-share-figures/) · [LSEG IR](https://www.lseg.com/en/investor-relations)

실제 적재 전에는 데이터 공급사 심볼 규칙과 `Equity` 접미사를 일괄 검증해야 합니다.""",
    },
]

blocks_without_geo = [
    block
    for block in manifest["blocks"]
    if block["id"] not in {"geo-story", "geo-chart", "country-table", "listing-note"}
]
insert_at = next(index for index, block in enumerate(blocks_without_geo) if block["id"] == "candidate-story")
manifest["blocks"] = blocks_without_geo[:insert_at] + geo_blocks + blocks_without_geo[insert_at:]

upsert_by_id(
    manifest["charts"],
    {
        "id": "chart-geo",
        "title": "현재 및 제안 유니버스의 지역별 종목 수 비중",
        "subtitle": "경제적 본거지 기준; 신규 35개 중 미국 9개, 영국 8개, 유럽(영국 제외) 17개, 일본 1개",
        "intent": "comparison",
        "question": "글로벌 35개 추가 후 지역 편중은 얼마나 완화되는가?",
        "type": "bar",
        "dataset": "region_mix_long",
        "sourceId": "src-geo-sql",
        "encodings": {
            "x": {"field": "region", "type": "nominal", "label": "지역"},
            "y": {"field": "value", "type": "quantitative", "format": "percent", "label": "종목 수 비중"},
            "color": {"field": "metric", "type": "nominal", "label": "구분"},
            "tooltip": [
                {"field": "current_names", "type": "quantitative", "label": "현재 종목 수"},
                {"field": "add_names", "type": "quantitative", "label": "신규 종목 수"},
                {"field": "final_names", "type": "quantitative", "label": "최종 종목 수"},
            ],
        },
        "valueFormat": "percent",
        "layout": "full",
        "palette": {"kind": "categorical"},
        "settings": {
            "orientation": "horizontal",
            "groupMode": "grouped",
            "sort": "descending",
            "categoryLabelPolicy": "wrap",
            "showValues": True,
        },
        "legend": {"position": "bottom", "title": "구분"},
        "surface": {"surface": "card", "viewMode": "visualization"},
    },
)

upsert_by_id(
    manifest["tables"],
    {
        "id": "table-country",
        "title": "최종 100종목의 국가별 구성",
        "subtitle": "거래소가 아니라 발행기업의 경제적 본거지 기준",
        "dataset": "proposed_country",
        "sourceId": "src-country-sql",
        "defaultSort": {"field": "final_names", "direction": "desc"},
        "density": "dense",
        "layout": "full",
        "columns": [
            {"field": "country", "label": "국가", "type": "text"},
            {"field": "current_names", "label": "현재", "format": "number"},
            {"field": "add_names", "label": "신규", "format": "number"},
            {"field": "final_names", "label": "최종", "format": "number"},
            {"field": "final_share", "label": "최종 비중", "format": "percent"},
        ],
    },
)

replace_by_id(
    manifest["blocks"],
    "candidate-story",
    {
        "id": "candidate-story",
        "type": "markdown",
        "sourceId": "src-analysis",
        "body": """## 추천 신규 35종목

아래 명단은 현재 유니버스와 중복되지 않으며 35개 티커가 모두 고유합니다. 미국 종목은 9개로 제한하고, 영국 8개, 독일 5개, 스위스 4개, 프랑스 3개, 네덜란드 2개, 덴마크·이탈리아·일본·스페인 각 1개를 추가합니다. 기술 후보는 메모리·노광·EDA·네트워킹·서버·유럽 소프트웨어로 분산하고, 비기술 후보는 유럽 산업재·금융·헬스케어·소비재를 중심으로 채웠습니다.""",
    },
)

candidate_table = next(table for table in manifest["tables"] if table["id"] == "table-candidates")
candidate_table["subtitle"] = "키옥시아와 샌디스크는 필수 편입; 미국 9개와 비미국 26개로 구성"
candidate_table["columns"] = [
    {"field": "selection_order", "label": "순번", "format": "number"},
    {"field": "ticker", "label": "티커", "type": "text"},
    {"field": "name", "label": "종목명", "type": "text"},
    {"field": "sector", "label": "섹터", "type": "text"},
    {"field": "subindustry", "label": "하위산업", "type": "text"},
    {"field": "country", "label": "국가", "type": "text"},
    {"field": "region", "label": "지역", "type": "text"},
    {"field": "role_label", "label": "역할", "type": "text"},
    {"field": "history_gate_label", "label": "이력 게이트", "type": "text"},
    {"field": "rationale", "label": "선정 이유", "type": "text"},
]

replace_by_id(
    manifest["blocks"],
    "history-gate",
    {
        "id": "history-gate",
        "type": "markdown",
        "body": """## 키옥시아와 샌디스크는 명단 포함, 거래는 이력 게이트

키옥시아는 도쿄증권거래소 프라임시장에 **2024년 12월 18일** 상장했고 증권코드는 **285A**입니다. [키옥시아 공식 상장 발표](https://www.kioxia-holdings.com/en-jp/news/2024/20241218-1.html)

샌디스크는 Western Digital에서 분리되어 **2025년 2월 24일**부터 Nasdaq에서 **SNDK**로 거래를 시작했습니다. [샌디스크 공식 상장 발표](https://investor.sandisk.com/news-releases/news-release-details/sandisk-celebrates-nasdaq-listing-after-completing-separation)

두 종목 모두 현재 모델의 약 5년 학습창보다 이력이 짧습니다. 유니버스 메타데이터에는 포함하되, 상장 전 가격·시가총액·펀더멘털 백필을 금지하고 최소 이력 기준을 충족할 때까지 거래 가능 여부를 별도 플래그로 관리하는 방식을 권고합니다.""",
    },
)

replace_by_id(
    manifest["blocks"],
    "next-steps",
    {
        "id": "next-steps",
        "type": "markdown",
        "sourceId": "src-config",
        "body": """## 권장 적용 순서

1. **35개 종목을 `Universe_Meta`에 먼저 추가**하고 섹터, 국가, 지역, 통화, 거래소, 현지 티커와 공급사 식별자를 고정합니다.
2. **현지 통화 가격을 기준통화로 환산**하고 GBP·CHF·DKK 등 현재 팩터 원천에 없는 환율 계열을 보강합니다.
3. **거래소별 휴장일과 시간대를 정렬**합니다. 현재 `weekday_index` 방식만으로는 런던·유럽·도쿄 휴장 차이를 충분히 표현하지 못합니다.
4. **모든 입력 시트의 컬럼 일치와 데이터 커버리지를 검증**하고, 상장 전 구간 마스킹과 최소 이력 게이트를 적용합니다.
5. **65종목 대 100종목을 같은 기간·같은 비용 가정으로 재백테스트**해 IR, 회전율, 추적오차, 섹터·국가 액티브 위험, 최적화 실패율을 비교합니다.
6. **게이트를 통과한 뒤에만 운영 유니버스를 전환**합니다. 이번 결과는 종목 선정안이지 운영 전환 승인안은 아닙니다.""",
    },
)

replace_by_id(
    manifest["blocks"],
    "questions",
    {
        "id": "questions",
        "type": "markdown",
        "body": """## 추가 확인할 질문

- 기준통화를 USD로 통일할지, 지역별 현지통화 수익률을 유지한 뒤 포트폴리오 단계에서 환산할지?
- GBP·CHF·DKK 환율과 런던·SIX·코펜하겐·유로넥스트·Xetra 휴장일을 어떤 원천으로 공급할지?
- 신규 35종목 모두에 대해 가격, 시가총액, 컨센서스, 뉴스, 공매도, 리비전 계열이 동일한 날짜 범위로 공급되는지?
- 100종목 시가총액 벤치마크가 기술주와 미국 주식 편중을 어느 정도 유지하는지?
- 샌디스크의 분할 이전 Western Digital 이력을 연결할지, 독립 상장 이후 데이터만 사용할지?""",
    },
)

replace_by_id(
    manifest["blocks"],
    "caveats",
    {
        "id": "caveats",
        "type": "markdown",
        "sourceId": "src-analysis",
        "body": """## 가정과 유의사항

- 종목 수 비중은 2026년 7월 14일 원천 엑셀 기준입니다. 실제 포트폴리오 비중은 2026년 5월 22일 마지막 운영 리밸런싱 산출물입니다.
- 현재 원천에는 국가 필드가 없어 경제적 본거지를 별도로 매핑했습니다. Linde는 영국 주요 본사, Fabrinet은 태국 중심 사업 기준으로 분류했으며, 운영 적재 시 공급사의 국가 분류 정책과 재확인해야 합니다.
- 섹터 분류는 프로젝트 원천 파일을 그대로 사용했습니다. 일부 종목은 표준 GICS 분류와 다를 수 있습니다.
- 팩터 원천에는 EURUSD·USDJPY·USDKRW가 있지만 GBP·CHF·DKK 환율이 없습니다. 또한 운영 캘린더는 `weekday_index`이므로 국가별 휴장일 검증이 필수입니다.
- 신규 35종목은 유동성·섹터·지역·하위산업·데이터 이력을 우선한 구조적 후보군입니다. 개별 종목의 목표가, 밸류에이션, 단기 모멘텀을 반영한 매수 추천은 아닙니다.
- 최종 자본 비중은 이번 종목 수 배분으로 확정되지 않습니다. 100종목 시가총액 벤치마크와 최적화 제약을 다시 계산해야 합니다.""",
    },
)

source_map = {source["id"]: source for source in manifest["sources"]}
source_map["src-kioxia"] = {
    "id": "src-kioxia",
    "label": "Kioxia Holdings Tokyo Stock Exchange listing announcement",
    "href": "https://www.kioxia-holdings.com/en-jp/news/2024/20241218-1.html",
}
source_map["src-sandisk"] = {
    "id": "src-sandisk",
    "label": "Sandisk Nasdaq listing announcement",
    "href": "https://investor.sandisk.com/news-releases/news-release-details/sandisk-celebrates-nasdaq-listing-after-completing-separation",
}
source_map["src-asml"] = {
    "id": "src-asml",
    "label": "ASML official share listing information",
    "href": "https://www.asml.com/en/en/investors/shares",
}
source_map["src-sap"] = {
    "id": "src-sap",
    "label": "SAP official basic stock data",
    "href": "https://www.sap.com/investors/en/stock/basic-data.html",
}
source_map["src-siemens"] = {
    "id": "src-siemens",
    "label": "Siemens official share identifiers",
    "href": "https://www.siemens.com/de-de/company/investor-relations/share-bonds-rating/basic-data-key-share-figures/",
}
source_map["src-lseg"] = {
    "id": "src-lseg",
    "label": "LSEG official investor relations page",
    "href": "https://www.lseg.com/en/investor-relations",
}
source_map["src-linde"] = {
    "id": "src-linde",
    "label": "Linde official corporate domicile and principal office information",
    "href": "https://www.linde.com/imprint",
}
source_map["src-fabrinet"] = {
    "id": "src-fabrinet",
    "label": "Fabrinet 2025 Form 10-K",
    "href": "https://www.sec.gov/Archives/edgar/data/1408710/000140871025000039/fn-20250627.htm",
}

source_map["src-summary-sql"] = {
    "id": "src-summary-sql",
    "label": "Universe expansion summary query",
    "path": "outputs/universe_100_recommendation/universe_100_analysis.sqlite",
    "query": {
        "engine": "SQLite",
        "language": "sql",
        "sql": "SELECT current_names, new_names, final_names, tech_name_share, tech_portfolio, tech_benchmark, final_tech_share, tech_share_change, current_us_share, final_us_share, current_non_us_share, final_non_us_share FROM summary;",
        "description": "Headline counts, technology concentration, and geographic concentration metrics.",
        "executed_at": generated_at,
        "tables_used": ["summary"],
        "metric_definitions": [
            "tech_name_share = current technology names / 65",
            "tech_portfolio = last operating portfolio technology weight",
            "final_tech_share = proposed technology names / 100",
            "current_non_us_share = 1 - current_us_share",
            "final_non_us_share = 1 - final_us_share",
        ],
    },
}
source_map["src-geo-sql"] = {
    "id": "src-geo-sql",
    "label": "Current and proposed regional mix query",
    "path": "outputs/universe_100_recommendation/universe_100_analysis.sqlite",
    "query": {
        "engine": "SQLite",
        "language": "sql",
        "sql": "SELECT region, metric, value, current_names, add_names, final_names, current_share, final_share FROM region_mix_long ORDER BY CASE region WHEN 'United States' THEN 1 WHEN 'United Kingdom' THEN 2 WHEN 'Europe ex-UK' THEN 3 ELSE 4 END, metric;",
        "description": "Current and proposed regional name counts and shares using economic domicile.",
        "executed_at": generated_at,
        "tables_used": ["region_mix_long"],
        "filters": ["Current universe = 65 names", "Proposed universe = 100 names"],
        "metric_definitions": [
            "current_share = current_names / 65",
            "final_share = final_names / 100",
        ],
    },
}
source_map["src-country-sql"] = {
    "id": "src-country-sql",
    "label": "Proposed country mix query",
    "path": "outputs/universe_100_recommendation/universe_100_analysis.sqlite",
    "query": {
        "engine": "SQLite",
        "language": "sql",
        "sql": "SELECT country, current_names, add_names, final_names, final_share FROM proposed_country ORDER BY final_names DESC, country;",
        "description": "Final country composition after the reviewed 35-name additions.",
        "executed_at": generated_at,
        "tables_used": ["proposed_country"],
        "filters": ["Economic domicile classification", "Final universe = 100 names"],
        "metric_definitions": ["final_share = final_names / 100"],
    },
}
source_map["src-candidates-sql"] = {
    "id": "src-candidates-sql",
    "label": "Recommended global 35-name candidate query",
    "path": "outputs/universe_100_recommendation/universe_100_analysis.sqlite",
    "query": {
        "engine": "SQLite",
        "language": "sql",
        "sql": "SELECT selection_order, ticker, name, sector, subindustry, country, region, priority, history_gate_label, rationale, listing_date FROM candidates ORDER BY selection_order;",
        "description": "Reviewed global 35-name recommendation, including required Kioxia and Sandisk additions.",
        "executed_at": generated_at,
        "tables_used": ["candidates"],
        "filters": [
            "No overlap with current 65-name universe",
            "Unique ticker count = 35",
            "Kioxia and Sandisk required",
            "New additions = 9 United States and 26 non-United States names",
        ],
        "metric_definitions": ["history_gate_label = Required for short independent listing history, otherwise Standard"],
    },
}

for source_id in ("src-current-sql", "src-proposed-sql"):
    if source_id in source_map and "query" in source_map[source_id]:
        source_map[source_id]["query"]["executed_at"] = generated_at

manifest["sources"] = list(source_map.values())
artifact["sources"] = copy.deepcopy(manifest["sources"])

assert set(old_block_ids) <= {block["id"] for block in manifest["blocks"]}
assert set(old_chart_ids) <= {chart["id"] for chart in manifest["charts"]}
assert set(old_table_ids) <= {table["id"] for table in manifest["tables"]}
assert set(old_card_ids) <= {card["id"] for card in manifest["cards"]}
assert len(snapshot["datasets"]["candidates"]) == 35
assert sum(row["final_names"] for row in snapshot["datasets"]["proposed_country"]) == 100

ARTIFACT_PATH.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Updated full artifact: {ARTIFACT_PATH}")
print(f"Blocks: {len(manifest['blocks'])}; charts: {len(manifest['charts'])}; tables: {len(manifest['tables'])}; cards: {len(manifest['cards'])}")
