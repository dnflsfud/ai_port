"""Build the canonical Data Analytics report artifact for universe comparison."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "universe_100_comparison"
RESULTS_PATH = OUT / "universe_100_comparison_results.json"
ARTIFACT_PATH = OUT / "artifact.json"

results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
generated_at = results["generated_at"]
headline = results["headline"]

sources = [
    {"id": "src-analysis", "label": "Universe proposal comparison results", "path": "outputs/universe_100_comparison/universe_100_comparison_results.json"},
    {"id": "src-prior", "label": "Prior 65-to-100 universe analysis", "path": "outputs/universe_100_recommendation/universe_100_results.json"},
    {"id": "src-fable", "label": "User-provided Fable 35-name candidate list"},
    {"id": "src-config", "label": "Model configuration: 1,260-session training window", "path": "src/config.py"},
    {"id": "src-arm", "label": "Arm IPO closing and Nasdaq trading date", "href": "https://newsroom.arm.com/news/arm-announces-closing-of-initial-public-offering"},
    {"id": "src-ceg", "label": "Constellation separation from Exelon", "href": "https://investors.constellationenergy.com/news-releases/news-release-details/constellation-reports-first-quarter-2022-results"},
    {"id": "src-advantest", "label": "Advantest investor overview", "href": "https://www.advantest.com/en/investors/individual-investors/"},
    {"id": "src-dte", "label": "Deutsche Telekom ownership of T-Mobile US", "href": "https://www.telekom.com/en/media/media-information/archive/t-mobile-us-1102078"},
    {"id": "src-kioxia", "label": "Kioxia listing announcement", "href": "https://www.kioxia-holdings.com/en-jp/news/2024/20241218-1.html"},
    {"id": "src-sandisk", "label": "Sandisk separation and Nasdaq listing", "href": "https://investor.sandisk.com/news-releases/news-release-details/sandisk-celebrates-nasdaq-listing-after-completing-separation"},
    {"id": "src-spotify", "label": "Spotify stock information", "href": "https://investors.spotify.com/stock-info/default.aspx"},
    {"id": "src-iberdrola", "label": "Iberdrola shareholder and investor information", "href": "https://www.iberdrola.com/shareholders-investors"},
]

proposal_summary_columns = [
    {"field": "proposal", "label": "안", "type": "text"},
    {"field": "us_add_names", "label": "신규 미국", "format": "number"},
    {"field": "non_us_add_names", "label": "신규 비미국", "format": "number"},
    {"field": "final_us_share", "label": "최종 미국", "format": "percent"},
    {"field": "final_uk_share", "label": "최종 영국", "format": "percent"},
    {"field": "final_europe_ex_uk_share", "label": "최종 유럽(영국 제외)", "format": "percent"},
    {"field": "final_asia_share", "label": "최종 아시아", "format": "percent"},
]

hybrid_columns = [
    {"field": "selection_order", "label": "순번", "format": "number"},
    {"field": "ticker", "label": "티커", "type": "text"},
    {"field": "name", "label": "종목명", "type": "text"},
    {"field": "sector", "label": "섹터", "type": "text"},
    {"field": "country", "label": "국가", "type": "text"},
    {"field": "source_status", "label": "선정 출처", "type": "text"},
    {"field": "history_gate", "label": "이력 게이트", "type": "text"},
    {"field": "selection_note", "label": "최종 판단", "type": "text"},
]

manifest = {
    "version": 1,
    "surface": "report",
    "title": "100종목 유니버스: 두 제안 비교와 하이브리드 추천",
    "description": "기존 35종목 제안과 Fable 제안을 현재 65종목 대비 중복, 지역·섹터 분산, 상대가치 페어, 데이터 이력으로 비교한 보고서",
    "generatedAt": generated_at,
    "sources": sources,
    "cards": [
        {"id": "card-consensus", "dataset": "headline", "sourceId": "src-analysis", "description": "두 제안이 함께 선택한 종목", "metrics": [{"label": "합의 종목", "field": "consensus_names", "format": "number"}]},
        {"id": "card-fable-us", "dataset": "headline", "sourceId": "src-analysis", "description": "Fable안 적용 후 미국 기업 종목 수 비중", "metrics": [{"label": "Fable 최종 미국", "field": "fable_final_us_share", "format": "percent"}]},
        {"id": "card-hybrid-us", "dataset": "headline", "sourceId": "src-analysis", "description": "하이브리드안 적용 후 미국 기업 종목 수 비중", "metrics": [{"label": "하이브리드 최종 미국", "field": "hybrid_final_us_share", "format": "percent"}]},
        {"id": "card-gates", "dataset": "headline", "sourceId": "src-analysis", "description": "1,260거래일 학습창을 아직 채우지 못한 추천 종목", "metrics": [{"label": "이력 게이트", "field": "hybrid_history_gates", "format": "number"}]},
    ],
    "charts": [
        {
            "id": "chart-region",
            "title": "제안별 최종 지역 구성",
            "subtitle": "하이브리드안은 Fable안보다 미국 종목을 5개 줄이면서 기존안보다 AI·상대가치 페어를 보강한다.",
            "intent": "comparison",
            "question": "각 제안을 적용한 최종 100종목의 경제적 본거지 구성은 어떻게 달라지는가?",
            "type": "bar",
            "dataset": "region_comparison",
            "sourceId": "src-analysis",
            "encodings": {
                "x": {"field": "region", "type": "nominal", "label": "지역"},
                "y": {"field": "final_share", "type": "quantitative", "format": "percent", "label": "최종 종목 수 비중"},
                "color": {"field": "proposal", "type": "nominal", "label": "제안"},
                "tooltip": [
                    {"field": "current_names", "type": "quantitative", "label": "현재 종목 수"},
                    {"field": "add_names", "type": "quantitative", "label": "추가 종목 수"},
                    {"field": "final_names", "type": "quantitative", "label": "최종 종목 수"},
                ],
            },
            "valueFormat": "percent",
            "layout": "full",
            "palette": {"kind": "categorical"},
            "settings": {"orientation": "horizontal", "groupMode": "grouped", "sort": "descending", "categoryLabelPolicy": "wrap", "showValues": True},
            "legend": {"position": "bottom", "title": "제안"},
            "surface": {"surface": "card", "viewMode": "visualization"},
        },
        {
            "id": "chart-sector",
            "title": "제안별 최종 섹터 종목 수",
            "subtitle": "하이브리드안은 산업재 1개를 헬스케어로 옮기고 유틸리티 2개를 유지해 방어·금리 팩터를 보강한다.",
            "intent": "comparison",
            "question": "최종 100종목의 섹터별 종목 수는 제안에 따라 어떻게 달라지는가?",
            "type": "bar",
            "dataset": "sector_comparison",
            "sourceId": "src-analysis",
            "encodings": {
                "x": {"field": "sector", "type": "nominal", "label": "섹터"},
                "y": {"field": "final_names", "type": "quantitative", "format": "number", "label": "최종 종목 수"},
                "color": {"field": "proposal", "type": "nominal", "label": "제안"},
                "tooltip": [
                    {"field": "current_names", "type": "quantitative", "label": "현재"},
                    {"field": "add_names", "type": "quantitative", "label": "추가"},
                    {"field": "final_names", "type": "quantitative", "label": "최종"},
                ],
            },
            "valueFormat": "number",
            "layout": "full",
            "palette": {"kind": "categorical"},
            "settings": {"orientation": "horizontal", "groupMode": "grouped", "sort": "descending", "categoryLabelPolicy": "wrap", "showValues": True},
            "legend": {"position": "bottom", "title": "제안"},
            "surface": {"surface": "card", "viewMode": "visualization"},
        },
    ],
    "tables": [
        {"id": "table-proposals", "title": "두 제안과 하이브리드안의 지역 비교", "subtitle": "발행기업의 경제적 본거지 기준", "dataset": "proposal_summaries", "sourceId": "src-analysis", "defaultSort": {"field": "final_us_share", "direction": "desc"}, "density": "dense", "layout": "full", "columns": proposal_summary_columns},
        {"id": "table-consensus", "title": "두 제안이 공통으로 선택한 18종목", "subtitle": "공통 종목은 기본 코어로 유지", "dataset": "consensus", "sourceId": "src-analysis", "defaultSort": {"field": "sector", "direction": "asc"}, "density": "dense", "layout": "full", "columns": [
            {"field": "ticker", "label": "티커", "type": "text"}, {"field": "name", "label": "종목명", "type": "text"}, {"field": "sector", "label": "섹터", "type": "text"}, {"field": "region", "label": "지역", "type": "text"}
        ]},
        {"id": "table-hybrid", "title": "최종 추천 신규 35종목", "subtitle": "공통 18개 + 재선정 17개; 키옥시아와 샌디스크 필수 포함", "dataset": "hybrid_candidates", "sourceId": "src-analysis", "defaultSort": {"field": "selection_order", "direction": "asc"}, "density": "dense", "layout": "full", "columns": hybrid_columns},
        {"id": "table-conflicts", "title": "충돌 종목 결정표", "subtitle": "종목 스토리보다 현재 65종목 대비 한계 분산효과를 우선", "dataset": "conflict_decisions", "sourceId": "src-analysis", "defaultSort": {"field": "area", "direction": "asc"}, "density": "dense", "layout": "full", "columns": [
            {"field": "area", "label": "판단 영역", "type": "text"}, {"field": "selected", "label": "선택", "type": "text"}, {"field": "not_selected", "label": "제외", "type": "text"}, {"field": "decision", "label": "이유", "type": "text"}
        ]},
        {"id": "table-framework", "title": "후보 선정 점수표", "subtitle": "정성 스토리를 동일한 검증 규칙으로 바꾸는 권고안", "dataset": "selection_framework", "sourceId": "src-analysis", "defaultSort": {"field": "weight", "direction": "desc"}, "density": "dense", "layout": "full", "columns": [
            {"field": "criterion", "label": "기준", "type": "text"}, {"field": "weight", "label": "가중치", "format": "percent"}, {"field": "test", "label": "검증 내용", "type": "text"}
        ]},
        {"id": "table-gates", "title": "상장 이력 게이트 대상", "subtitle": "상장 전 마스킹과 최소 이력 충족 전 거래 제한", "dataset": "history_gates", "sourceId": "src-analysis", "defaultSort": {"field": "listing_date", "direction": "desc"}, "density": "dense", "layout": "full", "columns": [
            {"field": "ticker", "label": "티커", "type": "text"}, {"field": "name", "label": "종목명", "type": "text"}, {"field": "listing_date", "label": "거래 시작/독립 상장", "type": "date"}, {"field": "reason", "label": "처리", "type": "text"}
        ]},
    ],
    "blocks": [
        {"id": "title", "type": "markdown", "body": "# 100종목 유니버스: 두 제안 비교와 하이브리드 추천", "layout": "full"},
        {"id": "executive-summary", "type": "markdown", "sourceId": "src-analysis", "body": """## Executive Summary

- **결론은 두 안 중 하나를 그대로 채택하는 것이 아니라 공통 18종목을 코어로 두고 나머지 17종목만 재선정하는 것입니다.** 두 목록의 일치율은 51.4%입니다.
- **Fable안의 장점은 상대가치 페어와 AI 인프라 공급망 연결성입니다.** Advantest–기존 Teradyne, Schneider–기존 Eaton/Vertiv, Novo Nordisk–기존 Eli Lilly 같은 비교군이 좋아집니다. 반면 신규 35개 중 미국 기업이 18개여서 최종 미국 비중이 78%입니다.
- **기존안의 장점은 지역 분산입니다.** 최종 미국 비중은 69%지만 Deutsche Telekom–기존 T-Mobile US 같은 실질 중복과 통신·저성장 종목이 약점입니다.
- **권고 하이브리드안은 최종 미국 73%, 영국 8%, 유럽(영국 제외) 13%, 아시아 6%입니다.** 키옥시아와 샌디스크는 필수 포함하고, AI 공급망의 깊이는 유지하면서 미국·동일 테마 집중을 줄입니다.
- **검증 상태는 ‘조건부 공유’입니다.** 구조·중복·상장 이력은 검증했지만 35종목 전체의 최신 밸류에이션, 유동성, 시점 일치 재무데이터를 아직 동일 패널로 넣지 않았으므로 기대수익 순위로 해석하면 안 됩니다."""},
        {"id": "metrics", "type": "metric-strip", "cardIds": ["card-consensus", "card-fable-us", "card-hybrid-us", "card-gates"], "layout": "full"},
        {"id": "region-story", "type": "markdown", "sourceId": "src-analysis", "body": """## Fable안은 테마가 강하고, 기존안은 지역 분산이 강하다

현재 65종목은 경제적 본거지 기준 미국 기업이 60개입니다. Fable안은 AI·데이터센터와 미국 성장주 연결성이 강한 대신 최종 미국 기업이 78개가 됩니다. 기존안은 69개까지 낮추지만 일부 후보의 사업 중복이 큽니다. 하이브리드안은 미국 73개로 중간점을 택하되, 영국 8개·유럽 13개·아시아 6개를 확보합니다."""},
        {"id": "region-chart", "type": "chart", "chartId": "chart-region", "layout": "full"},
        {"id": "proposal-table", "type": "table", "tableId": "table-proposals", "layout": "full"},
        {"id": "consensus-story", "type": "markdown", "sourceId": "src-analysis", "body": """## 18종목은 논쟁하지 않고 공통 코어로 유지한다

공통 종목은 키옥시아, 샌디스크, ASML, SAP, KLA, Schneider, Siemens, Rolls-Royce, Quanta, HSBC, Novo Nordisk, AstraZeneca, Thermo Fisher, Booking, LVMH, Nestlé, Shell, Rio Tinto입니다. 이 18개는 두 독립 제안이 동시에 선택했고, 현재 유니버스의 유럽·일본 공백과 산업 공급망 공백을 함께 메웁니다."""},
        {"id": "consensus-table", "type": "table", "tableId": "table-consensus", "layout": "full"},
        {"id": "hybrid-story", "type": "markdown", "sourceId": "src-analysis", "body": """## 최종 35종목은 AI 공급망과 방어 팩터를 함께 보강한다

기술 10개는 ARM·Arista·Advantest를 받아 상대가치 페어를 강화하고, Cadence·STMicroelectronics를 남겨 EDA와 유럽 아날로그 반도체를 보완합니다. ServiceNow와 Seagate는 기존 소프트웨어·메모리 노출과의 중복 때문에 제외합니다.

비기술 25개에서는 Progressive·Boston Scientific·TJX·Philip Morris·Williams·Constellation·Spotify를 채택합니다. 동시에 Allianz·LSEG·Roche·Iberdrola·Vonovia를 유지해 보험, 시장 인프라, 진단, 규제 유틸리티, 유럽 주거용 부동산을 확보합니다."""},
        {"id": "sector-chart", "type": "chart", "chartId": "chart-sector", "layout": "full"},
        {"id": "hybrid-table", "type": "table", "tableId": "table-hybrid", "layout": "full"},
        {"id": "conflict-story", "type": "markdown", "sourceId": "src-analysis", "body": """## 제외 기준은 종목의 질이 아니라 현재 65종목과의 중복이다

Digital Realty는 좋은 데이터센터 자산이지만 현재 Equinix가 이미 있습니다. Vistra와 Constellation을 함께 넣으면 미국 merchant-power/AI 전력 테마가 겹치므로 Constellation 하나만 두고 Iberdrola를 짝으로 둡니다. Deutsche Telekom은 2026년 2월 기준 T-Mobile US 지분 52.8%를 보유해 기존 TMUS와 look-through 중복이 큽니다. [Deutsche Telekom 공식 자료](https://www.telekom.com/en/media/media-information/archive/t-mobile-us-1102078)"""},
        {"id": "conflict-table", "type": "table", "tableId": "table-conflicts", "layout": "full"},
        {"id": "framework-story", "type": "markdown", "sourceId": "src-analysis", "body": """## 종목 스토리를 5개 공통 점수로 바꾼다

후보 선정은 한계 분산효과 30%, 데이터 준비도 20%, 상대가치 비교군 기여 20%, 유동성·품질 15%, 테마·지배구조 중복 패널티 15%로 평가하는 방식을 권고합니다. 밸류에이션·모멘텀 데이터가 준비되기 전에는 이 점수로 우선순위를 정하되, 기대수익 점수처럼 사용하지 않습니다."""},
        {"id": "framework-table", "type": "table", "tableId": "table-framework", "layout": "full"},
        {"id": "history-story", "type": "markdown", "sourceId": "src-config", "body": """## 네 종목은 유니버스에는 넣되 즉시 거래 자격을 주지 않는다

현재 설정의 학습창은 1,260거래일입니다. 키옥시아는 2024년 12월 18일 상장했고, 샌디스크는 2025년 2월 24일 독립 거래를 시작했습니다. ARM은 2023년 9월 14일 Nasdaq 거래를 시작했고, Constellation은 2022년 2월 2일 Exelon 분사 후 정규 거래를 시작했습니다. 따라서 네 종목 모두 상장 전 마스킹과 최소 이력 게이트가 필요합니다. [Kioxia](https://www.kioxia-holdings.com/en-jp/news/2024/20241218-1.html) · [Sandisk](https://investor.sandisk.com/news-releases/news-release-details/sandisk-celebrates-nasdaq-listing-after-completing-separation) · [Arm](https://newsroom.arm.com/news/arm-announces-closing-of-initial-public-offering) · [Constellation](https://investors.constellationenergy.com/news-releases/news-release-details/constellation-reports-first-quarter-2022-results)"""},
        {"id": "history-table", "type": "table", "tableId": "table-gates", "layout": "full"},
        {"id": "next-steps", "type": "markdown", "sourceId": "src-analysis", "body": """## Recommended Next Steps

1. **35종목 전체의 데이터 적재를 먼저 완료합니다.** 현지 티커, 통화, FX, 거래소 달력, 기업행사, 재무항목 커버리지를 동일 체크리스트로 검증합니다.
2. **두 단계 자격 규칙을 적용합니다.** 유니버스에는 포함하되 가격·재무·학습 이력 기준을 통과하기 전에는 모델 학습과 실제 거래 대상에서 제외합니다.
3. **세 안을 같은 조건으로 재백테스트합니다.** 동일 기간·거래비용·회전율·종목 상한·섹터/국가 제약에서 IR, turnover, tracking error, optimizer failure, sector/country active risk를 비교합니다.
4. **하이브리드안을 기본안으로 사용합니다.** 재백테스트에서 Fable안이 비용 후 IR을 유의하게 개선하고 지역·테마 리스크 한도를 지킬 때만 Fable-only 종목을 추가합니다."""},
        {"id": "questions", "type": "markdown", "sourceId": "src-analysis", "body": """## Further Questions

- 최종 100종목은 종목 수 균형이 목표인가, 또는 국가·섹터별 자본 비중 제한도 함께 둘 것인가?
- ADR과 현지 상장 중 어떤 라인을 표준으로 사용할 것인가?
- 최소 이력 기준을 1,260거래일로 고정할지, 짧은 이력 종목에 별도 축소 모델을 허용할 것인가?
- 방위산업과 담배처럼 정책·ESG 민감 업종에 별도 허용 한도가 필요한가?"""},
        {"id": "caveats", "type": "markdown", "sourceId": "src-analysis", "body": """## Caveats

- 이 보고서의 지역 비중은 자본 비중이 아니라 **종목 수 비중**이며, 거래소가 아닌 발행기업의 경제적 본거지 기준입니다.
- 현재 운용 비중과 섹터 편중은 2026년 5월 22일 운영 산출물을 사용했고, 후보 35종목의 최신 시가총액·밸류에이션·유동성 패널은 포함하지 않았습니다.
- 종목 선정은 투자수익 보장이나 개별 매수 권고가 아니라 모델 유니버스 설계안입니다.
- 최종 검증 상태는 **Share with caveats**입니다. 데이터 적재와 동일 조건 재백테스트 전에는 운영 승인안으로 사용하면 안 됩니다."""},
    ],
}


def widget_source(dataset: str) -> dict:
    return {
        "id": f"src-{dataset}-sql",
        "label": f"Reviewed comparison dataset: {dataset}",
        "path": "outputs/universe_100_comparison/universe_100_comparison.sqlite",
        "query": {
            "id": f"query-{dataset}",
            "engine": "sqlite",
            "sql": f'SELECT * FROM "{dataset}"',
            "description": f"Read the reviewed {dataset} result table.",
            "tables_used": [dataset],
            "executed_at": generated_at,
            "metric_definitions": [
                "All shares are name-count shares, not capital weights.",
                "Regions use issuer economic domicile rather than listing exchange.",
            ],
        },
    }


for card in manifest["cards"]:
    card["source"] = widget_source(card["dataset"])
for chart in manifest["charts"]:
    chart["source"] = widget_source(chart["dataset"])
for table in manifest["tables"]:
    table["source"] = widget_source(table["dataset"])

snapshot = {
    "version": 1,
    "generatedAt": generated_at,
    "status": "ready",
    "datasets": {
        "headline": [headline],
        "proposal_summaries": results["proposal_summaries"],
        "region_comparison": results["region_comparison"],
        "sector_comparison": results["sector_comparison"],
        "consensus": results["consensus"],
        "hybrid_candidates": results["hybrid_candidates"],
        "conflict_decisions": results["conflict_decisions"],
        "selection_framework": results["selection_framework"],
        "history_gates": results["history_gates"],
    },
}

artifact = {"surface": "report", "manifest": manifest, "snapshot": snapshot, "sources": sources}
ARTIFACT_PATH.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"artifact={ARTIFACT_PATH}")
print(f"blocks={len(manifest['blocks'])}")
print(f"charts={len(manifest['charts'])}")
print(f"tables={len(manifest['tables'])}")
