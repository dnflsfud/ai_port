# -*- coding: utf-8 -*-
"""§S13 (2026-07-23) 유니버스 150→200 — universe_config.py 적용 대기 블록.

리프레시 당일(§S13 잔여 단계 ②)에 `machine/re_study/universe_config.py`에
적용한다. 오늘(목) 선적용하지 않는 이유: 일일 run_data_pipeline이 UNIVERSE를
소비하므로 가격 소스(Data/S&P500.xlsx)에 신규 50열이 생기기 전에 200으로
올리면 평일 런이 Missing/실패한다(§S11.1은 일요일 선적용이라 무해했음).

적용 3단계:
1. `EXPECTED_UNIVERSE_SIZE = 150` → `200`
2. UNIVERSE dict 끝("IBE SM Equity" 항목 뒤)에 아래 S13_ENTRIES 50개를
   이 순서 그대로 삽입 (insertion order = canonical column order,
   oppor.xlsx tickers EW1..GT1과 동일).
3. `build_factset_ticker_map()`의 vendor-spelling update에 추가:
       "BRK.B-US^": "BRK/B",
       "BRK-B-US^": "BRK/B",
   (워크북 r2는 BRK.B-US^ 표기 사용 — NOVO.B 선례.)
그리고 test_universe_config.py 핀 갱신: 150→200, tail-5 사전등록 값을
["CVX US Equity", "TTE FP Equity", "NEM US Equity", "SHW US Equity",
 "SO US Equity"]로 교체.
"""

S13_ENTRIES = {
    # S13 expansion 2026-07-23 (50) — 결정 로그 §S13, oppor.xlsx tickers 순서
    "INTU US Equity": {"name": "Intuit", "sector": "Technology", "color": "#365EBF"},
    "SNPS US Equity": {"name": "Synopsys", "sector": "Technology", "color": "#5A2D82"},
    "APH US Equity": {"name": "Amphenol", "sector": "Technology", "color": "#003087"},
    "MSI US Equity": {"name": "Motorola Solutions", "sector": "Technology", "color": "#005EB8"},
    "CRWD US Equity": {"name": "CrowdStrike", "sector": "Technology", "color": "#FC0000"},
    "NXPI US Equity": {"name": "NXP Semiconductors", "sector": "Technology", "color": "#F9B500"},
    "KEYS US Equity": {"name": "Keysight Technologies", "sector": "Technology", "color": "#E90029"},
    "ADSK US Equity": {"name": "Autodesk", "sector": "Technology", "color": "#0696D7"},
    "WDAY US Equity": {"name": "Workday", "sector": "Technology", "color": "#F38B00"},
    "WDC US Equity": {"name": "Western Digital", "sector": "Technology", "color": "#005DAA"},
    "DDOG US Equity": {"name": "Datadog", "sector": "Technology", "color": "#632CA6"},
    "FTNT US Equity": {"name": "Fortinet", "sector": "Technology", "color": "#EE3124"},
    "DSY FP Equity": {"name": "Dassault Systemes", "sector": "Technology", "color": "#005386"},
    "6146 JP Equity": {"name": "Disco", "sector": "Technology", "color": "#004098"},
    "6981 JP Equity": {"name": "Murata Manufacturing", "sector": "Technology", "color": "#C8003C"},
    "BRK/B US Equity": {"name": "Berkshire Hathaway", "sector": "Financials", "color": "#2C2A29"},
    "CB US Equity": {"name": "Chubb", "sector": "Financials", "color": "#01518A"},
    "ICE US Equity": {"name": "Intercontinental Exchange", "sector": "Financials", "color": "#002D62"},
    "MCO US Equity": {"name": "Moody's", "sector": "Financials", "color": "#0C51A3"},
    "PYPL US Equity": {"name": "PayPal", "sector": "Financials", "color": "#003087"},
    "COF US Equity": {"name": "Capital One", "sector": "Financials", "color": "#D03027"},
    "BNP FP Equity": {"name": "BNP Paribas", "sector": "Financials", "color": "#00915A"},
    "MUV2 GR Equity": {"name": "Munich Re", "sector": "Financials", "color": "#0A5A96"},
    "RTX US Equity": {"name": "RTX", "sector": "Industrials", "color": "#CE1126"},
    "EMR US Equity": {"name": "Emerson Electric", "sector": "Industrials", "color": "#004B8D"},
    "AXON US Equity": {"name": "Axon Enterprise", "sector": "Industrials", "color": "#FFB300"},
    "UBER US Equity": {"name": "Uber Technologies", "sector": "Industrials", "color": "#000000"},
    "AIR FP Equity": {"name": "Airbus", "sector": "Industrials", "color": "#00205B"},
    "SAF FP Equity": {"name": "Safran", "sector": "Industrials", "color": "#002F6C"},
    "MRK US Equity": {"name": "Merck", "sector": "Healthcare", "color": "#007A73"},
    "AMGN US Equity": {"name": "Amgen", "sector": "Healthcare", "color": "#0063C3"},
    "SYK US Equity": {"name": "Stryker", "sector": "Healthcare", "color": "#FFB500"},
    "MCK US Equity": {"name": "McKesson", "sector": "Healthcare", "color": "#F26522"},
    "NOVN SW Equity": {"name": "Novartis", "sector": "Healthcare", "color": "#0460A9"},
    "NKE US Equity": {"name": "Nike", "sector": "Consumer Discretionary", "color": "#111111"},
    "ORLY US Equity": {"name": "O'Reilly Automotive", "sector": "Consumer Discretionary", "color": "#007934"},
    "DASH US Equity": {"name": "DoorDash", "sector": "Consumer Discretionary", "color": "#FF3008"},
    "CFR SW Equity": {"name": "Richemont", "sector": "Consumer Discretionary", "color": "#6E1E2F"},
    "TTD US Equity": {"name": "Trade Desk", "sector": "Communication Services", "color": "#0099FA"},
    "RBLX US Equity": {"name": "Roblox", "sector": "Communication Services", "color": "#232527"},
    "9432 JP Equity": {"name": "NTT", "sector": "Communication Services", "color": "#0068B7"},
    "9433 JP Equity": {"name": "KDDI", "sector": "Communication Services", "color": "#EB5505"},
    "PEP US Equity": {"name": "PepsiCo", "sector": "Consumer Staples", "color": "#004B93"},
    "MDLZ US Equity": {"name": "Mondelez", "sector": "Consumer Staples", "color": "#4F2170"},
    "OR FP Equity": {"name": "L'Oreal", "sector": "Consumer Staples", "color": "#3C3C3B"},
    "CVX US Equity": {"name": "Chevron", "sector": "Energy", "color": "#0054A4"},
    "TTE FP Equity": {"name": "TotalEnergies", "sector": "Energy", "color": "#E1000F"},
    "NEM US Equity": {"name": "Newmont", "sector": "Materials", "color": "#FDB913"},
    "SHW US Equity": {"name": "Sherwin-Williams", "sector": "Materials", "color": "#0069AF"},
    "SO US Equity": {"name": "Southern Company", "sector": "Utilities", "color": "#1E4F9C"},
}

assert len(S13_ENTRIES) == 50
