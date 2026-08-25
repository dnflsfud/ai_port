# -*- coding: utf-8 -*-
"""§S14 (2026-08-25) 유니버스 200→250 — universe_config.py 적용 대기 블록.

리프레시 당일(§S14 잔여 단계 ③)에 `machine/re_study/universe_config.py`에
적용한다. 선적용하지 않는 이유는 §S13과 동일: 일일 run_data_pipeline이
UNIVERSE를 소비하므로 가격 소스(Data/S&P500.xlsx)에 신규 50열이 생기기 전에
250으로 올리면 평일 런이 Missing/실패한다.

적용 5단계:
1. `EXPECTED_UNIVERSE_SIZE = 200` → `250`
2. UNIVERSE dict 끝("SO US Equity" 항목 뒤)에 아래 S14_ENTRIES 50개를
   이 순서 그대로 삽입 (insertion order = canonical column order,
   oppor.xlsx tickers GU1..IR1과 동일).
3. `MARKET_TO_CURRENCY`에 `"IM": "EUR"` 추가 (밀라노 — UCG/ENEL.
   신규 FX 페어 0: EUR은 기지원, 거래소 코드만 신규).
4. `FACTSET_MARKET_CODE`에 `"IM": "IT"` 추가 (워크북 r2 = UCG-IT^/ENEL-IT^).
5. test_universe_config.py 핀 갱신: 200→250, tail-5 사전등록 값을
   ["APD US Equity", "4063 JP Equity", "NRG US Equity",
    "ENEL IM Equity", "WELL US Equity"]로 교체.
그리고 ai_port 측(잔여 단계 ⑤, TICKERS 250 확장 시):
`src/data_loader.py` MARKET_TO_CURRENCY에 `"IM": "EUR"` 추가.

조건부 명단(결정 로그 §S14 게이트 ③): † PIT/연속성 정책 등록 후 편입 —
APP·COHR·APO·HWM·ROP(로슈 재진입: ROG 3/16 종료→ROP 3/17, 백필 확인 필수)·
MDT·6758·RDDT·TKO. ‡ 기업행사 데이터 QA — PH(Filtration Group 8/13 종결)·
ZTS·MNST(8/11 2:1 분할)·SLB(ChampionX)·NRG(LS Power).
"""

S14_ENTRIES = {
    # S14 expansion 2026-08-25 (50) — 결정 로그 §S14, oppor.xlsx tickers 순서
    "APP US Equity": {"name": "AppLovin", "sector": "Technology", "color": "#00A5E0"},
    "SNOW US Equity": {"name": "Snowflake", "sector": "Technology", "color": "#29B5E8"},
    "NET US Equity": {"name": "Cloudflare", "sector": "Technology", "color": "#F38020"},
    "ZS US Equity": {"name": "Zscaler", "sector": "Technology", "color": "#005CA9"},
    "TEAM US Equity": {"name": "Atlassian", "sector": "Technology", "color": "#0052CC"},
    "COHR US Equity": {"name": "Coherent", "sector": "Technology", "color": "#0057B8"},
    "ON US Equity": {"name": "onsemi", "sector": "Technology", "color": "#D6001C"},
    "FICO US Equity": {"name": "Fair Isaac", "sector": "Technology", "color": "#147BB3"},
    "HPE US Equity": {"name": "Hewlett Packard Enterprise", "sector": "Technology", "color": "#01A982"},
    "VRSN US Equity": {"name": "VeriSign", "sector": "Technology", "color": "#003A70"},
    "ASM NA Equity": {"name": "ASM International", "sector": "Technology", "color": "#00A0DF"},
    "MRSH US Equity": {"name": "Marsh", "sector": "Financials", "color": "#002C77"},
    "APO US Equity": {"name": "Apollo Global Management", "sector": "Financials", "color": "#002855"},
    "MSCI US Equity": {"name": "MSCI", "sector": "Financials", "color": "#0D2240"},
    "BNY US Equity": {"name": "BNY", "sector": "Financials", "color": "#007D8A"},
    "IBKR US Equity": {"name": "Interactive Brokers", "sector": "Financials", "color": "#D81222"},
    "UBSG SW Equity": {"name": "UBS Group", "sector": "Financials", "color": "#EC0016"},
    "UCG IM Equity": {"name": "UniCredit", "sector": "Financials", "color": "#E2001A"},
    "8316 JP Equity": {"name": "Sumitomo Mitsui Financial Group", "sector": "Financials", "color": "#004831"},
    "ADP US Equity": {"name": "Automatic Data Processing", "sector": "Industrials", "color": "#D0271D"},
    "HWM US Equity": {"name": "Howmet Aerospace", "sector": "Industrials", "color": "#2C5697"},
    "WM US Equity": {"name": "Waste Management", "sector": "Industrials", "color": "#00834D"},
    "PH US Equity": {"name": "Parker Hannifin", "sector": "Industrials", "color": "#FFB81C"},
    "REL LN Equity": {"name": "RELX", "sector": "Industrials", "color": "#FF6C00"},
    "7011 JP Equity": {"name": "Mitsubishi Heavy Industries", "sector": "Industrials", "color": "#E60012"},
    "ROP SW Equity": {"name": "Roche", "sector": "Healthcare", "color": "#0B41CD"},
    "GILD US Equity": {"name": "Gilead Sciences", "sector": "Healthcare", "color": "#C8102E"},
    "PFE US Equity": {"name": "Pfizer", "sector": "Healthcare", "color": "#0093D0"},
    "MDT US Equity": {"name": "Medtronic", "sector": "Healthcare", "color": "#004B87"},
    "HCA US Equity": {"name": "HCA Healthcare", "sector": "Healthcare", "color": "#005EB8"},
    "ZTS US Equity": {"name": "Zoetis", "sector": "Healthcare", "color": "#EF7622"},
    "RMS FP Equity": {"name": "Hermes", "sector": "Consumer Discretionary", "color": "#F37021"},
    "6758 JP Equity": {"name": "Sony Group", "sector": "Consumer Discretionary", "color": "#000000"},
    "CMG US Equity": {"name": "Chipotle Mexican Grill", "sector": "Consumer Discretionary", "color": "#A81612"},
    "MAR US Equity": {"name": "Marriott International", "sector": "Consumer Discretionary", "color": "#A70023"},
    "GM US Equity": {"name": "General Motors", "sector": "Consumer Discretionary", "color": "#0170CE"},
    "GRMN US Equity": {"name": "Garmin", "sector": "Consumer Discretionary", "color": "#007CC3"},
    "RDDT US Equity": {"name": "Reddit", "sector": "Communication Services", "color": "#FF4500"},
    "LYV US Equity": {"name": "Live Nation Entertainment", "sector": "Communication Services", "color": "#E31837"},
    "TKO US Equity": {"name": "TKO Group", "sector": "Communication Services", "color": "#D20A0A"},
    "MO US Equity": {"name": "Altria", "sector": "Consumer Staples", "color": "#EE3524"},
    "CL US Equity": {"name": "Colgate-Palmolive", "sector": "Consumer Staples", "color": "#D2010D"},
    "MNST US Equity": {"name": "Monster Beverage", "sector": "Consumer Staples", "color": "#95D600"},
    "COP US Equity": {"name": "ConocoPhillips", "sector": "Energy", "color": "#E4002B"},
    "SLB US Equity": {"name": "SLB", "sector": "Energy", "color": "#0014DC"},
    "APD US Equity": {"name": "Air Products", "sector": "Materials", "color": "#0057B8"},
    "4063 JP Equity": {"name": "Shin-Etsu Chemical", "sector": "Materials", "color": "#0068B7"},
    "NRG US Equity": {"name": "NRG Energy", "sector": "Utilities", "color": "#F26722"},
    "ENEL IM Equity": {"name": "Enel", "sector": "Utilities", "color": "#172983"},
    "WELL US Equity": {"name": "Welltower", "sector": "Real Estate", "color": "#007A6E"},
}

assert len(S14_ENTRIES) == 50
