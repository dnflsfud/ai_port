# -*- coding: utf-8 -*-
"""universe_config 200종 유니버스(§S11/§S13) 계약 테스트.

기대 명단은 결정 로그 §S11·§S13 슬레이트(=Data/oppor.xlsx tickers 시트 셀 순서)를
사전등록 값으로 고정한 것이다. §S11.2(2026-07-20)에서 MMC -> AON 교체,
§S13.1(2026-07-23)에서 150 -> 200 확장. universe_config가 이 계약과 어긋나면
데이터 파이프라인 전체(뉴스 감성·RL_Universe_Data·ai_signal_data)가 어긋난다.
"""

from universe_config import (
    EXPECTED_UNIVERSE_SIZE,
    FACTSET_MARKET_CODE,
    MARKET_TO_CURRENCY,
    UNIVERSE,
    build_factset_ticker_map,
    ticker_currency,
    validate_universe,
)

# §S11 슬레이트 (Tech 10 → Fin 13 → Comm 9 → CD 6 → HC 4 → Ind 3 → St 2 → Mat 2 → Util 1)
S11_EXPECTED_TAIL_50 = [
    "QCOM US Equity", "TXN US Equity", "ADBE US Equity", "NOW US Equity",
    "ACN US Equity", "IBM US Equity", "ADI US Equity", "DELL US Equity",
    "8035 JP Equity", "IFX GR Equity",
    "MS US Equity", "SCHW US Equity", "AXP US Equity", "PGR US Equity",
    "AON US Equity", "CME US Equity", "KKR US Equity", "BN US Equity",
    "C US Equity", "LSEG LN Equity", "ZURN SW Equity", "CS FP Equity",
    "8306 JP Equity",
    "DIS US Equity", "CMCSA US Equity", "VZ US Equity", "T US Equity",
    "TTWO US Equity", "PUB FP Equity", "7974 JP Equity", "DTE GR Equity",
    "UMG NA Equity",
    "LOW US Equity", "TJX US Equity", "SBUX US Equity", "ABNB US Equity",
    "7203 JP Equity", "ITX SM Equity",
    "ABT US Equity", "DHR US Equity", "VRTX US Equity", "SAN FP Equity",
    "GE US Equity", "TT US Equity", "ABBN SW Equity",
    "KO US Equity", "ULVR LN Equity",
    "ECL US Equity", "AI FP Equity",
    "IBE SM Equity",
]

# §S13 슬레이트 (Tech 15 → Fin 8 → Ind 6 → HC 5 → CD 4 → Comm 4 → St 3 → En 2 → Mat 2 → Ut 1)
S13_EXPECTED_TAIL_50 = [
    "INTU US Equity", "SNPS US Equity", "APH US Equity", "MSI US Equity",
    "CRWD US Equity", "NXPI US Equity", "KEYS US Equity", "ADSK US Equity",
    "WDAY US Equity", "WDC US Equity", "DDOG US Equity", "FTNT US Equity",
    "DSY FP Equity", "6146 JP Equity", "6981 JP Equity",
    "BRK/B US Equity", "CB US Equity", "ICE US Equity", "MCO US Equity",
    "PYPL US Equity", "COF US Equity", "BNP FP Equity", "MUV2 GR Equity",
    "RTX US Equity", "EMR US Equity", "AXON US Equity", "UBER US Equity",
    "AIR FP Equity", "SAF FP Equity",
    "MRK US Equity", "AMGN US Equity", "SYK US Equity", "MCK US Equity",
    "NOVN SW Equity",
    "NKE US Equity", "ORLY US Equity", "DASH US Equity", "CFR SW Equity",
    "TTD US Equity", "RBLX US Equity", "9432 JP Equity", "9433 JP Equity",
    "PEP US Equity", "MDLZ US Equity", "OR FP Equity",
    "CVX US Equity", "TTE FP Equity",
    "NEM US Equity", "SHW US Equity",
    "SO US Equity",
]

SECTOR_TAXONOMY = {
    "Technology", "Financials", "Communication Services",
    "Consumer Discretionary", "Consumer Staples", "Healthcare",
    "Industrials", "Energy", "Materials", "Utilities", "Real Estate",
}


def test_universe_size_is_200():
    assert EXPECTED_UNIVERSE_SIZE == 200
    assert len(UNIVERSE) == 200
    validate_universe()  # 예외 없이 통과해야 한다


def test_s11_block_members_and_order():
    # §S13 확장 후 §S11 블록은 101~150번째 (계약 불변)
    block = list(UNIVERSE)[-100:-50]
    assert block == S11_EXPECTED_TAIL_50


def test_s13_tail_50_members_and_order():
    tail = list(UNIVERSE)[-50:]
    assert tail == S13_EXPECTED_TAIL_50


def test_sm_market_mappings():
    assert MARKET_TO_CURRENCY["SM"] == "EUR"
    assert FACTSET_MARKET_CODE["SM"] == "ES"


def test_s11_currency_mix():
    counts = {}
    for ticker in S11_EXPECTED_TAIL_50:
        currency = ticker_currency(ticker)
        counts[currency] = counts.get(currency, 0) + 1
    assert counts == {"USD": 33, "EUR": 9, "JPY": 4, "CHF": 2, "GBP": 2}


def test_s13_currency_mix():
    # §S13: 신규 FX 페어 0 — USD 37·EUR 7·JPY 4·CHF 2
    counts = {}
    for ticker in S13_EXPECTED_TAIL_50:
        currency = ticker_currency(ticker)
        counts[currency] = counts.get(currency, 0) + 1
    assert counts == {"USD": 37, "EUR": 7, "JPY": 4, "CHF": 2}


def test_factset_map_covers_new_markets():
    factset_map = build_factset_ticker_map()  # 시장 코드 누락 시 KeyError
    expected_aliases = {
        "ITX-ES^": "ITX",
        "IBE-ES^": "IBE",
        "CS-FR^": "CS",
        "8035-JP^": "8035",
        "8306-JP^": "8306",
        "UMG-NL^": "UMG",
        "SAN-FR^": "SAN",
        "ABBN-CH^": "ABBN",
        "LSEG-GB^": "LSEG",
        "ULVR-GB^": "ULVR",
        "IFX-DE^": "IFX",
        "DTE-DE^": "DTE",
        "PUB-FR^": "PUB",
        "AI-FR^": "AI",
        # §S13 신규 시장/특수 표기
        "DSY-FR^": "DSY",
        "MUV2-DE^": "MUV2",
        "NOVN-CH^": "NOVN",
        "CFR-CH^": "CFR",
        "6146-JP^": "6146",
        "9432-JP^": "9432",
        # BRK/B: 워크북 r2는 FactSet 점 표기(BRK.B) — NOVO.B 선례
        "BRK.B-US^": "BRK/B",
        "BRK-B-US^": "BRK/B",
    }
    for alias, simple in expected_aliases.items():
        assert factset_map.get(alias) == simple


def test_new_members_use_known_sector_taxonomy():
    for ticker in S11_EXPECTED_TAIL_50 + S13_EXPECTED_TAIL_50:
        metadata = UNIVERSE.get(ticker)
        assert metadata is not None, f"missing: {ticker}"
        assert metadata["sector"] in SECTOR_TAXONOMY, ticker
