# -*- coding: utf-8 -*-
"""Shared 200-stock universe configuration for the data pipeline.

§S13.1 (2026-07-23): 150 -> 200 확장 (결정 로그 §S13, oppor.xlsx tickers 순서).
"""

from __future__ import annotations

from collections.abc import Iterable


EXPECTED_UNIVERSE_SIZE = 200


# Insertion order is the canonical output-column order.
UNIVERSE = {
    "AAPL US Equity": {"name": "Apple", "sector": "Technology", "color": "#A2AAAD"},
    "MSFT US Equity": {"name": "Microsoft", "sector": "Technology", "color": "#00A4EF"},
    "GOOGL US Equity": {"name": "Alphabet", "sector": "Technology", "color": "#4285F4"},
    "AMZN US Equity": {"name": "Amazon", "sector": "Consumer Discretionary", "color": "#FF9900"},
    "META US Equity": {"name": "Meta", "sector": "Technology", "color": "#0668E1"},
    "NVDA US Equity": {"name": "Nvidia", "sector": "Technology", "color": "#76B900"},
    "TSLA US Equity": {"name": "Tesla", "sector": "Consumer Discretionary", "color": "#E82127"},
    "PLTR US Equity": {"name": "Palantir", "sector": "Technology", "color": "#000000"},
    "AVGO US Equity": {"name": "Broadcom", "sector": "Technology", "color": "#CC0000"},
    "MU US Equity": {"name": "Micron", "sector": "Technology", "color": "#0033A0"},
    "GEV US Equity": {"name": "GE Vernova", "sector": "Industrials", "color": "#0066CC"},
    "VRT US Equity": {"name": "Vertiv", "sector": "Industrials", "color": "#9B59B6"},
    "BE US Equity": {"name": "Bloom Energy", "sector": "Industrials", "color": "#20B2AA"},
    "LITE US Equity": {"name": "Lumentum", "sector": "Technology", "color": "#FF6F00"},
    "000660 KS Equity": {"name": "SK Hynix", "sector": "Technology", "color": "#E4002B"},
    "005930 KS Equity": {"name": "Samsung Electronics", "sector": "Technology", "color": "#1428A0"},
    "UNH US Equity": {"name": "UnitedHealth", "sector": "Healthcare", "color": "#002677"},
    "LLY US Equity": {"name": "Eli Lilly", "sector": "Healthcare", "color": "#D52B1E"},
    "ISRG US Equity": {"name": "Intuitive Surgical", "sector": "Healthcare", "color": "#003DA5"},
    "ABBV US Equity": {"name": "AbbVie", "sector": "Healthcare", "color": "#071D49"},
    "REGN US Equity": {"name": "Regeneron", "sector": "Healthcare", "color": "#003B5C"},
    "JPM US Equity": {"name": "JPMorgan", "sector": "Financials", "color": "#003594"},
    "V US Equity": {"name": "Visa", "sector": "Financials", "color": "#1A1F71"},
    "MA US Equity": {"name": "Mastercard", "sector": "Financials", "color": "#EB001B"},
    "BLK US Equity": {"name": "BlackRock", "sector": "Financials", "color": "#000000"},
    "SPGI US Equity": {"name": "S&P Global", "sector": "Financials", "color": "#CC0000"},
    "GS US Equity": {"name": "Goldman Sachs", "sector": "Financials", "color": "#6FA8DC"},
    "COST US Equity": {"name": "Costco", "sector": "Consumer Staples", "color": "#E31837"},
    "HD US Equity": {"name": "Home Depot", "sector": "Consumer Discretionary", "color": "#F96302"},
    "PG US Equity": {"name": "Procter & Gamble", "sector": "Consumer Staples", "color": "#003DA5"},
    "MCD US Equity": {"name": "McDonald's", "sector": "Consumer Discretionary", "color": "#FFC72C"},
    "WMT US Equity": {"name": "Walmart", "sector": "Consumer Staples", "color": "#0071CE"},
    "CAT US Equity": {"name": "Caterpillar", "sector": "Industrials", "color": "#FFCD11"},
    "HON US Equity": {"name": "Honeywell", "sector": "Industrials", "color": "#E10600"},
    "DE US Equity": {"name": "Deere", "sector": "Industrials", "color": "#367C2B"},
    "UNP US Equity": {"name": "Union Pacific", "sector": "Industrials", "color": "#003B71"},
    "LMT US Equity": {"name": "Lockheed Martin", "sector": "Industrials", "color": "#003B6F"},
    "ETN US Equity": {"name": "Eaton", "sector": "Industrials", "color": "#003DA5"},
    "XOM US Equity": {"name": "Exxon Mobil", "sector": "Energy", "color": "#ED1C24"},
    "LNG US Equity": {"name": "Cheniere Energy", "sector": "Energy", "color": "#00529B"},
    "FCX US Equity": {"name": "Freeport-McMoRan", "sector": "Materials", "color": "#003366"},
    "LIN US Equity": {"name": "Linde", "sector": "Materials", "color": "#005591"},
    "NEE US Equity": {"name": "NextEra Energy", "sector": "Utilities", "color": "#5B2C6F"},
    "AMT US Equity": {"name": "American Tower", "sector": "Real Estate", "color": "#003087"},
    "EQIX US Equity": {"name": "Equinix", "sector": "Real Estate", "color": "#ED1C24"},
    "TMUS US Equity": {"name": "T-Mobile", "sector": "Communication Services", "color": "#E20074"},
    "PLD US Equity": {"name": "Prologis", "sector": "Real Estate", "color": "#0C2340"},
    "AMD US Equity": {"name": "AMD", "sector": "Technology", "color": "#ED1C24"},
    "CRM US Equity": {"name": "Salesforce", "sector": "Technology", "color": "#00A1E0"},
    "NFLX US Equity": {"name": "Netflix", "sector": "Communication Services", "color": "#E50914"},
    "TER US Equity": {"name": "Teradyne", "sector": "Technology", "color": "#0033A0"},
    "GLW US Equity": {"name": "Corning", "sector": "Technology", "color": "#A50034"},
    "JNJ US Equity": {"name": "Johnson & Johnson", "sector": "Healthcare", "color": "#CC0000"},
    "WFC US Equity": {"name": "Wells Fargo", "sector": "Financials", "color": "#D71E2B"},
    "MPC US Equity": {"name": "Marathon Petroleum", "sector": "Energy", "color": "#00529B"},
    "LRCX US Equity": {"name": "Lam Research", "sector": "Technology", "color": "#003DA5"},
    "AMAT US Equity": {"name": "Applied Materials", "sector": "Technology", "color": "#00A9E0"},
    "PANW US Equity": {"name": "Palo Alto Networks", "sector": "Technology", "color": "#FA582D"},
    "FN US Equity": {"name": "Fabrinet", "sector": "Technology", "color": "#1F3864"},
    "MPWR US Equity": {"name": "Monolithic Power", "sector": "Technology", "color": "#0066B3"},
    "BAC US Equity": {"name": "Bank of America", "sector": "Financials", "color": "#E31837"},
    "CSCO US Equity": {"name": "Cisco", "sector": "Technology", "color": "#1BA0D7"},
    "INTC US Equity": {"name": "Intel", "sector": "Technology", "color": "#0071C5"},
    "ORCL US Equity": {"name": "Oracle", "sector": "Technology", "color": "#F80000"},
    "TSM US Equity": {"name": "Taiwan Semiconductor", "sector": "Technology", "color": "#CC0000"},
    "SNDK US Equity": {"name": "Sandisk", "sector": "Technology", "color": "#E2231A"},
    "KLAC US Equity": {"name": "KLA", "sector": "Technology", "color": "#6F2C91"},
    "ANET US Equity": {"name": "Arista Networks", "sector": "Technology", "color": "#0099CC"},
    "MRVL US Equity": {"name": "Marvell Technology", "sector": "Technology", "color": "#7A1E8A"},
    "CDNS US Equity": {"name": "Cadence Design Systems", "sector": "Technology", "color": "#0067B1"},
    "STX US Equity": {"name": "Seagate Technology", "sector": "Technology", "color": "#6EBE44"},
    "PWR US Equity": {"name": "Quanta Services", "sector": "Industrials", "color": "#F28C28"},
    "BX US Equity": {"name": "Blackstone", "sector": "Financials", "color": "#000000"},
    "TMO US Equity": {"name": "Thermo Fisher Scientific", "sector": "Healthcare", "color": "#E31937"},
    "BSX US Equity": {"name": "Boston Scientific", "sector": "Healthcare", "color": "#005EB8"},
    "BKNG US Equity": {"name": "Booking Holdings", "sector": "Consumer Discretionary", "color": "#003580"},
    "PM US Equity": {"name": "Philip Morris International", "sector": "Consumer Staples", "color": "#1C3F60"},
    "WMB US Equity": {"name": "Williams Companies", "sector": "Energy", "color": "#0072CE"},
    "CEG US Equity": {"name": "Constellation Energy", "sector": "Utilities", "color": "#00AEEF"},
    "VST US Equity": {"name": "Vistra", "sector": "Utilities", "color": "#F5A623"},
    "DLR US Equity": {"name": "Digital Realty", "sector": "Real Estate", "color": "#0067A0"},
    "ARM US Equity": {"name": "Arm Holdings", "sector": "Technology", "color": "#00C1DE"},
    "SPOT US Equity": {"name": "Spotify", "sector": "Communication Services", "color": "#1DB954"},
    "RACE US Equity": {"name": "Ferrari", "sector": "Consumer Discretionary", "color": "#D40000"},
    "285A JP Equity": {"name": "Kioxia Holdings", "sector": "Technology", "color": "#005BAC"},
    "6857 JP Equity": {"name": "Advantest", "sector": "Technology", "color": "#E60012"},
    "SU FP Equity": {"name": "Schneider Electric", "sector": "Industrials", "color": "#3DCD58"},
    "SIE GR Equity": {"name": "Siemens", "sector": "Industrials", "color": "#009999"},
    "RHM GR Equity": {"name": "Rheinmetall", "sector": "Industrials", "color": "#0066B3"},
    "ALV GR Equity": {"name": "Allianz", "sector": "Financials", "color": "#003781"},
    "MC FP Equity": {"name": "LVMH", "sector": "Consumer Discretionary", "color": "#1A1A1A"},
    "NESN SW Equity": {"name": "Nestle", "sector": "Consumer Staples", "color": "#8B5E3C"},
    "RR/ LN Equity": {"name": "Rolls-Royce Holdings", "sector": "Industrials", "color": "#1E3A8A"},
    "SAP GR Equity": {"name": "SAP", "sector": "Technology", "color": "#0FAAFF"},
    "ASML NA Equity": {"name": "ASML", "sector": "Technology", "color": "#0B1F8C"},
    "AZN LN Equity": {"name": "AstraZeneca", "sector": "Healthcare", "color": "#8A1B61"},
    "SHEL LN Equity": {"name": "Shell", "sector": "Energy", "color": "#FFD500"},
    "HSBA LN Equity": {"name": "HSBC", "sector": "Financials", "color": "#DB0011"},
    "NOVOB DC Equity": {"name": "Novo Nordisk", "sector": "Healthcare", "color": "#003D7C"},
    "RIO LN Equity": {"name": "Rio Tinto", "sector": "Materials", "color": "#9B1B30"},
    # S11 expansion 2026-07-18 (50) — 결정 로그 §S11, oppor.xlsx tickers 순서
    # §S11.2 (2026-07-20): MMC -> AON 교체 (MMC는 가격 소스 데이터 부재)
    "QCOM US Equity": {"name": "Qualcomm", "sector": "Technology", "color": "#3253DC"},
    "TXN US Equity": {"name": "Texas Instruments", "sector": "Technology", "color": "#CC0000"},
    "ADBE US Equity": {"name": "Adobe", "sector": "Technology", "color": "#FA0F00"},
    "NOW US Equity": {"name": "ServiceNow", "sector": "Technology", "color": "#62D84E"},
    "ACN US Equity": {"name": "Accenture", "sector": "Technology", "color": "#A100FF"},
    "IBM US Equity": {"name": "IBM", "sector": "Technology", "color": "#0F62FE"},
    "ADI US Equity": {"name": "Analog Devices", "sector": "Technology", "color": "#0067B9"},
    "DELL US Equity": {"name": "Dell Technologies", "sector": "Technology", "color": "#007DB8"},
    "8035 JP Equity": {"name": "Tokyo Electron", "sector": "Technology", "color": "#E60027"},
    "IFX GR Equity": {"name": "Infineon Technologies", "sector": "Technology", "color": "#0A8276"},
    "MS US Equity": {"name": "Morgan Stanley", "sector": "Financials", "color": "#002B5C"},
    "SCHW US Equity": {"name": "Charles Schwab", "sector": "Financials", "color": "#00A0DF"},
    "AXP US Equity": {"name": "American Express", "sector": "Financials", "color": "#006FCF"},
    "PGR US Equity": {"name": "Progressive", "sector": "Financials", "color": "#0057B8"},
    "AON US Equity": {"name": "Aon", "sector": "Financials", "color": "#EB0029"},
    "CME US Equity": {"name": "CME Group", "sector": "Financials", "color": "#003B6A"},
    "KKR US Equity": {"name": "KKR", "sector": "Financials", "color": "#002D62"},
    "BN US Equity": {"name": "Brookfield", "sector": "Financials", "color": "#0B2E4F"},
    "C US Equity": {"name": "Citigroup", "sector": "Financials", "color": "#056DAE"},
    "LSEG LN Equity": {"name": "London Stock Exchange Group", "sector": "Financials", "color": "#001A70"},
    "ZURN SW Equity": {"name": "Zurich Insurance", "sector": "Financials", "color": "#2167AE"},
    "CS FP Equity": {"name": "AXA", "sector": "Financials", "color": "#00008F"},
    "8306 JP Equity": {"name": "Mitsubishi UFJ Financial", "sector": "Financials", "color": "#E60012"},
    "DIS US Equity": {"name": "Walt Disney", "sector": "Communication Services", "color": "#113CCF"},
    "CMCSA US Equity": {"name": "Comcast", "sector": "Communication Services", "color": "#0089CF"},
    "VZ US Equity": {"name": "Verizon", "sector": "Communication Services", "color": "#CD040B"},
    "T US Equity": {"name": "AT&T", "sector": "Communication Services", "color": "#00A8E0"},
    "TTWO US Equity": {"name": "Take-Two Interactive", "sector": "Communication Services", "color": "#D22630"},
    "PUB FP Equity": {"name": "Publicis Groupe", "sector": "Communication Services", "color": "#F04E37"},
    "7974 JP Equity": {"name": "Nintendo", "sector": "Communication Services", "color": "#E4000F"},
    "DTE GR Equity": {"name": "Deutsche Telekom", "sector": "Communication Services", "color": "#CC0066"},
    "UMG NA Equity": {"name": "Universal Music Group", "sector": "Communication Services", "color": "#2D2E83"},
    "LOW US Equity": {"name": "Lowe's", "sector": "Consumer Discretionary", "color": "#004990"},
    "TJX US Equity": {"name": "TJX Companies", "sector": "Consumer Discretionary", "color": "#D0103A"},
    "SBUX US Equity": {"name": "Starbucks", "sector": "Consumer Discretionary", "color": "#00704A"},
    "ABNB US Equity": {"name": "Airbnb", "sector": "Consumer Discretionary", "color": "#FF5A5F"},
    "7203 JP Equity": {"name": "Toyota Motor", "sector": "Consumer Discretionary", "color": "#EB0A1E"},
    "ITX SM Equity": {"name": "Inditex", "sector": "Consumer Discretionary", "color": "#435363"},
    "ABT US Equity": {"name": "Abbott Laboratories", "sector": "Healthcare", "color": "#009CDE"},
    "DHR US Equity": {"name": "Danaher", "sector": "Healthcare", "color": "#005587"},
    "VRTX US Equity": {"name": "Vertex Pharmaceuticals", "sector": "Healthcare", "color": "#6E2585"},
    "SAN FP Equity": {"name": "Sanofi", "sector": "Healthcare", "color": "#7A00E6"},
    "GE US Equity": {"name": "GE Aerospace", "sector": "Industrials", "color": "#3B73B9"},
    "TT US Equity": {"name": "Trane Technologies", "sector": "Industrials", "color": "#C8102E"},
    "ABBN SW Equity": {"name": "ABB", "sector": "Industrials", "color": "#FF000F"},
    "KO US Equity": {"name": "Coca-Cola", "sector": "Consumer Staples", "color": "#F40009"},
    "ULVR LN Equity": {"name": "Unilever", "sector": "Consumer Staples", "color": "#1F36C7"},
    "ECL US Equity": {"name": "Ecolab", "sector": "Materials", "color": "#007AC9"},
    "AI FP Equity": {"name": "Air Liquide", "sector": "Materials", "color": "#164194"},
    "IBE SM Equity": {"name": "Iberdrola", "sector": "Utilities", "color": "#78BE20"},
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


MARKET_TO_CURRENCY = {
    "US": "USD",
    "KS": "KRW",
    "JP": "JPY",
    "FP": "EUR",
    "GR": "EUR",
    "NA": "EUR",
    "SW": "CHF",
    "LN": "GBP",
    "DC": "DKK",
    "SM": "EUR",
}

FACTSET_MARKET_CODE = {
    "US": "US",
    "KS": "KR",
    "JP": "JP",
    "FP": "FR",
    "GR": "DE",
    "NA": "NL",
    "SW": "CH",
    "LN": "GB",
    "DC": "DK",
    "SM": "ES",
}


def split_bloomberg_ticker(ticker: str) -> tuple[str, str]:
    """Return ``(symbol, market_code)`` from ``SYMBOL XX Equity``."""
    parts = ticker.rsplit(" ", 2)
    if len(parts) != 3 or parts[2] != "Equity":
        raise ValueError(f"지원하지 않는 Bloomberg 티커 형식입니다: {ticker}")
    return parts[0], parts[1]


def simple_ticker(ticker: str) -> str:
    """Remove the Bloomberg market and security suffix."""
    return split_bloomberg_ticker(ticker)[0]


def ticker_currency(ticker: str) -> str:
    """Return the local trading currency implied by the Bloomberg market code."""
    _, market = split_bloomberg_ticker(ticker)
    return MARKET_TO_CURRENCY.get(market, "USD")


def build_factset_ticker_map() -> dict[str, str]:
    """Build FactSet header aliases for every member of the shared universe."""
    result = {}
    for ticker in UNIVERSE:
        symbol, market = split_bloomberg_ticker(ticker)
        country = FACTSET_MARKET_CODE[market]
        result[f"{symbol}-{country}^"] = symbol

    # Common vendor spellings for symbols containing punctuation/classes.
    result.update(
        {
            "RR-GB^": "RR/",
            "RR.-GB^": "RR/",
            "RR/-GB^": "RR/",
            "NOVO.B-DK^": "NOVOB",
            "NOVO-B-DK^": "NOVOB",
            "NOVOB-DK^": "NOVOB",
            # §S13.1: BRK/B — 워크북 r2는 FactSet 점 표기(BRK.B-US^, NOVO.B 선례)
            "BRK.B-US^": "BRK/B",
            "BRK-B-US^": "BRK/B",
        }
    )
    return result


def tickers_for_currency(currency: str, tickers: Iterable[str] | None = None) -> list[str]:
    """Return universe members whose local trading currency matches ``currency``."""
    members = UNIVERSE if tickers is None else tickers
    return [ticker for ticker in members if ticker_currency(ticker) == currency]


def validate_universe(expected_size: int = EXPECTED_UNIVERSE_SIZE) -> None:
    """Fail fast when the canonical universe is malformed or unexpectedly changed."""
    if len(UNIVERSE) != expected_size:
        raise ValueError(f"유니버스 종목 수 오류: expected={expected_size}, actual={len(UNIVERSE)}")

    simple_names = [simple_ticker(ticker) for ticker in UNIVERSE]
    duplicates = sorted({name for name in simple_names if simple_names.count(name) > 1})
    if duplicates:
        raise ValueError(f"간소화 티커가 중복됩니다: {duplicates}")

    required_fields = {"name", "sector", "color"}
    incomplete = [
        ticker for ticker, metadata in UNIVERSE.items()
        if not required_fields.issubset(metadata)
    ]
    if incomplete:
        raise ValueError(f"유니버스 메타데이터가 불완전합니다: {incomplete}")


validate_universe()
