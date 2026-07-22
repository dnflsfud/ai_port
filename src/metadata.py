"""
Canonical ticker metadata for the AI Signal portfolio universe.

Every module that needs sector, style, or sub-industry information should
import from here instead of maintaining its own copy.
"""

from typing import Dict

# ---- Full Ticker Metadata: Sector / Style / Sub-Industry ----
TICKER_META: Dict[str, Dict[str, str]] = {
    # Mega-Cap Tech
    "AAPL":   {"sector": "Technology",       "style": "Quality Growth",   "sub": "Consumer Electronics"},
    "MSFT":   {"sector": "Technology",       "style": "Quality Growth",   "sub": "Enterprise Software"},
    "GOOGL":  {"sector": "Communication",    "style": "Quality Growth",   "sub": "Digital Advertising"},
    "AMZN":   {"sector": "Consumer Disc.",   "style": "Growth",           "sub": "E-Commerce / Cloud"},
    "META":   {"sector": "Communication",    "style": "Quality Growth",   "sub": "Social Media"},
    # Semiconductors
    "NVDA":   {"sector": "Semiconductors",   "style": "Growth",           "sub": "AI / GPU"},
    "AVGO":   {"sector": "Semiconductors",   "style": "GARP",             "sub": "Networking Chips"},
    "MU":     {"sector": "Semiconductors",   "style": "Cyclical",         "sub": "Memory"},
    "AMD":    {"sector": "Semiconductors",   "style": "Growth",           "sub": "CPU / GPU"},
    "000660": {"sector": "Semiconductors",   "style": "Cyclical",         "sub": "Memory (KR)"},
    "005930": {"sector": "Semiconductors",   "style": "Cyclical",         "sub": "Foundry / Memory (KR)"},
    # Growth / Platform
    "TSLA":   {"sector": "Consumer Disc.",   "style": "Growth",           "sub": "EV / Energy Storage"},
    "PLTR":   {"sector": "Technology",       "style": "Growth",           "sub": "Data Analytics / AI"},
    "CRM":    {"sector": "Technology",       "style": "Growth",           "sub": "CRM / Cloud SaaS"},
    "NFLX":   {"sector": "Communication",    "style": "Growth",           "sub": "Streaming"},
    # Power / Energy Infra
    "GEV":    {"sector": "Industrials",      "style": "Cyclical",         "sub": "Power Equipment"},
    "VRT":    {"sector": "Industrials",      "style": "Cyclical",         "sub": "Data Center Cooling"},
    "BE":     {"sector": "Industrials",      "style": "Growth",           "sub": "Fuel Cells"},
    "LITE":   {"sector": "Technology",       "style": "Cyclical",         "sub": "Photonics / Fiber"},
    # Healthcare
    "UNH":    {"sector": "Healthcare",       "style": "Quality",          "sub": "Managed Care"},
    "LLY":    {"sector": "Healthcare",       "style": "Growth",           "sub": "Pharma / GLP-1"},
    "ISRG":   {"sector": "Healthcare",       "style": "Growth",           "sub": "Surgical Robotics"},
    "ABBV":   {"sector": "Healthcare",       "style": "Value",            "sub": "Pharma / Biotech"},
    "REGN":   {"sector": "Healthcare",       "style": "GARP",             "sub": "Biotech"},
    # Financials
    "JPM":    {"sector": "Financials",       "style": "Value",            "sub": "Banking"},
    "V":      {"sector": "Financials",       "style": "Quality Growth",   "sub": "Payments"},
    "MA":     {"sector": "Financials",       "style": "Quality Growth",   "sub": "Payments"},
    "BLK":    {"sector": "Financials",       "style": "Quality",          "sub": "Asset Management"},
    "SPGI":   {"sector": "Financials",       "style": "Quality",          "sub": "Data / Rating"},
    "GS":     {"sector": "Financials",       "style": "Value",            "sub": "Investment Banking"},
    # Consumer
    "COST":   {"sector": "Consumer Staples", "style": "Quality Growth",   "sub": "Warehouse Retail"},
    "HD":     {"sector": "Consumer Disc.",   "style": "Quality",          "sub": "Home Improvement"},
    "PG":     {"sector": "Consumer Staples", "style": "Defensive",        "sub": "Household Products"},
    "MCD":    {"sector": "Consumer Disc.",   "style": "Defensive",        "sub": "QSR"},
    "WMT":    {"sector": "Consumer Staples", "style": "Defensive",        "sub": "Discount Retail"},
    # Industrials / Defense
    "CAT":    {"sector": "Industrials",      "style": "Cyclical",         "sub": "Heavy Equipment"},
    "HON":    {"sector": "Industrials",      "style": "Quality",          "sub": "Diversified Industrials"},
    "DE":     {"sector": "Industrials",      "style": "Cyclical",         "sub": "Agriculture Equipment"},
    "UNP":    {"sector": "Industrials",      "style": "Quality",          "sub": "Railroads"},
    "LMT":    {"sector": "Industrials",      "style": "Defensive",        "sub": "Defense / Aerospace"},
    "ETN":    {"sector": "Industrials",      "style": "Quality Growth",   "sub": "Electrical Equipment"},
    # Energy / Materials / Utilities
    "XOM":    {"sector": "Energy",           "style": "Value",            "sub": "Oil Major"},
    "LNG":    {"sector": "Energy",           "style": "Value",            "sub": "LNG / Natural Gas"},
    "FCX":    {"sector": "Materials",        "style": "Cyclical",         "sub": "Copper Mining"},
    "LIN":    {"sector": "Materials",        "style": "Quality",          "sub": "Industrial Gas"},
    "NEE":    {"sector": "Utilities",        "style": "Defensive",        "sub": "Renewables / Utilities"},
    # Real Estate / Infra / Telecom
    "AMT":    {"sector": "Real Estate",      "style": "Defensive",        "sub": "Tower REIT"},
    "EQIX":   {"sector": "Real Estate",      "style": "Growth",           "sub": "Data Center REIT"},
    "TMUS":   {"sector": "Communication",    "style": "GARP",             "sub": "Wireless Telecom"},
    "PLD":    {"sector": "Real Estate",      "style": "Quality",          "sub": "Logistics REIT"},
    # Expansion 2026-04-13
    "TER":    {"sector": "Semiconductors",   "style": "Cyclical",         "sub": "Semi Test Equipment"},
    "GLW":    {"sector": "Technology",       "style": "Cyclical",         "sub": "Optical / Specialty Glass"},
    "JNJ":    {"sector": "Healthcare",       "style": "Defensive",        "sub": "Diversified Pharma"},
    "WFC":    {"sector": "Financials",       "style": "Value",            "sub": "Banking"},
    "MPC":    {"sector": "Energy",           "style": "Cyclical",         "sub": "Refining"},
    "LRCX":   {"sector": "Semiconductors",   "style": "Cyclical",         "sub": "Semi Capital Equipment"},
    "AMAT":   {"sector": "Semiconductors",   "style": "Cyclical",         "sub": "Semi Capital Equipment"},
    "PANW":   {"sector": "Technology",       "style": "Growth",           "sub": "Cybersecurity"},
    "FN":     {"sector": "Technology",       "style": "Growth",           "sub": "Optical Manufacturing"},
    "MPWR":   {"sector": "Semiconductors",   "style": "Quality Growth",   "sub": "Power ICs"},
}

# Convenience: ticker -> sector string (flat mapping)
TICKER_SECTOR: Dict[str, str] = {t: v["sector"] for t, v in TICKER_META.items()}


def get_sector_map_from_meta() -> Dict[str, str]:
    """Return a copy of the ticker-to-sector flat mapping."""
    return TICKER_SECTOR.copy()


def build_ticker_meta(meta) -> Dict[str, Dict[str, str]]:
    """Universe_Meta 기반 티커 메타 (§S11.4 Phase 3).

    Sector는 워크북 Universe_Meta(정본)에서, style/sub는 정적 TICKER_META
    fallback에서 가져온다. 워크북에 없는 정보는 중립값('Other'/'N/A').
    ``meta``는 UniverseData.meta 형태(index=티커, 'sector' 컬럼)를 기대한다.
    """
    result: Dict[str, Dict[str, str]] = {}
    sectors = meta["sector"] if "sector" in meta.columns else None
    for ticker in meta.index:
        static = TICKER_META.get(str(ticker), {})
        sector = None
        if sectors is not None:
            value = sectors.get(ticker)
            if value is not None and str(value) not in ("nan", ""):
                sector = str(value)
        result[str(ticker)] = {
            "sector": sector or static.get("sector", "Other"),
            "style": static.get("style", "Other"),
            "sub": static.get("sub", "N/A"),
        }
    return result


# ---------------------------------------------------------------------------
# Asset-rotation buckets (used by analytics.classify_sector_rotation).
# -----------------------------------------------------------------------------
# Previously hard-coded inside analytics.py. Centralised here so the universe
# list + rotation taxonomy stays in one place. These are intentionally flat
# lists (not derived from sector alone) because the "asset light vs heavy"
# rotation cuts across sector boundaries: software platforms are light, power-
# infrastructure industrials and semiconductor hardware are heavy.
# ---------------------------------------------------------------------------
ASSET_LIGHT_TICKERS = ["MSFT", "GOOGL", "META", "PLTR", "CRM", "NFLX"]
ASSET_HEAVY_TICKERS = ["NVDA", "AVGO", "MU", "GEV", "VRT", "BE", "LITE", "000660"]


def get_asset_rotation_buckets():
    """Return (asset_light, asset_heavy) ticker lists."""
    return list(ASSET_LIGHT_TICKERS), list(ASSET_HEAVY_TICKERS)
