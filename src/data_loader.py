"""
Phase 1: 데이터 로드 및 전처리
- config.data_path (기본 re_study/ai_signal_data.xlsx) 의 모든 시트를 로드
- 날짜 인덱스 통일 (BusinessDays)
- Sent_Trend 시트 회사명 -> 티커 매핑
- 결측치 처리: ffill -> cross-sectional median
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple

from src.config import DEFAULT_CONFIG, PipelineConfig

logger = logging.getLogger(__name__)

# Sent_Trend 시트의 회사명 -> 티커 매핑
COMPANY_TO_TICKER = {
    # 기존
    "Apple": "AAPL", "Microsoft": "MSFT", "Alphabet": "GOOGL",
    "Amazon": "AMZN", "Meta": "META", "Nvidia": "NVDA",
    "Tesla": "TSLA", "Palantir": "PLTR", "Broadcom": "AVGO",
    "Micron": "MU", "GE Vernova": "GEV", "Vertiv": "VRT",
    "Bloom Energy": "BE", "Lumentum": "LITE",
    "SK Hynix": "000660", "Samsung Electronics": "005930",
    # Healthcare
    "UnitedHealth": "UNH", "Eli Lilly": "LLY",
    "Intuitive Surgical": "ISRG", "AbbVie": "ABBV", "Regeneron": "REGN",
    # Financials
    "JPMorgan": "JPM", "Visa": "V", "Mastercard": "MA",
    "BlackRock": "BLK", "S&P Global": "SPGI", "Goldman Sachs": "GS",
    # Consumer
    "Costco": "COST", "Home Depot": "HD", "Procter & Gamble": "PG",
    "McDonald's": "MCD", "Walmart": "WMT",
    # Industrials
    "Caterpillar": "CAT", "Honeywell": "HON", "Deere": "DE",
    "Union Pacific": "UNP", "Lockheed Martin": "LMT", "Eaton": "ETN",
    # Energy/Materials/Utilities
    "Exxon Mobil": "XOM", "Cheniere Energy": "LNG",
    "Freeport-McMoRan": "FCX", "Linde": "LIN", "NextEra Energy": "NEE",
    # Real Estate/Infra/Telecom
    "American Tower": "AMT", "Equinix": "EQIX",
    "T-Mobile": "TMUS", "Prologis": "PLD",
    # Tech Diversifier
    "AMD": "AMD", "Salesforce": "CRM", "Netflix": "NFLX",
    # Expansion 2026-04-13
    "Teradyne": "TER", "Corning": "GLW", "Johnson & Johnson": "JNJ",
    "Wells Fargo": "WFC", "Marathon Petroleum": "MPC",
    "Lam Research": "LRCX", "Applied Materials": "AMAT",
    "Palo Alto Networks": "PANW", "Fabrinet": "FN",
    "Monolithic Power": "MPWR",
    # Expansion 2026-04-23
    "Bank of America": "BAC", "Cisco": "CSCO", "Intel": "INTC",
    "Oracle": "ORCL", "Taiwan Semiconductor": "TSM",
}

# 시트명 -> 피처 카테고리 매핑
SHEET_CATEGORY = {
    "PX_LAST": "Price",
    "Daily_Returns": "Price",
    "BEST_EPS": "Accounting",
    "BEST_SALES": "Accounting",
    "BEST_PE_RATIO": "Valuation",
    "BEST_PEG_RATIO": "Valuation",
    "BEST_CALCULATED_FCF": "Accounting",
    "BEST_GROSS_MARGIN": "Accounting",
    "CUR_MKT_CAP": "Conditioning",
    "OPER_MARGIN": "Accounting",
    "BEST_CAPEX": "Accounting",
    "BEST_ROE": "Accounting",
    "BEST_PX_BPS_RATIO": "Valuation",
    "BEST_EV_TO_BEST_EBITDA": "Valuation",
    "NEWS_SENTIMENT_DAILY_AVG": "Sentiment",
    "EQY_REC_CONS": "Sellside",
    "Sent_Trend_Momentum_Timeseries": "Sentiment",
    "Sent_Trend_21d_Timeseries": "Sentiment",
    "Factset_EPS_Revision": "Sellside",
    "Factset_Sales_Revision": "Sellside",
    "Factset_TG_Price": "Sellside",
    "Universe_Meta": "Meta",
}

TICKERS = [
    # 기존 16개
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "PLTR", "AVGO", "MU",
    "GEV", "VRT", "BE", "LITE", "000660", "005930",
    # Healthcare
    "UNH", "LLY", "ISRG", "ABBV", "REGN",
    # Financials
    "JPM", "V", "MA", "BLK", "SPGI", "GS",
    # Consumer
    "COST", "HD", "PG", "MCD", "WMT",
    # Industrials / Defense
    "CAT", "HON", "DE", "UNP", "LMT", "ETN",
    # Energy / Materials / Utilities
    "XOM", "LNG", "FCX", "LIN", "NEE",
    # Real Estate / Infra / Telecom
    "AMT", "EQIX", "TMUS", "PLD",
    # Growth / Platform / Tech Diversifier
    "AMD", "CRM", "NFLX",
    # Expansion 2026-04-13 (10)
    "TER", "GLW", "JNJ", "WFC", "MPC",
    "LRCX", "AMAT", "PANW", "FN", "MPWR",
    # Expansion 2026-04-23 (5)
    "BAC", "CSCO", "INTC", "ORCL", "TSM",
    # Expansion 2026-07-16 (35) -- exact Universe_Meta workbook order
    "SNDK", "KLAC", "ANET", "MRVL", "CDNS", "STX", "PWR", "BX",
    "TMO", "BSX", "BKNG", "PM", "WMB", "CEG", "VST", "DLR", "ARM",
    "SPOT", "RACE", "285A", "6857", "SU", "SIE", "RHM", "ALV", "MC",
    "NESN", "RR/", "SAP", "ASML", "AZN", "SHEL", "HSBA", "NOVOB", "RIO",
]

# Bloomberg listing suffix -> local trading currency. Universe_Meta is the
# authoritative source of each suffix (``SYMBOL XX Equity``).
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
    "SM": "EUR",  # Spain (ITX, IBE) — S11 expansion, decision log §S11
}

# Raw quote convention -> USD per one unit of local currency. ``inverse``
# means the source is local-per-USD (USDKRW, USDJPY, USDCHF, USDDKK).
FX_QUOTE_SPECS = {
    "KRW": {"column": "USDKRW", "direction": "inverse"},
    "JPY": {"column": "USDJPY", "direction": "inverse"},
    "EUR": {"column": "EURUSD", "direction": "direct"},
    "CHF": {"column": "USDCHF", "direction": "inverse"},
    "GBP": {"column": "GBPUSD", "direction": "direct"},
    "DKK": {"column": "USDDKK", "direction": "inverse"},
}

# Used only when an old/synthetic workbook has no Bloomberg market suffix.
# Production currency always comes from Universe_Meta.
FALLBACK_TICKER_CURRENCY = {
    "000660": "KRW", "005930": "KRW",
    "285A": "JPY", "6857": "JPY",
    "SU": "EUR", "SIE": "EUR", "RHM": "EUR", "ALV": "EUR",
    "MC": "EUR", "SAP": "EUR", "ASML": "EUR",
    "NESN": "CHF",
    "RR/": "GBP", "AZN": "GBP", "SHEL": "GBP", "HSBA": "GBP",
    "RIO": "GBP",
    "NOVOB": "DKK",
}

SENT_TREND_SHEETS = {
    "Sent_Trend_Momentum_Timeseries",
    "Sent_Trend_21d_Timeseries",
}

# 날짜 인덱스가 아닌 메타/요약 시트 (전처리에서 제외)
SKIP_SHEETS = {"Universe_Meta", "Summary_Stats", "BusinessDays", "Factor_Meta", "Earnings_Timeline"}

# Factor 시트 (ticker 기반이 아닌 별도 컬럼 구조)
FACTOR_SHEETS = {"Factor_PX_LAST", "Factor_Returns", "Factor_Meta"}

# Bloomberg "XXX US Equity" 형식 컬럼을 쓰는 시트
BLOOMBERG_EQUITY_SHEETS = {"SHORT_INT_RATIO"}

FACTOR_CATEGORIES = {
    "Market_Index": ["SPX", "NDX", "RTY", "MXWD", "MXEF", "SX5E", "NKY", "HSI", "SHCOMP"],
    "Volatility": ["VIX", "SKEW"],
    "Rates": ["UST_3M", "UST_2Y", "UST_10Y", "US_BEI10", "GER_10Y"],
    "FX": [
        "DXY", "USDKRW", "USDJPY", "EURUSD", "USDCHF", "GBPUSD",
        "USDDKK", "USDCNH",
    ],
    "Commodity": ["WTI", "GOLD", "COPPER", "BCOM"],
    "Factor_ETF": ["F_MinVol", "F_Quality", "F_HiDiv", "F_Growth", "F_Value", "F_SmCap", "F_HiBeta"],
    "GS_Thematic": ["GS_AI", "GS_Nuclear", "GS_SemiHW"],
    "Macro_Sentiment": ["CESI_US", "AAII_Bull", "AAII_Bear"],
}
ALL_FACTOR_COLUMNS = [col for cols in FACTOR_CATEGORIES.values() for col in cols]


def load_all_sheets(data_path: str) -> Dict[str, pd.DataFrame]:
    """엑셀 파일의 모든 시트를 Dict[시트명, DataFrame]으로 로드."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")

    xls = pd.ExcelFile(path, engine="openpyxl")
    raw: Dict[str, pd.DataFrame] = {}

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name, index_col=0)
        raw[sheet_name] = df

    return raw


def _normalize_company_name(value) -> str:
    """Normalize a vendor company label for collision-free exact matching."""
    return " ".join(str(value).strip().split()).casefold()


def _split_bloomberg_ticker(value) -> Tuple[str, Optional[str]]:
    """Return ``(simple_ticker, exchange_code)`` from a Bloomberg identifier."""
    text = str(value).strip()
    parts = text.rsplit(" ", 2)
    if len(parts) == 3 and parts[2].casefold() == "equity":
        return parts[0].strip(), parts[1].strip().upper()
    return text, None


def build_company_to_ticker(
    meta: pd.DataFrame,
    tickers: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Build an exact normalized company-name -> ticker map from Universe_Meta.

    Universe_Meta names win. Legacy aliases are retained only as fallbacks for
    historical vendor spellings. Substring matching is deliberately forbidden.
    """
    allowed = set(tickers or list(meta.index) or TICKERS)
    mapping = {
        _normalize_company_name(company): ticker
        for company, ticker in COMPANY_TO_TICKER.items()
        if ticker in allowed
    }
    name_col = next(
        (c for c in meta.columns if str(c).strip().casefold() == "name"),
        None,
    )
    if name_col is not None:
        for ticker, company in meta[name_col].items():
            if ticker not in allowed or pd.isna(company):
                continue
            key = _normalize_company_name(company)
            prior = mapping.get(key)
            if prior is not None and prior != ticker:
                raise ValueError(
                    "Universe_Meta contains a duplicate normalized company name "
                    f"{company!r} for {prior} and {ticker}."
                )
            mapping[key] = ticker
    return mapping


def _rename_sent_trend_columns(
    df: pd.DataFrame,
    tickers: Optional[List[str]] = None,
    company_to_ticker: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Map Sent_Trend headers using normalized exact names, never substrings."""
    tickers = list(tickers or TICKERS)
    allowed = set(tickers)
    company_to_ticker = company_to_ticker or {
        _normalize_company_name(company): ticker
        for company, ticker in COMPANY_TO_TICKER.items()
        if ticker in allowed
    }
    rename_map = {}
    for col in df.columns:
        col_str = str(col).strip()
        if col_str in allowed:
            rename_map[col] = col_str
            continue
        ticker = company_to_ticker.get(_normalize_company_name(col_str))
        if ticker is not None:
            rename_map[col] = ticker
    renamed = df.rename(columns=rename_map)
    if renamed.columns.duplicated().any():
        duplicates = sorted(set(renamed.columns[renamed.columns.duplicated()].tolist()))
        raise ValueError(
            "Sent_Trend exact-name mapping produced duplicate ticker columns: "
            f"{duplicates}"
        )
    unmapped = [c for c in renamed.columns if str(c).strip() not in allowed]
    if unmapped:
        logger.warning(
            "Sent_Trend columns missing from Universe_Meta Name mapping: %s. "
            "These columns will be dropped downstream.",
            unmapped,
        )
    return renamed


def _rename_bloomberg_equity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Bloomberg 'AAPL US Equity' / '000660 KS Equity' 형식 컬럼을 티커로 변환."""
    rename_map = {}
    for col in df.columns:
        col_str = str(col).strip()
        parts = col_str.split()
        if len(parts) >= 2 and parts[-1].lower() == "equity":
            rename_map[col] = parts[0]
    return df.rename(columns=rename_map)


def _standardize_index(df: pd.DataFrame) -> pd.DataFrame:
    """인덱스를 DatetimeIndex로 변환."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼명을 문자열로 통일하고, 공백 제거."""
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """결측치 처리: ffill -> 남은 NaN은 해당 날짜 cross-sectional median."""
    df = df.ffill()
    row_medians = df.median(axis=1)
    for col in df.columns:
        mask = df[col].isna()
        if mask.any():
            df.loc[mask, col] = row_medians[mask]
    return df


def _filter_tickers(
    df: pd.DataFrame,
    tickers: Optional[List[str]] = None,
) -> pd.DataFrame:
    """TICKERS에 포함된 컬럼만 남김. 컬럼 순서 통일."""
    tickers = list(tickers or TICKERS)
    available = [t for t in tickers if t in df.columns]
    return df[available]


def load_universe_meta(raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Load ordered Universe_Meta and derive simple ticker/exchange/currency."""
    if "Universe_Meta" not in raw:
        return pd.DataFrame({
            "ticker": TICKERS,
            "Name": TICKERS,
            "sector": "Unknown",
            "bloomberg_ticker": TICKERS,
            "exchange_code": None,
            "currency": [FALLBACK_TICKER_CURRENCY.get(t, "USD") for t in TICKERS],
        }, index=pd.Index(TICKERS, name="ticker"))

    meta = _standardize_columns(raw["Universe_Meta"].copy())
    ticker_col = next(
        (c for c in meta.columns if str(c).strip().casefold() == "ticker"),
        None,
    )
    simple_tickers = []
    bloomberg_tickers = []
    exchange_codes = []
    for idx, row in meta.iterrows():
        raw_ticker = row[ticker_col] if ticker_col is not None else idx
        simple, exchange = _split_bloomberg_ticker(raw_ticker)
        idx_simple, idx_exchange = _split_bloomberg_ticker(idx)
        if exchange is None and idx_exchange is not None:
            exchange = idx_exchange
        if (not simple or simple.casefold() == "nan") and idx_simple:
            simple = idx_simple
        bloomberg = (
            str(raw_ticker).strip()
            if _split_bloomberg_ticker(raw_ticker)[1] is not None
            else str(idx).strip()
            if idx_exchange is not None
            else str(raw_ticker).strip()
        )
        simple_tickers.append(simple)
        bloomberg_tickers.append(bloomberg)
        exchange_codes.append(exchange)

    ticker_index = pd.Index(simple_tickers, name="ticker")
    if ticker_index.duplicated().any():
        duplicates = ticker_index[ticker_index.duplicated()].unique().tolist()
        raise ValueError(f"Universe_Meta contains duplicate tickers: {duplicates}")
    meta.index = ticker_index
    meta["ticker"] = simple_tickers
    meta["bloomberg_ticker"] = bloomberg_tickers
    meta["exchange_code"] = exchange_codes
    meta["currency"] = [
        MARKET_TO_CURRENCY.get(code) if code is not None else None
        for code in exchange_codes
    ]
    if "Sector" in meta.columns:
        meta = meta.rename(columns={"Sector": "sector"})
    return meta


def preprocess_sheets(
    raw: Dict[str, pd.DataFrame],
    tickers: Optional[List[str]] = None,
    company_to_ticker: Optional[Dict[str, str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    모든 시트를 전처리하여 반환.
    - DatetimeIndex 통일
    - Sent_Trend 컬럼 매핑
    - 티커 필터링
    - 결측치 처리
    """
    processed: Dict[str, pd.DataFrame] = {}
    tickers = list(tickers or TICKERS)

    for sheet_name, df in raw.items():
        if sheet_name in SKIP_SHEETS or sheet_name in FACTOR_SHEETS:
            continue

        df = df.copy()
        df = _standardize_columns(df)

        # Sent_Trend 시트는 회사명 -> 티커 매핑
        if sheet_name in SENT_TREND_SHEETS:
            df = _rename_sent_trend_columns(
                df,
                tickers=tickers,
                company_to_ticker=company_to_ticker,
            )

        # Bloomberg Equity 시트는 'AAPL US Equity' -> 'AAPL' 매핑
        if sheet_name in BLOOMBERG_EQUITY_SHEETS:
            df = _rename_bloomberg_equity_columns(df)

        df = _standardize_index(df)
        df = _filter_tickers(df, tickers=tickers)

        # 수치형 변환
        df = df.apply(pd.to_numeric, errors="coerce")

        # 결측치 처리
        df = _fill_missing(df)

        processed[sheet_name] = df

    return processed


def align_dates(
    processed: Dict[str, pd.DataFrame],
    config: PipelineConfig = None,
    diagnostics: Optional[Dict] = None,
) -> Dict[str, pd.DataFrame]:
    """모든 시트의 날짜 인덱스를 정렬.

    전략: 교집합(intersection)으로 안전한 날짜 범위를 확보한 뒤,
    PX_LAST 기준으로 끝부분(tail)만 ffill 확장한다.
    이렇게 하면 업데이트가 느린 시트(Factset 등)가 전체를 잘라먹지
    않으면서도, 과거 데이터 품질은 intersection 수준으로 유지된다.
    """
    config = config or DEFAULT_CONFIG
    diagnostics = diagnostics if diagnostics is not None else {}

    # 1) 교집합 산출 (기존 방식 — 안전한 코어 날짜)
    common_idx = None
    for df in processed.values():
        if common_idx is None:
            common_idx = df.index
        else:
            common_idx = common_idx.intersection(df.index)

    if common_idx is None or len(common_idx) == 0:
        raise ValueError("시트 간 공통 날짜가 없습니다.")

    common_idx = common_idx.sort_values()

    # Rebalance cadence is defined in trading-day rows. Bloomberg PX_LAST can
    # carry Friday prices onto Saturday/Sunday in the recent tail; allowing
    # those rows into the model calendar makes a 21-day rebalance count
    # calendar rows. Remove weekends before constructing either calendar arm.
    weekend_common_dates = common_idx[common_idx.dayofweek >= 5]
    common_idx = common_idx[common_idx.dayofweek < 5]
    if len(common_idx) == 0:
        raise ValueError("No weekday dates remain after calendar filtering.")

    # 2) PX_LAST 끝 날짜까지 tail 확장
    if "PX_LAST" in processed:
        ref_idx = processed["PX_LAST"].index
    elif "Daily_Returns" in processed:
        ref_idx = processed["Daily_Returns"].index
    else:
        ref_idx = common_idx

    ref_idx = pd.DatetimeIndex(ref_idx).sort_values().unique()
    ref_weekday_idx = ref_idx[ref_idx.dayofweek < 5]
    if len(ref_weekday_idx) == 0:
        raise ValueError("Reference price calendar has no weekday dates.")
    px_end = ref_weekday_idx.max()

    n_extended = 0
    tail_from = None
    tail_to = None
    weekend_tail_dates = pd.DatetimeIndex([])
    if px_end > common_idx[-1]:
        # PX_LAST 캘린더에서 교집합 끝 이후~PX_LAST 끝까지의 날짜 추가
        raw_tail_dates = ref_idx[(ref_idx > common_idx[-1]) & (ref_idx <= ref_idx.max())]
        weekend_tail_dates = raw_tail_dates[raw_tail_dates.dayofweek >= 5]
        tail_dates = ref_weekday_idx[
            (ref_weekday_idx > common_idx[-1]) & (ref_weekday_idx <= px_end)
        ]
        extended_idx = common_idx.append(tail_dates).sort_values().unique()
        n_extended = len(tail_dates)
        tail_from = common_idx[-1]
        tail_to = px_end
        logger.warning(
            "[DataLoader] Extending %d tail dates beyond intersection "
            "(%s -> %s) via ffill.",
            n_extended,
            common_idx[-1].strftime("%Y-%m-%d"),
            px_end.strftime("%Y-%m-%d"),
        )
    else:
        extended_idx = common_idx

    # 3) 경고: 교집합이 최대 시트 대비 줄었으면 알림
    max_len = max(len(df) for df in processed.values())
    intersection_pct = len(common_idx) / max_len if max_len else 1.0
    diagnostics.update({
        "intersection_dates": int(len(common_idx)),
        "longest_sheet_dates": int(max_len),
        "intersection_pct_of_longest": float(intersection_pct),
        "tail_ffill_days": int(n_extended),
        "tail_from": tail_from.strftime("%Y-%m-%d") if tail_from is not None else None,
        "tail_to": tail_to.strftime("%Y-%m-%d") if tail_to is not None else None,
        "tail_extended_dates": int(len(extended_idx) - len(common_idx)),
        "weekend_dates_removed": int(
            len(weekend_common_dates.append(weekend_tail_dates).unique())
        ),
        "calendar_type": "weekday_index",
        "max_tail_ffill_days": int(getattr(config, "max_tail_ffill_days", 10)),
        "fail_on_stale_tail_ffill": bool(getattr(config, "fail_on_stale_tail_ffill", False)),
    })

    max_tail = int(getattr(config, "max_tail_ffill_days", 10))
    if n_extended > max_tail:
        msg = (
            f"[DataLoader] Tail ffill length {n_extended} exceeds "
            f"max_tail_ffill_days={max_tail}."
        )
        logger.warning(msg)
        if getattr(config, "fail_on_stale_tail_ffill", False):
            raise ValueError(msg)

    if len(common_idx) < max_len * 0.9:
        logger.warning(
            "[DataLoader] Date intersection (%d) is %.0f%% of longest sheet (%d). "
            "Tail-extended to %d dates.",
            len(common_idx), intersection_pct * 100, max_len,
            len(extended_idx),
        )

    # 4) 정렬: 교집합 구간은 원본, tail 구간만 ffill
    # CLAUDE.md 사양: ffill -> 남은 NaN은 해당 날짜 cross-sectional median.
    # 예전엔 fillna(0)을 쓰고 있었는데, P/E, 센티먼트, 마진 같은 level 변수는
    # 0이 경제적 중립값이 아니라 조용한 편향을 만든다.
    aligned = {}
    for name, df in processed.items():
        df_f = df.reindex(extended_idx).ffill()
        if df_f.isna().any().any():
            # 남은 NaN (시리즈 시작 이전 구간)을 해당 날짜 cross-sectional median으로
            row_median = df_f.median(axis=1)
            for col in df_f.columns:
                df_f[col] = df_f[col].fillna(row_median)
            # 해당 날짜 전체가 NaN인 경우만 0으로 fallback
            df_f = df_f.fillna(0.0)
        aligned[name] = df_f

    return aligned


def load_factor_sheets(raw: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Factor_PX_LAST / Factor_Returns를 별도 파이프라인으로 로드.

    결측치 처리: ffill ONLY. 과거엔 ffill().fillna(0) 이었으나,
    이는 level 데이터(Factor_PX_LAST) 의 leading NaN 을 literal 0 으로 바꿔
    raw broadcast 피처(features/factor.py) 에 심각한 편향을 만든다.
    leading NaN 은 그대로 두고, rolling-window 피처와 patch assembly 단계에서
    per-date median fill 로 처리되도록 맡긴다.
    """
    factor_data: Dict[str, pd.DataFrame] = {}
    for sheet_name in ["Factor_PX_LAST", "Factor_Returns"]:
        if sheet_name not in raw:
            continue
        df = raw[sheet_name].copy()
        df = _standardize_columns(df)
        df = _standardize_index(df)
        available = [c for c in ALL_FACTOR_COLUMNS if c in df.columns]
        df = df[available]
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.ffill()  # NOTE: do NOT fillna(0) here — see docstring.
        factor_data[sheet_name] = df
    return factor_data


def _raw_factor_prices_for_fx(raw: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Return unfilled Factor_PX_LAST observations for FX construction.

    ``load_factor_sheets`` forward-fills factor levels for model features.  FX
    freshness must instead retain the source observation gaps so a filled
    factor value cannot mask a newer quote from the external FX workbook.
    """
    if "Factor_PX_LAST" not in raw:
        return None
    frame = raw["Factor_PX_LAST"].copy()
    frame = _standardize_columns(frame)
    frame = _standardize_index(frame)
    available = [column for column in ALL_FACTOR_COLUMNS if column in frame.columns]
    return frame[available].apply(pd.to_numeric, errors="coerce")


def _canonical_fx_quote_name(value) -> str:
    """Normalize Bloomberg FX headers (for example ``USDKRW Curncy``)."""
    name = " ".join(str(value).strip().upper().split())
    if name.endswith(" CURNCY"):
        name = name[:-7].strip()
    return name


def normalize_fx_quotes_to_usd_per_local(quotes: pd.DataFrame) -> pd.DataFrame:
    """Convert supported raw FX quotes to USD-per-local-currency levels.

    ``USDKRW``, ``USDJPY``, ``USDCHF`` and ``USDDKK`` are local-per-USD and
    are inverted. ``EURUSD`` and ``GBPUSD`` are already USD-per-local.
    Non-positive quotes are invalid and remain missing; they are never filled
    with cross-sectional medians.
    """
    if quotes is None or quotes.empty:
        return pd.DataFrame(index=getattr(quotes, "index", None))
    frame = quotes.copy()
    frame = _standardize_index(frame)
    canonical = {_canonical_fx_quote_name(col): col for col in frame.columns}
    converted = {}
    for currency, spec in FX_QUOTE_SPECS.items():
        source_col = canonical.get(spec["column"])
        if source_col is None:
            continue
        values = pd.to_numeric(frame[source_col], errors="coerce")
        values = values.where(values > 0)
        if spec["direction"] == "inverse":
            values = 1.0 / values
        converted[currency] = values.astype(float)
    return pd.DataFrame(converted, index=frame.index)


def load_external_fx_quotes(
    data_path: str,
    currencies: List[str],
) -> pd.DataFrame:
    """Load only required FX columns from the external Index.xlsx source."""
    currencies = [c for c in currencies if c in FX_QUOTE_SPECS]
    if not currencies:
        return pd.DataFrame()
    path = Path(data_path)
    if not path.exists():
        return pd.DataFrame()

    header = pd.read_excel(path, sheet_name="PX_LAST", nrows=0).columns.tolist()
    canonical_to_actual = {_canonical_fx_quote_name(col): col for col in header}
    date_col = next(
        (col for col in header if str(col).strip().casefold() == "date"),
        None,
    )
    quote_cols = [
        canonical_to_actual[FX_QUOTE_SPECS[currency]["column"]]
        for currency in currencies
        if FX_QUOTE_SPECS[currency]["column"] in canonical_to_actual
    ]
    if date_col is None or not quote_cols:
        return pd.DataFrame()
    quotes = pd.read_excel(
        path,
        sheet_name="PX_LAST",
        usecols=[date_col, *quote_cols],
    )
    quotes[date_col] = pd.to_datetime(quotes[date_col], errors="coerce")
    quotes = (
        quotes.dropna(subset=[date_col])
        .drop_duplicates(subset=[date_col], keep="last")
        .set_index(date_col)
        .sort_index()
    )
    return quotes


def build_fx_rates_usd_per_local(
    target_index: pd.DatetimeIndex,
    currencies: List[str],
    config: PipelineConfig,
    factor_prices: Optional[pd.DataFrame] = None,
    external_quotes: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Build date x currency USD-per-local levels with freshness diagnostics.

    Factor_PX_LAST is preferred wherever it has an observed value. The
    external source fills missing currencies/dates, including its fresher tail.
    Forward fill is time-series-only and guarded by calendar-day staleness.
    """
    target_index = pd.DatetimeIndex(target_index).sort_values().unique()
    required = list(dict.fromkeys(str(c).upper() for c in currencies))
    non_usd = [c for c in required if c != "USD"]
    factor_levels = normalize_fx_quotes_to_usd_per_local(factor_prices)
    if external_quotes is None and non_usd:
        external_quotes = load_external_fx_quotes(config.fx_source_path, non_usd)
    external_levels = normalize_fx_quotes_to_usd_per_local(external_quotes)

    rates = pd.DataFrame(index=target_index)
    diagnostics = {
        "required_currencies": required,
        "source_by_currency": {},
        "coverage_by_currency": {},
        "latest_source_date_by_currency": {},
        "max_staleness_days_by_currency": {},
        "missing_currencies": [],
        "stale_currencies": [],
        "quote_directions": {
            currency: spec["direction"]
            for currency, spec in FX_QUOTE_SPECS.items()
            if currency in non_usd
        },
    }

    if "USD" in required:
        rates["USD"] = 1.0
        diagnostics["source_by_currency"]["USD"] = "identity"
        diagnostics["coverage_by_currency"]["USD"] = 1.0
        diagnostics["latest_source_date_by_currency"]["USD"] = (
            target_index.max().strftime("%Y-%m-%d") if len(target_index) else None
        )
        diagnostics["max_staleness_days_by_currency"]["USD"] = 0

    for currency in non_usd:
        if currency not in FX_QUOTE_SPECS:
            diagnostics["missing_currencies"].append(currency)
            continue
        factor_series = (
            factor_levels[currency]
            if currency in factor_levels.columns
            else pd.Series(dtype=float)
        )
        external_series = (
            external_levels[currency]
            if currency in external_levels.columns
            else pd.Series(dtype=float)
        )
        observations = factor_series.combine_first(external_series)
        observations = observations[~observations.index.duplicated(keep="last")]
        observations = observations.sort_index().dropna()
        if len(target_index):
            observations = observations.loc[observations.index <= target_index.max()]
        source_parts = []
        if not factor_series.dropna().empty:
            source_parts.append("Factor_PX_LAST")
        if not external_series.dropna().empty:
            source_parts.append("Index.xlsx/PX_LAST")
        diagnostics["source_by_currency"][currency] = "+".join(source_parts) or None
        if observations.empty:
            diagnostics["missing_currencies"].append(currency)
            diagnostics["coverage_by_currency"][currency] = 0.0
            diagnostics["latest_source_date_by_currency"][currency] = None
            diagnostics["max_staleness_days_by_currency"][currency] = None
            rates[currency] = np.nan
            continue

        union_index = observations.index.union(target_index).sort_values()
        aligned = observations.reindex(union_index).ffill().reindex(target_index)
        observed_dates = pd.Series(
            observations.index,
            index=observations.index,
            dtype="datetime64[ns]",
        )
        last_observed = observed_dates.reindex(union_index).ffill().reindex(target_index)
        age_days = pd.Series(
            (target_index - pd.DatetimeIndex(last_observed)).days,
            index=target_index,
            dtype=float,
        )
        rates[currency] = aligned
        coverage = float(aligned.notna().mean()) if len(aligned) else 1.0
        max_age = int(age_days.max()) if age_days.notna().any() else None
        diagnostics["coverage_by_currency"][currency] = coverage
        diagnostics["latest_source_date_by_currency"][currency] = (
            observations.index.max().strftime("%Y-%m-%d")
        )
        diagnostics["max_staleness_days_by_currency"][currency] = max_age
        if coverage < 1.0:
            diagnostics["missing_currencies"].append(currency)
        if age_days.gt(config.max_fx_staleness_days).any():
            diagnostics["stale_currencies"].append(currency)

    diagnostics["missing_currencies"] = sorted(
        set(diagnostics["missing_currencies"])
    )
    diagnostics["stale_currencies"] = sorted(
        set(diagnostics["stale_currencies"])
    )
    issues = []
    if diagnostics["missing_currencies"]:
        issues.append(f"missing={diagnostics['missing_currencies']}")
    if diagnostics["stale_currencies"]:
        issues.append(
            f"stale>{config.max_fx_staleness_days}d="
            f"{diagnostics['stale_currencies']}"
        )
    if issues:
        message = "FX coverage validation failed: " + "; ".join(issues)
        if config.fail_on_missing_fx:
            raise ValueError(message)
        logger.warning(message)
    return rates.reindex(columns=required), diagnostics


def mask_pre_listing(
    df: pd.DataFrame,
    listing_dates: Dict[str, str],
    inclusive: bool,
) -> pd.DataFrame:
    """Mask pre-listing backfilled cells to NaN, returning a NEW frame.

    ``df`` is a date-index × ticker-column frame. For every ticker in
    ``listing_dates`` that appears in the frame, cells dated before the
    listing date are set to NaN. Tickers absent from the frame are ignored;
    an empty ``listing_dates`` yields a plain copy. The input is never mutated
    (``df.copy()``).

    - inclusive=True  → mask ``index <= listing_date`` (returns / targets /
      predictions: the listing-day return is measured against the phantom
      backfilled reference price, so it too is fake).
    - inclusive=False → mask ``index < listing_date`` (level sheets: the
      listing-day price / market cap is the first real observation).
    """
    out = df.copy()
    for ticker, listing in listing_dates.items():
        if ticker not in out.columns:
            continue
        listing_ts = pd.Timestamp(listing)
        if inclusive:
            mask = out.index <= listing_ts
        else:
            mask = out.index < listing_ts
        out.loc[mask, ticker] = np.nan
    return out


class UniverseData:
    """전처리된 유니버스 데이터를 담는 컨테이너."""

    def __init__(self, data_path: str, config: PipelineConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.data_path = data_path
        self.raw = load_all_sheets(data_path)
        self.meta = load_universe_meta(self.raw)
        self.data_quality: Dict = {}
        self.full_universe = list(self.meta.index)
        self.company_to_ticker = build_company_to_ticker(
            self.meta,
            tickers=self.full_universe,
        )
        self._full_currency_map: Dict[str, str] = {}
        for ticker in self.full_universe:
            currency = self.meta.loc[ticker, "currency"] if "currency" in self.meta else None
            exchange = (
                self.meta.loc[ticker, "exchange_code"]
                if "exchange_code" in self.meta
                else None
            )
            if pd.isna(currency) or not str(currency).strip():
                if exchange is not None and not pd.isna(exchange):
                    raise ValueError(
                        f"Unsupported Bloomberg exchange code {exchange!r} for {ticker}."
                    )
                currency = FALLBACK_TICKER_CURRENCY.get(ticker, "USD")
            self._full_currency_map[ticker] = str(currency).upper()

        # Preserve RAW (un-imputed) Daily_Returns BEFORE preprocess_sheets
        # fills NaN. Survivorship checks (run_selection_bias.py) need this
        # to detect late entrants — if we only expose the imputed panel,
        # first_valid_index() always returns the alignment start and the
        # check silently passes even when a ticker genuinely joined late.
        self.raw_returns = self._extract_raw_returns()

        self.sheets = preprocess_sheets(
            self.raw,
            tickers=self.full_universe,
            company_to_ticker=self.company_to_ticker,
        )
        self.sheets = align_dates(
            self.sheets,
            config=self.config,
            diagnostics=self.data_quality,
        )

        # Pre-listing backfill masking (OFF by default). Applied AFTER
        # align/impute so align_dates' cross-sectional median fill can't
        # back-fill the masked pre-listing cells. See config.listing_mask_enabled.
        if self.config.listing_mask_enabled:
            listing_dates = self.config.listing_dates
            n_masked = 0
            # Daily_Returns 시트는 PCA 타깃 엔진이 dense 횡단면을 요구해 시트 마스킹
            # 제외 — 라벨 오염은 run_backtest의 targets 셀 마스킹이, PnL은 예측
            # 마스킹(w=0)이 차단 (2026-07-02 ablation에서 확인).
            for sheet_key, inclusive in (
                ("PX_LAST", False),
                ("CUR_MKT_CAP", False),
            ):
                if sheet_key not in self.sheets:
                    continue
                before = int(self.sheets[sheet_key].isna().sum().sum())
                self.sheets[sheet_key] = mask_pre_listing(
                    self.sheets[sheet_key], listing_dates, inclusive=inclusive
                )
                after = int(self.sheets[sheet_key].isna().sum().sum())
                n_masked += after - before
            if self.raw_returns is not None:
                self.raw_returns = mask_pre_listing(
                    self.raw_returns, listing_dates, inclusive=True
                )
            logger.info(
                "[UniverseData] listing mask applied: %d cell(s) masked across "
                "PX_LAST/CUR_MKT_CAP (+ raw_returns)", n_masked,
            )

        # ---------------------------------------------------------------
        # Universe construction (REDESIGN I, 2026-04-12)
        # ---------------------------------------------------------------
        # Previous logic: intersection across ALL sheets → banks (JPM, GS)
        # were dropped because BEST_CALCULATED_FCF / BEST_CAPEX /
        # BEST_EV_TO_BEST_EBITDA don't exist for deposit-taking institutions.
        #
        # New logic: "essential sheets" intersection only. A ticker must
        # appear in EVERY essential sheet to stay in the universe. Non-
        # essential sheets (bank-incompatible fundamentals) are allowed to
        # have missing tickers — the feature layer fills those with per-date
        # cross-sectional median in assembly.py, which is the correct
        # "neutral" treatment for metrics that don't apply.
        # ---------------------------------------------------------------
        ESSENTIAL_SHEETS = {
            "PX_LAST", "Daily_Returns", "CUR_MKT_CAP",
            "BEST_EPS", "BEST_SALES",
            "BEST_PE_RATIO",
            "OPER_MARGIN",
            "BEST_ROE",
            "NEWS_SENTIMENT_DAILY_AVG", "EQY_REC_CONS",
            "Factset_EPS_Revision", "Factset_Sales_Revision",
            "Factset_TG_Price",
        }
        # Sheets where some tickers legitimately have no data:
        OPTIONAL_SHEETS = {
            "BEST_CALCULATED_FCF",      # banks: no traditional FCF
            "BEST_CAPEX",               # banks: no physical capex
            "BEST_EV_TO_BEST_EBITDA",   # banks: EBITDA undefined
            "BEST_GROSS_MARGIN",        # banks (WFC): no gross margin concept
            "BEST_PEG_RATIO",           # low-coverage names (FN): no LT growth estimate
            "BEST_PX_BPS_RATIO",        # PM and some non-US names: no usable BPS
        }

        if self.sheets:
            # Intersect only across essential sheets
            essential_sets = [
                set(df.columns) for name, df in self.sheets.items()
                if name in ESSENTIAL_SHEETS
            ]
            if essential_sets:
                loaded_intersection = set.intersection(*essential_sets)
            else:
                # Fallback: intersect across all sheets (old behavior)
                loaded_intersection = set.intersection(
                    *(set(df.columns) for df in self.sheets.values())
                )

            self.tickers = [t for t in self.full_universe if t in loaded_intersection]

            # Diagnostic: which tickers are missing from which sheets
            self.missing_tickers = [
                t for t in self.full_universe if t not in loaded_intersection
            ]
            self.missing_by_sheet: Dict[str, list] = {}
            self.optional_missing: Dict[str, list] = {}
            for name, df in self.sheets.items():
                gone = [t for t in self.tickers if t not in df.columns]
                if gone:
                    if name in OPTIONAL_SHEETS:
                        self.optional_missing[name] = gone
                    else:
                        self.missing_by_sheet[name] = gone
        else:
            self.tickers = list(self.full_universe)
            self.missing_tickers = []
            self.missing_by_sheet = {}
            self.optional_missing = {}

        if self.missing_tickers:
            import logging
            logging.warning(
                "[UniverseData] %d ticker(s) dropped (missing from essential sheets): %s",
                len(self.missing_tickers), self.missing_tickers,
            )
        if self.optional_missing:
            import logging
            logging.info(
                "[UniverseData] Optional sheet gaps (filled with cross-sectional median): %s",
                {k: v for k, v in self.optional_missing.items()},
            )

        self.currency_map = {
            ticker: self._full_currency_map[ticker] for ticker in self.tickers
        }
        currency_counts = pd.Series(self.currency_map, dtype=object).value_counts().to_dict()
        self.data_quality["universe"] = {
            "meta_count": int(len(self.meta)) if "Universe_Meta" in self.raw else None,
            "full_universe_count": int(len(self.full_universe)),
            "essential_ticker_count": int(len(self.tickers)),
            "loaded_ticker_count": int(len(self.tickers)),
            "missing_tickers": list(self.missing_tickers),
            "essential_sheet_ticker_counts": {
                name: int(len(set(df.columns).intersection(self.full_universe)))
                for name, df in self.sheets.items()
                if name in ESSENTIAL_SHEETS
            },
        }
        self.data_quality["currency"] = {
            "base_currency": self.config.base_currency,
            "counts": {str(k): int(v) for k, v in currency_counts.items()},
            "non_usd_ticker_count": int(
                sum(currency != "USD" for currency in self.currency_map.values())
            ),
        }

        self.dates = self.sheets[next(iter(self.sheets))].index

        # Factor 데이터 (별도 파이프라인)
        self.factor_data = load_factor_sheets(self.raw)
        if self.factor_data:
            for name, df in self.factor_data.items():
                common = df.index.intersection(self.dates)
                self.factor_data[name] = df.loc[common]

        # Earnings Timeline (종목별 실적발표일 0/1 매트릭스)
        self._apply_usd_conversion()

        self.earnings_timeline = self._load_earnings_timeline()

    @staticmethod
    def _max_abs_finite(frame: pd.DataFrame) -> float:
        values = frame.to_numpy(dtype=float, copy=False)
        finite = np.isfinite(values)
        return float(np.max(np.abs(values[finite]))) if finite.any() else 0.0

    def _apply_usd_conversion(self) -> None:
        """Preserve local data, then expose USD prices/returns and FX effects."""
        self.local_prices = self.sheets["PX_LAST"].copy()
        self.local_returns = self.sheets["Daily_Returns"].copy()
        self.raw_local_returns = (
            self.raw_returns.copy() if self.raw_returns is not None else None
        )

        conversion_columns = list(dict.fromkeys(
            list(self.local_prices.columns)
            + list(self.local_returns.columns)
            + (
                list(self.raw_local_returns.columns)
                if self.raw_local_returns is not None
                else []
            )
        ))
        all_dates = pd.DatetimeIndex(self.dates)
        if self.raw_local_returns is not None:
            all_dates = all_dates.union(self.raw_local_returns.index)
        all_dates = all_dates.sort_values().unique()

        if not self.config.convert_returns_to_usd:
            self.fx_rates_usd_per_local = pd.DataFrame(
                {
                    ticker: (
                        1.0
                        if self.currency_map[ticker] == "USD"
                        else np.nan
                    )
                    for ticker in self.tickers
                },
                index=self.dates,
            )
            self.fx_returns = pd.DataFrame(
                0.0,
                index=self.dates,
                columns=self.tickers,
            )
            fx_quality = {
                "conversion_enabled": False,
                "base_currency": self.config.base_currency,
                "required_currencies": sorted(set(self.currency_map.values())),
                "missing_currencies": [],
                "stale_currencies": [],
                "price_return_reconciliation_max_abs_error": None,
            }
            self.data_quality["fx"] = fx_quality
            self.data_quality["fx_data_as_of"] = None
            self.data_quality["fx_missing_currencies"] = []
            self.data_quality["fx_stale_currencies"] = []
            self.data_quality["universe_funnel"] = dict(self.data_quality["universe"])
            return

        required_currencies = [
            self._full_currency_map[ticker] for ticker in conversion_columns
        ]
        # Use raw observations here.  The model-facing factor panel is
        # forward-filled, which is appropriate for features but would make an
        # old FX quote look newly observed and could hide the fresher external
        # Index.xlsx value.
        factor_prices = _raw_factor_prices_for_fx(self.raw)
        currency_rates, fx_quality = build_fx_rates_usd_per_local(
            all_dates,
            required_currencies,
            config=self.config,
            factor_prices=factor_prices,
        )
        ticker_rates_all = pd.DataFrame(
            {
                ticker: currency_rates[self._full_currency_map[ticker]]
                for ticker in conversion_columns
            },
            index=all_dates,
        )
        ticker_fx_returns_all = (
            ticker_rates_all.pct_change(fill_method=None).fillna(0.0)
        )

        self.fx_rates_usd_per_local = ticker_rates_all.reindex(
            index=self.dates,
            columns=self.tickers,
        )
        self.fx_returns = ticker_fx_returns_all.reindex(
            index=self.dates,
            columns=self.tickers,
        )

        local_price_rates = ticker_rates_all.reindex(
            index=self.local_prices.index,
            columns=self.local_prices.columns,
        )
        self.sheets["PX_LAST"] = self.local_prices * local_price_rates

        local_fx_returns = ticker_fx_returns_all.reindex(
            index=self.local_returns.index,
            columns=self.local_returns.columns,
        )
        usd_returns = (
            (1.0 + self.local_returns) * (1.0 + local_fx_returns) - 1.0
        )
        self.sheets["Daily_Returns"] = usd_returns

        if self.raw_local_returns is not None:
            raw_fx_returns = ticker_fx_returns_all.reindex(
                index=self.raw_local_returns.index,
                columns=self.raw_local_returns.columns,
            )
            self.raw_returns = (
                (1.0 + self.raw_local_returns) * (1.0 + raw_fx_returns) - 1.0
            )

        price_implied_returns = self.sheets["PX_LAST"].pct_change(fill_method=None)
        price_reconciliation = price_implied_returns - usd_returns.reindex(
            index=price_implied_returns.index,
            columns=price_implied_returns.columns,
        )
        fx_quality.update({
            "conversion_enabled": True,
            "base_currency": self.config.base_currency,
            "price_return_reconciliation_max_abs_error": self._max_abs_finite(
                price_reconciliation
            ),
        })
        latest_dates = [
            value
            for currency, value in fx_quality["latest_source_date_by_currency"].items()
            if currency != "USD" and value is not None
        ]
        fx_data_as_of = min(latest_dates) if latest_dates else (
            self.dates.max().strftime("%Y-%m-%d") if len(self.dates) else None
        )
        self.data_quality["fx"] = fx_quality
        self.data_quality["fx_data_as_of"] = fx_data_as_of
        self.data_quality["fx_missing_currencies"] = list(
            fx_quality["missing_currencies"]
        )
        self.data_quality["fx_stale_currencies"] = list(
            fx_quality["stale_currencies"]
        )
        self.data_quality["universe_funnel"] = dict(self.data_quality["universe"])
        self.data_quality["data_freshness"] = {
            "portfolio_data_as_of": (
                self.dates.max().strftime("%Y-%m-%d") if len(self.dates) else None
            ),
            "fx_data_as_of": fx_data_as_of,
            "max_fx_staleness_days": int(self.config.max_fx_staleness_days),
        }

    def _extract_raw_returns(self) -> Optional[pd.DataFrame]:
        """Return a standardized but UN-imputed copy of Daily_Returns.

        Survivorship / late-entrant checks must see the true first valid
        observation per ticker, so we mirror just the column/index
        standardization that preprocess_sheets applies and skip _fill_missing.
        """
        if "Daily_Returns" not in self.raw:
            return None
        df = self.raw["Daily_Returns"].copy()
        df = _standardize_columns(df)
        df = _standardize_index(df)
        df = _filter_tickers(df, tickers=self.full_universe)
        df = df.apply(pd.to_numeric, errors="coerce")
        return df

    def _load_earnings_timeline(self) -> Optional[pd.DataFrame]:
        """Earnings_Timeline 시트 로드 (date × ticker, 발표일=1).

        REDESIGN U (2026-04-14): "Earnings_Date" fallback 추가. Iter 14에서
        timeline activation이 model column space를 변경해 IR -0.290 손실을
        보였으나, 이는 (1) earn_cycle_pos가 whitelist에 있었던 것과 (2)
        revision cleaning이 정밀 모드로 전환된 것 두 가지 부수효과 때문.
        Iter 15에서는 이 두 부수효과를 격리 — earn_cycle_pos를 whitelist에서
        제거하고 sellside.py가 force-None을 clean_revision_spikes에 전달.
        결과적으로 timeline은 메모리에만 로드되고, model은 iter 9와 정확히
        동일한 56-feature panel을 사용. PEAD signal은 post-process에서만 주입.
        """
        sheet_key = None
        for name in ("Earnings_Timeline", "Earnings_Date"):
            if name in self.raw:
                sheet_key = name
                break
        if sheet_key is None:
            print("  [!] Earnings_Timeline/Earnings_Date 시트 없음 - 캘린더 기반 실적시즌 사용")
            return None
        df = self.raw[sheet_key].copy()
        df = _standardize_columns(df)
        df = _standardize_index(df)
        df = _filter_tickers(df, tickers=self.full_universe)
        df = df.fillna(0).astype(int)
        common = df.index.intersection(self.dates)
        df = df.loc[common]
        n_events = int((df == 1).sum().sum())
        print(f"  [O] Earnings_Timeline 로드: {df.shape}, 총 {n_events}개 발표일")
        return df

    @property
    def prices(self) -> pd.DataFrame:
        return self.sheets["PX_LAST"]

    @property
    def returns(self) -> pd.DataFrame:
        return self.sheets["Daily_Returns"]

    @property
    def market_cap(self) -> pd.DataFrame:
        return self.sheets["CUR_MKT_CAP"]

    @property
    def factor_prices(self) -> Optional[pd.DataFrame]:
        return self.factor_data.get("Factor_PX_LAST")

    @property
    def factor_returns(self) -> Optional[pd.DataFrame]:
        return self.factor_data.get("Factor_Returns")

    def has_factor_data(self) -> bool:
        return bool(self.factor_data) and "Factor_Returns" in self.factor_data

    def get_sheet(self, name: str) -> pd.DataFrame:
        if name not in self.sheets:
            raise KeyError(f"시트 '{name}'을 찾을 수 없습니다. 사용 가능: {list(self.sheets.keys())}")
        return self.sheets[name]

    def summary(self) -> str:
        lines = [
            f"데이터 경로: {self.data_path}",
            f"기간: {self.dates[0].strftime('%Y-%m-%d')} ~ {self.dates[-1].strftime('%Y-%m-%d')}",
            f"영업일 수: {len(self.dates)}",
            f"종목 수 (실제 로드): {len(self.tickers)} / {len(self.full_universe)}",
            f"시트 수: {len(self.sheets)}",
        ]
        if self.missing_tickers:
            lines.append(
                f"  [!] 누락 종목 (essential sheet 부재): {', '.join(self.missing_tickers)}"
            )
            for sheet_name, gone in self.missing_by_sheet.items():
                lines.append(f"      - {sheet_name}: {', '.join(gone)}")
        if getattr(self, "optional_missing", {}):
            lines.append(
                "  [i] Optional sheet 결측 (per-date median 대체):"
            )
            for sheet_name, gone in self.optional_missing.items():
                lines.append(f"      - {sheet_name}: {', '.join(gone)}")
        lines += [
            "",
            "시트별 shape:",
        ]
        for name, df in self.sheets.items():
            missing_pct = df.isna().sum().sum() / df.size * 100
            lines.append(f"  {name:40s} {str(df.shape):>15s}  결측: {missing_pct:.2f}%")
        if self.factor_data:
            lines.append("")
            lines.append("Factor 데이터:")
            for name, df in self.factor_data.items():
                lines.append(f"  {name:40s} {str(df.shape):>15s}")
        if self.earnings_timeline is not None:
            n_events = int((self.earnings_timeline == 1).sum().sum())
            lines.append(f"\nEarnings Timeline: {self.earnings_timeline.shape}, {n_events}개 발표일")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"UniverseData(tickers={len(self.tickers)}, "
            f"dates={len(self.dates)}, sheets={len(self.sheets)})"
        )
