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
from typing import Dict, Optional, List

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
]

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
    "FX": ["DXY", "USDKRW", "USDJPY", "EURUSD", "USDCNH"],
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


def _rename_sent_trend_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Sent_Trend 시트의 회사명 컬럼을 티커로 변환.

    C3: COMPANY_TO_TICKER 에 없는 회사명이 들어오면 warning 으로 알린다.
    (silent drop 방지 — 새 종목이 추가될 때 feature pipeline 이 조용히 누락하는 것을 차단)
    """
    rename_map = {}
    for col in df.columns:
        col_str = str(col).strip()
        for company, ticker in COMPANY_TO_TICKER.items():
            if company.lower() in col_str.lower():
                rename_map[col] = ticker
                break
    renamed = df.rename(columns=rename_map)
    # Post-rename verification: any column that is not a known ticker is unmapped.
    unmapped = [c for c in renamed.columns if str(c).strip() not in TICKERS]
    if unmapped:
        logger.warning(
            "Sent_Trend columns missing from COMPANY_TO_TICKER: %s. "
            "Add them to data_loader.COMPANY_TO_TICKER or the signal for these "
            "companies will be silently dropped downstream.",
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


def _filter_tickers(df: pd.DataFrame) -> pd.DataFrame:
    """TICKERS에 포함된 컬럼만 남김. 컬럼 순서 통일."""
    available = [t for t in TICKERS if t in df.columns]
    return df[available]


def load_universe_meta(raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Universe_Meta 시트에서 종목별 섹터 정보 추출."""
    if "Universe_Meta" not in raw:
        sectors = pd.Series("Unknown", index=TICKERS, name="sector")
        return sectors.to_frame()

    meta = raw["Universe_Meta"].copy()
    meta = _standardize_columns(meta)

    # 인덱스가 'AAPL US Equity' 형태이면 티커만 추출
    new_idx = []
    for idx in meta.index:
        idx_str = str(idx).strip()
        # "AAPL US Equity" -> "AAPL", "000660 KS Equity" -> "000660"
        parts = idx_str.split()
        new_idx.append(parts[0] if parts else idx_str)
    meta.index = new_idx

    # Sector 컬럼 표준화
    if "Sector" in meta.columns:
        meta = meta.rename(columns={"Sector": "sector"})

    return meta


def preprocess_sheets(
    raw: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    모든 시트를 전처리하여 반환.
    - DatetimeIndex 통일
    - Sent_Trend 컬럼 매핑
    - 티커 필터링
    - 결측치 처리
    """
    processed: Dict[str, pd.DataFrame] = {}

    for sheet_name, df in raw.items():
        if sheet_name in SKIP_SHEETS or sheet_name in FACTOR_SHEETS:
            continue

        df = df.copy()
        df = _standardize_columns(df)

        # Sent_Trend 시트는 회사명 -> 티커 매핑
        if sheet_name in SENT_TREND_SHEETS:
            df = _rename_sent_trend_columns(df)

        # Bloomberg Equity 시트는 'AAPL US Equity' -> 'AAPL' 매핑
        if sheet_name in BLOOMBERG_EQUITY_SHEETS:
            df = _rename_bloomberg_equity_columns(df)

        df = _standardize_index(df)
        df = _filter_tickers(df)

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

        # Preserve RAW (un-imputed) Daily_Returns BEFORE preprocess_sheets
        # fills NaN. Survivorship checks (run_selection_bias.py) need this
        # to detect late entrants — if we only expose the imputed panel,
        # first_valid_index() always returns the alignment start and the
        # check silently passes even when a ticker genuinely joined late.
        self.raw_returns = self._extract_raw_returns()

        self.sheets = preprocess_sheets(self.raw)
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
            "BEST_ROE", "BEST_PX_BPS_RATIO",
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
        }

        self.full_universe = list(TICKERS)
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

            self.tickers = [t for t in TICKERS if t in loaded_intersection]

            # Diagnostic: which tickers are missing from which sheets
            self.missing_tickers = [t for t in TICKERS if t not in loaded_intersection]
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
            self.tickers = list(TICKERS)
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

        self.dates = self.sheets[next(iter(self.sheets))].index

        # Factor 데이터 (별도 파이프라인)
        self.factor_data = load_factor_sheets(self.raw)
        if self.factor_data:
            for name, df in self.factor_data.items():
                common = df.index.intersection(self.dates)
                self.factor_data[name] = df.loc[common]

        # Earnings Timeline (종목별 실적발표일 0/1 매트릭스)
        self.earnings_timeline = self._load_earnings_timeline()

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
        df = _filter_tickers(df)
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
        df = _filter_tickers(df)
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
                f"  [i] Optional sheet 결측 (per-date median 대체):"
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
