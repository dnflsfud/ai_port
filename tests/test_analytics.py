# -*- coding: utf-8 -*-
"""analytics의 ticker_meta 주입 계약 테스트 (§S11.4 Phase 3, 2026-07-21 강화).

compute_style_sector_tilt_rows / compute_monthly_ow_explanation_rows는
ticker_meta를 명시적으로 주입받아야 한다. 미주입(None)이면 60/150 스테일
정적 TICKER_META를 무음 사용하는 잠재 결함이므로 ValueError로 거부한다.
레거시 정적 동작이 필요하면 ticker_meta=TICKER_META를 명시적으로 넘긴다.
"""

import pandas as pd
import pytest

from src.analytics import (
    compute_monthly_ow_explanation_rows,
    compute_style_sector_tilt_rows,
)
from src.metadata import TICKER_META


def _weights():
    tickers = ["AAA", "BBB"]
    w = pd.Series([0.7, 0.3], index=tickers)
    return {pd.Timestamp("2026-01-05"): w}, tickers


def test_tilt_rows_use_injected_ticker_meta():
    portfolio_weights, tickers = _weights()
    ticker_meta = {
        "AAA": {"sector": "Financials", "style": "Value", "sub": "X"},
        "BBB": {"sector": "Technology", "style": "Growth", "sub": "Y"},
    }
    rows = compute_style_sector_tilt_rows(
        portfolio_weights, tickers, ticker_meta=ticker_meta
    )
    assert len(rows) == 1
    assert "sector_Financials" in rows[0]
    assert "sector_Technology" in rows[0]
    assert "sector_Other" not in rows[0]


def test_tilt_rows_reject_missing_meta():
    portfolio_weights, tickers = _weights()
    with pytest.raises(ValueError, match="ticker_meta"):
        compute_style_sector_tilt_rows(portfolio_weights, tickers)


def test_tilt_rows_static_meta_requires_explicit_opt_in():
    # 레거시 정적 동작은 ticker_meta=TICKER_META 명시로만 사용 가능.
    portfolio_weights, tickers = _weights()
    rows = compute_style_sector_tilt_rows(
        portfolio_weights, tickers, ticker_meta=TICKER_META
    )
    # AAA/BBB는 정적 TICKER_META에 없음 -> 'Other' 버킷 (기존 fallback 동작)
    assert "sector_Other" in rows[0]


def test_monthly_ow_rows_reject_missing_meta():
    with pytest.raises(ValueError, match="ticker_meta"):
        compute_monthly_ow_explanation_rows(
            portfolio_weights={},
            predictions=pd.DataFrame(),
            returns=pd.DataFrame(),
            dates=pd.DatetimeIndex([]),
            tickers=[],
            group_contributions={},
        )
