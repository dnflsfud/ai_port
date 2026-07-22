import numpy as np
import pandas as pd
import pytest

from src.config import PipelineConfig
from src.data_loader import (
    TICKERS,
    UniverseData,
    _rename_sent_trend_columns,
    build_fx_rates_usd_per_local,
    build_company_to_ticker,
    load_universe_meta,
    normalize_fx_quotes_to_usd_per_local,
)


ESSENTIAL_SHEETS = {
    "PX_LAST",
    "Daily_Returns",
    "CUR_MKT_CAP",
    "BEST_EPS",
    "BEST_SALES",
    "BEST_PE_RATIO",
    "OPER_MARGIN",
    "BEST_ROE",
    "NEWS_SENTIMENT_DAILY_AVG",
    "EQY_REC_CONS",
    "Factset_EPS_Revision",
    "Factset_Sales_Revision",
    "Factset_TG_Price",
}


def _raw_fixture(bloomberg_tickers, names, prices, factor_prices=None):
    dates = prices.index
    simple = [ticker.split()[0] for ticker in bloomberg_tickers]
    meta = pd.DataFrame(
        {
            "Ticker": simple,
            "Name": names,
            "Sector": ["Test"] * len(simple),
            "Status": ["Active"] * len(simple),
        },
        index=bloomberg_tickers,
    )
    local_returns = prices.pct_change(fill_method=None).fillna(0.0)
    raw = {
        "Universe_Meta": meta,
        "PX_LAST": prices,
        "Daily_Returns": local_returns,
    }
    for sheet in ESSENTIAL_SHEETS - {"PX_LAST", "Daily_Returns"}:
        raw[sheet] = pd.DataFrame(
            1.0,
            index=dates,
            columns=simple,
        )
    if factor_prices is not None:
        raw["Factor_PX_LAST"] = factor_prices
    return raw


def _patch_raw(monkeypatch, raw):
    monkeypatch.setattr(
        "src.data_loader.load_all_sheets",
        lambda _path: {name: frame.copy() for name, frame in raw.items()},
    )


def test_fallback_universe_has_exact_150_workbook_order():
    assert len(TICKERS) == 150
    assert len(set(TICKERS)) == 150
    assert TICKERS[:5] == ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    assert TICKERS[-5:] == ["KO", "ULVR", "ECL", "AI", "IBE"]


def test_usd_and_listing_guardrails_are_enabled_for_100_name_regime():
    config = PipelineConfig()
    assert config.base_currency == "USD"
    assert config.convert_returns_to_usd is True
    assert config.fail_on_missing_fx is True
    assert config.listing_mask_enabled is True
    assert config.listing_dates == {
        "PLTR": "2020-09-30",
        "GEV": "2024-04-02",
        "BE": "2018-07-25",
        "285A": "2024-12-18",
        "SNDK": "2025-02-24",
        "ARM": "2023-09-14",
        "CEG": "2022-02-02",
        # S11 150-name expansion (decision log §S11)
        "DELL": "2018-12-28",
        "ABNB": "2020-12-10",
        "UMG": "2021-09-21",
        "GE": "2024-04-02",
        "TT": "2020-03-02",
        "BN": "2022-12-12",
        # S11.4 coverage audit — unregistered leading-constant backfills
        "ANET": "2014-06-06",
        "RACE": "2015-10-21",
        "LITE": "2015-07-27",
        "VST": "2016-10-05",
        "SPOT": "2018-04-03",
        "VRT": "2018-08-01",
    }


def test_fx_quote_directions_are_normalized_to_usd_per_local():
    idx = pd.to_datetime(["2026-07-15"])
    quotes = pd.DataFrame(
        {
            "USDKRW Curncy": [1400.0],
            "USDJPY Curncy": [160.0],
            "EURUSD Curncy": [1.2],
            "USDCHF Curncy": [0.8],
            "GBPUSD Curncy": [1.4],
            "USDDKK Curncy": [7.0],
        },
        index=idx,
    )
    result = normalize_fx_quotes_to_usd_per_local(quotes)

    assert result.loc[idx[0], "KRW"] == pytest.approx(1.0 / 1400.0)
    assert result.loc[idx[0], "JPY"] == pytest.approx(1.0 / 160.0)
    assert result.loc[idx[0], "EUR"] == pytest.approx(1.2)
    assert result.loc[idx[0], "CHF"] == pytest.approx(1.0 / 0.8)
    assert result.loc[idx[0], "GBP"] == pytest.approx(1.4)
    assert result.loc[idx[0], "DKK"] == pytest.approx(1.0 / 7.0)


def test_fx_freshness_guard_fails_on_stale_required_currency():
    target = pd.to_datetime(["2026-07-10"])
    external = pd.DataFrame(
        {"GBPUSD Curncy": [1.35]},
        index=pd.to_datetime(["2026-07-01"]),
    )
    with pytest.raises(ValueError, match="stale>3d=.*GBP"):
        build_fx_rates_usd_per_local(
            target,
            ["GBP"],
            config=PipelineConfig(
                fx_source_path="missing.xlsx",
                max_fx_staleness_days=3,
            ),
            external_quotes=external,
        )


def test_meta_drives_exact_sentiment_mapping_and_currency_without_substrings():
    raw = {
        "Universe_Meta": pd.DataFrame(
            {
                "Ticker": ["V", "ASML", "000660"],
                "Name": ["Visa", "ASML", "SK Hynix"],
                "Sector": ["Financials", "Technology", "Technology"],
                "Status": ["Active", "Active", "Active"],
            },
            index=["V US Equity", "ASML NA Equity", "000660 KS Equity"],
        )
    }
    meta = load_universe_meta(raw)
    assert list(meta.index) == ["V", "ASML", "000660"]
    assert meta["currency"].to_dict() == {
        "V": "USD",
        "ASML": "EUR",
        "000660": "KRW",
    }

    mapping = build_company_to_ticker(meta)
    sent = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0]],
        columns=[" Visa ", "ASML", "SK Hynix", "VisaSomething"],
    )
    renamed = _rename_sent_trend_columns(
        sent,
        tickers=list(meta.index),
        company_to_ticker=mapping,
    )
    assert list(renamed.columns[:3]) == ["V", "ASML", "000660"]
    assert renamed.columns[3] == "VisaSomething"


def test_usd_return_identity_preserves_local_and_raw_panels(monkeypatch):
    dates = pd.bdate_range("2026-07-13", periods=3)
    local_prices = pd.DataFrame(
        {"KOR": [100.0, 110.0, 121.0]},
        index=dates,
    )
    factor_prices = pd.DataFrame(
        {"USDKRW": [1000.0, 1100.0, 1210.0]},
        index=dates,
    )
    raw = _raw_fixture(
        ["KOR KS Equity"],
        ["Korea Test"],
        local_prices,
        factor_prices=factor_prices,
    )
    _patch_raw(monkeypatch, raw)
    data = UniverseData(
        "unused.xlsx",
        config=PipelineConfig(fx_source_path="missing.xlsx"),
    )

    expected = (1.0 + data.local_returns) * (1.0 + data.fx_returns) - 1.0
    pd.testing.assert_frame_equal(data.returns, expected)
    pd.testing.assert_frame_equal(data.local_prices, local_prices)
    pd.testing.assert_frame_equal(data.raw_local_returns, raw["Daily_Returns"])
    assert data.returns.iloc[1, 0] == pytest.approx(0.0)
    assert data.raw_returns.iloc[1, 0] == pytest.approx(0.0)
    assert data.prices.iloc[1, 0] == pytest.approx(110.0 / 1100.0)
    assert data.market_cap.iloc[1, 0] == pytest.approx(1.0)


def test_external_fx_tail_is_not_hidden_by_model_factor_forward_fill(monkeypatch):
    dates = pd.bdate_range("2026-07-13", periods=3)
    local_prices = pd.DataFrame(
        {"KOR": [100.0, 110.0, 121.0]},
        index=dates,
    )
    factor_prices = pd.DataFrame(
        {"USDKRW": [1000.0, np.nan, np.nan]},
        index=dates,
    )
    external_quotes = pd.DataFrame(
        {"USDKRW Curncy": [1000.0, 1100.0, 1210.0]},
        index=dates,
    )
    raw = _raw_fixture(
        ["KOR KS Equity"],
        ["Korea Test"],
        local_prices,
        factor_prices=factor_prices,
    )
    _patch_raw(monkeypatch, raw)
    monkeypatch.setattr(
        "src.data_loader.load_external_fx_quotes",
        lambda _path, _currencies: external_quotes.copy(),
    )

    data = UniverseData(
        "unused.xlsx",
        config=PipelineConfig(fx_source_path="unused_fx.xlsx"),
    )

    # Model factors remain forward-filled, while FX accounting uses the raw
    # gaps and therefore accepts the fresher external observations.
    assert data.factor_data["Factor_PX_LAST"]["USDKRW"].tolist() == [
        1000.0,
        1000.0,
        1000.0,
    ]
    assert data.fx_rates_usd_per_local["KOR"].tolist() == pytest.approx(
        [1.0 / 1000.0, 1.0 / 1100.0, 1.0 / 1210.0]
    )
    assert data.returns["KOR"].tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_best_px_bps_is_optional_so_pm_survives_and_all_usd_needs_no_fx_file(
    monkeypatch,
):
    dates = pd.bdate_range("2026-07-13", periods=3)
    prices = pd.DataFrame(
        {"AAPL": [100.0, 101.0, 102.0], "PM": [90.0, 91.0, 92.0]},
        index=dates,
    )
    raw = _raw_fixture(
        ["AAPL US Equity", "PM US Equity"],
        ["Apple", "Philip Morris International"],
        prices,
    )
    raw["BEST_PX_BPS_RATIO"] = pd.DataFrame(
        {"AAPL": [10.0, 10.1, 10.2]},
        index=dates,
    )
    raw["Sent_Trend_Momentum_Timeseries"] = pd.DataFrame(
        {
            "Apple": [1.0, 2.0, 3.0],
            "Philip Morris International": [4.0, 5.0, 6.0],
        },
        index=dates,
    )
    _patch_raw(monkeypatch, raw)
    data = UniverseData(
        "unused.xlsx",
        config=PipelineConfig(fx_source_path="definitely_missing.xlsx"),
    )

    assert data.tickers == ["AAPL", "PM"]
    assert data.data_quality["universe"]["essential_ticker_count"] == 2
    assert data.optional_missing["BEST_PX_BPS_RATIO"] == ["PM"]
    assert list(data.get_sheet("Sent_Trend_Momentum_Timeseries").columns) == [
        "AAPL",
        "PM",
    ]
    assert data.currency_map == {"AAPL": "USD", "PM": "USD"}
    assert np.allclose(data.fx_rates_usd_per_local.to_numpy(), 1.0)
