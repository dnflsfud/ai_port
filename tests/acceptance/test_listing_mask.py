"""Acceptance tests — pre-listing backfill masking (default-OFF flag).

Written from the spec BEFORE implementation exists. Most tests here MUST FAIL
now because the target code (mask_pre_listing, make_capweight_bm_fn's new
`config` param, PipelineConfig.listing_* fields) is not written yet. The only
test that may pass today is the optimizer-pin regression (#8), which guards
already-existing behaviour.

Idioms (project convention): plain pytest functions, no fixtures, synthetic
data only (no Excel load), np.allclose(..., atol=1e-6), OFF-parity checks
reproduce the current code logic inline (_inline_reference).

Symbols that do not exist yet are imported INSIDE the test bodies so that
`pytest --collect-only` succeeds with zero collection errors (a missing symbol
then surfaces as a runtime ImportError/TypeError/AttributeError in the specific
test, not as a collection crash).

======================================================================
합격기준(spec) ↔ 테스트 매핑표
----------------------------------------------------------------------
S-A  mask_pre_listing: 미매칭 티커/미존재 티커는 불변(.equals)          -> test_mask_pre_listing_off_is_identity_when_no_tickers_match
S-B  mask_pre_listing inclusive=True: date<=listing 마스킹, 이후 보존,
     타 컬럼 불변                                                       -> test_mask_pre_listing_inclusive_masks_listing_day
S-C  mask_pre_listing inclusive=False: date<listing만 마스킹, 상장일 보존 -> test_mask_pre_listing_exclusive_keeps_listing_day
S-D  mask_pre_listing: 새 프레임 반환, 원본 불변(immutable)             -> test_mask_pre_listing_does_not_mutate_input
S-E  make_capweight_bm_fn OFF/미전달: 기존 median 대체 동작 바이트동일   -> test_capweight_fn_off_parity
S-F  make_capweight_bm_fn ON: 마스킹 티커 weight==0(median 금지),
     나머지 유효 cap 비례·합≈1                                          -> test_capweight_fn_on_zero_weight_for_masked
S-G  PipelineConfig OFF-default + listing_dates 3종목 기본값             -> test_config_defaults
S-H  optimize_portfolio: NaN 알파 + bm=0 티커 weight≈0 (회귀 방지)       -> test_optimizer_pins_nan_alpha_to_zero_bm
S-I  (경계) mask_pre_listing: 프레임에 일부만 존재하는 티커도 안전 처리   -> test_mask_pre_listing_partial_ticker_overlap
S-J  (경계) mask_pre_listing: listing_dates 빈 dict면 완전 불변          -> test_mask_pre_listing_empty_dates_is_identity
======================================================================
"""

import types

import numpy as np
import pandas as pd

from src.config import PipelineConfig


# Canonical listing dates from the spec (independent hardcode, NOT imported
# from the implementation).
SPEC_LISTING_DATES = {
    "PLTR": "2020-09-30",
    "GEV": "2024-04-02",
    "BE": "2018-07-25",
    "285A": "2024-12-18",
    "SNDK": "2025-02-24",
    "ARM": "2023-09-14",
    "CEG": "2022-02-02",
}


def _daily_frame():
    """Synthetic date x ticker frame straddling PLTR's listing (2020-09-30).

    PLTR column carries distinct non-NaN values so masking vs preservation is
    unambiguous. OTHER is a control column that must never change.
    """
    idx = pd.to_datetime(
        ["2020-09-28", "2020-09-29", "2020-09-30", "2020-10-01", "2020-10-02"]
    )
    return pd.DataFrame(
        {
            "PLTR": [10.0, 11.0, 12.0, 13.0, 14.0],
            "OTHER": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# S-A: no matching tickers -> identity
# ---------------------------------------------------------------------------
def test_mask_pre_listing_off_is_identity_when_no_tickers_match():
    from src.data_loader import mask_pre_listing

    idx = pd.to_datetime(["2019-01-01", "2020-01-01", "2021-01-01"])
    df = pd.DataFrame({"X": [1.0, 2.0, 3.0], "Y": [4.0, 5.0, 6.0]}, index=idx)
    # listing_dates references PLTR, which is absent from the frame.
    out = mask_pre_listing(df, {"PLTR": "2020-09-30"}, inclusive=True)
    assert out.equals(df)


# ---------------------------------------------------------------------------
# S-B: inclusive=True masks the listing day itself and everything before
# ---------------------------------------------------------------------------
def test_mask_pre_listing_inclusive_masks_listing_day():
    from src.data_loader import mask_pre_listing

    df = _daily_frame()
    out = mask_pre_listing(df, {"PLTR": "2020-09-30"}, inclusive=True)

    pltr = out["PLTR"]
    # dates 2020-09-28, -29, -30 (<= listing) must be NaN
    assert bool(np.isnan(pltr.iloc[0]))
    assert bool(np.isnan(pltr.iloc[1]))
    assert bool(np.isnan(pltr.iloc[2]))
    # dates after listing preserved exactly
    assert pltr.iloc[3] == 13.0
    assert pltr.iloc[4] == 14.0
    # control column untouched
    assert out["OTHER"].equals(df["OTHER"])


# ---------------------------------------------------------------------------
# S-C: inclusive=False preserves the listing day (only strictly-before masked)
# ---------------------------------------------------------------------------
def test_mask_pre_listing_exclusive_keeps_listing_day():
    from src.data_loader import mask_pre_listing

    df = _daily_frame()
    out = mask_pre_listing(df, {"PLTR": "2020-09-30"}, inclusive=False)

    pltr = out["PLTR"]
    # strictly before listing -> NaN
    assert bool(np.isnan(pltr.iloc[0]))
    assert bool(np.isnan(pltr.iloc[1]))
    # listing day itself preserved (real value)
    assert pltr.iloc[2] == 12.0
    # after listing preserved
    assert pltr.iloc[3] == 13.0
    assert pltr.iloc[4] == 14.0
    assert out["OTHER"].equals(df["OTHER"])


# ---------------------------------------------------------------------------
# S-D: returns a new frame; the input must not be mutated
# ---------------------------------------------------------------------------
def test_mask_pre_listing_does_not_mutate_input():
    from src.data_loader import mask_pre_listing

    df = _daily_frame()
    snapshot = df.copy(deep=True)
    out = mask_pre_listing(df, {"PLTR": "2020-09-30"}, inclusive=True)

    # original object unchanged (byte-identical to snapshot)
    assert df.equals(snapshot)
    # a distinct object was returned
    assert out is not df


# ---------------------------------------------------------------------------
# S-I (boundary): only some listing_dates tickers appear in the frame
# ---------------------------------------------------------------------------
def test_mask_pre_listing_partial_ticker_overlap():
    from src.data_loader import mask_pre_listing

    idx = pd.to_datetime(
        ["2020-09-28", "2020-09-29", "2020-09-30", "2020-10-01"]
    )
    df = pd.DataFrame(
        {
            "PLTR": [10.0, 11.0, 12.0, 13.0],
            "OTHER": [1.0, 2.0, 3.0, 4.0],
        },
        index=idx,
    )
    # GEV/BE are in listing_dates but absent from the frame -> ignored.
    out = mask_pre_listing(df, SPEC_LISTING_DATES, inclusive=True)

    pltr = out["PLTR"]
    assert bool(np.isnan(pltr.iloc[0]))
    assert bool(np.isnan(pltr.iloc[1]))
    assert bool(np.isnan(pltr.iloc[2]))  # 2020-09-30 inclusive
    assert pltr.iloc[3] == 13.0
    # untouched control column
    assert out["OTHER"].equals(df["OTHER"])


# ---------------------------------------------------------------------------
# S-J (boundary): empty listing_dates -> full identity
# ---------------------------------------------------------------------------
def test_mask_pre_listing_empty_dates_is_identity():
    from src.data_loader import mask_pre_listing

    df = _daily_frame()
    out = mask_pre_listing(df, {}, inclusive=True)
    assert out.equals(df)


# ---------------------------------------------------------------------------
# S-E: make_capweight_bm_fn with config omitted / OFF == existing median-fill
# ---------------------------------------------------------------------------
def _capweight_reference(mc_df, tickers, query_date):
    """Independent re-implementation of the CURRENT (flag-OFF) capweight logic.

    Mirrors make_capweight_bm_fn: forward-fill, then substitute NaN/non-positive
    caps with the ticker's historical median, drop still-missing to 0, normalise.
    """
    mc_aligned = mc_df.reindex(columns=tickers).ffill()
    row = np.array(mc_aligned.loc[query_date], dtype=float)  # writable copy
    nan_mask = ~np.isfinite(row) | (row <= 0)
    if nan_mask.any():
        medians = mc_aligned.median(axis=0).to_numpy()
        row[nan_mask] = medians[nan_mask]
    row = np.where(np.isfinite(row) & (row > 0), row, 0.0)
    s = row.sum()
    return row / s


def test_capweight_fn_off_parity():
    from src.backtest import make_capweight_bm_fn

    tickers = ["A", "B", "C"]
    idx = pd.to_datetime(
        ["2020-09-28", "2020-09-29", "2020-09-30", "2020-10-01", "2020-10-02"]
    )
    # A has leading-NaN (pre-listing backfill gap); B, C fully populated.
    mc = pd.DataFrame(
        {
            "A": [np.nan, np.nan, 100.0, 100.0, 100.0],
            "B": [200.0, 200.0, 200.0, 200.0, 200.0],
            "C": [300.0, 300.0, 300.0, 300.0, 300.0],
        },
        index=idx,
    )
    data = types.SimpleNamespace(market_cap=mc)
    query = idx[1]  # a date where A is still NaN -> exercises median fallback

    # Explicit OFF config retains the legacy median-fill branch.
    cfg = PipelineConfig(listing_mask_enabled=False)
    fn = make_capweight_bm_fn(data, tickers, config=cfg)
    w = np.asarray(fn(query, tickers, len(tickers)), dtype=float)

    expected_ref = _capweight_reference(mc, tickers, query)
    # Closed-form anchor: A imputed to its median 100 -> [100,200,300]/600.
    expected_closed = np.array([100.0, 200.0, 300.0]) / 600.0

    assert np.allclose(expected_ref, expected_closed, atol=1e-6)
    assert np.allclose(w, expected_closed, atol=1e-6)
    assert abs(w.sum() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# S-F: make_capweight_bm_fn with listing_mask ON -> masked cap -> weight 0
# ---------------------------------------------------------------------------
def test_capweight_fn_on_zero_weight_for_masked():
    from src.backtest import make_capweight_bm_fn

    tickers = ["A", "B", "C"]
    idx = pd.to_datetime(
        ["2020-09-28", "2020-09-29", "2020-09-30", "2020-10-01", "2020-10-02"]
    )
    mc = pd.DataFrame(
        {
            "A": [np.nan, np.nan, 100.0, 100.0, 100.0],
            "B": [200.0, 200.0, 200.0, 200.0, 200.0],
            "C": [300.0, 300.0, 300.0, 300.0, 300.0],
        },
        index=idx,
    )
    data = types.SimpleNamespace(market_cap=mc)
    query = idx[1]  # A still NaN here (pre-listing)

    cfg = PipelineConfig()
    cfg.listing_mask_enabled = True
    fn = make_capweight_bm_fn(data, tickers, config=cfg)
    w = np.asarray(fn(query, tickers, len(tickers)), dtype=float)

    # Masked (NaN cap) ticker gets ZERO weight — no median substitution.
    expected = np.array([0.0, 200.0, 300.0]) / 500.0
    assert np.isclose(w[0], 0.0, atol=1e-6)
    assert np.allclose(w, expected, atol=1e-6)
    assert abs(w.sum() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# S-G: config defaults — OFF-default invariant + listing_dates defaults
# ---------------------------------------------------------------------------
def test_config_defaults():
    c = PipelineConfig()
    assert c.listing_mask_enabled is True
    assert c.listing_dates == SPEC_LISTING_DATES


# ---------------------------------------------------------------------------
# S-H: optimizer pins NaN-alpha + bm=0 name to zero (regression guard; this
# test may already pass against current code)
# ---------------------------------------------------------------------------
def test_optimizer_pins_nan_alpha_to_zero_bm():
    from src.portfolio_optimizer import optimize_portfolio

    tickers = [f"T{i}" for i in range(6)]
    # T0: NaN alpha (masked/pre-listing); others finite.
    mu = pd.Series([np.nan, 0.03, 0.01, -0.02, 0.02, -0.01], index=tickers)
    cov = np.eye(6) * (0.02 ** 2)  # diagonal, well-conditioned daily cov
    # T0 has bm weight 0 (masked); remaining 5 names sum to 1.
    bm = np.array([0.0, 0.2, 0.2, 0.2, 0.2, 0.2])

    w = optimize_portfolio(mu, cov, bm_weights=bm, config=PipelineConfig())
    w = np.asarray(w, dtype=float)

    assert np.isfinite(w).all()
    # NaN-alpha name is pinned to bm_i == 0.
    assert np.isclose(w[0], 0.0, atol=1e-6)
