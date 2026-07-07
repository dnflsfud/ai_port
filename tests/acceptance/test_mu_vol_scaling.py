"""Acceptance tests — A1 z→mu volatility scaling overlay (default-OFF).

Written from the spec (spec-a1-mu-vol-scaling.md) BEFORE any implementation
exists. Every test here MUST be RED now: the target symbol
``src.backtest.apply_mu_vol_scaling`` and the config flag
``PipelineConfig.mu_vol_scaling_enabled`` are not written yet, so the tests
fail with ImportError / AttributeError (NOT collection errors).

Pinned implementation contract (this is the interface the implementer MUST
provide — the name/signature/semantics below become the contract):

    src.backtest.apply_mu_vol_scaling(
        predictions: pd.DataFrame,     # date x ticker, post-overlay-chain z
        risk_returns: pd.DataFrame,    # date x ticker, non-interpolated raw
                                       #   returns on the FULL daily grid
                                       #   (data.raw_returns reindexed);
                                       #   predictions.index ⊆ risk_returns.index
        config: PipelineConfig,
    ) -> pd.DataFrame                  # date x ticker, transformed z (mu)

Pre-registered transform (spec §"사전등록 변환"), per date t, ticker i:

    mu_i(t) = z_i(t) · σ_i(t) / median_CS{ σ_j(t) : j valid }

    σ_i(t) = std of ticker i's raw returns over the trailing `cov_lookback`
             (=126) rows STRICTLY BEFORE t  — matching the covariance window
             convention `risk_source.iloc[hist_start:t_idx]` (exclusive of t).
    valid  = (# non-NaN obs in window ≥ 63) AND finite σ.
    Guards (all inert-leaning):
      (a) valid obs < 63  → σ_i := CS median  (scale 1)
      (b) σ_i non-finite  → σ_i := CS median  (scale 1)
      (c) no valid ticker → whole date is identity (mu = z)
      (d) NaN prediction  → stays NaN
    No clipping. Median-scaled so the median-σ ticker's mu == its z.

The independent expected values are computed here with plain numpy/pandas
(pandas .std on the same synthetic window) — the implementation's transform
code is NOT reused. The σ/median ratio is invariant to the std ddof choice
when the valid tickers share the same observation count (a common constant
factor cancels), so these tests do not pin ddof; the synthetic panels used
for exact-value checks give every valid ticker an identical obs count.

Idioms (project convention): plain pytest functions, no fixtures, synthetic
data only, np.allclose(..., atol=1e-6). Target symbols imported INSIDE test
bodies so `pytest --collect-only` succeeds with zero collection errors.

======================================================================
합격기준(spec/task) ↔ 테스트 매핑표
----------------------------------------------------------------------
1  OFF 항등 (default & enabled=False → 값·dtype 그대로)
       -> test_off_default_is_identity
       -> test_off_explicit_false_is_identity
2  수식 정확성 (mu = z·σ/median_CS(σ); median 종목 mu==z)
       -> test_formula_matches_independent_computation
       -> test_median_stock_mu_equals_z
3  룩어헤드 금지 (t 이후 returns 변경 → t 결과 불변)
       -> test_no_lookahead_on_future_returns
4a 가드: 유효 관측 <63 종목 → 스케일 1
       -> test_guard_insufficient_obs_scale_one
4b 가드: σ 비유한 종목 → 스케일 1
       -> test_guard_nonfinite_sigma_scale_one
4c 가드: 날짜 전체 σ 없음 → 그 날짜 항등
       -> test_guard_whole_date_identity
       -> test_early_dates_below_min_obs_are_identity  (dense-panel boundary)
4d 가드: NaN 예측 → NaN 유지
       -> test_nan_prediction_preserved
5  config 플래그 존재 + 기본 False
       -> test_config_flag_default_off
6  lookback: cov_lookback(126) 필드 사용 (마지막 126개만 반영)
       -> test_only_trailing_lookback_used
       -> test_cov_lookback_field_is_honored
======================================================================
"""

import numpy as np
import pandas as pd

from src.config import PipelineConfig


LOOKBACK_DEFAULT = 126
MIN_OBS = 63


# ---------------------------------------------------------------------------
# Synthetic-data + independent-reference helpers (NOT the implementation)
# ---------------------------------------------------------------------------
def _grid(n_rows):
    return pd.date_range("2015-01-02", periods=n_rows, freq="B")


def _raw_returns(n_rows, n_tickers=5, seed=0, vol_step=0.5):
    """date x ticker raw-return panel with DISTINCT per-ticker volatility.

    Ticker j's returns are scaled by (1 + vol_step*j) so the cross-sectional
    σ's are distinct → the σ/median ratio is a non-trivial (≠1) scale for
    every non-median ticker.
    """
    rng = np.random.default_rng(seed)
    tickers = [f"T{j}" for j in range(n_tickers)]
    base = rng.normal(0.0, 0.01, size=(n_rows, n_tickers))
    scales = 1.0 + vol_step * np.arange(n_tickers)
    return pd.DataFrame(base * scales, index=_grid(n_rows), columns=tickers)


def _z_panel(index, tickers, seed=7):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0.0, 1.0, size=(len(index), len(tickers))),
        index=index,
        columns=tickers,
    )


def _cfg(enabled=True, cov_lookback=None):
    c = PipelineConfig()
    c.mu_vol_scaling_enabled = enabled  # dataclass instance is not frozen
    if cov_lookback is not None:
        c.cov_lookback = cov_lookback
    return c


def _expected_scale_row(raw, k, lookback=LOOKBACK_DEFAULT, min_obs=MIN_OBS):
    """Independent σ/median scale vector for the date at grid position k.

    Window = rows STRICTLY BEFORE position k (exclusive of t), matching the
    covariance convention. Returns a ticker-indexed Series, or None when the
    whole date has no valid ticker (guard c → identity).
    """
    start = max(0, k - lookback)
    window = raw.iloc[start:k]
    sig = window.std(axis=0, ddof=1)          # per-ticker, NaN-skipping
    counts = window.notna().sum(axis=0)
    valid = (counts >= min_obs) & np.isfinite(sig)
    if not bool(valid.any()):
        return None
    med = float(np.median(sig[valid].to_numpy()))
    sig_eff = sig.where(valid, med)           # guarded tickers → median (scale 1)
    return sig_eff / med


def _expected_mu(raw, z, lookback=LOOKBACK_DEFAULT, min_obs=MIN_OBS):
    """Independent full expected mu panel (NaN z stays NaN; identity dates = z)."""
    out = z.copy()
    for t in z.index:
        k = raw.index.get_loc(t)
        scale = _expected_scale_row(raw, k, lookback, min_obs)
        if scale is None:
            continue  # identity for this date
        out.loc[t] = z.loc[t] * scale.reindex(z.columns)
    return out


# ---------------------------------------------------------------------------
# 1. OFF identity
# ---------------------------------------------------------------------------
def test_off_default_is_identity():
    """A fresh PipelineConfig has the flag OFF → output byte-equals input."""
    from src.backtest import apply_mu_vol_scaling

    raw = _raw_returns(130)
    z = _z_panel(raw.index, raw.columns)
    out = apply_mu_vol_scaling(z, raw, PipelineConfig())
    assert out.equals(z)  # values + dtype + NaN positions identical


def test_off_explicit_false_is_identity():
    from src.backtest import apply_mu_vol_scaling

    raw = _raw_returns(130, seed=3)
    z = _z_panel(raw.index, raw.columns, seed=4)
    out = apply_mu_vol_scaling(z, raw, _cfg(enabled=False))
    assert out.equals(z)


# ---------------------------------------------------------------------------
# 2. Formula correctness
# ---------------------------------------------------------------------------
def test_formula_matches_independent_computation():
    """mu = z·σ/median_CS(σ) reproduced on a dense panel (late rows active,
    early rows identity via guard c)."""
    from src.backtest import apply_mu_vol_scaling

    raw = _raw_returns(130)
    z = _z_panel(raw.index, raw.columns)
    out = apply_mu_vol_scaling(z, raw, _cfg(enabled=True))

    expected = _expected_mu(raw, z)
    assert np.allclose(out.to_numpy(), expected.to_numpy(), atol=1e-6, equal_nan=True)

    # Spot-check that the transform is actually ACTIVE on late rows (not a
    # silent identity) — at least one late cell must differ from raw z.
    late = z.index[MIN_OBS:]  # rows with a ≥63 window
    assert not np.allclose(
        out.loc[late].to_numpy(), z.loc[late].to_numpy(), atol=1e-6
    )


def test_median_stock_mu_equals_z():
    """The cross-sectional median-σ ticker has scale exactly 1 → mu == z."""
    from src.backtest import apply_mu_vol_scaling

    raw = _raw_returns(130)               # 5 tickers, distinct σ (odd count)
    z = _z_panel(raw.index, raw.columns)
    out = apply_mu_vol_scaling(z, raw, _cfg(enabled=True))

    t = z.index[-1]                       # a fully-active late date
    scale = _expected_scale_row(raw, raw.index.get_loc(t))
    med_ticker = (scale - 1.0).abs().idxmin()   # the σ==median ticker (scale 1)
    assert np.isclose(float(scale[med_ticker]), 1.0, atol=1e-9)  # sanity
    assert np.isclose(out.loc[t, med_ticker], z.loc[t, med_ticker], atol=1e-6)


# ---------------------------------------------------------------------------
# 3. No look-ahead
# ---------------------------------------------------------------------------
def test_no_lookahead_on_future_returns():
    """Mutating raw returns at dates strictly AFTER t must not change the
    transform result at t."""
    from src.backtest import apply_mu_vol_scaling

    raw = _raw_returns(160, seed=11)
    z = _z_panel(raw.index, raw.columns, seed=12)
    cfg = _cfg(enabled=True)

    out_base = apply_mu_vol_scaling(z, raw, cfg)

    # Corrupt every row strictly after position 100 with huge values.
    raw_future = raw.copy()
    raw_future.iloc[101:] = 9.0
    out_future = apply_mu_vol_scaling(z, raw_future, cfg)

    t = raw.index[100]  # date whose window is entirely at/ before position 100
    assert np.allclose(
        out_base.loc[t].to_numpy(),
        out_future.loc[t].to_numpy(),
        atol=1e-9,
        equal_nan=True,
    )


# ---------------------------------------------------------------------------
# 4a. Guard: fewer than 63 valid observations → scale 1
# ---------------------------------------------------------------------------
def test_guard_insufficient_obs_scale_one():
    from src.backtest import apply_mu_vol_scaling

    raw = _raw_returns(70, seed=5)                 # window at last row = 69 rows
    # Ticker T0: NaN in 10 of the window rows → 59 valid (<63) → guarded.
    raw.iloc[0:10, raw.columns.get_loc("T0")] = np.nan

    t = raw.index[-1]
    z = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0, 5.0]], index=[t], columns=list(raw.columns)
    )
    out = apply_mu_vol_scaling(z, raw, _cfg(enabled=True))

    # Guarded ticker: scale 1 → mu == z.
    assert np.isclose(out.loc[t, "T0"], z.loc[t, "T0"], atol=1e-6)
    # A valid, non-median ticker is genuinely scaled (mu != z).
    assert not np.isclose(out.loc[t, "T1"], z.loc[t, "T1"], atol=1e-6)
    # And it matches the independent reference exactly.
    expected = _expected_mu(raw, z)
    assert np.allclose(out.to_numpy(), expected.to_numpy(), atol=1e-6, equal_nan=True)


# ---------------------------------------------------------------------------
# 4b. Guard: non-finite σ → scale 1 (isolated from the count guard)
# ---------------------------------------------------------------------------
def test_guard_nonfinite_sigma_scale_one():
    from src.backtest import apply_mu_vol_scaling

    raw = _raw_returns(70, seed=6)
    # Ticker T0 keeps a FULL observation count (69) but one +inf poisons its
    # std → σ non-finite → guarded via the finiteness leg, not the count leg.
    raw.iloc[5, raw.columns.get_loc("T0")] = np.inf

    t = raw.index[-1]
    z = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0, 5.0]], index=[t], columns=list(raw.columns)
    )
    out = apply_mu_vol_scaling(z, raw, _cfg(enabled=True))

    assert np.isclose(out.loc[t, "T0"], z.loc[t, "T0"], atol=1e-6)   # scale 1
    assert not np.isclose(out.loc[t, "T1"], z.loc[t, "T1"], atol=1e-6)
    expected = _expected_mu(raw, z)
    assert np.allclose(out.to_numpy(), expected.to_numpy(), atol=1e-6, equal_nan=True)


# ---------------------------------------------------------------------------
# 4c. Guard: whole date has no valid σ → identity
# ---------------------------------------------------------------------------
def test_guard_whole_date_identity():
    """A prediction date with < 63 rows of history before it → identity for
    every ticker (no ticker can reach the 63-obs floor)."""
    from src.backtest import apply_mu_vol_scaling

    raw = _raw_returns(30, seed=8)          # only ≤29 rows precede any date
    t = raw.index[-1]                       # window = 29 rows < 63 for all
    z = pd.DataFrame(
        [[0.5, -1.0, 2.0, -0.3, 1.7]], index=[t], columns=list(raw.columns)
    )
    out = apply_mu_vol_scaling(z, raw, _cfg(enabled=True))
    assert np.allclose(out.to_numpy(), z.to_numpy(), atol=1e-9, equal_nan=True)


def test_early_dates_below_min_obs_are_identity():
    """On a dense panel, rows with a <63-row trailing window are pass-through
    (position 62 → 62-row window → identity)."""
    from src.backtest import apply_mu_vol_scaling

    raw = _raw_returns(130)
    z = _z_panel(raw.index, raw.columns)
    out = apply_mu_vol_scaling(z, raw, _cfg(enabled=True))

    early = z.index[62]                     # window = positions [0:62] = 62 rows
    assert np.allclose(out.loc[early].to_numpy(), z.loc[early].to_numpy(), atol=1e-9)


# ---------------------------------------------------------------------------
# 4d. Guard: NaN predictions preserved
# ---------------------------------------------------------------------------
def test_nan_prediction_preserved():
    from src.backtest import apply_mu_vol_scaling

    raw = _raw_returns(130, seed=9)
    z = _z_panel(raw.index, raw.columns, seed=10)
    t = z.index[-1]                         # fully-active late date
    z.loc[t, "T2"] = np.nan                 # one NaN prediction

    out = apply_mu_vol_scaling(z, raw, _cfg(enabled=True))
    assert bool(np.isnan(out.loc[t, "T2"]))                 # NaN preserved
    assert np.isfinite(out.loc[t, "T0"])                    # finite cells stay finite
    assert np.isfinite(out.loc[t, "T4"])


# ---------------------------------------------------------------------------
# 5. Config flag exists and defaults OFF
# ---------------------------------------------------------------------------
def test_config_flag_default_off():
    assert PipelineConfig().mu_vol_scaling_enabled is False


# ---------------------------------------------------------------------------
# 6. Lookback window semantics (uses cov_lookback field, last 126 only)
# ---------------------------------------------------------------------------
def test_only_trailing_lookback_used():
    """With default cov_lookback=126, only rows inside [k-126, k) affect the
    result at position k: an outside row is inert, an inside row is not."""
    from src.backtest import apply_mu_vol_scaling

    raw = _raw_returns(200, seed=13)
    t = raw.index[199]                       # window = [73:199]
    z = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0, 5.0]], index=[t], columns=list(raw.columns)
    )
    cfg = _cfg(enabled=True)                  # cov_lookback defaults to 126

    base = apply_mu_vol_scaling(z, raw, cfg)

    # (i) perturb a row OUTSIDE the window (position 50 < 73) → no change.
    raw_out = raw.copy()
    raw_out.iloc[50] = 7.0
    out_outside = apply_mu_vol_scaling(z, raw_out, cfg)
    assert np.allclose(base.to_numpy(), out_outside.to_numpy(), atol=1e-12)

    # (ii) perturb a row INSIDE the window (position 150 ∈ [73,199)) → changes.
    raw_in = raw.copy()
    raw_in.iloc[150] = 7.0
    out_inside = apply_mu_vol_scaling(z, raw_in, cfg)
    assert not np.allclose(base.to_numpy(), out_inside.to_numpy(), atol=1e-6)


def test_cov_lookback_field_is_honored():
    """Shrinking config.cov_lookback shrinks the window: a row that is inside
    the 126-window but outside a 70-window is inert only under cov_lookback=70,
    proving the transform reads the field rather than a hardcoded 126."""
    from src.backtest import apply_mu_vol_scaling

    raw = _raw_returns(200, seed=14)
    t = raw.index[199]
    z = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0, 5.0]], index=[t], columns=list(raw.columns)
    )

    # Perturb position 100: inside 126-window [73:199], outside 70-window [129:199].
    raw_pert = raw.copy()
    raw_pert.iloc[100] = 7.0

    cfg70 = _cfg(enabled=True, cov_lookback=70)
    base70 = apply_mu_vol_scaling(z, raw, cfg70)
    pert70 = apply_mu_vol_scaling(z, raw_pert, cfg70)
    assert np.allclose(base70.to_numpy(), pert70.to_numpy(), atol=1e-12)  # inert

    cfg126 = _cfg(enabled=True, cov_lookback=126)
    base126 = apply_mu_vol_scaling(z, raw, cfg126)
    pert126 = apply_mu_vol_scaling(z, raw_pert, cfg126)
    assert not np.allclose(base126.to_numpy(), pert126.to_numpy(), atol=1e-6)  # active
