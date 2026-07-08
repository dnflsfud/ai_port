"""Acceptance tests — "Stock Drivers" tab: per-stock feature SHAP attribution.

Written from the spec (spec-feature-attribution-tab.md, §4) BEFORE any
implementation exists. Every test here MUST be RED now: the target symbols
``scripts.export_operating_data.build_feature_attribution`` and
``streamlit_app.prepare_stock_drivers`` are not written yet, so the tests fail
with ImportError / AttributeError (NOT collection errors — targets are imported
INSIDE each test body so ``pytest --collect-only`` succeeds cleanly).

======================================================================
PINNED IMPLEMENTATION CONTRACT (this is the interface the implementer MUST
provide; the spec pins the SIGNATURES and the §2 OUTPUT schema, these tests
pin the concrete shapes):

  scripts.export_operating_data.build_feature_attribution(result) -> dict
    `result` is duck-typed and carries (single argument — file I/O / pkl load
    happen OUTSIDE this pure function):
      result.models            : {pd.Timestamp -> LGBMRegressor}, each model
                                 exposes `._active_features` (list[str]) — the
                                 subset of panel columns it was fit on.
      result.panel             : DataFrame, MultiIndex[date, ticker] x features
                                 (a SUPERSET of any single model's features).
      result.portfolio_weights : {pd.Timestamp -> pd.Series(ticker -> weight)}.
      result.feature_groups    : {group_name -> [feature, ...]}  (the native
                                 orientation used across src/*; the OUTPUT
                                 inverts this to feature -> group per §2).
      result.sector_map        : {ticker -> sector}   (holdings source field).
      result.bm_weights        : {ticker -> bm weight at as_of}  (holdings
                                 source field == bm_fn(as_of, tickers, n)).
    Returns the §2 dict:
      {"as_of","model_date","feature_groups"(feat->group),"additivity_ok",
       "tickers": {tkr: {"weight","bm_weight","active","sector","mu",
                         "base_value","shap": {feat: float}}}}
    where as_of == max(portfolio_weights), model_date == max model date <= as_of,
    X = panel.loc[date==as_of, model._active_features], SHAP via
    src.attribution.compute_shap_values.

  streamlit_app.prepare_stock_drivers(attr: dict | None, ticker: str | None)
      -> dict | None
    None / empty attr -> None; else
      {"options": [(ticker, active), ... active DESC],
       "default": <max-active (OW) ticker>,
       "top_features": [(feat, group, shap), ... |shap| DESC, at most 12],
       "metrics": {"weight","bm_weight","active","mu"}}   # for selected/default
    `ticker` selects that stock's metrics/top_features; None -> default (max OW).

INDEPENDENCE: expected values are computed here with plain numpy / python
`sorted` (NOT by importing the implementation's helpers). The SHAP additivity
gate (base_value + sum(shap) ~= mu) is checked on the OUTPUT only, and `mu` is
cross-checked against an independent `model.predict` call in the test.
Idioms (project convention): plain pytest functions, no fixtures, synthetic
data only, fast (small LGBM ~ a couple seconds, no 66MB pkl).

======================================================================
합격기준(spec §4) ↔ 테스트 매핑표
----------------------------------------------------------------------
cov-1  build: 스키마 키(as_of/model_date/feature_groups/tickers) 존재
         -> test_build_schema_keys_present
cov-1  build: 전 종목(패널 as_of) 포함
         -> test_build_includes_all_panel_tickers
cov-1  build: shap 키 == model._active_features (패널 초과열 f7 제외)
         -> test_build_shap_keys_equal_active_features
cov-1  build: additivity base+sum(shap)~=mu (rtol 1e-3, OUTPUT만으로 독립검증)
         -> test_build_additivity_holds
cov-1  build: mu == 독립 model.predict(X_row)
         -> test_build_mu_matches_independent_predict
cov-1  build: feature_groups 출력이 feat->group 역매핑
         -> test_build_feature_groups_inverted
(pre-mortem 모델일≠리밸일 off-by-one) build: model_date == max(model<=as_of), 미래모델 무시
         -> test_build_model_date_ignores_future_model
cov-2  build: 가중치에 없는 패널 종목 weight=0 (존재는 유지)
         -> test_build_missing_weight_is_zero
cov-2  build: active == weight - bm_weight 항등 + sector == sector_map
         -> test_build_active_identity_and_sector
cov-3  prepare: None -> None
         -> test_prepare_none_returns_none
cov-3  prepare: 빈/무종목 attr -> None (load_json {} 대비 경계)
         -> test_prepare_empty_returns_none
cov-3  prepare: options active desc 정렬 + default = 최대 OW
         -> test_prepare_options_sorted_default_max_ow
cov-3  prepare: top_features |shap| desc + 12개 상한(사전약정 N=12)
         -> test_prepare_top_features_capped_and_sorted
cov-4  prepare: ticker 지정 -> 해당 종목 metrics/top_features
         -> test_prepare_ticker_selects_that_stock
cov-5  회귀(부분): streamlit_app import 부작용 없음 + 신규 심볼/ main 존재
       (전체 회귀 test_streamlit_app.py / test_streamlit_report.py 재실행은 verifier)
         -> test_streamlit_import_clean_and_symbols_present
======================================================================
"""

import types

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Synthetic-data helpers (NOT the implementation).
# ---------------------------------------------------------------------------
TICKERS = ["NVDA", "AVGO", "MSFT", "AAPL", "TSLA", "XOM"]
PANEL_FEATURES = [f"f{i}" for i in range(8)]           # f0..f7 (panel superset)
ACTIVE_FEATURES = [f"f{i}" for i in range(7)]          # f0..f6 (model subset; f7 excluded)
AS_OF = pd.Timestamp("2026-05-21")
EARLIER_REB = pd.Timestamp("2026-04-23")
FUTURE_MODEL = pd.Timestamp("2026-06-10")              # > as_of, must be ignored
FEATURE_GROUPS = {                                     # group -> [features] (native)
    "Alpha": ["f0", "f1", "f2"],
    "Beta": ["f3", "f4"],
    "Gamma": ["f5", "f6"],
    "Extra": ["f7"],                                   # maps a non-active feature
}
SECTOR_MAP = {
    "NVDA": "Semiconductors", "AVGO": "Semiconductors", "MSFT": "Software",
    "AAPL": "Hardware", "TSLA": "Autos", "XOM": "Energy",
}


def _train_model(seed=0):
    """A small, genuinely-fit LGBMRegressor on ACTIVE_FEATURES (7 cols).

    lightgbm is imported here (not at module top) so ``--collect-only`` stays
    clean regardless of the environment.
    """
    import lightgbm as lgb

    rng = np.random.default_rng(seed)
    n = 400
    X = pd.DataFrame(rng.normal(size=(n, len(ACTIVE_FEATURES))), columns=ACTIVE_FEATURES)
    y = (0.5 * X["f0"] - 0.3 * X["f3"] + 0.2 * X["f1"] * X["f2"]
         + rng.normal(0, 0.1, n))
    model = lgb.LGBMRegressor(n_estimators=40, num_leaves=8,
                              min_child_samples=5, verbose=-1)
    model.fit(X, y)
    model._active_features = list(ACTIVE_FEATURES)
    return model


def _panel(seed=1):
    """MultiIndex[date, ticker] x PANEL_FEATURES for {EARLIER_REB, AS_OF}."""
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([[EARLIER_REB, AS_OF], TICKERS],
                                     names=["date", "ticker"])
    data = rng.normal(size=(len(idx), len(PANEL_FEATURES)))
    return pd.DataFrame(data, index=idx, columns=PANEL_FEATURES)


def _make_result(drop_weight_ticker="XOM", seed=1):
    """Duck-typed result carrying the pinned build_feature_attribution inputs."""
    model = _train_model()
    panel = _panel(seed)

    # portfolio_weights at AS_OF omits `drop_weight_ticker` -> weight 0 in output.
    held = [t for t in TICKERS if t != drop_weight_ticker]
    rng = np.random.default_rng(7)
    w_vals = 1.0 / len(TICKERS) + rng.normal(0, 0.01, len(held))
    weights_asof = pd.Series(w_vals, index=held)
    weights_earlier = pd.Series(1.0 / len(TICKERS), index=TICKERS)

    return types.SimpleNamespace(
        # The chosen model is AS_OF; earlier/future dates reuse the same fitted
        # object (only the chosen date's model is exercised — model_date is what
        # distinguishes them, so the future model MUST be excluded by date).
        models={EARLIER_REB: model, AS_OF: model, FUTURE_MODEL: model},
        panel=panel,
        portfolio_weights={EARLIER_REB: weights_earlier, AS_OF: weights_asof},
        feature_groups={k: list(v) for k, v in FEATURE_GROUPS.items()},
        sector_map=dict(SECTOR_MAP),
        bm_weights={t: 1.0 / len(TICKERS) for t in TICKERS},
    )


# ===========================================================================
# build_feature_attribution  (export side)
# ===========================================================================
def test_build_schema_keys_present():
    from scripts.export_operating_data import build_feature_attribution

    out = build_feature_attribution(_make_result())
    for key in ("as_of", "model_date", "feature_groups", "tickers"):
        assert key in out, f"missing top-level key {key!r}"
    assert out["as_of"] == "2026-05-21"            # max(portfolio_weights)
    assert out["model_date"] == "2026-05-21"       # max model date <= as_of


def test_build_includes_all_panel_tickers():
    from scripts.export_operating_data import build_feature_attribution

    out = build_feature_attribution(_make_result())
    assert set(out["tickers"].keys()) == set(TICKERS)  # every panel@as_of ticker


def test_build_shap_keys_equal_active_features():
    """SHAP is keyed by the MODEL's _active_features (7), NOT the 8-col panel;
    the extra panel column f7 must be sliced out (pre-mortem §5)."""
    from scripts.export_operating_data import build_feature_attribution

    out = build_feature_attribution(_make_result())
    for tkr, rec in out["tickers"].items():
        assert set(rec["shap"].keys()) == set(ACTIVE_FEATURES), tkr
        assert "f7" not in rec["shap"], tkr


def test_build_additivity_holds():
    """SHAP local-accuracy: base_value + sum(shap) ~= mu for every stock.
    Checked on the OUTPUT ONLY (rtol 1e-3), independent of how it was computed.
    Also asserts additivity_ok is flagged True for clean data (§2 gate)."""
    from scripts.export_operating_data import build_feature_attribution

    out = build_feature_attribution(_make_result())
    for tkr, rec in out["tickers"].items():
        recon = float(rec["base_value"]) + float(sum(rec["shap"].values()))
        mu = float(rec["mu"])
        assert abs(recon - mu) <= 1e-3 * abs(mu) + 1e-9, (
            f"{tkr}: base+sum(shap)={recon} vs mu={mu}"
        )
    assert out.get("additivity_ok") is True


def test_build_mu_matches_independent_predict():
    """`mu` equals an INDEPENDENT model.predict on the as_of feature row."""
    from scripts.export_operating_data import build_feature_attribution

    result = _make_result()
    out = build_feature_attribution(result)

    model = result.models[AS_OF]
    panel = result.panel
    for tkr, rec in out["tickers"].items():
        x_row = panel.loc[(AS_OF, tkr), ACTIVE_FEATURES].to_numpy(dtype=float)
        expected_mu = float(model.predict(x_row.reshape(1, -1))[0])
        assert np.isclose(float(rec["mu"]), expected_mu, rtol=1e-3, atol=1e-9), tkr


def test_build_feature_groups_inverted():
    """Output feature_groups is feat->group (inverse of result.feature_groups)."""
    from scripts.export_operating_data import build_feature_attribution

    out = build_feature_attribution(_make_result())
    fg = out["feature_groups"]
    assert fg["f0"] == "Alpha"
    assert fg["f2"] == "Alpha"
    assert fg["f3"] == "Beta"
    assert fg["f5"] == "Gamma"


def test_build_model_date_ignores_future_model():
    """A retrain dated AFTER as_of must not be selected (max model <= as_of)."""
    from scripts.export_operating_data import build_feature_attribution

    out = build_feature_attribution(_make_result())
    assert out["model_date"] == "2026-05-21"      # NOT 2026-06-10 (future)


def test_build_missing_weight_is_zero():
    """A ticker present in the panel@as_of but absent from portfolio_weights is
    still emitted, with weight == 0 (spec §2)."""
    from scripts.export_operating_data import build_feature_attribution

    out = build_feature_attribution(_make_result(drop_weight_ticker="XOM"))
    assert "XOM" in out["tickers"]
    assert float(out["tickers"]["XOM"]["weight"]) == 0.0


def test_build_active_identity_and_sector():
    """active == weight - bm_weight for every stock; sector from sector_map."""
    from scripts.export_operating_data import build_feature_attribution

    out = build_feature_attribution(_make_result())
    for tkr, rec in out["tickers"].items():
        w = float(rec["weight"]); bm = float(rec["bm_weight"]); a = float(rec["active"])
        assert np.isclose(a, w - bm, atol=1e-6), tkr
        assert rec["sector"] == SECTOR_MAP[tkr], tkr


# ===========================================================================
# prepare_stock_drivers  (app side)
# ===========================================================================
def _attr(n_features=15):
    """A §2-shaped attribution dict for the app-side pure helper.

    5 stocks with DISTINCT active values (mix of OW>0 and UW<0). One stock
    ("BIG") carries `n_features` shap entries with strictly increasing |shap|
    so the |shap|-desc top-12 cap is exercised with a hand-checkable order.
    """
    feats = [f"f{i}" for i in range(n_features)]
    feature_groups = {f: ("Alpha" if int(f[1:]) % 2 == 0 else "Beta") for f in feats}
    # |shap_i| = (i+1)*1e-3, sign alternates -> top-12 by |shap| = f14..f3.
    big_shap = {f"f{i}": ((-1) ** i) * (i + 1) * 1e-3 for i in range(n_features)}

    def rec(active, weight, mu, shap):
        return {"weight": weight, "bm_weight": weight - active, "active": active,
                "sector": "X", "mu": mu, "base_value": 0.0, "shap": shap}

    tickers = {
        "BIG": rec(0.030, 0.090, 0.02, big_shap),          # max OW -> default
        "MID": rec(0.010, 0.070, 0.01, {"f0": 0.005, "f1": -0.002}),
        "FLAT": rec(0.000, 0.060, 0.00, {"f0": 0.001}),
        "UW1": rec(-0.008, 0.052, -0.01, {"f0": -0.004, "f1": 0.001}),
        "UW2": rec(-0.020, 0.040, -0.02, {"f0": -0.006}),
    }
    return {"as_of": "2026-05-21", "model_date": "2026-05-21",
            "feature_groups": feature_groups, "additivity_ok": True,
            "tickers": tickers}


def test_prepare_none_returns_none():
    from streamlit_app import prepare_stock_drivers

    assert prepare_stock_drivers(None, None) is None


def test_prepare_empty_returns_none():
    """load_json returns {} for a missing file -> must map to None (app shows
    st.info and the whole app keeps working, spec §3)."""
    from streamlit_app import prepare_stock_drivers

    assert prepare_stock_drivers({}, None) is None
    assert prepare_stock_drivers({"tickers": {}}, None) is None


def test_prepare_options_sorted_default_max_ow():
    from streamlit_app import prepare_stock_drivers

    view = prepare_stock_drivers(_attr(), None)
    assert view is not None

    order = [t for t, _a in view["options"]]
    expected_order = ["BIG", "MID", "FLAT", "UW1", "UW2"]   # active DESC
    assert order == expected_order
    # option payload carries the active value alongside the ticker.
    assert dict(view["options"])["BIG"] == 0.030
    assert view["default"] == "BIG"                          # max OW


def test_prepare_top_features_capped_and_sorted():
    from streamlit_app import prepare_stock_drivers

    attr = _attr(n_features=15)
    view = prepare_stock_drivers(attr, None)                 # default = BIG

    top = view["top_features"]
    assert len(top) == 12                                    # pre-registered cap

    # Independently derive the expected top-12 (|shap| desc) with plain sorted.
    big_shap = attr["tickers"]["BIG"]["shap"]
    exp = sorted(big_shap.items(), key=lambda kv: abs(kv[1]), reverse=True)[:12]
    assert [f for f, _g, _v in top] == [f for f, _v in exp]
    # |shap| is non-increasing down the list.
    mags = [abs(v) for _f, _g, v in top]
    assert all(mags[i] >= mags[i + 1] for i in range(len(mags) - 1))
    # group label comes from attr["feature_groups"].
    top_feat, top_group, _top_val = top[0]
    assert top_group == attr["feature_groups"][top_feat]


def test_prepare_ticker_selects_that_stock():
    from streamlit_app import prepare_stock_drivers

    attr = _attr()
    view = prepare_stock_drivers(attr, "UW2")
    assert view is not None

    m = view["metrics"]
    src = attr["tickers"]["UW2"]
    assert np.isclose(m["weight"], src["weight"], atol=1e-12)
    assert np.isclose(m["bm_weight"], src["bm_weight"], atol=1e-12)
    assert np.isclose(m["active"], src["active"], atol=1e-12)
    assert np.isclose(m["mu"], src["mu"], atol=1e-12)
    # top_features reflect the SELECTED stock (UW2 has a single shap entry).
    assert [f for f, _g, _v in view["top_features"]] == ["f0"]


# ===========================================================================
# Regression (partial): import cleanliness + new symbol presence.
# Full re-run of tests/test_streamlit_app.py + test_streamlit_report.py is the
# verifier's job (spec §4 coverage 5).
# ===========================================================================
def test_streamlit_import_clean_and_symbols_present():
    import streamlit_app  # must not execute any Streamlit UI at import time

    assert hasattr(streamlit_app, "prepare_stock_drivers")
    assert callable(streamlit_app.prepare_stock_drivers)
    assert hasattr(streamlit_app, "main") and callable(streamlit_app.main)
