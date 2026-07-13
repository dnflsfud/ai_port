"""Unit smoke — build_feature_attribution (satisfies the filename TDD guard).

This file exists so the guard that requires ``tests/test_<module>.py`` for edits
to ``scripts/export_operating_data.py`` is satisfied. It is a MINIMAL smoke of
``build_feature_attribution`` (schema + SHAP additivity) — the exhaustive
coverage lives in ``tests/acceptance/test_feature_attribution_tab.py``.

Written by test-designer BEFORE implementation; copied verbatim by the
implementer (sha256-verified). Target imported INSIDE test bodies so
``pytest --collect-only`` stays clean; RED now (build_feature_attribution
does not exist yet -> ImportError).
"""

import types

import numpy as np
import pandas as pd


_TICKERS = ["AAA", "BBB", "CCC", "DDD"]
_ACTIVE = [f"f{i}" for i in range(6)]           # model subset
_PANEL_COLS = _ACTIVE + ["f6"]                  # panel superset (f6 extra)
_AS_OF = pd.Timestamp("2026-05-21")


def _result():
    import lightgbm as lgb

    rng = np.random.default_rng(0)
    n = 300
    Xtr = pd.DataFrame(rng.normal(size=(n, len(_ACTIVE))), columns=_ACTIVE)
    ytr = 0.4 * Xtr["f0"] - 0.25 * Xtr["f2"] + rng.normal(0, 0.1, n)
    model = lgb.LGBMRegressor(n_estimators=30, num_leaves=8,
                              min_child_samples=5, verbose=-1)
    model.fit(Xtr, ytr)
    model._active_features = list(_ACTIVE)

    idx = pd.MultiIndex.from_product([[_AS_OF], _TICKERS], names=["date", "ticker"])
    panel = pd.DataFrame(rng.normal(size=(len(idx), len(_PANEL_COLS))),
                         index=idx, columns=_PANEL_COLS)
    weights = pd.Series(1.0 / len(_TICKERS), index=_TICKERS)

    return types.SimpleNamespace(
        models={_AS_OF: model},
        panel=panel,
        portfolio_weights={_AS_OF: weights},
        feature_groups={"G1": ["f0", "f1", "f2"], "G2": ["f3", "f4", "f5"]},
        sector_map={t: "S" for t in _TICKERS},
        bm_weights={t: 1.0 / len(_TICKERS) for t in _TICKERS},
    )


def test_smoke_schema():
    from scripts.export_operating_data import build_feature_attribution

    out = build_feature_attribution(_result())
    assert {"as_of", "model_date", "feature_groups", "tickers"} <= set(out)
    assert set(out["tickers"]) == set(_TICKERS)
    for rec in out["tickers"].values():
        assert set(rec["shap"]) == set(_ACTIVE)      # sliced to model features


def test_smoke_additivity():
    from scripts.export_operating_data import build_feature_attribution

    out = build_feature_attribution(_result())
    for rec in out["tickers"].values():
        recon = float(rec["base_value"]) + float(sum(rec["shap"].values()))
        assert abs(recon - float(rec["mu"])) <= 1e-3 * abs(float(rec["mu"])) + 1e-9
    assert out.get("additivity_ok") is True


def test_rebalance_metadata_for_non_rebalance_and_rebalance_asof():
    from scripts.export_operating_data import build_rebalance_metadata

    dates = pd.bdate_range("2026-01-02", periods=25)
    non_rebalance = build_rebalance_metadata(
        [dates[0], dates[21]],
        dates,
        21,
    )
    assert non_rebalance["last_rebalance_date"] == dates[21].strftime("%Y-%m-%d")
    assert non_rebalance["rows_since_last_rebalance"] == 3
    assert non_rebalance["rows_until_next_rebalance"] == 18
    assert non_rebalance["is_rebalance_data_as_of"] is False
    assert pd.Timestamp(non_rebalance["next_expected_rebalance_date"]).dayofweek < 5

    rebalance_asof = build_rebalance_metadata(
        [dates[0], dates[21]],
        dates[:22],
        21,
    )
    assert rebalance_asof["is_rebalance_data_as_of"] is True
    assert rebalance_asof["rows_since_last_rebalance"] == 0
    assert rebalance_asof["rows_until_next_rebalance"] == 21


def test_rebalance_metadata_rejects_weekend_calendar():
    from scripts.export_operating_data import build_rebalance_metadata

    dates = pd.to_datetime(["2026-01-02", "2026-01-03"])
    with np.testing.assert_raises_regex(ValueError, "weekend"):
        build_rebalance_metadata([dates[0]], dates, 21)
