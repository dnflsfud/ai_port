"""Acceptance tests — operating-dashboard real-ops additions (drift / cost / sector band).

Written by test-designer BEFORE implementation, from the FROZEN spec
``spec-dashboard-ops-additions.md`` only (implementation not read). RED now:
targets do not exist yet, so every test fails with ImportError / AttributeError /
AssertionError / TypeError (NOT a collection error). Targets are imported INSIDE
test bodies so ``pytest --collect-only`` stays clean.

Expected numbers are recomputed INDEPENDENTLY inside each test (explicit numpy
loop / plain arithmetic) and cross-checked against values hardcoded from a
separate Python session — the implementation's own math is never imported or
reused. Numeric gate: exact math at ``abs=1e-12`` (drift) / ``1e-10`` (cost);
the 0.5% default tolerance is not needed because these are exact identities.

Acceptance-criterion -> test mapping
------------------------------------
spec ref                                             | test
-----------------------------------------------------|------------------------------------------------------------
§2.1 build_current_drift schema + sequential drift    | test_build_current_drift_matches_independent_drift_math
§2.1 by_ticker |drift|-desc + weight_sum≈1 + l1 tie   | test_build_current_drift_matches_independent_drift_math
§2.1 / §3-1 empty post_rebalance_returns -> flat/0     | test_build_current_drift_empty_returns_is_flat
§2.1 no_trade_band strict `>` boundary + threshold cnt | test_build_current_drift_names_outside_band_strict_boundary
§2.1 guard: degenerate target sum -> ValueError        | test_build_current_drift_rejects_degenerate_target
§2.2 build_transaction_cost_history cumulative identity | test_build_transaction_cost_history_matches_independent_cost_math
§2.2 series monotone + last==cumulative + drag formula  | test_build_transaction_cost_history_matches_independent_cost_math
§2.2 / §3-4 empty turnover -> 0 / [] / no crash        | test_build_transaction_cost_history_empty_turnover_no_crash
§2.2 annualized drag max(n_return_days,1) guard         | test_build_transaction_cost_history_zero_return_days_guard
§2.4 / §4-6 collect_operating_alerts backward compat    | test_collect_operating_alerts_backward_compat_and_sector_binding
§2.4 / §2.3 sector_active binding -> new alert (only)   | test_collect_operating_alerts_backward_compat_and_sector_binding
§4-4 regenerated bundles: keys + weight_sum + cost gate | test_regenerated_bundle_has_ops_additions[<bundle>]
§2.3 / §2.5 sector_active + sector_deviation_limit data | test_regenerated_bundle_has_ops_additions[<bundle>]

Judged NOT-translatable in THIS acceptance file ("판정 불가"), returned to lead:
  * §4-1 full-suite count (224 + new all PASS): suite-level, verifier runs `pytest tests/ -q`.
  * §4-2 bundle regeneration commands: an action the implementer runs, not an assertion;
         its RESULT is checked by test_regenerated_bundle_has_ops_additions.
  * §4-3 validate_portfolio_bundles.py exit 0: verifier subprocess step; its numeric
         gates are mirrored here as pure-data JSON checks.
  * §4-5 IR/TE/realized_beta regression guard: needs a pre-change snapshot captured at
         implement time; no independent baseline exists at test-authoring time.
"""

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# Repo root = ai_port (this file is ai_port/tests/acceptance/<file>.py).
_AI_PORT = Path(__file__).resolve().parents[2]
_BUNDLE_DIRS = ["outputs/operating", "outputs/operating_codex_causal_rank_65"]


# --------------------------------------------------------------------------- #
# Independent reference helpers (NOT the implementation).                      #
# --------------------------------------------------------------------------- #
def _reference_current_drift(target, returns):
    """Sequential w <- w*(1+r)/Σ(w*(1+r)), plain numpy, independent of src."""
    w = np.asarray(target, dtype=float).copy()
    for _, row in returns.iterrows():
        w = w * (1.0 + np.asarray(row, dtype=float))
        w = w / w.sum()
    return w


def _drift_fixture():
    tickers = ["AAA", "BBB", "CCC"]
    target = pd.Series([0.5, 0.3, 0.2], index=tickers)
    returns = pd.DataFrame(
        [[0.10, -0.05, 0.00],
         [0.02, 0.03, -0.01]],
        index=pd.bdate_range("2026-06-02", periods=2),
        columns=tickers,
    )
    return tickers, target, returns


# --------------------------------------------------------------------------- #
# §2.1  build_current_drift                                                    #
# --------------------------------------------------------------------------- #
def test_build_current_drift_matches_independent_drift_math():
    from scripts.export_operating_data import build_current_drift

    tickers, target, returns = _drift_fixture()
    out = build_current_drift(
        target_weights=target,
        post_rebalance_returns=returns,
        no_trade_band=0.02,
        as_of="2026-06-04",
        last_rebalance_date="2026-06-01",
    )

    # Independent reference + values hardcoded from a separate session.
    ref_current = _reference_current_drift(target.values, returns)
    ref_drift = ref_current - target.values
    assert ref_drift[0] == pytest.approx(0.032991306826279154, abs=1e-12)
    assert float(np.abs(ref_drift).sum()) == pytest.approx(0.0659826136525582, abs=1e-12)

    # Scalar schema.
    assert out["as_of"] == "2026-06-04"
    assert out["last_rebalance_date"] == "2026-06-01"
    assert out["days_since_rebalance"] == 2
    assert out["no_trade_band"] == pytest.approx(0.02)
    assert out["drift_l1"] == pytest.approx(float(np.abs(ref_drift).sum()), abs=1e-12)
    assert out["max_single_drift"] == pytest.approx(float(np.abs(ref_drift).max()), abs=1e-12)
    assert out["weight_sum"] == pytest.approx(1.0, abs=1e-9)
    assert out["weight_sum"] == pytest.approx(float(ref_current.sum()), abs=1e-12)
    # band 0.02: |drift| = [.0330, .0211, .0119] -> AAA,BBB outside.
    assert out["names_outside_band"] == 2

    # by_ticker: full list, |drift| descending, exact per-ticker reconciliation.
    rows = out["by_ticker"]
    assert len(rows) == len(tickers)
    for rec in rows:
        assert set(rec) == {"ticker", "target", "current", "drift"}
    mags = [abs(rec["drift"]) for rec in rows]
    assert mags == sorted(mags, reverse=True)
    assert [rec["ticker"] for rec in rows] == ["AAA", "BBB", "CCC"]
    by_t = {rec["ticker"]: rec for rec in rows}
    for i, t in enumerate(tickers):
        assert by_t[t]["target"] == pytest.approx(target.values[i], abs=1e-12)
        assert by_t[t]["current"] == pytest.approx(ref_current[i], abs=1e-12)
        assert by_t[t]["drift"] == pytest.approx(ref_drift[i], abs=1e-12)
        assert by_t[t]["drift"] == pytest.approx(
            by_t[t]["current"] - by_t[t]["target"], abs=1e-12
        )
    # drift_l1 == Σ|by_ticker.drift|  (validator identity, §2.5).
    assert out["drift_l1"] == pytest.approx(
        sum(abs(rec["drift"]) for rec in rows), abs=1e-12
    )


def test_build_current_drift_empty_returns_is_flat():
    from scripts.export_operating_data import build_current_drift

    tickers, target, _ = _drift_fixture()
    empty = pd.DataFrame(columns=tickers)
    out = build_current_drift(
        target_weights=target,
        post_rebalance_returns=empty,
        no_trade_band=0.02,
        as_of="2026-06-01",
        last_rebalance_date="2026-06-01",
    )
    assert out["days_since_rebalance"] == 0
    assert out["drift_l1"] == pytest.approx(0.0, abs=1e-12)
    assert out["max_single_drift"] == pytest.approx(0.0, abs=1e-12)
    assert out["names_outside_band"] == 0
    assert out["weight_sum"] == pytest.approx(1.0, abs=1e-9)
    for rec in out["by_ticker"]:
        assert rec["drift"] == pytest.approx(0.0, abs=1e-12)
        assert rec["current"] == pytest.approx(rec["target"], abs=1e-12)


def test_build_current_drift_names_outside_band_strict_boundary():
    from scripts.export_operating_data import build_current_drift

    _, target, returns = _drift_fixture()

    def _count(band):
        return build_current_drift(
            target_weights=target,
            post_rebalance_returns=returns,
            no_trade_band=band,
            as_of="2026-06-04",
            last_rebalance_date="2026-06-01",
        )["names_outside_band"]

    # Threshold counts against |drift| = [.0330, .0211, .0119].
    assert _count(0.015) == 2
    assert _count(0.025) == 1

    # Strict `>` boundary: band == the reported max must EXCLUDE that name.
    probe = build_current_drift(
        target_weights=target,
        post_rebalance_returns=returns,
        no_trade_band=0.5,
        as_of="2026-06-04",
        last_rebalance_date="2026-06-01",
    )
    m = probe["max_single_drift"]
    assert _count(m) == 0                # equal -> not counted (strict >)
    assert _count(m * (1.0 - 1e-6)) >= 1  # just below the max -> counted


def test_build_current_drift_rejects_degenerate_target():
    from scripts.export_operating_data import build_current_drift

    tickers, _, returns = _drift_fixture()
    zero_target = pd.Series([0.0, 0.0, 0.0], index=tickers)
    with pytest.raises(ValueError):
        build_current_drift(
            target_weights=zero_target,
            post_rebalance_returns=returns,
            no_trade_band=0.02,
            as_of="2026-06-04",
            last_rebalance_date="2026-06-01",
        )


# --------------------------------------------------------------------------- #
# §2.2  build_transaction_cost_history                                         #
# --------------------------------------------------------------------------- #
def _cost_fixture():
    dates = pd.bdate_range("2026-01-05", periods=3)
    turnover = pd.Series([0.20, 0.15, 0.10], index=dates)
    return dates, turnover


def test_build_transaction_cost_history_matches_independent_cost_math():
    from scripts.export_operating_data import build_transaction_cost_history

    dates, turnover = _cost_fixture()
    rate = 0.001
    n_return_days = 500
    out = build_transaction_cost_history(
        turnover, rate, n_return_days=n_return_days
    )

    # Independent reference + hardcoded cross-check.
    ref_cost = turnover.values * rate
    ref_cum = float(ref_cost.sum())
    ref_cum_turn = float(turnover.values.sum())
    ref_series_cum = np.cumsum(ref_cost)
    ref_drag = ref_cum * 252.0 / max(n_return_days, 1)
    assert ref_cum == pytest.approx(0.00045, abs=1e-12)
    assert ref_drag == pytest.approx(0.0002268, abs=1e-12)

    assert out["one_way_transaction_cost_rate"] == pytest.approx(rate)
    assert out["one_way_transaction_cost_bps"] == pytest.approx(rate * 1e4)  # 10.0
    assert out["n_rebalances"] == 3
    assert out["cumulative_two_way_turnover"] == pytest.approx(ref_cum_turn, abs=1e-12)
    assert out["cumulative_transaction_cost"] == pytest.approx(ref_cum, abs=1e-12)
    # Core validator identity (§2.5): cumulative == Σturnover × rate.
    assert out["cumulative_transaction_cost"] == pytest.approx(
        out["cumulative_two_way_turnover"] * out["one_way_transaction_cost_rate"],
        abs=1e-10,
    )
    assert out["annualized_cost_drag"] == pytest.approx(ref_drag, abs=1e-12)

    series = out["series"]
    assert len(series) == 3
    for rec in series:
        assert set(rec) == {"date", "turnover", "cost", "cumulative_cost"}
    cum_curve = [rec["cumulative_cost"] for rec in series]
    assert cum_curve == sorted(cum_curve)  # monotone non-decreasing
    assert all(np.diff(cum_curve) >= -1e-15)
    assert cum_curve[-1] == pytest.approx(out["cumulative_transaction_cost"], abs=1e-10)
    for i, rec in enumerate(series):
        assert rec["cost"] == pytest.approx(ref_cost[i], abs=1e-12)
        assert rec["cumulative_cost"] == pytest.approx(ref_series_cum[i], abs=1e-12)
        assert rec["turnover"] == pytest.approx(turnover.values[i], abs=1e-12)


def test_build_transaction_cost_history_empty_turnover_no_crash():
    from scripts.export_operating_data import build_transaction_cost_history

    empty = pd.Series([], dtype=float)
    out = build_transaction_cost_history(empty, 0.001, n_return_days=250)
    assert out["n_rebalances"] == 0
    assert out["cumulative_two_way_turnover"] == pytest.approx(0.0, abs=1e-12)
    assert out["cumulative_transaction_cost"] == pytest.approx(0.0, abs=1e-12)
    assert out["annualized_cost_drag"] == pytest.approx(0.0, abs=1e-12)
    assert out["series"] == []
    assert out["one_way_transaction_cost_rate"] == pytest.approx(0.001)


def test_build_transaction_cost_history_zero_return_days_guard():
    from scripts.export_operating_data import build_transaction_cost_history

    _, turnover = _cost_fixture()
    rate = 0.001
    # max(n_return_days, 1): with 0 -> divide by 1 -> drag == cumulative*252.
    out = build_transaction_cost_history(turnover, rate, n_return_days=0)
    expected = float((turnover.values * rate).sum()) * 252.0 / 1.0
    assert expected == pytest.approx(0.1134, abs=1e-12)
    assert out["annualized_cost_drag"] == pytest.approx(expected, abs=1e-12)


# --------------------------------------------------------------------------- #
# §2.4 / §4-6  collect_operating_alerts backward compat + sector binding       #
# --------------------------------------------------------------------------- #
def test_collect_operating_alerts_backward_compat_and_sector_binding():
    # streamlit_app is imported inside the body so collect-only stays clean and
    # a RED failure is scoped to this test rather than the whole module.
    from streamlit_app import collect_operating_alerts

    monitoring = {"guardrails": {"estimated_te_breached": True,
                                 "latest_estimated_te": 0.036}}

    # Backward compat: 1-arg, (monitoring, None) and 2-arg positional all work.
    assert isinstance(collect_operating_alerts({}), list)
    assert isinstance(collect_operating_alerts({}, None), list)  # §4-6 exact call
    two_arg = collect_operating_alerts(monitoring, {"coverage": {}})
    assert isinstance(two_arg, list)
    assert "estimated_te_breached" in {row["key"] for row in two_arg}

    # New 3rd param must exist (RED here until implemented -> AssertionError).
    params = inspect.signature(collect_operating_alerts).parameters
    assert "operations" in params
    assert params["operations"].default is None  # default None keeps 2-arg calls valid

    binding_ops = {"sector_active": [
        {"sector": "Tech", "port": 0.30, "bm": 0.25, "active": 0.05,
         "limit": 0.04, "binding": True},
        {"sector": "Health", "port": 0.10, "bm": 0.11, "active": -0.01,
         "limit": 0.04, "binding": False},
    ]}
    alerts = collect_operating_alerts(monitoring, None, operations=binding_ops)
    by_key = {row["key"]: row for row in alerts}
    assert "sector_deviation_binding" in by_key
    row = by_key["sector_deviation_binding"]
    assert row["label"] == "Sector deviation at limit"
    assert "Tech" in row["value"]
    assert "Health" not in row["value"]

    # No binding row -> alert absent (and no crash).
    no_binding = {"sector_active": [
        {"sector": "Tech", "port": 0.26, "bm": 0.25, "active": 0.01,
         "limit": 0.04, "binding": False},
    ]}
    keys = {row["key"] for row in collect_operating_alerts(monitoring, None,
                                                           operations=no_binding)}
    assert "sector_deviation_binding" not in keys

    # operations=None -> no crash, no sector alert.
    keys_none = {row["key"] for row in collect_operating_alerts(monitoring, None,
                                                                operations=None)}
    assert "sector_deviation_binding" not in keys_none


# --------------------------------------------------------------------------- #
# §4-4  Integration smoke — regenerated bundles carry the new keys + gates.    #
# RED while bundles are old-schema; PASS only after implement + regenerate.    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bundle", _BUNDLE_DIRS)
def test_regenerated_bundle_has_ops_additions(bundle):
    ops_path = _AI_PORT / bundle / "operations.json"
    mon_path = _AI_PORT / bundle / "monitoring.json"
    assert ops_path.exists(), f"missing {ops_path}"
    assert mon_path.exists(), f"missing {mon_path}"
    ops = json.loads(ops_path.read_text(encoding="utf-8"))
    mon = json.loads(mon_path.read_text(encoding="utf-8"))

    # ---- operations.json: current_drift ----
    assert "current_drift" in ops
    drift = ops["current_drift"]
    assert 0.999999 <= drift["weight_sum"] <= 1.000001
    assert drift["days_since_rebalance"] >= 0
    assert abs(drift["drift_l1"]
               - sum(abs(r["drift"]) for r in drift["by_ticker"])) <= 1e-8

    # ---- operations.json: sector_active + sector_deviation_limit ----
    assert "sector_deviation_limit" in ops
    assert "sector_active" in ops
    limit = float(ops["sector_deviation_limit"])
    for r in ops["sector_active"]:
        assert r["limit"] == pytest.approx(limit)
        assert r["binding"] == bool(abs(r["active"]) >= limit - 1e-9)

    # ---- monitoring.json: transaction_costs ----
    assert "transaction_costs" in mon
    tc = mon["transaction_costs"]
    assert tc["cumulative_transaction_cost"] == pytest.approx(
        tc["cumulative_two_way_turnover"] * tc["one_way_transaction_cost_rate"],
        abs=1e-10,
    )
    if tc["series"]:
        assert tc["series"][-1]["cumulative_cost"] == pytest.approx(
            tc["cumulative_transaction_cost"], abs=1e-10
        )
