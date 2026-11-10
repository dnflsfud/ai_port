# -*- coding: utf-8 -*-
"""§S27.2 paper ledger (decision log §S27 ① — no real trading, follow the rebalance-date target in full):
append-only paper book whose w_prev is the drifted PAPER book, trading each new rebalance target at the data
date of its first publication; re-publications never trade (jitter is logged)."""

import json

import numpy as np
import pandas as pd
import pytest

from scripts.paper_ledger import ONE_WAY_COST_BPS, advance_book, apply_trade, run, update

DATES = pd.bdate_range("2026-09-01", periods=30)
T = ["A", "B"]


def _returns(a=0.01, b=-0.01):
    return pd.DataFrame({"A": a, "B": b}, index=DATES)


def _bm(r=0.0):
    return pd.Series(r, index=DATES)


def test_advance_book_drifts_and_compounds():
    w, nav, bm_nav, n = advance_book(pd.Series([0.5, 0.5], index=T), 1.0, 1.0, _returns(), _bm(0.001),
                                     DATES[0], DATES[2])
    assert n == 2
    # day 1: 50/50 x (+1%, -1%) = 0; day 2 on the drifted 0.505/0.495 book = +0.01%
    assert nav == pytest.approx(1.0001, rel=0, abs=1e-12)
    assert w["A"] > 0.5 and w.sum() == pytest.approx(1.0)
    assert bm_nav == pytest.approx(1.001 ** 2)


def test_apply_trade_charges_two_way_l1_turnover_times_one_way_bps():
    # §S28 H1: backtest convention (src/backtest.py tc_cost = sum|dw| * one_way_tc) — L1 0.2, one-way 0.1
    w, nav, one_way = apply_trade(pd.Series([0.6, 0.4], index=T), pd.Series({"A": 0.5, "B": 0.5}), 2.0)
    assert one_way == pytest.approx(0.1) and nav == pytest.approx(2.0 * (1 - 0.2 * ONE_WAY_COST_BPS / 1e4))
    assert w.to_dict() == {"A": 0.5, "B": 0.5}


def test_apply_trade_trades_over_the_union_of_book_and_target():
    # §S28 H1: a name in the target but not in the book used to be dropped (the book could never add a name)
    w, nav, one_way = apply_trade(pd.Series([0.6, 0.4], index=T), pd.Series({"A": 0.5, "C": 0.5}), 1.0)
    assert w.to_dict() == {"A": 0.5, "B": 0.0, "C": 0.5}
    assert one_way == pytest.approx(0.5)                                           # 0.5 * (0.1 + 0.4 + 0.5)


def _target_fn(calls):
    def fn(r, prev):
        calls.append((r, prev.copy()))
        return pd.Series({"A": 0.3, "B": 0.7}), {"solver": "ECOS"}
    return fn


def test_init_starts_from_the_backtest_book_without_trading():
    calls = []
    st, rec = update(None, data_as_of=DATES[5], last_rebalance=DATES[3], sim_book=pd.Series([0.6, 0.4], index=T),
                     returns=_returns(), bm_returns=_bm(), target_fn=_target_fn(calls), sim_target_at=lambda r: None)
    assert rec["action"] == "init" and not calls
    assert st["as_of"] == str(DATES[5].date()) and st["last_frozen_rebalance"] == str(DATES[3].date())
    assert st["weights"] == {"A": 0.6, "B": 0.4} and st["nav"] == 1.0 and st["bm_nav"] == 1.0


def _state():
    st, _ = update(None, data_as_of=DATES[5], last_rebalance=DATES[3], sim_book=pd.Series([0.5, 0.5], index=T),
                   returns=_returns(), bm_returns=_bm(), target_fn=_target_fn([]), sim_target_at=lambda r: None)
    return st


def test_accrue_without_a_new_rebalance_only_drifts():
    calls = []
    st, rec = update(_state(), data_as_of=DATES[8], last_rebalance=DATES[3], sim_book=pd.Series([0.9, 0.1], index=T),
                     returns=_returns(), bm_returns=_bm(), target_fn=_target_fn(calls), sim_target_at=lambda r: None)
    assert rec["action"] == "accrue" and rec["days"] == 3 and not calls
    assert st["as_of"] == str(DATES[8].date()) and st["weights"]["A"] > 0.5      # drifted paper book, not sim_book


def test_new_rebalance_uses_the_paper_book_entering_r_and_trades_at_the_publication_date():
    calls = []
    st0 = _state()
    st, rec = update(st0, data_as_of=DATES[12], last_rebalance=DATES[9], sim_book=pd.Series([0.9, 0.1], index=T),
                     returns=_returns(), bm_returns=_bm(), target_fn=_target_fn(calls), sim_target_at=lambda r: None)
    expected_prev, *_ = advance_book(pd.Series(st0["weights"]), 1.0, 1.0, _returns(), _bm(), DATES[5], DATES[9])
    assert len(calls) == 1 and calls[0][0] == DATES[9]
    pd.testing.assert_series_equal(calls[0][1], expected_prev)
    assert rec["action"] == "freeze" and rec["trade_data_date"] == str(DATES[12].date()) and rec["days"] == 7
    assert st["weights"] == {"A": 0.3, "B": 0.7} and st["last_frozen_rebalance"] == str(DATES[9].date())
    assert st["frozen"][str(DATES[9].date())]["published_data_as_of"] == str(DATES[12].date())
    assert st0["last_frozen_rebalance"] == str(DATES[3].date())                    # input state not mutated


def test_republication_never_trades_and_logs_jitter():
    # §S28 H1: like-for-like — this run's simulated target at the frozen date vs the simulated target stored
    # when it was frozen (not vs the paper target, which is solved from the paper book and always differs).
    calls = []
    sim = pd.Series({"A": 0.4, "B": 0.6})                                          # paper target is 0.3/0.7
    kw = dict(last_rebalance=DATES[9], sim_book=pd.Series([0.5, 0.5], index=T), returns=_returns(),
              bm_returns=_bm(), target_fn=_target_fn(calls))
    st, rec0 = update(_state(), data_as_of=DATES[12], sim_target_at=lambda r: sim, **kw)
    assert st["frozen"][str(DATES[9].date())]["sim_target"] == {"A": 0.4, "B": 0.6}
    assert "sim_target" not in rec0                                                # state only, not the ledger line
    st2, rec = update(st, data_as_of=DATES[14], sim_target_at=lambda r: sim, **kw)
    assert len(calls) == 1 and rec["action"] == "accrue"
    assert rec["republication_jitter_one_way"] == pytest.approx(0.0)
    _, rec = update(st2, data_as_of=DATES[16], sim_target_at=lambda r: pd.Series({"A": 0.5, "C": 0.5}), **kw)
    assert len(calls) == 1 and rec["republication_jitter_one_way"] == pytest.approx(0.6)   # 0.5 * (0.1 + 0.6 + 0.5)
    _, rec = update(st2, data_as_of=DATES[16], sim_target_at=lambda r: None, **kw)
    assert "republication_jitter_one_way" not in rec
    # frozen without a stored simulated target: nothing to compare with
    st, _ = update(_state(), data_as_of=DATES[12], sim_target_at=lambda r: None, **kw)
    assert "sim_target" not in st["frozen"][str(DATES[9].date())]
    _, rec = update(st, data_as_of=DATES[14], sim_target_at=lambda r: sim, **kw)
    assert "republication_jitter_one_way" not in rec


def test_no_new_data_leaves_the_state_unchanged():
    st = _state()
    st2, rec = update(st, data_as_of=DATES[5], last_rebalance=DATES[3], sim_book=pd.Series([0.9, 0.1], index=T),
                      returns=_returns(), bm_returns=_bm(), target_fn=_target_fn([]), sim_target_at=lambda r: None)
    assert rec["action"] == "no_new_data" and st2 == st


def test_run_appends_one_ledger_line_per_call_and_persists_state(tmp_path):
    kw = dict(last_rebalance=DATES[3], sim_book=pd.Series([0.5, 0.5], index=T), returns=_returns(), bm_returns=_bm(),
              target_fn=_target_fn([]), sim_target_at=lambda r: None)
    run(tmp_path, run_info={"run": 1}, data_as_of=DATES[5], **kw)
    run(tmp_path, run_info={"run": 2}, data_as_of=DATES[8], **kw)
    lines = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["run"] for x in lines] == [1, 2]
    assert [json.loads(x)["action"] for x in lines] == ["init", "accrue"]
    assert json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))["as_of"] == str(DATES[8].date())
    assert not list(tmp_path.glob("*.tmp"))


def test_run_appends_the_ledger_line_before_an_atomic_state_replace(tmp_path, monkeypatch):
    # §S28 M7: a failed state write leaves the OLD state (never a torn one) plus the audit line
    kw = dict(last_rebalance=DATES[3], sim_book=pd.Series([0.5, 0.5], index=T), returns=_returns(), bm_returns=_bm(),
              target_fn=_target_fn([]), sim_target_at=lambda r: None)
    run(tmp_path, run_info={"run": 1}, data_as_of=DATES[5], **kw)
    before = (tmp_path / "state.json").read_bytes()

    def boom(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr("scripts.paper_ledger.os.replace", boom)
    with pytest.raises(OSError, match="replace failed"):
        run(tmp_path, run_info={"run": 2}, data_as_of=DATES[8], **kw)
    assert (tmp_path / "state.json").read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))
    lines = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["run"] for x in lines] == [1, 2]
