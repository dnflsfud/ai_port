# -*- coding: utf-8 -*-
"""§S27.2 paper ledger (decision log §S27 ① — user answers 2026-11-03: no real trading
(paper), follow the rebalance-date target in full).

Append-only record of what the published process would have held. The paper book
switches to the target of each new rebalance date R at the data date D of the FIRST
run that publishes R (the real publication delay), with w_prev = the drifted PAPER
book entering R — not the backtest's simulated path — and drifts with daily returns
in between. Past NAV is never rewritten when a workbook regeneration revises history:
each run only appends the dates (state.as_of, data_as_of]. Re-publications of an
already-frozen R never trade; their one-way distance to the frozen target is logged
as re-publication jitter.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

ONE_WAY_COST_BPS = 10.0      # no cost given by the user: 10bp one-way (operating export default)


def advance_book(weights: pd.Series, nav: float, bm_nav: float, returns: pd.DataFrame,
                 bm_returns: pd.Series, start_exclusive, end_inclusive
                 ) -> Tuple[pd.Series, float, float, int]:
    """Hold ``weights`` (close of ``start_exclusive``) through ``end_inclusive``; NaN return = 0."""
    tickers = list(weights.index)
    idx = returns.index
    days = idx[(idx > pd.Timestamp(start_exclusive)) & (idx <= pd.Timestamp(end_inclusive))]
    w = weights.to_numpy(dtype=float)
    for d in days:
        r = np.nan_to_num(returns.loc[d].reindex(tickers).to_numpy(dtype=float), nan=0.0)
        nav *= 1.0 + float(w @ r)
        bm_nav *= 1.0 + float(np.nan_to_num(bm_returns.get(d, 0.0)))
        grown = w * (1.0 + r)                       # backtest._drift_weights arithmetic
        w = grown / grown.sum() if grown.sum() > 0 else grown
    return pd.Series(w, index=tickers), nav, bm_nav, int(len(days))


def apply_trade(weights: pd.Series, target: pd.Series, nav: float,
                cost_bps: float = ONE_WAY_COST_BPS) -> Tuple[pd.Series, float, float]:
    target = target.reindex(weights.index).fillna(0.0)
    one_way = 0.5 * float(np.abs(target.to_numpy() - weights.to_numpy()).sum())
    return target.copy(), nav * (1.0 - one_way * cost_bps / 1e4), one_way


def update(state: Optional[dict], *, data_as_of, last_rebalance, sim_book: pd.Series,
           returns: pd.DataFrame, bm_returns: pd.Series,
           target_fn: Callable[[pd.Timestamp, pd.Series], Tuple[pd.Series, dict]],
           sim_target_at: Callable[[pd.Timestamp], Optional[pd.Series]]) -> Tuple[dict, dict]:
    """One run → (new state, ledger record). ``target_fn(R, prev)`` → (target, info) through the
    production rebalance path; ``sim_target_at(R)`` → this run's backtest target at R (jitter)."""
    D, R = pd.Timestamp(data_as_of), pd.Timestamp(last_rebalance)
    if state is None:
        st = {"schema_version": 1, "inception_data_as_of": str(D.date()), "as_of": str(D.date()),
              "weights": {k: float(v) for k, v in sim_book.items()}, "nav": 1.0, "bm_nav": 1.0,
              "last_frozen_rebalance": str(R.date()), "frozen": {}}
        return st, {"action": "init", "paper_nav": 1.0, "bm_nav": 1.0,
                    "note": "paper book starts as the backtest book at data_as_of (no trade)"}
    rec: dict = {}
    last_frozen = pd.Timestamp(state["last_frozen_rebalance"])
    frozen = state["frozen"].get(str(last_frozen.date()))
    sim_at_frozen = sim_target_at(last_frozen)
    if frozen is not None and sim_at_frozen is not None:
        ft = pd.Series(frozen["target"], dtype=float)
        rec["republication_jitter_one_way"] = 0.5 * float(
            np.abs(sim_at_frozen.reindex(ft.index).fillna(0.0) - ft).sum())
    S = pd.Timestamp(state["as_of"])
    if D <= S:
        rec["action"] = "no_new_data"
        return state, rec
    w = pd.Series(state["weights"], dtype=float)
    nav, bm_nav = float(state["nav"]), float(state["bm_nav"])
    new = {k: v for k, v in state.items()}
    if R > last_frozen and S < R <= D:
        w, nav, bm_nav, n1 = advance_book(w, nav, bm_nav, returns, bm_returns, S, R)
        target, info = target_fn(R, w)
        w, nav, bm_nav, n2 = advance_book(w, nav, bm_nav, returns, bm_returns, R, D)
        w, nav, one_way = apply_trade(w, target, nav)
        new["frozen"] = dict(state["frozen"])
        new["frozen"][str(R.date())] = {"published_data_as_of": str(D.date()),
                                        "target": {k: float(v) for k, v in target.items()},
                                        "one_way_turnover": one_way, **info}
        new["last_frozen_rebalance"] = str(R.date())
        rec.update({"action": "freeze", "rebalance": str(R.date()), "trade_data_date": str(D.date()),
                    "one_way_turnover": one_way, "days": n1 + n2, **info})
    else:
        w, nav, bm_nav, n = advance_book(w, nav, bm_nav, returns, bm_returns, S, D)
        rec.update({"action": "accrue", "days": n})
    new.update({"as_of": str(D.date()), "weights": {k: float(v) for k, v in w.items()},
                "nav": nav, "bm_nav": bm_nav})
    rec.update({"paper_nav": nav, "bm_nav": bm_nav, "paper_active_since_inception": nav / bm_nav - 1.0})
    return new, rec


def run(ledger_dir, *, run_info: dict, **kwargs) -> dict:
    """Load ``state.json``, apply :func:`update`, rewrite the state and append one line to
    ``ledger.jsonl`` (the ledger is append-only; the state is the current paper book)."""
    ledger_dir = Path(ledger_dir)
    ledger_dir.mkdir(parents=True, exist_ok=True)
    state_path = ledger_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
    new_state, rec = update(state, **kwargs)
    record = {**run_info, "data_as_of": str(pd.Timestamp(kwargs["data_as_of"]).date()),
              "last_rebalance": str(pd.Timestamp(kwargs["last_rebalance"]).date()), **rec}
    state_path.write_text(json.dumps(new_state, indent=1, default=str), encoding="utf-8")
    with open(ledger_dir / "ledger.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
    return record
