# -*- coding: utf-8 -*-
"""§S18.1 arm 판정 스크립트 헬퍼 단위테스트 (경량 — pkl 로드 없음)."""
import types

import numpy as np
import pandas as pd

from scripts.eval_s18_arm import (
    BASE, FRAMES, NEG_EQUITY_NAMES, TG_BASIS_GATES, TURNOVER_NEUTRAL_BAND, Z_SD_MIN,
    mechanism_static, mechanism_static_neutral, mechanism_tg_basis, mechanism_tilt,
)


def test_frames_and_base():
    assert set(FRAMES) == {"s18_1_tilt_negative_equity", "s18_2_tg_basis_events", "s18_3_static_execution",
                           "s18_5_static_execution_eta042"}
    assert BASE == "s18_s0recert"
    # §S18.3 re-attempt is judged against the two-flip re-certification, not the S18.1 base.
    assert FRAMES["s18_5_static_execution_eta042"]["base"] == "s18_4_flip2_recert"
    assert FRAMES["s18_5_static_execution_eta042"]["mechanism"] == "static_neutral"
    assert TURNOVER_NEUTRAL_BAND == (0.85, 1.15)
    assert FRAMES["s18_3_static_execution"]["alpha_identical"] is True
    assert all(v["turnover_max"] == 1.25 for v in FRAMES.values())
    assert set(TG_BASIS_GATES) == {"RTX", "T", "DELL", "DHR"} and Z_SD_MIN == 0.90
    assert len(NEG_EQUITY_NAMES) == 21


def _res(pre_exec=None, panel=None):
    return types.SimpleNamespace(pre_execution_predictions=pre_exec, panel=panel)


def test_mechanism_tilt_requires_penalty_release_on_the_named_cells():
    idx = pd.bdate_range("2024-01-01", periods=120)
    cols = NEG_EQUITY_NAMES[:5] + ["AAPL"]
    base = pd.DataFrame(0.0, index=idx, columns=cols)
    arm = base.copy()
    arm.loc[:, NEG_EQUITY_NAMES[:5]] = 0.3           # 600 cells released by +0.3
    out = mechanism_tilt(_res(base), _res(arm))
    assert out["pass"] is True and out["changed_cells"] == 600
    weak = base.copy(); weak.loc[idx[:10], NEG_EQUITY_NAMES[0]] = 0.5   # 10 cells only
    assert mechanism_tilt(_res(base), _res(weak))["pass"] is False
    assert mechanism_tilt(_res(None), _res(arm))["pass"] is False


def _panel(values_by_ticker, years=(2014, 2022), n_other=30, spread=1.0):
    dates = pd.bdate_range(f"{years[0]}-01-01", f"{years[1]}-12-31")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(0, spread, (len(dates), n_other)), index=dates,
                         columns=[f"X{i}" for i in range(n_other)])
    for t, v in values_by_ticker.items():
        frame[t] = v
    stacked = frame.stack()
    stacked.index.names = ["date", "ticker"]
    return pd.DataFrame({"tg_upside": stacked})


def test_mechanism_tg_basis_gates():
    base = _panel({"RTX": -3.5, "T": -2.2, "DELL": 5.0, "DHR": 3.0}, spread=0.8)
    arm = _panel({"RTX": -0.3, "T": 0.1, "DELL": 0.5, "DHR": 0.2}, spread=1.0)
    out = mechanism_tg_basis(_res(panel=base), _res(panel=arm))
    assert out["pass"] is True
    assert out["names"]["RTX"]["base"] < -3 and out["names"]["RTX"]["arm"] > -1.5
    assert out["tg_upside_cs_sd_2014_2021"]["arm"] >= Z_SD_MIN
    still_pinned = _panel({"RTX": -3.5, "T": 0.1, "DELL": 0.5, "DHR": 0.2}, spread=1.0)
    assert mechanism_tg_basis(_res(panel=base), _res(panel=still_pinned))["pass"] is False


def test_mechanism_static_execution_bounds():
    g0_ok = {"avg_ic_bit_identical": True, "degenerate_equal": True}
    assert mechanism_static(1.10, g0_ok)["pass"] is True
    assert mechanism_static(0.98, g0_ok)["pass"] is False      # execution-only change must not trade less
    assert mechanism_static(1.30, g0_ok)["pass"] is False
    assert mechanism_static(1.10, {"avg_ic_bit_identical": False, "degenerate_equal": True})["pass"] is False


def test_mechanism_static_neutral_bounds():
    g0_ok = {"avg_ic_bit_identical": True, "degenerate_equal": True}
    assert mechanism_static_neutral(1.00, g0_ok)["pass"] is True
    assert mechanism_static_neutral(0.90, g0_ok)["pass"] is True      # neutral: trading less is allowed
    assert mechanism_static_neutral(0.80, g0_ok)["pass"] is False
    assert mechanism_static_neutral(1.28, g0_ok)["pass"] is False     # the S18.1 arm-3 outcome fails here
    assert mechanism_static_neutral(1.00, {"avg_ic_bit_identical": False, "degenerate_equal": True})["pass"] is False
