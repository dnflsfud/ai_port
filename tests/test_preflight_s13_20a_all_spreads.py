# -*- coding: utf-8 -*-
"""§S13.20a 전수 재검정 스크립트 순수 함수 단위테스트."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.preflight_s13_20a_all_spreads import (SPREAD_CONFIGS,
                                                   spread_innovation)


def test_spread_innovation_hand_match():
    bd = pd.bdate_range("2024-01-02", periods=6)
    rng = np.random.default_rng(2)
    px = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, (6, 3)), axis=0)),
        index=bd, columns=["SPX_FWD_EPS", "NDX_FWD_EPS", "MXWD_FWD_EPS"])
    d = {k: np.log(px[f"{k}_FWD_EPS"]).diff() for k in ("SPX", "NDX", "MXWD")}
    np.testing.assert_allclose(
        spread_innovation(px, "fac_eps_g63"), d["MXWD"], atol=1e-12)
    np.testing.assert_allclose(
        spread_innovation(px, "fac_eps_us_lead63"),
        d["SPX"] - d["MXWD"], atol=1e-12)
    np.testing.assert_allclose(
        spread_innovation(px, "fac_eps_us_lead252"),
        d["SPX"] - d["MXWD"], atol=1e-12)
    np.testing.assert_allclose(
        spread_innovation(px, "fac_eps_tech_lead63"),
        d["NDX"] - d["SPX"], atol=1e-12)


def test_spread_configs_cover_all_four():
    assert [c["state"] for c in SPREAD_CONFIGS] == [
        "fac_eps_g63", "fac_eps_us_lead63",
        "fac_eps_us_lead252", "fac_eps_tech_lead63"]
