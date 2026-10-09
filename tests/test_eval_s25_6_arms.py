# -*- coding: utf-8 -*-
"""§S25.6 성과 arm 판정 스크립트의 순수 함수 테스트 (결정 로그 §S25.6 사전등록 규칙 핀)."""

import numpy as np
import pandas as pd

from scripts.eval_s25_6_arms import (
    ARMS,
    REFERENCE_BASE_IR,
    mechanism_boost_cells,
    mechanism_sector_share,
    top_decile_weight_share,
    verdicts,
)


def _preds(n_dates=4, n=50, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_dates)
    return pd.DataFrame(rng.normal(size=(n_dates, n)), index=dates, columns=[f"t{i}" for i in range(n)])


def test_arm_table_pins_the_two_single_flag_arms_and_base():
    assert set(ARMS) == {"s25_6_top_decile_boost", "s25_6_sector_risk_penalty"}
    assert ARMS["s25_6_top_decile_boost"]["flag"] == "top_decile_boost_enabled"
    assert ARMS["s25_6_sector_risk_penalty"]["flag"] == "sector_active_risk_penalty_enabled"
    assert REFERENCE_BASE_IR == 1.8229575020692097


def test_mechanism_boost_cells_requires_changes_exactly_on_the_base_top_decile():
    base = _preds(); arm = base.copy()
    for d in base.index:
        row = base.loc[d]; top = row.ge(row.quantile(0.9))
        arm.loc[d, top[top].index] += 0.5 * row.std(ddof=1)
    out = mechanism_boost_cells(base, arm, quantile=0.9)
    assert out["pass"] is True and out["changed_outside_top"] == 0 and out["top_unchanged"] == 0 and out["changed_cells"] > 0
    arm2 = arm.copy(); arm2.iloc[0, 0] += 1.0  # touch a cell outside the top decile (very likely) -> fail unless it is top
    row = base.iloc[0]; outside = row.index[~row.ge(row.quantile(0.9))][0]
    arm3 = arm.copy(); arm3.loc[base.index[0], outside] += 1.0
    assert mechanism_boost_cells(base, arm3, quantile=0.9)["pass"] is False
    assert mechanism_boost_cells(base, base, quantile=0.9)["pass"] is False


def test_top_decile_weight_share_counts_portfolio_weight_in_base_top_decile_names():
    base = _preds(n_dates=1)
    d = base.index[0]; row = base.loc[d]; top = row.ge(row.quantile(0.9))
    w = pd.Series(0.0, index=row.index); w[top[top].index[:3]] = 0.1; w["t0"] = 0.7 if not top["t0"] else 0.0
    share = top_decile_weight_share({d: w.to_dict()}, base, quantile=0.9)
    assert abs(share.loc[d] - 0.3 / w.sum()) < 1e-12


def test_mechanism_sector_share_requires_lower_mean_and_fewer_breaches():
    base = pd.Series([0.9, 0.86, 0.7, 0.6]); arm = pd.Series([0.8, 0.7, 0.6, 0.55])
    out = mechanism_sector_share(base, arm, cap=0.85)
    assert out["pass"] is True and out["breaches_base"] == 2 and out["breaches_arm"] == 0 and out["mean_delta"] < 0
    assert mechanism_sector_share(base, base, cap=0.85)["pass"] is False
    worse = pd.Series([0.9, 0.9, 0.5, 0.5])  # mean lower (0.70 < 0.765) but breaches not fewer (2 == 2)
    assert mechanism_sector_share(base, worse, cap=0.85)["pass"] is False


def test_verdicts_performance_frame():
    v = verdicts(True, True, True, 0.40, [0.1, 0.2, 0.3])
    assert v["e1_formal_pass"] and v["adoption_candidate"] and v["override_candidate"]
    v = verdicts(True, True, True, 0.10, [0.1, 0.2, 0.3])
    assert not v["adoption_candidate"] and v["override_candidate"] and v["no_harm_pass"]
    v = verdicts(True, True, True, -0.10, [-0.1, -0.2, -0.3])
    assert not v["no_harm_pass"] and not v["override_candidate"]
    v = verdicts(True, False, True, 0.50, [0.1, 0.2, 0.3])
    assert not v["adoption_candidate"] and not v["override_candidate"]
