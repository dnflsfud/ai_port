# -*- coding: utf-8 -*-
"""§S24.2 arm 판정 스크립트의 순수 함수 테스트 (결정 로그 §S24.2 사전등록 규칙 핀)."""

import numpy as np

from scripts.eval_s24_slope_arm import (
    CONSUMED_MIN,
    REFERENCE_BASE_IR,
    TURNOVER_MAX,
    mechanism_consumption_from_gains,
    mechanism_feature_set,
    verdicts,
)
from src.features.fwd_sales_slope import FWD_SALES_SLOPE_FEATURES

BASE_FEATURES = ["momentum_252d", "best_sales_chg_252d", "idio_vol_63d"]


def test_feature_set_pass_when_exactly_four_slope_features_added():
    arm = BASE_FEATURES + list(FWD_SALES_SLOPE_FEATURES)
    out = mechanism_feature_set(BASE_FEATURES, arm)
    assert out["set_pass"] is True
    assert out["added"] == sorted(FWD_SALES_SLOPE_FEATURES)
    assert out["removed"] == []


def test_feature_set_fails_on_extra_or_removed_feature():
    arm_extra = BASE_FEATURES + list(FWD_SALES_SLOPE_FEATURES) + ["stray_feature"]
    assert mechanism_feature_set(BASE_FEATURES, arm_extra)["set_pass"] is False
    arm_removed = BASE_FEATURES[:-1] + list(FWD_SALES_SLOPE_FEATURES)
    assert mechanism_feature_set(BASE_FEATURES, arm_removed)["set_pass"] is False
    assert mechanism_feature_set(BASE_FEATURES, BASE_FEATURES + list(FWD_SALES_SLOPE_FEATURES[:3]))["set_pass"] is False


def _gains(slope_gain: float, n: int = 10):
    recs = []
    for _ in range(n):
        rec = {f: 10.0 for f in BASE_FEATURES}
        rec.update({f: slope_gain for f in FWD_SALES_SLOPE_FEATURES})
        recs.append(rec)
    return recs


def test_consumption_pass_when_all_four_used_in_every_retrain():
    out = mechanism_consumption_from_gains(_gains(5.0))
    assert out["recorded"] is True
    assert out["consumption_pass"] is True
    assert all(v == 1.0 for v in out["consumed_share_by_feature"].values())
    # block share = 4*5 / (3*10 + 4*5) = 0.4
    assert np.isclose(out["block_gain_share_mean"], 0.4)


def test_consumption_fails_when_a_slope_feature_is_never_split_on():
    recs = _gains(5.0)
    for rec in recs:
        rec[FWD_SALES_SLOPE_FEATURES[0]] = 0.0
    out = mechanism_consumption_from_gains(recs)
    assert out["consumed_share_by_feature"][FWD_SALES_SLOPE_FEATURES[0]] == 0.0
    assert out["consumption_pass"] is False
    assert CONSUMED_MIN == 0.9


def test_consumption_not_recorded_when_no_gains():
    out = mechanism_consumption_from_gains([])
    assert out["recorded"] is False


def test_verdicts_formal_e1_and_override_rules():
    # formal E1: dIR > +0.36 and all splits positive
    v = verdicts(g0=True, mechanism=True, e2=True, d_ir=0.40, split_deltas=[0.1, 0.2, 0.05])
    assert v["adoption_candidate"] is True and v["override_candidate"] is True
    # positive but below the bar -> override candidate only
    v = verdicts(g0=True, mechanism=True, e2=True, d_ir=0.10, split_deltas=[0.1, 0.2, 0.05])
    assert v["adoption_candidate"] is False and v["override_candidate"] is True
    # one negative split kills both
    v = verdicts(g0=True, mechanism=True, e2=True, d_ir=0.40, split_deltas=[0.5, -0.01, 0.3])
    assert v["adoption_candidate"] is False and v["override_candidate"] is False
    # any gate failure kills both
    for gate in ("g0", "mechanism", "e2"):
        kw = dict(g0=True, mechanism=True, e2=True)
        kw[gate] = False
        v = verdicts(d_ir=0.40, split_deltas=[0.1, 0.2, 0.05], **kw)
        assert v["adoption_candidate"] is False and v["override_candidate"] is False


def test_preregistered_constants_pinned():
    assert np.isclose(REFERENCE_BASE_IR, 1.6951803327080086)
    assert TURNOVER_MAX == 1.25
