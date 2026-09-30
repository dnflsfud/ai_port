# -*- coding: utf-8 -*-
"""§S24.3 arm 판정 스크립트(T·RTX tg_basis_events 제거)의 순수 함수 테스트 — 결정 로그 §S24.3 사전등록 규칙 핀."""

import numpy as np
import pandas as pd

from scripts.eval_s24_3_arm import (
    ARM_PRE_Z_ABS_MAX,
    EVENTS_REMOVED,
    EVENTS_KEPT,
    KEEP_DRIFT_MAX,
    POST_CUT,
    PRE_Z_MIN,
    REFERENCE_BASE_IR,
    TG_FEATURES,
    flip_verdict,
    mechanism_tg_events,
)

TICKERS = ["RTX", "T", "DELL", "DHR", "AAA"]
DATES = pd.date_range("2019-01-31", "2023-12-31", freq="ME")


def _panel(pre_post, other=1.0, tg_mom=0.0):
    """pre_post: ticker -> (pre-event value, post-event value) for tg_upside; others constant."""
    rows = []
    for d in DATES:
        for t in TICKERS:
            ev = EVENTS_REMOVED.get(t) or EVENTS_KEPT.get(t)
            pre, post = pre_post.get(t, (0.0, 0.0))
            val = pre if (ev is not None and d < pd.Timestamp(ev)) else post
            rows.append((d, t, val, tg_mom, other))
    df = pd.DataFrame(rows, columns=["date", "ticker", "tg_upside", "tg_mom_63d", "other_feat"])
    return df.set_index(["date", "ticker"])


BASE = _panel({"RTX": (5.0, -0.2), "T": (2.6, -0.3), "DELL": (0.3, 0.1), "DHR": (0.2, 0.0), "AAA": (0.5, 0.5)})


def test_constants_pinned():
    assert EVENTS_REMOVED == {"RTX": "2020-04-03", "T": "2022-04-11"}
    assert EVENTS_KEPT == {"DELL": "2021-11-02", "DHR": "2016-07-05"}
    assert POST_CUT == "2022-07-15" and ARM_PRE_Z_ABS_MAX == 1.0 and KEEP_DRIFT_MAX == 0.1
    assert PRE_Z_MIN == {"RTX": 2.0, "T": 1.5}
    assert set(TG_FEATURES) == {"tg_upside", "tg_mom_63d"}
    assert np.isclose(REFERENCE_BASE_IR, 1.6951803327080086)


def test_mechanism_passes_when_only_pre_event_t_rtx_move_to_centre():
    arm = _panel({"RTX": (0.1, -0.2), "T": (-0.2, -0.3), "DELL": (0.32, 0.1), "DHR": (0.18, 0.0), "AAA": (0.5, 0.5)})
    out = mechanism_tg_events(BASE, arm)
    assert out["pass"] is True
    assert out["names"]["RTX"]["base_pre_median"] == 5.0 and abs(out["names"]["RTX"]["arm_pre_median"]) <= 1.0
    assert out["kept"]["DELL"]["drift"] <= KEEP_DRIFT_MAX
    assert out["post_cut_max_abs_delta"] <= 1e-9 and out["other_features_max_abs_delta"] <= 1e-9


def test_mechanism_fails_when_rtx_still_inflated():
    arm = _panel({"RTX": (3.0, -0.2), "T": (-0.2, -0.3), "DELL": (0.3, 0.1), "DHR": (0.2, 0.0), "AAA": (0.5, 0.5)})
    assert mechanism_tg_events(BASE, arm)["pass"] is False


def test_mechanism_fails_when_kept_event_drifts_or_other_feature_changes():
    arm = _panel({"RTX": (0.1, -0.2), "T": (-0.2, -0.3), "DELL": (0.9, 0.1), "DHR": (0.2, 0.0), "AAA": (0.5, 0.5)})
    assert mechanism_tg_events(BASE, arm)["pass"] is False
    arm2 = _panel({"RTX": (0.1, -0.2), "T": (-0.2, -0.3), "DELL": (0.3, 0.1), "DHR": (0.2, 0.0), "AAA": (0.5, 0.5)}, other=2.0)
    assert mechanism_tg_events(BASE, arm2)["pass"] is False


def test_mechanism_fails_when_post_cut_tg_cells_change():
    arm = _panel({"RTX": (0.1, 0.4), "T": (-0.2, -0.3), "DELL": (0.3, 0.1), "DHR": (0.2, 0.0), "AAA": (0.5, 0.5)})
    out = mechanism_tg_events(BASE, arm)
    assert out["post_cut_max_abs_delta"] > 1e-9 and out["pass"] is False


def test_flip_verdict_requires_all_four_gates():
    assert flip_verdict(True, True, True, True) is True
    for i in range(4):
        args = [True, True, True, True]
        args[i] = False
        assert flip_verdict(*args) is False
