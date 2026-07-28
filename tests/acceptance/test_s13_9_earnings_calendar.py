# -*- coding: utf-8 -*-
"""S13.9 earnings-calendar admission arm.

conditioning.py builds 8 announcement-timing features from Earnings_Timeline,
but CORE_FEATURE_WHITELIST drops every one of them. Earnings timing therefore
reaches the book only through the PEAD overlay, which runs AFTER the model has
already ranked the cross-section — the model itself has never seen "this name
reports in 3 days".

Contract:
  * config.EARNINGS_CALENDAR_FEATURES names the 8 existing features.
  * config.earnings_calendar_feature_enabled is default-OFF.
  * assembly.build_all_features admits them through the S8 extra_whitelist
    path when the flag is ON, and is byte-identical when OFF.
  * None of the 8 may already be in CORE_FEATURE_WHITELIST — if one were,
    the arm would be measuring a smaller delta than it claims.
"""

import inspect

from src.config import EARNINGS_CALENDAR_FEATURES, PipelineConfig


def test_flag_is_default_off():
    assert PipelineConfig().earnings_calendar_feature_enabled is False


def test_block_is_the_eight_conditioning_features():
    assert len(EARNINGS_CALENDAR_FEATURES) == 8
    assert len(set(EARNINGS_CALENDAR_FEATURES)) == 8

    from src.features import conditioning

    source = inspect.getsource(conditioning)
    for name in EARNINGS_CALENDAR_FEATURES:
        assert f'features["{name}"]' in source, f"{name} is not built by conditioning.py"


def test_none_of_the_block_is_already_in_the_core_whitelist():
    """The arm's whole premise is that these are currently excluded."""
    from src.features.assembly import CORE_FEATURE_WHITELIST

    overlap = set(EARNINGS_CALENDAR_FEATURES) & CORE_FEATURE_WHITELIST
    assert not overlap, f"already in core, arm delta would be overstated: {overlap}"


def test_assembly_gates_the_block_on_the_flag():
    from src.features import assembly

    source = inspect.getsource(assembly.build_all_features)
    assert "earnings_calendar_feature_enabled" in source
    assert "EARNINGS_CALENDAR_FEATURES" in source


def test_day_counts_are_zscored_so_the_outlier_clip_cannot_flatten_them():
    """earn_days_since / earn_days_to_next span 0..999 (NO_EVENT sentinel).

    Conditioning features skip the cross-sectional z-score, but every feature
    still goes through clip_outliers(+-5). Left in skip_zscore these two would
    be clipped to 0..5 — i.e. silently degraded into duplicates of the
    earn_pre_5d / earn_post_5d flags, and the arm would be measuring 6
    features while claiming 8.
    """
    from src.features import assembly

    source = inspect.getsource(assembly.build_all_features)
    head, _, tail = source.partition("skip_zscore")
    assert "earn_days_since" in tail and "earn_days_to_next" in tail, (
        "day-count features are not lifted out of skip_zscore"
    )


def test_extra_whitelist_admits_the_block():
    """apply_core_filter must keep the 8 only when they are passed as extra."""
    import pandas as pd

    from src.features.assembly import apply_core_filter

    feats = {name: pd.DataFrame({"A": [1.0]}) for name in EARNINGS_CALENDAR_FEATURES}
    feats["momentum_252d"] = pd.DataFrame({"A": [1.0]})  # a real core member
    groups = {"Conditioning": list(EARNINGS_CALENDAR_FEATURES),
              "Price": ["momentum_252d"]}

    off = dict(feats)
    apply_core_filter(off, dict(groups))
    assert set(off) == {"momentum_252d"}

    on = dict(feats)
    apply_core_filter(on, dict(groups), extra_whitelist=set(EARNINGS_CALENDAR_FEATURES))
    assert set(on) == set(EARNINGS_CALENDAR_FEATURES) | {"momentum_252d"}
