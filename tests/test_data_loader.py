"""TDD-guard stem test for src/data_loader.mask_pre_listing.

Authoritative coverage lives in tests/acceptance/test_listing_mask.py. This
stem-named file exists so the pytest-tdd PreToolUse guard permits editing
src/data_loader.py.
"""

import numpy as np
import pandas as pd
import pytest


def test_mask_pre_listing_inclusive_masks_listing_day():
    from src.data_loader import mask_pre_listing

    idx = pd.to_datetime(["2020-09-29", "2020-09-30", "2020-10-01"])
    df = pd.DataFrame({"PLTR": [1.0, 2.0, 3.0]}, index=idx)
    out = mask_pre_listing(df, {"PLTR": "2020-09-30"}, inclusive=True)
    assert bool(np.isnan(out["PLTR"].iloc[0]))
    assert bool(np.isnan(out["PLTR"].iloc[1]))
    assert out["PLTR"].iloc[2] == 3.0
    # input frame is not mutated
    assert df["PLTR"].iloc[0] == 1.0


def test_align_dates_removes_weekends_from_intersection_and_tail():
    from src.config import PipelineConfig
    from src.data_loader import align_dates

    px_dates = pd.to_datetime([
        "2026-06-04", "2026-06-05", "2026-06-06", "2026-06-07", "2026-06-08",
    ])
    slow_dates = pd.to_datetime(["2026-06-04", "2026-06-05", "2026-06-06"])
    processed = {
        "PX_LAST": pd.DataFrame({"AAA": range(len(px_dates))}, index=px_dates),
        "SLOW": pd.DataFrame({"AAA": range(len(slow_dates))}, index=slow_dates),
    }
    diagnostics = {}
    aligned = align_dates(
        processed,
        config=PipelineConfig(max_tail_ffill_days=10),
        diagnostics=diagnostics,
    )

    assert list(aligned["PX_LAST"].index.strftime("%Y-%m-%d")) == [
        "2026-06-04", "2026-06-05", "2026-06-08",
    ]
    assert diagnostics["weekend_dates_removed"] == 2
    assert diagnostics["calendar_type"] == "weekday_index"
    assert diagnostics["tail_ffill_days"] == 1


def test_align_dates_can_fail_fast_on_stale_weekday_tail():
    from src.config import PipelineConfig
    from src.data_loader import align_dates

    px_dates = pd.bdate_range("2026-06-01", periods=5)
    processed = {
        "PX_LAST": pd.DataFrame({"AAA": range(5)}, index=px_dates),
        "SLOW": pd.DataFrame({"AAA": range(2)}, index=px_dates[:2]),
    }
    with pytest.raises(ValueError, match="Tail ffill length 3 exceeds"):
        align_dates(
            processed,
            config=PipelineConfig(
                max_tail_ffill_days=2,
                fail_on_stale_tail_ffill=True,
            ),
        )
