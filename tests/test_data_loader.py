"""TDD-guard stem test for src/data_loader.mask_pre_listing.

Authoritative coverage lives in tests/acceptance/test_listing_mask.py. This
stem-named file exists so the pytest-tdd PreToolUse guard permits editing
src/data_loader.py.
"""

import numpy as np
import pandas as pd


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
