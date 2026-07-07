"""Co-located smoke test for scripts/run_seed_ensemble_arm.py (house convention:
each driver script has a tests/test_<script>.py companion, cf. test_dr_alpha.py,
test_run_adaptive_ema_arm.py).

The AUTHORITATIVE contract for the two pure functions lives in
tests/acceptance/test_seed_ensemble.py (test-designer owned, not modified here).
This file only smoke-checks that the module imports and that the two most basic
invariants hold, so a broken module fails fast under the plain `tests/` sweep.
RED (ModuleNotFoundError) until scripts/run_seed_ensemble_arm.py exists.
"""

import numpy as np
import pandas as pd

from scripts.run_seed_ensemble_arm import (
    combine_seed_panels,
    nan_mask_mismatch_rate,
)


def _bdays(n, start="2020-01-01"):
    return pd.bdate_range(start, periods=n)


def _tickers(n):
    return [f"T{i}" for i in range(n)]


def test_identical_panels_combine_to_per_date_z():
    """k copies of one panel combine to that panel re-standardized per date."""
    dates = _bdays(3)
    tks = _tickers(4)
    rng = np.random.default_rng(0)
    p = pd.DataFrame(
        rng.integers(-4, 5, size=(len(dates), len(tks))).astype(float),
        index=dates, columns=tks,
    )
    got = combine_seed_panels([p, p, p, p, p])
    # Each re-standardized row (non-constant) has finite mean ~0, sample std ~1.
    for d in got.index:
        row = got.loc[d].values.astype(float)
        fin = row[np.isfinite(row)]
        if len(fin) >= 2 and fin.std(ddof=1) > 0:
            assert abs(fin.mean()) <= 1e-9
            assert abs(fin.std(ddof=1) - 1.0) <= 1e-9


def test_all_nan_cell_yields_nan():
    dates = _bdays(2)
    tks = _tickers(3)
    p0 = pd.DataFrame(np.ones((2, 3)), index=dates, columns=tks)
    p1 = pd.DataFrame(np.ones((2, 3)) * 2, index=dates, columns=tks)
    for p in (p0, p1):
        p.iloc[0, 2] = np.nan
    got = combine_seed_panels([p0, p1])
    assert np.isnan(got.iloc[0, 2])


def test_mismatch_rate_consistent_is_zero():
    dates = _bdays(3)
    tks = _tickers(3)
    p = pd.DataFrame(np.ones((3, 3)), index=dates, columns=tks)
    p.iloc[0, 0] = np.nan
    q = p.copy()
    assert nan_mask_mismatch_rate([p, q]) == 0.0
