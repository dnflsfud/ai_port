"""Co-located smoke test for scripts/run_adaptive_ema_arm.py (house convention:
each driver script has a tests/test_<script>.py companion, cf. test_dr_alpha.py).

The AUTHORITATIVE contract for the two pure functions lives in
tests/acceptance/test_adaptive_ema.py (test-designer owned, not modified here).
This file only smoke-checks that the module imports and that the two most basic
invariants hold, so a broken module fails fast under the plain `tests/` sweep.
RED (ModuleNotFoundError) until scripts/run_adaptive_ema_arm.py exists.
"""

import numpy as np
import pandas as pd

from scripts.run_adaptive_ema_arm import (
    compute_adaptive_alpha,
    apply_adaptive_ema,
)
from src.model_trainer import apply_prediction_ema


def _bdays(n, start="2020-01-01"):
    return pd.bdate_range(start, periods=n)


def test_no_events_alpha_is_half():
    dates = _bdays(10)
    empty = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)
    a = compute_adaptive_alpha(empty, dates, 0.04, 0.075)
    assert list(a.index) == list(dates)
    assert np.all(a.values == 0.5)


def test_adaptive_ema_half_equals_prediction_ema():
    rng = np.random.default_rng(0)
    dates = _bdays(20)
    raw = pd.DataFrame(rng.normal(size=(20, 4)),
                       index=dates, columns=list("ABCD"))
    raw.iloc[:3] = np.nan
    got = apply_adaptive_ema(raw, pd.Series(0.5, index=dates))
    ref = apply_prediction_ema(raw, 0.5)
    pd.testing.assert_frame_equal(got, ref, check_exact=True)
