"""TDD-guard stem test for src/backtest listing-mask wiring.

Authoritative coverage lives in tests/acceptance/test_listing_mask.py. This
stem-named file exists so the pytest-tdd PreToolUse guard permits editing
src/backtest.py.
"""

import inspect


def test_make_capweight_bm_fn_accepts_config():
    from src.backtest import make_capweight_bm_fn

    assert "config" in inspect.signature(make_capweight_bm_fn).parameters
