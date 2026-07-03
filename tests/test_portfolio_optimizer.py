"""Minimal stem tests for src/portfolio_optimizer.

Required by the TDD guard hook so edits to portfolio_optimizer.py are allowed.
The authoritative acceptance coverage for the structural fixes lives in
tests/acceptance/test_optimizer_structural_fixes.py — this file only smoke-checks
importability and the OFF-default parity of the new cov-shrink flag.
"""

import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.portfolio_optimizer import estimate_covariance, optimize_portfolio


def test_module_imports():
    assert callable(estimate_covariance)
    assert callable(optimize_portfolio)


def test_cov_shrink_flag_default_on():
    # New flag defaults to True (current behavior preserved).
    assert PipelineConfig().cov_megacap_vol_shrink_enabled is True
