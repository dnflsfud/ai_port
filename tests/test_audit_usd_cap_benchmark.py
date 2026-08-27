# -*- coding: utf-8 -*-
"""audit_usd_cap_benchmark 유니버스 게이트 계약 테스트 (§S14: 200→250)."""

import pandas as pd
import pytest

import scripts.audit_usd_cap_benchmark as audit_mod
from scripts.audit_usd_cap_benchmark import (
    EXPECTED_UNIVERSE_SIZE,
    _cap_sample,
    _check_universe,
)


def test_universe_gate_expects_250():
    assert EXPECTED_UNIVERSE_SIZE == 250


def test_universe_gate_accepts_250_unique():
    _check_universe([f"T{i}" for i in range(250)])  # no raise


def test_universe_gate_rejects_200_and_duplicates():
    with pytest.raises(ValueError):
        _check_universe([f"T{i}" for i in range(200)])
    with pytest.raises(ValueError):
        _check_universe(["DUP"] * 250)


def test_cap_sample_skips_missing_tickers_instead_of_raising():
    # 슬레이트 교체로 샘플 티커가 빠져도 실질 검사 통과 후 KeyError로 죽으면 안 됨.
    caps = pd.Series({"AAPL": 3.5e6, "ZZZ": 1.0})
    assert _cap_sample(caps) == {"AAPL": 3.5e6}
    assert _cap_sample(pd.Series(dtype=float)) == {}


def test_module_docstring_reflects_250_universe():
    assert "250" in audit_mod.__doc__
    assert "150" not in audit_mod.__doc__
