# -*- coding: utf-8 -*-
"""audit_usd_cap_benchmark 유니버스 게이트 계약 테스트 (§S11.3: 100→150)."""

import pytest

from scripts.audit_usd_cap_benchmark import EXPECTED_UNIVERSE_SIZE, _check_universe


def test_universe_gate_expects_150():
    assert EXPECTED_UNIVERSE_SIZE == 150


def test_universe_gate_accepts_150_unique():
    _check_universe([f"T{i}" for i in range(150)])  # no raise


def test_universe_gate_rejects_100_and_duplicates():
    with pytest.raises(ValueError):
        _check_universe([f"T{i}" for i in range(100)])
    with pytest.raises(ValueError):
        _check_universe(["DUP"] * 150)
