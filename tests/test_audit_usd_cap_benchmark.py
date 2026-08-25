# -*- coding: utf-8 -*-
"""audit_usd_cap_benchmark 유니버스 게이트 계약 테스트 (§S14: 200→250)."""

import pytest

from scripts.audit_usd_cap_benchmark import EXPECTED_UNIVERSE_SIZE, _check_universe


def test_universe_gate_expects_250():
    assert EXPECTED_UNIVERSE_SIZE == 250


def test_universe_gate_accepts_250_unique():
    _check_universe([f"T{i}" for i in range(250)])  # no raise


def test_universe_gate_rejects_200_and_duplicates():
    with pytest.raises(ValueError):
        _check_universe([f"T{i}" for i in range(200)])
    with pytest.raises(ValueError):
        _check_universe(["DUP"] * 250)
