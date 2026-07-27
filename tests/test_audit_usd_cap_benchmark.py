# -*- coding: utf-8 -*-
"""audit_usd_cap_benchmark 유니버스 게이트 계약 테스트 (§S13.3: 150→200)."""

import pytest

from scripts.audit_usd_cap_benchmark import EXPECTED_UNIVERSE_SIZE, _check_universe


def test_universe_gate_expects_200():
    assert EXPECTED_UNIVERSE_SIZE == 200


def test_universe_gate_accepts_200_unique():
    _check_universe([f"T{i}" for i in range(200)])  # no raise


def test_universe_gate_rejects_150_and_duplicates():
    with pytest.raises(ValueError):
        _check_universe([f"T{i}" for i in range(150)])
    with pytest.raises(ValueError):
        _check_universe(["DUP"] * 200)
