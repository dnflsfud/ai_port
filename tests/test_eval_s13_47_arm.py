# -*- coding: utf-8 -*-
"""§S13.47 E1 판정 스크립트의 순수 헬퍼 테스트."""
from scripts.eval_s13_47_arm import degeneracy_comparison


def test_degeneracy_comparison_worsened():
    base = {"degenerate_retrains": 13, "total_retrains": 33}
    arm = {"degenerate_retrains": 20, "total_retrains": 33}
    out = degeneracy_comparison(base, arm)
    assert out["rate_base"] == 13 / 33
    assert out["rate_arm"] == 20 / 33
    assert out["worsened"] is True


def test_degeneracy_comparison_improved_or_equal():
    base = {"degenerate_retrains": 13, "total_retrains": 33}
    assert degeneracy_comparison(
        base, {"degenerate_retrains": 10, "total_retrains": 33})["worsened"] is False
    assert degeneracy_comparison(
        base, {"degenerate_retrains": 13, "total_retrains": 33})["worsened"] is False
