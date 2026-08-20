# -*- coding: utf-8 -*-
"""§S13.46 E1 판정 스크립트의 순수 헬퍼 테스트."""
import numpy as np
import pandas as pd

from scripts.eval_s13_46_arm import E1_DELTA_IR, _ir, evaluate_e1


def test_ir_annualization():
    s = pd.Series([0.001] * 100 + [0.002] * 100)
    expected = s.mean() / s.std() * np.sqrt(252.0)
    assert abs(_ir(s) - expected) < 1e-12


def test_ir_zero_std_is_nan():
    assert np.isnan(_ir(pd.Series([0.001] * 50)))


def test_e1_gate_requires_bar_and_sign_consistency():
    assert evaluate_e1(E1_DELTA_IR + 0.01, [0.1, 0.2, 0.3]) is True
    assert evaluate_e1(E1_DELTA_IR + 0.01, [0.1, -0.01, 0.3]) is False  # 부호 비일관
    assert evaluate_e1(E1_DELTA_IR, [0.1, 0.2, 0.3]) is False  # 바 초과 아님(>)
    assert evaluate_e1(0.0, [0.1, 0.2, 0.3]) is False
