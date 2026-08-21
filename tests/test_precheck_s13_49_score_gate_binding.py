# -*- coding: utf-8 -*-
"""§S13.49 score-gate 바인딩 사전점검 — plain-function 단위 테스트."""
import numpy as np

from scripts.precheck_s13_49_score_gate_binding import (binding_share,
                                                        gate_targets,
                                                        gate_width)


def test_gate_targets_includes_score_equal_to_threshold():
    # portfolio_optimizer.py:421은 `score_i <= score_threshold` — 등호 포함.
    mu = np.array([-0.5, 0.0, 1e-12, 0.3])
    bm = np.zeros(4)
    mask = gate_targets(mu, bm, threshold=0.0, mega_bm_thr=0.04)
    assert mask.tolist() == [True, True, False, False]


def test_gate_targets_excludes_mega_names():
    mu = np.array([-1.0, -1.0, -1.0])
    bm = np.array([0.039999, 0.04, 0.08])   # 두 번째부터 mega (>= 임계, 등호 포함)
    mask = gate_targets(mu, bm, threshold=0.0, mega_bm_thr=0.04)
    assert mask.tolist() == [True, False, False]


def test_gate_targets_excludes_nonfinite_mu():
    mu = np.array([np.nan, np.inf, -np.inf, -0.2])
    bm = np.zeros(4)
    mask = gate_targets(mu, bm, threshold=0.0, mega_bm_thr=0.04)
    assert mask.tolist() == [False, False, False, True]


def test_binding_share_counts_tolerance_and_returns_nan_on_empty_mask():
    w = np.array([0.010000_0000, 0.0100005, 0.02, 0.005])
    bm = np.array([0.01, 0.01, 0.01, 0.01])
    mask = np.array([True, True, True, False])
    # |w-bm| = [0, 5e-7, 1e-2] -> tol 1e-6 안이 2/3
    assert np.isclose(binding_share(w, bm, mask, tol=1e-6), 2.0 / 3.0, atol=1e-12)
    assert binding_share(w, bm, np.zeros(4, dtype=bool), tol=1e-6) != \
        binding_share(w, bm, np.zeros(4, dtype=bool), tol=1e-6)  # nan != nan


def test_gate_width_excludes_nonfinite_mu_from_denominator():
    mu = np.array([-1.0, 0.0, 0.5, np.nan, np.inf])
    # 유효 3개 중 mu <= 0 은 2개
    assert np.isclose(gate_width(mu, threshold=0.0), 2.0 / 3.0, atol=1e-12)
    assert np.isnan(gate_width(np.array([np.nan, np.nan]), threshold=0.0))
