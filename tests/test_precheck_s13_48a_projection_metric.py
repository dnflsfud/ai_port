# -*- coding: utf-8 -*-
"""§S13.48-A 균일 클로백 사전점검 — plain-function 단위 테스트."""
import numpy as np

from scripts.precheck_s13_48a_projection_metric import (clawback_cluster,
                                                        split_by_mu, thirds)


def test_clawback_cluster_finds_modal_block():
    delta = np.array([0.5, 0.2, 0.2, 0.2, 0.7, np.nan, 0.7])
    c, m, mask = clawback_cluster(delta)
    assert np.allclose(c, 0.2, atol=1e-6)
    assert m == 3
    assert list(mask) == [False, True, True, True, False, False, False]


def test_clawback_cluster_merges_ninth_decimal_noise():
    delta = np.array([-0.000140083, -0.000140084, -0.000140085, 0.004, -0.003])
    c, m, mask = clawback_cluster(delta, decimals=7)
    assert np.allclose(c, -0.0001401, atol=1e-6)
    assert m == 3
    assert list(mask) == [True, True, True, False, False]


def test_split_by_mu_drops_median_member_when_odd():
    mu = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
    mask = np.array([True] * 5)
    lo, hi = split_by_mu(mu, mask)
    assert len(lo) == len(hi) == 2                    # m=5 -> h=2, 중앙 1개 폐기
    assert not set(lo.tolist()) & set(hi.tolist())    # 서로소
    assert sorted(lo.tolist()) == [1, 3]              # mu 1.0, 2.0
    assert sorted(hi.tolist()) == [0, 4]              # mu 5.0, 4.0
    assert 2 not in lo.tolist() and 2 not in hi.tolist()  # 중앙(mu 3.0) 폐기


def test_split_by_mu_excludes_non_finite_mu_members():
    mu = np.array([1.0, np.nan, 2.0, 3.0, np.inf, 4.0])
    mask = np.array([True] * 6)
    lo, hi = split_by_mu(mu, mask)
    assert len(lo) == len(hi) == 2                    # 유효 4개 -> h=2
    members = set(lo.tolist()) | set(hi.tolist())
    assert 1 not in members and 4 not in members      # NaN/inf 멤버 제외
    assert sorted(lo.tolist()) == [0, 2]
    assert sorted(hi.tolist()) == [3, 5]


def test_thirds_partitions_97_and_preserves_all_elements():
    vals = [float(i) for i in range(97)]
    blocks = thirds(vals)
    assert [len(b) for b in blocks] == [32, 32, 33]
    assert np.allclose(np.concatenate(blocks), np.asarray(vals), atol=1e-6)
