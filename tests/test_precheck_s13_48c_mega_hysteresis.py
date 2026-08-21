# -*- coding: utf-8 -*-
"""§S13.48-C mega funding_set 히스테리시스 사전점검 — plain-function 단위 테스트.

funding_set은 src/portfolio_optimizer.py:373-385의 선택 로직 미러링이므로,
임계 부등호·비유한 처리·정렬/절단이 원본과 일치하는지를 고정한다.
"""
import numpy as np

from scripts.precheck_s13_48c_mega_hysteresis import funding_set, thirds


def test_funding_set_excludes_names_below_bm_threshold():
    mu = np.array([-1.0, -2.0, -3.0])
    bm = np.array([0.05, 0.03, 0.04])          # idx1만 임계 미만
    assert funding_set(mu, bm, 0.04, 0.0, 4) == {0, 2}   # bm >= thr는 포함(등호)
    assert funding_set(mu, np.array([0.03, 0.03, 0.03]), 0.04, 0.0, 4) == set()


def test_funding_set_score_max_is_strict_inequality():
    mu = np.array([0.0, -0.1, 0.001])
    bm = np.full(3, 0.05)
    # mu == score_max는 제외(엄격부등), mu > score_max도 제외
    assert funding_set(mu, bm, 0.04, 0.0, 4) == {1}


def test_funding_set_treats_nonfinite_mu_as_zero():
    mu = np.array([np.nan, -0.1, np.inf])
    bm = np.full(3, 0.05)
    # score_max=0.0 → 대체값 0.0은 0.0 < 0.0 이 거짓이라 제외
    assert funding_set(mu, bm, 0.04, 0.0, 4) == {1}
    # score_max=0.5 → 대체값 0.0이 통과하므로 비유한 종목도 편입
    assert funding_set(mu, bm, 0.04, 0.5, 4) == {0, 1, 2}


def test_funding_set_takes_k_lowest_scores():
    mu = np.array([-0.5, -3.0, -1.0, -2.0, -4.0])
    bm = np.full(5, 0.05)
    assert funding_set(mu, bm, 0.04, 0.0, 2) == {4, 1}   # -4.0, -3.0
    assert funding_set(mu, bm, 0.04, 0.0, 4) == {4, 1, 3, 2}
    assert funding_set(mu, bm, 0.04, 0.0, 0) == set()    # k <= 0 → 빈 set


def test_thirds_splits_97_preserving_every_element():
    vals = list(range(97))
    blocks = thirds(vals)
    assert [len(b) for b in blocks] == [32, 32, 33]
    assert np.allclose(np.concatenate(blocks),
                       np.asarray(vals, dtype=float), atol=1e-6)
