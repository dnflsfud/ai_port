# -*- coding: utf-8 -*-
"""§S16.8 사전점검 헬퍼 단위테스트 (경량 — 데이터 로드 없음)."""
import numpy as np
import pandas as pd

from scripts.precheck_s16_8 import (
    daily_rank_ic,
    evaluate_precheck_b,
    nw_tstat,
    residualize_on,
)


def _frames():
    idx = pd.date_range("2026-01-01", periods=5, freq="B")
    cols = list("ABCDE")
    rng = np.random.default_rng(0)
    f = pd.DataFrame(rng.normal(size=(5, 5)), index=idx, columns=cols)
    return idx, cols, f


def test_daily_rank_ic_perfect_and_inverse():
    idx, cols, f = _frames()
    ic_pos = daily_rank_ic(f, f * 2 + 1)          # 단조 변환 → IC +1
    assert np.allclose(ic_pos.values, 1.0)
    ic_neg = daily_rank_ic(f, -f)
    assert np.allclose(ic_neg.values, -1.0)


def test_residualize_removes_score_component():
    idx, cols, f = _frames()
    score = f * 3.0                                # feature ∝ score
    resid = residualize_on(f, score)
    assert float(resid.abs().max().max()) < 1e-10  # 완전 설명 → 잔차 0
    # score 와 직교한 성분은 보존
    rng = np.random.default_rng(1)
    noise = pd.DataFrame(rng.normal(size=f.shape), index=f.index, columns=f.columns)
    resid2 = residualize_on(f + noise, f)
    daily_corr = [
        np.corrcoef(resid2.loc[d], f.loc[d])[0, 1] for d in f.index
    ]
    assert max(abs(c) for c in daily_corr) < 1e-8


def test_nw_tstat_zero_mean_series_small():
    rng = np.random.default_rng(2)
    s = pd.Series(rng.normal(0, 1, 500))
    assert abs(nw_tstat(s, lag=20)) < 3.0
    s_pos = pd.Series(rng.normal(0.5, 0.1, 500))
    assert nw_tstat(s_pos, lag=20) > 10.0


def test_evaluate_precheck_b():
    assert evaluate_precheck_b(2.5)["proceed"] is True
    assert evaluate_precheck_b(-2.5)["proceed"] is True   # 부호 무관(트리)
    assert evaluate_precheck_b(1.9)["proceed"] is False
