# -*- coding: utf-8 -*-
"""§S13.45-D ndcg@k 사용률 감사 사전점검 — plain-function 단위 테스트."""
import numpy as np
import pandas as pd

from scripts.precheck_s13_45d_rank_usage import (bucket_contribs,
                                                 bucket_shares, mu_ranks,
                                                 thirds_signs)


def test_mu_ranks_descending_top_and_ascending_bottom():
    mu = pd.Series({"A": 3.0, "B": 1.0, "C": 2.0, "D": np.nan})
    top, bottom = mu_ranks(mu)
    assert list(top.sort_index().values) == [1, 3, 2]      # A=1위, B=3위, C=2위
    assert list(bottom.sort_index().values) == [3, 1, 2]   # B가 하위 1위
    assert "D" not in top.index                            # NaN mu는 랭크 제외


def test_bucket_shares_ow_uw_and_unranked_sum_to_one():
    names = [f"T{i}" for i in range(30)] + ["NR"]           # NR = mu NaN 종목
    mu = pd.Series(np.arange(30, 0, -1, dtype=float), index=names[:30])
    mu["NR"] = np.nan
    top, bottom = mu_ranks(mu)
    active = pd.Series(0.0, index=names)
    active["T0"] = 0.02    # top-rank 1 (버킷 1-5)
    active["T7"] = 0.03    # top-rank 8 (버킷 6-20)
    active["NR"] = 0.05    # 랭크 없음 → unranked
    active["T29"] = -0.04  # bottom-rank 1 (UW 버킷 1-5)
    ow = bucket_shares(active, top, "ow")
    uw = bucket_shares(active, bottom, "uw")
    assert abs(ow["gross"] - 0.10) < 1e-12
    assert abs(ow["shares"]["1-5"] - 0.2) < 1e-12
    assert abs(ow["shares"]["6-20"] - 0.3) < 1e-12
    assert abs(ow["shares"]["unranked"] - 0.5) < 1e-12
    assert abs(sum(ow["shares"].values()) - 1.0) < 1e-12
    assert abs(uw["gross"] - 0.04) < 1e-12
    assert abs(uw["shares"]["1-5"] - 1.0) < 1e-12          # UW 전액이 하위 1-5


def test_bucket_shares_zero_gross_returns_nan():
    mu = pd.Series({"A": 2.0, "B": 1.0})
    top, _ = mu_ranks(mu)
    ow = bucket_shares(pd.Series({"A": -0.01, "B": 0.0}), top, "ow")
    assert ow["gross"] == 0.0
    assert np.isnan(ow["shares"]["1-5"])


def test_bucket_contribs_signed_active_decomposes_active_return():
    mu = pd.Series({f"T{i}": float(10 - i) for i in range(10)})  # T0=1위 … T9=10위
    top, _ = mu_ranks(mu)
    active = pd.Series(0.0, index=mu.index)
    active["T0"] = 0.02    # 버킷 1-5 OW
    active["T2"] = -0.01   # 버킷 1-5 UW — 부호 그대로 상쇄돼야 함
    active["T6"] = 0.03    # 버킷 6-20 OW
    fwd = pd.Series(0.10, index=mu.index)
    c = bucket_contribs(active, top, fwd)
    assert abs(c["1-5"] - (0.02 - 0.01) * 0.10) < 1e-12
    assert abs(c["6-20"] - 0.03 * 0.10) < 1e-12
    assert abs(c["21-50"]) < 1e-12
    # 전 버킷 합 = Σ active×fwd (실현 active 수익 분해)
    assert abs(sum(c.values()) - float((active * fwd).sum())) < 1e-12
    # fwd NaN 종목은 기여에서 제외
    fwd2 = fwd.copy()
    fwd2["T6"] = np.nan
    assert abs(bucket_contribs(active, top, fwd2)["6-20"]) < 1e-12


def test_thirds_signs_time_ordered_blocks():
    vals = [1.0, 1.0, 1.0, -2.0, -2.0, -2.0, 0.5, 0.5, 0.5]
    assert thirds_signs(vals) == [1, -1, 1]
    assert thirds_signs([1.0, -3.0, 2.0]) == [1, -1, 1]
