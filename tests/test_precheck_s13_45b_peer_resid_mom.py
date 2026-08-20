# -*- coding: utf-8 -*-
"""§S13.45-B 피어 잔차 모멘텀 사전점검 — plain-function 단위 테스트."""
import numpy as np
import pandas as pd

from scripts.precheck_s13_45b_peer_resid_mom import (peer_sets,
                                                     peer_signal,
                                                     residual_returns,
                                                     rolling_beta)
from scripts.precheck_s13_44_volspike_rev import residual_ic_row


def _bm_and_index(n=300, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    bm = pd.Series(rng.normal(0, 0.01, n), index=idx)
    return rng, idx, bm


def test_rolling_beta_recovers_constant_beta_after_burnin():
    rng, idx, bm = _bm_and_index()
    rets = pd.DataFrame({"A": 2.0 * bm + rng.normal(0, 0.001, len(idx)),
                         "B": 0.5 * bm + rng.normal(0, 0.001, len(idx))})
    beta = rolling_beta(rets, bm, window=126, min_periods=63)
    # min_periods=63 → 첫 유효 행 = 62 (63번째 관측)
    assert beta.iloc[:62].isna().all().all()
    assert beta.iloc[62:].notna().all().all()
    assert abs(float(beta["A"].iloc[-1]) - 2.0) < 0.1
    assert abs(float(beta["B"].iloc[-1]) - 0.5) < 0.1


def test_residual_returns_zero_for_pure_beta_stock():
    _, idx, bm = _bm_and_index()
    rets = pd.DataFrame({"A": 1.5 * bm})
    resid = residual_returns(rets, bm, window=126, min_periods=63)
    # r = 1.5·bm 정확히 → β=1.5, 잔차 ≈ 0 (번인 후)
    assert resid["A"].iloc[62:].abs().max() < 1e-10
    assert resid["A"].iloc[:62].isna().all()


def test_peer_sets_top_correlated_excludes_self_and_low_obs():
    rng = np.random.default_rng(11)
    n = 126
    f, g = rng.normal(0, 1, n), rng.normal(0, 1, n)
    df = pd.DataFrame({
        "A1": f + 0.1 * rng.normal(0, 1, n),
        "A2": f + 0.1 * rng.normal(0, 1, n),
        "A3": f + 0.1 * rng.normal(0, 1, n),
        "B1": g + 0.1 * rng.normal(0, 1, n),
        "B2": g + 0.1 * rng.normal(0, 1, n),
        "C": f,
    })
    df.loc[df.index[:76], "C"] = np.nan  # C는 유효 관측 50 < 100 → 후보 탈락
    peers = peer_sets(df, top_n=2, min_pair_obs=100)
    assert set(peers["A1"]) == {"A2", "A3"}   # 같은 팩터 그룹이 상위
    assert "A1" not in peers["A1"]            # 자기 제외
    assert all("C" not in p for p in peers.values())
    assert peers["C"] == []                   # C 자신도 피어 없음


def test_peer_signal_averages_available_peers_nan_when_none():
    mom = pd.Series({"A": 9.0, "B": 0.1, "C": 0.3, "F": np.nan})
    s = peer_signal({"A": ["B", "C"], "D": [], "E": ["B", "F"]}, mom)
    assert abs(float(s["A"]) - 0.2) < 1e-12   # 자기 mom 무시, 피어 평균
    assert np.isnan(float(s["D"]))            # 피어 0 → NaN
    assert abs(float(s["E"]) - 0.1) < 1e-12   # NaN 피어 제외 nanmean


def test_two_sided_partial_spearman_with_nan_signal():
    # §S13.44 residual_ic_row 미러가 NaN 보유 연속 신호에서도 성립하는지
    rng = np.random.default_rng(7)
    n = 120
    names = pd.Index([f"T{i}" for i in range(n)])
    score = pd.Series(rng.normal(0, 1, n), index=names)
    sig = pd.Series(rng.normal(0, 1, n), index=names)
    sig.iloc[90:] = np.nan                    # 번인 종목의 NaN 신호
    fwd = (sig.fillna(0) * 0.02
           + pd.Series(rng.normal(0, 0.001, n), index=names))
    row = residual_ic_row(sig, score, fwd)
    assert row["resid_ic"] > 0.5              # 직교 정보는 잔차에서 생존
    fwd2 = score * 0.02 + pd.Series(rng.normal(0, 0.001, n), index=names)
    noinfo = residual_ic_row(sig, score, fwd2)
    assert abs(noinfo["resid_ic"]) < 0.25     # score만이 설명하면 잔차 IC ≈ 0
