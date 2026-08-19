# -*- coding: utf-8 -*-
"""§S13.44 vol-spike × 리비전 방향 사전점검 — plain-function 단위 테스트."""
import numpy as np
import pandas as pd

from scripts.precheck_s13_44_volspike_rev import (group_spread_row,
                                                  idio_vol_spike_z,
                                                  residual_ic_row,
                                                  sparse_signal_row)


def test_idio_vol_spike_z_flags_recent_burst():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2020-01-01", periods=500)
    cols = [f"T{i}" for i in range(19)] + ["A"]
    rets = pd.DataFrame(rng.normal(0, 0.01, (500, 20)), index=idx, columns=cols)
    # A만 마지막 21일 변동성 5배 폭증 (EW mkt 오염은 1/20로 희석)
    rets.iloc[-21:, -1] = rng.normal(0, 0.05, 21)
    z = idio_vol_spike_z(rets)
    # 번인: β(62) + vol(+20) + z min126(+125) → 첫 유효 행 = 207
    assert z.iloc[:207].isna().all().all()
    assert z.iloc[207].notna().all()
    assert float(z["A"].iloc[-1]) > 2.0          # 폭증 종목은 강한 spike
    assert float(z["T0"].iloc[-1]) < 1.0         # 평온 종목은 spike 아님


def test_sparse_signal_row_maps_spike_and_rev_sign():
    spike = pd.Series({"A": 1.5, "B": 0.5, "C": 2.0, "D": 1.2, "E": np.nan})
    rev = pd.Series({"A": 0.3, "B": 0.4, "C": -0.1, "D": 0.0, "E": 0.2})
    s = sparse_signal_row(spike, rev, pd.Index(list("ABCDE")))
    assert float(s["A"]) == 1.0     # spike ∩ 상향
    assert float(s["B"]) == 0.0     # spike 아님
    assert float(s["C"]) == -1.0    # spike ∩ 하향
    assert float(s["D"]) == 0.0     # rev == 0 → 제외
    assert float(s["E"]) == 0.0     # spike NaN → 제외


def test_group_spread_row_exact_means_and_counts():
    s = pd.Series({"A": 1.0, "B": 1.0, "C": -1.0, "D": 0.0})
    fwd = pd.Series({"A": 0.04, "B": 0.02, "C": -0.01, "D": 0.10})
    row = group_spread_row(s, fwd)
    assert row["n_spike"] == 3 and row["n_up"] == 2 and row["n_down"] == 1
    assert abs(row["mean_up"] - 0.03) < 1e-12
    assert abs(row["mean_down"] + 0.01) < 1e-12
    assert abs(row["spread"] - 0.04) < 1e-12
    # 한쪽 그룹 부재 → 스프레드 NaN
    only_up = pd.Series({"A": 1.0, "B": 0.0})
    assert np.isnan(group_spread_row(only_up, fwd)["spread"])


def test_residual_ic_row_survives_orthogonal_signal_only():
    rng = np.random.default_rng(11)
    names = pd.Index([f"T{i}" for i in range(60)])
    score = pd.Series(rng.normal(0, 1, 60), index=names)
    sig = pd.Series(0.0, index=names)
    sig.iloc[:10], sig.iloc[10:20] = 1.0, -1.0   # score와 독립인 희소 신호
    fwd = sig * 0.02 + pd.Series(rng.normal(0, 0.001, 60), index=names)
    row = residual_ic_row(sig, score, fwd)
    assert row["resid_ic"] > 0.5                 # 직교 정보는 잔차에서 생존
    # score만이 수익을 설명하면(신호 증분 0) 잔차 IC ≈ 0
    fwd2 = score * 0.02 + pd.Series(rng.normal(0, 0.001, 60), index=names)
    noinfo = residual_ic_row(sig, score, fwd2)
    assert abs(noinfo["resid_ic"]) < 0.25
