"""src/harness.py sub-period IR coverage.

P1/P2/P3 boundaries are frozen for comparability with every prior report.
Portfolio returns now extend past the P3 endpoint (2026-04-13), so
sub_period_irs additionally emits a 'P4_tail_ir' entry over
(P3_end, port_end] plus 'sub_period_coverage_end' — additive keys only,
P1-P3 values stay bit-identical.
"""

import numpy as np
import pandas as pd
import pytest

from src.harness import SUB_PERIODS, sub_ir, sub_period_irs


def _series(end):
    idx = pd.bdate_range("2018-11-23", end)
    rng = np.random.default_rng(7)
    port = pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx)
    bm = pd.Series(rng.normal(0.0004, 0.009, len(idx)), index=idx)
    return port, bm


def test_sub_period_boundaries_frozen():
    assert SUB_PERIODS == {
        "P1": ("2018-11-23", "2021-05-11"),
        "P2": ("2021-05-12", "2023-10-27"),
        "P3": ("2023-10-30", "2026-04-13"),
    }


def test_no_tail_keys_when_series_ends_within_p3():
    port, bm = _series("2026-04-13")
    out = sub_period_irs(port, bm)
    assert set(out) == {"P1_ir", "P2_ir", "P3_ir"}


def test_tail_added_and_p1_p3_bit_identical():
    port, bm = _series("2026-08-25")
    out = sub_period_irs(port, bm)
    # P1-P3 bit-identical to a direct frozen-window computation
    for label, (start, end) in SUB_PERIODS.items():
        assert out[f"{label}_ir"] == sub_ir(port, bm, start, end)
    # ... and to the dict computed on the series truncated at the P3 end
    short = sub_period_irs(port[port.index <= "2026-04-13"], bm)
    for key, value in short.items():
        assert out[key] == value
    # tail window is (P3_end, port_end], same IR formula as sub_ir
    active = (port - bm)[port.index > pd.Timestamp("2026-04-13")]
    expected = float(active.mean() / active.std(ddof=1) * np.sqrt(252))
    assert out["P4_tail_ir"] == pytest.approx(expected, rel=1e-12)
    assert out["sub_period_coverage_end"] == port.index.max().strftime("%Y-%m-%d")
