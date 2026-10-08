# -*- coding: utf-8 -*-
"""§S25.3 생성기 쌍 판정 스크립트의 순수 함수 테스트 (결정 로그 §S25.3 사전등록 규칙 핀)."""

import pandas as pd

from scripts.eval_s25_3_pair import (
    EXPECTED_MASKED,
    TURNOVER_MAX,
    mechanism_model,
    mechanism_workbook,
    parse_probe_summary,
    verdicts,
)

PROBE_OUT = """old loaded 60 300
new loaded 60 600
sheet sets equal: True | only old: [] | only new: []

OTHER changes (unmasked / value-changed cells) by sheet: NONE
elapsed 700
"""


PROBE_DERIVED = PROBE_OUT.replace("NONE", "{'Summary_Stats': (0, 1583), 'iv30_z': (4614, 31375)}") + (
    'Z-SHEET CLASSIFICATION: {"iv30_z": {"within_window_changes": 35817, "beyond_window_cells": 6137, '
    '"beyond_nan_to_zero": 4614, "beyond_zero_to_nan": 1523, "beyond_other": 0, "beyond_tickers": ["6981", "7011"]}}\n')


def test_parse_probe_summary_reads_sheet_set_and_other_changes():
    out = parse_probe_summary(PROBE_OUT)
    assert out["sheet_sets_equal"] and out["other_changes_none"] and out["other_changes_sheets"] == [] and not out["shape_or_index_issues"]
    bad = parse_probe_summary(PROBE_OUT.replace("NONE", "{'PX_LAST': (0, 3)}").replace("equal: True", "equal: False"))
    assert bad["sheet_sets_equal"] is False and bad["other_changes_none"] is False
    assert bad["other_changes_sheets"] == ["PX_LAST"] and bad["other_changes_only_derived"] is False
    assert parse_probe_summary(PROBE_OUT + "[SHAPE/COLUMNS DIFFER] X\n")["shape_or_index_issues"] is True


def test_parse_probe_summary_accepts_derived_sheet_propagation_only_when_beyond_window_is_nan_zero_flip():
    out = parse_probe_summary(PROBE_DERIVED)
    assert out["other_changes_none"] is False and out["other_changes_sheets"] == ["Summary_Stats", "iv30_z"]
    assert out["other_changes_only_derived"] is True and out["z_beyond_other_zero"] is True
    worse = parse_probe_summary(PROBE_DERIVED.replace('"beyond_other": 0', '"beyond_other": 7'))
    assert worse["z_beyond_other_zero"] is False
    # a raw sheet (not derived) with value changes is never admissible
    raw = parse_probe_summary(PROBE_DERIVED.replace("'Summary_Stats'", "'BEST_PE_RATIO'"))
    assert raw["other_changes_only_derived"] is False


def _csv(rows):
    return pd.DataFrame(rows, columns=["sheet", "ticker", "masked_cells", "first_masked", "last_masked",
                                       "first_valid_new", "old_value_at_first_masked", "all_masked"])


def test_mechanism_workbook_requires_probe_clean_and_expected_cases_present():
    rows = [(s, t, 100, "2014-01-24", "2020-01-01", "2020-01-02", 1.0, s == "OPER_MARGIN") for s, t in EXPECTED_MASKED]
    clean = parse_probe_summary(PROBE_OUT)
    good = mechanism_workbook(_csv(rows), clean)
    assert good["pass"] is True and good["missing_expected"] == [] and good["total_masked_cells"] == 100 * len(EXPECTED_MASKED)
    missing = mechanism_workbook(_csv(rows[:-1]), clean)
    assert missing["pass"] is False and missing["missing_expected"] == [list(EXPECTED_MASKED[-1])]
    derived = mechanism_workbook(_csv(rows), parse_probe_summary(PROBE_DERIVED))
    assert derived["pass"] is True
    dirty = mechanism_workbook(_csv(rows), parse_probe_summary(PROBE_DERIVED.replace('"beyond_other": 0', '"beyond_other": 7')))
    assert dirty["pass"] is False
    raw_dirty = mechanism_workbook(_csv(rows), parse_probe_summary(PROBE_OUT.replace("NONE", "{'PX_LAST': (0, 3)}")))
    assert raw_dirty["pass"] is False


def test_mechanism_model_requires_identical_feature_sets():
    assert mechanism_model(["a", "b"], ["b", "a"])["pass"] is True
    out = mechanism_model(["a", "b"], ["a", "c"])
    assert out["pass"] is False and out["only_old"] == ["b"] and out["only_new"] == ["c"]


def test_verdict_frame_is_correctness_not_ir():
    assert TURNOVER_MAX == 1.25
    v = verdicts(mechanism=True, e2=True, d_ir=-0.10, split_deltas=[-0.1, 0.05, -0.2])
    assert v["no_harm_pass"] is True and v["certified"] is True
    v = verdicts(mechanism=True, e2=True, d_ir=-0.40, split_deltas=[-0.1, -0.2, -0.3])
    assert v["no_harm_pass"] is False and v["certified"] is True  # observation only: correctness fix is not rolled back
    v = verdicts(mechanism=False, e2=True, d_ir=0.5, split_deltas=[0.1, 0.2, 0.3])
    assert v["certified"] is False
