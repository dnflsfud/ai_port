# -*- coding: utf-8 -*-
"""§S16.2 — revision 지속 연장(§S15 fix-pack #5)의 길이 상한
(revision_extension_max_days) OFF 파리티·ON 동작 테스트."""
import numpy as np
import pandas as pd
import pytest

from src.config import PipelineConfig
from src.features.sellside import clean_revision_spikes

# production 파라미터 (codex_causal_rank_65 / DEFAULT_CONFIG)
_PROD = dict(threshold=15.0, mode="reversion_gated",
             extreme_threshold=50.0, reversion_ratio=0.5)


def _collapse_series(n_after=200):
    """80 유지 5행 뒤 10으로 재베이스 — 붕괴가 중간대역에 계속 머무는 케이스."""
    vals = np.concatenate([np.full(5, 80.0), np.full(n_after, 10.0)])
    dates = pd.bdate_range("2020-03-02", periods=len(vals))
    return pd.DataFrame({"A": vals}, index=dates)


def test_extension_cap_flag_default_none():
    assert PipelineConfig().revision_extension_max_days is None


# ---------------------------------------------------------------------------
# 파리티: None / 0 이하 = 인자 추가 이전 경로
# ---------------------------------------------------------------------------

def test_extension_cap_none_is_parity_with_omitted_arg():
    rev = _collapse_series()
    for ext_on in (True, False):
        omitted = clean_revision_spikes(
            rev, persistent_rollover_extension=ext_on, **_PROD)
        explicit_none = clean_revision_spikes(
            rev, persistent_rollover_extension=ext_on,
            extension_max_days=None, **_PROD)
        pd.testing.assert_frame_equal(omitted, explicit_none)
        # 0 이하도 상한 없음(기존 경로)으로 취급
        for cap in (0, -1):
            pd.testing.assert_frame_equal(
                omitted,
                clean_revision_spikes(
                    rev, persistent_rollover_extension=ext_on,
                    extension_max_days=cap, **_PROD),
            )


# ---------------------------------------------------------------------------
# 결함 재현: 상한 없으면 무기한 동결
# ---------------------------------------------------------------------------

def test_uncapped_extension_freezes_indefinitely():
    rev = _collapse_series(n_after=200)
    out = clean_revision_spikes(
        rev, persistent_rollover_extension=True,
        extension_max_days=None, **_PROD)
    # 이벤트(idx 5) 이후 200행 전부가 붕괴 이전 값 80으로 복원된다.
    assert np.allclose(out["A"].to_numpy(), 80.0)


# ---------------------------------------------------------------------------
# 상한 동작
# ---------------------------------------------------------------------------

def test_extension_cap_releases_after_max_days():
    rev = _collapse_series(n_after=200)
    out = clean_revision_spikes(
        rev, persistent_rollover_extension=True,
        extension_max_days=21, **_PROD)["A"]
    # 이벤트 행(5) + 연장 21행(6..26)까지만 80, 그 다음 행부터 실제 값이 흐른다.
    assert np.allclose(out.iloc[5:27].to_numpy(), 80.0)
    assert np.allclose(out.iloc[27:].to_numpy(), 10.0)


def test_extension_cap_allows_reentry_on_new_event():
    vals = np.concatenate([np.full(5, 80.0), np.full(30, 10.0),
                           np.full(10, 80.0), np.full(30, 10.0)])
    dates = pd.bdate_range("2020-03-02", periods=len(vals))
    rev = pd.DataFrame({"A": vals}, index=dates)
    out = clean_revision_spikes(
        rev, persistent_rollover_extension=True,
        extension_max_days=5, **_PROD)["A"]
    # 1차 이벤트(5): 5..10 마스킹 후 해제
    assert np.allclose(out.iloc[5:11].to_numpy(), 80.0)
    assert np.allclose(out.iloc[11:35].to_numpy(), 10.0)
    # 2차 이벤트(45): 상한으로 해제된 뒤에도 새 이벤트에서 연장이 재개된다.
    assert np.allclose(out.iloc[45:51].to_numpy(), 80.0)
    assert np.allclose(out.iloc[51:].to_numpy(), 10.0)


def test_base_mask_row_masked_regardless_of_cap():
    rev = _collapse_series(n_after=200)
    out = clean_revision_spikes(
        rev, persistent_rollover_extension=True,
        extension_max_days=1, **_PROD)["A"]
    # 상한이 1이어도 이벤트 당일(5)은 통과하지 않는다 — 기존 base 마스킹 유지.
    assert out.iloc[5] == pytest.approx(80.0)
    assert out.iloc[6] == pytest.approx(80.0)      # 연장 1행
    assert out.iloc[7] == pytest.approx(10.0)      # 상한 도달 -> 해제
    # 연장 자체를 끈 경우와 이벤트 당일 값이 동일함(= base 경로 불변)
    base_only = clean_revision_spikes(rev, **_PROD)["A"]
    assert base_only.iloc[5] == pytest.approx(80.0)
