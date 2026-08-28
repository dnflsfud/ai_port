# -*- coding: utf-8 -*-
"""§S13.41 옵션 IV 변동성 예측 → 공분산 대각 스케일 모듈 테스트."""
import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.option_vol_cov import (
    CLIP_HI,
    CLIP_LO,
    FIRST_EST_POS,
    build_option_vol_scale,
)


def _synthetic(n_days=700, n_names=40, seed=7, diverge_after=None):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    cols = [f"T{i:02d}" for i in range(n_names)]
    vols = rng.uniform(0.01, 0.03, n_names)
    ret = pd.DataFrame(rng.normal(0.0, vols, (n_days, n_names)), idx, cols)
    z = pd.DataFrame(rng.normal(0.0, 1.0, (n_days, n_names)), idx, cols)
    if diverge_after is not None:
        ret.iloc[diverge_after:] += 0.05
        z.iloc[diverge_after:] += 3.0
    return ret, z


def test_flag_default_off():
    assert PipelineConfig().option_vol_covariance_enabled is False


def test_scale_panel_shape_clip_and_warmup():
    ret, z = _synthetic()
    s = build_option_vol_scale(ret, z)
    assert s.shape == ret.shape
    assert (s.iloc[:FIRST_EST_POS] == 1.0).all().all()  # 최초 추정 전 inert
    assert float(s.values.min()) >= CLIP_LO and float(s.values.max()) <= CLIP_HI
    # 추정 이후 구간은 실제로 1.0이 아닌 값이 존재해야 함 (모델이 작동)
    assert (s.iloc[FIRST_EST_POS + 63:] != 1.0).any().any()


def test_scale_nan_inputs_are_inert():
    ret, z = _synthetic()
    z.iloc[:, 0] = np.nan  # 첫 종목 z 전결측
    s = build_option_vol_scale(ret, z)
    assert (s.iloc[:, 0] == 1.0).all()


def test_causality_future_data_does_not_change_past_scale():
    ret_a, z_a = _synthetic()
    ret_b, z_b = _synthetic(diverge_after=450)
    s_a = build_option_vol_scale(ret_a, z_a)
    s_b = build_option_vol_scale(ret_b, z_b)
    # 450 이전 행은 (모델·입력 모두 과거 동일이므로) 바이트 동일해야 한다
    pd.testing.assert_frame_equal(s_a.iloc[:450], s_b.iloc[:450])
    # 미래 발산은 미래 행에는 반영된다 (온전성)
    assert not s_a.iloc[500:].equals(s_b.iloc[500:])


# ---------------------------------------------------------------------------
# 구조 리뷰 2026-08-27 — option_vol_scale_fix_enabled.
#
# (b) 커버리지 가드: 모듈 계약은 "커버리지 밖은 1.0"이지만, 로더의
# _fill_missing(ffill -> 날짜별 횡단면 median)이 iv30_z의 결측을 전부 메우기
# 때문에 열이 통째로 없을 때만 가드가 발동한다(실측 non-inert 86.1%).
# observed_mask를 주면 임퓨트된 셀이 다시 NaN이 되어 inert로 수렴한다.
# ---------------------------------------------------------------------------
def test_optvol_scale_fix_flag_default_off():
    assert PipelineConfig().option_vol_scale_fix_enabled is False


def test_observed_mask_none_is_byte_identical():
    """OFF 파리티: observed_mask=None은 기존 호출과 바이트 동일."""
    ret, z = _synthetic()
    pd.testing.assert_frame_equal(
        build_option_vol_scale(ret, z),
        build_option_vol_scale(ret, z, observed_mask=None),
    )


def test_observed_mask_forces_imputed_cells_inert():
    """로더 median-fill을 재현한 뒤 마스크를 주면 그 종목이 inert가 된다."""
    ret, z = _synthetic()
    # 첫 두 종목은 실제로는 미관측인데, 로더가 횡단면 median으로 메운 상태.
    imputed = z.copy()
    row_median = z.iloc[:, 2:].median(axis=1)
    for col in imputed.columns[:2]:
        imputed[col] = row_median

    # 마스크 없이는 (계약과 달리) inert가 아니다 — 이것이 실제 production 상태.
    unguarded = build_option_vol_scale(ret, imputed)
    assert (unguarded.iloc[FIRST_EST_POS + 63:, :2] != 1.0).any().any()

    mask = pd.DataFrame(True, index=z.index, columns=z.columns)
    mask.iloc[:, :2] = False
    guarded = build_option_vol_scale(ret, imputed, observed_mask=mask)
    # 미관측 종목은 전 구간 inert
    assert (guarded.iloc[:, :2] == 1.0).all().all()
    # 관측된 종목의 스케일은 살아 있다 (가드가 채널을 죽이지 않는다)
    assert (guarded.iloc[FIRST_EST_POS + 63:, 2:] != 1.0).any().any()
    # 그리고 임퓨트 행이 풀드 OLS 학습 표본에서 빠졌으므로 계수가 달라져
    # 나머지 종목의 스케일도 이동한다 — 이것이 (a) 오염 제거의 실증이다.
    assert not guarded.iloc[:, 2:].equals(
        build_option_vol_scale(ret, imputed).iloc[:, 2:]
    )


def test_observed_mask_partial_rows_are_inert():
    """열 단위가 아니라 셀 단위 결측(초기 이력 없음)도 inert로 수렴한다."""
    ret, z = _synthetic()
    mask = pd.DataFrame(True, index=z.index, columns=z.columns)
    mask.iloc[:600, 0] = False  # 첫 종목은 600행까지 옵션 데이터 없음
    s = build_option_vol_scale(ret, z, observed_mask=mask)
    assert (s.iloc[:600, 0] == 1.0).all()


def test_backtest_wires_both_fix_legs_behind_the_flag():
    """배선 확인 (house style: run_variant 계열 acceptance와 동일한 소스 단언).

    두 레그가 모두 플래그 뒤에 있어야 한다 — (a) risk_returns 원천 교체,
    (b) raw_sheet_observed_mask 커버리지 가드. 한쪽만 배선되면 계약이
    반만 복구되므로 여기서 실패시킨다."""
    import inspect

    from src.backtest import run_backtest

    src_text = inspect.getsource(run_backtest)
    assert "option_vol_scale_fix_enabled" in src_text
    assert "raw_sheet_observed_mask" in src_text
    assert "observed_mask=_optvol_mask" in src_text
    # OFF 경로가 여전히 dense 패널을 쓰는지 (파리티 근거)
    assert "_optvol_src = returns[tickers]" in src_text


def test_production_variant_does_not_enable_the_fix_yet():
    """§8: production flip은 사용자 결정 사항. 측정 전에는 OFF여야 한다."""
    import yaml

    with open("variants/codex_causal_rank_65.yaml", encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh)
    overrides = manifest.get("overrides") or {}
    assert overrides.get("option_vol_scale_fix_enabled") in (None, False)
    assert overrides.get("calendar_exempt_sheets_enabled") in (None, False)
