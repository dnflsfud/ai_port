# -*- coding: utf-8 -*-
"""§S13.28 직교화 잔여 신호 arm 순수 함수 단위테스트 (워크북/pkl I/O 없음)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_orthogonal_signal_arm import (
    CHARS, MIN_NAMES, _zscore, build_char_panels, split_signal)


def _panels(n_names=60, n_dates=4, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_dates)
    cols = [f"T{i}" for i in range(n_names)]
    chars = {
        name: pd.DataFrame(rng.normal(size=(n_dates, n_names)), index=dates, columns=cols)
        for name in CHARS
    }
    return dates, cols, chars


def test_zscore_standardises_and_winsorises():
    x = pd.DataFrame([list(np.linspace(1.0, 2.0, 99)) + [1000.0]])
    z = _zscore(x)
    np.testing.assert_allclose(z.mean(axis=1).to_numpy(), [0.0], atol=1e-12)
    np.testing.assert_allclose(z.std(axis=1, ddof=1).to_numpy(), [1.0], atol=1e-12)
    # 1/99 분위 클립이 실제로 걸려 미클립 z보다 극단값이 작아진다
    raw = x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1, ddof=1), axis=0)
    assert float(z.iloc[0, -1]) < float(raw.iloc[0, -1])


def test_residual_is_orthogonal_and_scale_preserving():
    dates, cols, chars = _panels()
    rng = np.random.default_rng(11)
    # 스코어 = 스타일 성분 + 직교 노이즈
    score = (1.5 * chars["MOM"] - 0.8 * chars["LOWVOL"]
             + pd.DataFrame(rng.normal(size=(len(dates), len(cols))),
                            index=dates, columns=cols))
    resid, fitted, diag = split_signal(score, chars)

    assert diag["dates_fitted"] == len(dates)
    assert diag["dates_passthrough"] == 0
    assert 0.0 < diag["mean_cross_sectional_r2"] < 1.0

    for dt in dates:
        # 잔여 신호는 모든 특성과 횡단면 직교 (affine 재스케일은 상관을 보존)
        for name in CHARS:
            corr = np.corrcoef(resid.loc[dt].to_numpy(), chars[name].loc[dt].to_numpy())[0, 1]
            assert abs(corr) < 1e-8, f"{dt} {name} corr={corr}"
        # 두 arm 모두 원 스코어의 평균·표준편차를 유지 (옵티마이저가 보는 스케일 불변)
        for arm in (resid, fitted):
            np.testing.assert_allclose(arm.loc[dt].mean(), score.loc[dt].mean(), atol=1e-9)
            np.testing.assert_allclose(arm.loc[dt].std(ddof=1), score.loc[dt].std(ddof=1),
                                       atol=1e-9)


def test_fully_spanned_score_passes_through_residual():
    dates, cols, chars = _panels(seed=3)
    score = 2.0 * chars["VALUE"] - 1.0 * chars["QUALITY"]   # 잔차 = 0
    resid, fitted, diag = split_signal(score, chars)
    assert diag["degenerate_rescale"] >= len(dates)
    # 잔차가 퇴화하면 원 스코어를 그대로 통과시킨다 (0으로 거래하지 않음)
    np.testing.assert_allclose(resid.to_numpy(), score.to_numpy(), atol=1e-9)
    np.testing.assert_allclose(diag["mean_cross_sectional_r2"], 1.0, atol=1e-9)


def test_thin_cross_section_and_nan_handling():
    dates, cols, chars = _panels(n_names=MIN_NAMES - 1, n_dates=2, seed=5)
    score = chars["MOM"] * 1.0
    _, _, diag = split_signal(score, chars)
    assert diag["dates_fitted"] == 0 and diag["dates_passthrough"] == 2

    # 결측 특성 셀은 중립(z=0)으로 처리되고 스코어 NaN은 보존된다
    dates, cols, chars = _panels(seed=9)
    rng = np.random.default_rng(2)
    score = pd.DataFrame(rng.normal(size=(len(dates), len(cols))), index=dates, columns=cols)
    score.iloc[0, :3] = np.nan
    chars["BAB"].iloc[0, 5:10] = np.nan
    resid, fitted, diag = split_signal(score, chars)
    assert resid.iloc[0, :3].isna().all() and fitted.iloc[0, :3].isna().all()
    assert np.isfinite(resid.iloc[0, 3:].to_numpy()).all()
    assert diag["dates_fitted"] == len(dates)


def test_build_char_panels_sign_convention():
    dates = pd.bdate_range("2024-01-02", periods=2)
    cols = [f"T{i}" for i in range(40)]
    rng = np.random.default_rng(4)
    mcap = pd.DataFrame(rng.uniform(1e9, 5e11, size=(2, 40)), index=dates, columns=cols)
    idx = pd.MultiIndex.from_product([dates, cols], names=["date", "ticker"])
    panel = pd.DataFrame(rng.normal(size=(len(idx), 2)), index=idx,
                         columns=["momentum_252d", "realized_vol_126d"])
    for col in ("best_px_bps_ratio_level_z", "best_roe_level_z", "beta_63d",
                "best_sales_chg_252d"):
        panel[col] = rng.normal(size=len(idx))

    out = build_char_panels(panel, mcap, dates, cols)
    assert set(out) == set(CHARS)
    # SMB = small minus big → 시총이 클수록 z가 낮아야 한다
    corr = np.corrcoef(out["SMB"].loc[dates[0]].to_numpy(),
                       np.log(mcap.loc[dates[0]].to_numpy()))[0, 1]
    assert corr < -0.9
    # LOWVOL = 저변동 롱 → realized_vol과 음의 상관
    vol = panel["realized_vol_126d"].unstack("ticker").loc[dates[0], cols].to_numpy()
    corr_vol = np.corrcoef(out["LOWVOL"].loc[dates[0]].to_numpy(), vol)[0, 1]
    assert corr_vol < -0.9
