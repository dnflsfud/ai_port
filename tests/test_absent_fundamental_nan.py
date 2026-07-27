# -*- coding: utf-8 -*-
"""§S13.6 은행권 구조적 결측 피처 NaN 보존 — 단위 계약.

BEST_CALCULATED_FCF / BEST_CAPEX / BEST_EV_TO_BEST_EBITDA / BEST_GROSS_MARGIN 은
은행권에서 컬럼 자체가 부재한다(banks: no FCF/capex/EBITDA/gross margin).
현재는 assembly 패널 단계의 per-date median fill 이 이 칸을 "시장 중앙값"으로
날조한다. 플래그 ON 이면 해당 (티커 × 피처) 칸만 NaN 으로 남겨 LightGBM 의
네이티브 결측 처리에 위임한다.

OFF 일 때는 무연산이어야 한다(§2.1 parity).
"""

import numpy as np
import pandas as pd

from src.config import (
    ABSENT_FUNDAMENTAL_SHEET_FEATURES,
    DEFAULT_CONFIG,
    NAN_TOLERANT_FEATURES,
    PipelineConfig,
)


def _panel(tickers=("JPM", "WFC", "AAPL"), n_dates=4):
    """(date, ticker) MultiIndex 패널 — 전 칸이 채워진 상태(현행 fill 이후)."""
    dates = pd.date_range("2026-01-05", periods=n_dates, freq="B")
    idx = pd.MultiIndex.from_product([dates, list(tickers)], names=["date", "ticker"])
    cols = [
        "best_calculated_fcf_level_z",
        "cash_conversion_z",
        "best_capex_level_z",
        "capex_intensity_z",
        "best_ev_to_best_ebitda_level_z",
        "best_gross_margin_chg_63d",
        "best_gross_margin_chg_252d",
        "op_leverage_63d",
        "momentum_252d",
    ]
    values = np.arange(len(idx) * len(cols), dtype=float).reshape(len(idx), len(cols))
    return pd.DataFrame(values, index=idx, columns=cols)


def test_flag_defaults_off():
    assert DEFAULT_CONFIG.absent_fundamental_nan_enabled is False
    assert PipelineConfig().absent_fundamental_nan_enabled is False


def test_sheet_feature_map_covers_eight_whitelisted_features():
    flat = [f for feats in ABSENT_FUNDAMENTAL_SHEET_FEATURES.values() for f in feats]
    assert len(flat) == 8
    assert len(set(flat)) == 8
    assert set(flat) == set(NAN_TOLERANT_FEATURES)
    assert set(ABSENT_FUNDAMENTAL_SHEET_FEATURES) == {
        "BEST_CALCULATED_FCF",
        "BEST_CAPEX",
        "BEST_EV_TO_BEST_EBITDA",
        "BEST_GROSS_MARGIN",
    }


def test_mask_off_is_byte_identical():
    from src.features.assembly import apply_absent_fundamental_nan

    panel = _panel()
    optional_missing = {"BEST_CALCULATED_FCF": ["JPM", "WFC"]}
    out = apply_absent_fundamental_nan(panel.copy(), optional_missing, enabled=False)
    assert out.notna().all().all()
    assert np.allclose(out.values, panel.values, atol=1e-12)


def test_mask_on_nans_only_absent_ticker_feature_pairs():
    from src.features.assembly import apply_absent_fundamental_nan

    panel = _panel()
    optional_missing = {
        "BEST_CALCULATED_FCF": ["JPM", "WFC"],
        "BEST_GROSS_MARGIN": ["WFC"],
    }
    out = apply_absent_fundamental_nan(panel.copy(), optional_missing, enabled=True)

    fcf_feats = ["best_calculated_fcf_level_z", "cash_conversion_z"]
    gm_feats = ["best_gross_margin_chg_63d", "best_gross_margin_chg_252d", "op_leverage_63d"]

    # JPM: FCF 유래 2개만 NaN
    jpm = out.xs("JPM", level="ticker")
    assert jpm[fcf_feats].isna().all().all()
    assert jpm[gm_feats].notna().all().all()

    # WFC: FCF 2개 + GROSS_MARGIN 3개 = 5개 NaN
    wfc = out.xs("WFC", level="ticker")
    assert wfc[fcf_feats + gm_feats].isna().all().all()
    assert wfc["momentum_252d"].notna().all()

    # AAPL: 부재 없음 -> 전 칸 보존
    aapl = out.xs("AAPL", level="ticker")
    assert aapl.notna().all().all()
    src_aapl = panel.xs("AAPL", level="ticker")
    assert np.allclose(aapl.values, src_aapl.values, atol=1e-12)

    # 마스킹 대상 밖 시트는 무영향
    assert out["best_capex_level_z"].notna().all()
    assert out["best_ev_to_best_ebitda_level_z"].notna().all()


def test_mask_on_ignores_sheets_outside_scope():
    from src.features.assembly import apply_absent_fundamental_nan

    panel = _panel()
    # BEST_PEG_RATIO / BEST_PX_BPS_RATIO 는 이번 범위가 아니다.
    out = apply_absent_fundamental_nan(
        panel.copy(), {"BEST_PEG_RATIO": ["JPM"], "BEST_PX_BPS_RATIO": ["WFC"]}, enabled=True
    )
    assert out.notna().all().all()


def test_train_mask_keeps_rows_whose_only_nan_is_tolerated():
    """listwise deletion 이 은행권을 학습에서 통째로 지우면 안 된다."""
    from src.model_trainer import _prepare_train_data

    panel = _panel(tickers=("JPM", "AAPL"), n_dates=3)
    panel.loc[panel.index.get_level_values("ticker") == "JPM",
              "best_calculated_fcf_level_z"] = np.nan

    dates = panel.index.get_level_values("date").unique()
    targets = pd.DataFrame(1.0, index=dates, columns=["JPM", "AAPL"])

    X, y = _prepare_train_data(panel, targets, list(panel.columns), dates)
    assert len(X) == len(panel)          # JPM 행이 살아 있어야 한다
    assert np.isnan(X).sum() == 3        # 관용 피처의 NaN 은 모델로 전달


def test_train_mask_still_drops_rows_with_required_nan():
    from src.model_trainer import _prepare_train_data

    panel = _panel(tickers=("JPM", "AAPL"), n_dates=3)
    panel.loc[panel.index.get_level_values("ticker") == "JPM", "momentum_252d"] = np.nan

    dates = panel.index.get_level_values("date").unique()
    targets = pd.DataFrame(1.0, index=dates, columns=["JPM", "AAPL"])

    X, y = _prepare_train_data(panel, targets, list(panel.columns), dates)
    assert len(X) == 3                   # AAPL 만 남는다
    assert not np.isnan(X).any()
