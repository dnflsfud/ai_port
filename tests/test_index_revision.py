# -*- coding: utf-8 -*-
"""§S13.23 국가-매핑 지수 리비전 피처 테스트 (default-OFF·게이트·매핑)."""

from types import SimpleNamespace

import pandas as pd

from scripts.preflight_s13_23_index_rev_mapping import (
    EXCHANGE_TO_REV as PREFLIGHT_MAPPING,
)
from src.data_loader import ALL_FACTOR_COLUMNS
from src.features.index_revision import (
    EXCHANGE_TO_REV,
    INDEX_REVISION_FEATURES,
    admitted_index_revision_features,
    build_index_revision_features,
)


def _synthetic_data():
    dates = pd.date_range("2024-01-01", periods=4, freq="B")
    factor_px = pd.DataFrame(
        {
            "SPX_REV": [1.0, 2.0, 3.0, 4.0],
            "CAC_REV": [10.0, 20.0, 30.0, 40.0],
            "DAX_REV": [5.0, 6.0, 7.0, 8.0],
            "SX5E_REV": [0.1, 0.2, 0.3, 0.4],
            "JPN_REV": [-1.0, -2.0, -3.0, -4.0],
        },
        index=dates,
    )
    tickers = ["AAA", "BBB", "CCC"]
    meta = pd.DataFrame(
        {"exchange_code": ["US", "FP", "JP"]},
        index=pd.Index(tickers, name="ticker"),
    )
    return SimpleNamespace(
        factor_prices=factor_px, meta=meta, tickers=tickers, dates=dates
    )


def test_admitted_gate_off_is_empty_on_is_full():
    off = SimpleNamespace(index_revision_feature_enabled=False)
    on = SimpleNamespace(index_revision_feature_enabled=True)
    assert admitted_index_revision_features(off) == set()
    assert admitted_index_revision_features(SimpleNamespace()) == set()
    assert admitted_index_revision_features(on) == set(INDEX_REVISION_FEATURES)


def test_mapping_matches_preflight_declaration():
    """사전등록(§S13.23) 매핑 테이블과 프로덕션 매핑의 드리프트 방지."""
    assert EXCHANGE_TO_REV == PREFLIGHT_MAPPING


def test_build_assigns_mapped_country_series():
    data = _synthetic_data()
    out = build_index_revision_features(data)
    assert set(out) == {"fac_idx_rev"}
    panel = out["fac_idx_rev"]
    assert list(panel.columns) == ["AAA", "BBB", "CCC"]
    assert panel["AAA"].tolist() == [1.0, 2.0, 3.0, 4.0]      # US -> SPX_REV
    assert panel["BBB"].tolist() == [10.0, 20.0, 30.0, 40.0]  # FP -> CAC_REV
    assert panel["CCC"].tolist() == [-1.0, -2.0, -3.0, -4.0]  # JP -> JPN_REV


def test_build_skips_when_columns_missing():
    data = _synthetic_data()
    data.factor_prices = data.factor_prices.drop(columns=["JPN_REV"])
    assert build_index_revision_features(data) == {}


def test_loader_whitelist_contains_revision_columns():
    for col in ("SPX_REV", "NDX_REV", "SX5E_REV", "DAX_REV", "CAC_REV", "JPN_REV"):
        assert col in ALL_FACTOR_COLUMNS
