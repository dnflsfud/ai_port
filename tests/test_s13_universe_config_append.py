# -*- coding: utf-8 -*-
"""§S13 universe_config 적용 대기 블록(outputs/s13_universe_config_append.py)
계약 테스트 — 사전등록 슬레이트(50종·섹터 배분·신규 FX 페어 0·중복 0) 고정."""
import importlib.util
from collections import Counter
from pathlib import Path


def _load_staged():
    path = (Path(__file__).resolve().parents[1] / "outputs"
            / "s13_universe_config_append.py")
    spec = importlib.util.spec_from_file_location("s13_append", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_s13_entries_contract():
    entries = _load_staged().S13_ENTRIES
    assert len(entries) == 50
    for ticker, meta in entries.items():
        assert {"name", "sector", "color"} <= set(meta), ticker
        parts = ticker.rsplit(" ", 2)
        assert len(parts) == 3 and parts[2] == "Equity", ticker
        # 신규 FX 페어 0: 전부 기지원 거래소 코드
        assert parts[1] in {"US", "FP", "GR", "SW", "JP"}, ticker
    simple = [t.rsplit(" ", 2)[0] for t in entries]
    assert len(set(simple)) == 50


def test_s13_no_overlap_and_allocation():
    entries = _load_staged().S13_ENTRIES
    from src.data_loader import TICKERS

    assert len(TICKERS) == 150  # 선적용 전 기준 (적용 후 이 테스트도 §S13.1에서 갱신)
    simple = {t.rsplit(" ", 2)[0] for t in entries}
    assert not (simple & set(TICKERS))
    # 사전등록 배분 (MSCI World 2026-06-30 비중 비례)
    assert Counter(m["sector"] for m in entries.values()) == {
        "Technology": 15, "Financials": 8, "Industrials": 6,
        "Healthcare": 5, "Consumer Discretionary": 4,
        "Communication Services": 4, "Consumer Staples": 3,
        "Energy": 2, "Materials": 2, "Utilities": 1,
    }
