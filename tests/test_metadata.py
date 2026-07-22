# -*- coding: utf-8 -*-
"""build_ticker_meta 계약 테스트 (§S11.4 Phase 3 — Universe_Meta 기반 통합).

섹터는 워크북 Universe_Meta가 정본, style/sub는 정적 TICKER_META fallback.
"""

import pandas as pd

from src.metadata import TICKER_META, build_ticker_meta


def _meta(rows):
    return pd.DataFrame({"sector": [s for _, s in rows]}, index=[t for t, _ in rows])


def test_sector_comes_from_universe_meta():
    tm = build_ticker_meta(_meta([("AON", "Financials"), ("GOOGL", "Communication Services")]))
    assert tm["AON"]["sector"] == "Financials"
    # 워크북이 정적 dict("Communication")와 다르면 워크북이 이긴다
    assert tm["GOOGL"]["sector"] == "Communication Services"


def test_style_sub_fallback_to_static_dict():
    tm = build_ticker_meta(_meta([("AAPL", "Technology"), ("AON", "Financials")]))
    assert tm["AAPL"]["style"] == TICKER_META["AAPL"]["style"]
    assert tm["AAPL"]["sub"] == TICKER_META["AAPL"]["sub"]
    # 정적 dict에 없는 신규 종목은 중립 fallback
    assert tm["AON"]["style"] == "Other"
    assert tm["AON"]["sub"] == "N/A"


def test_covers_every_meta_row():
    meta = _meta([("A1", "S1"), ("A2", "S2"), ("A3", "S3")])
    tm = build_ticker_meta(meta)
    assert set(tm.keys()) == {"A1", "A2", "A3"}
