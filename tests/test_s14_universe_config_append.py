# -*- coding: utf-8 -*-
"""§S14 universe_config 적용 대기 블록(outputs/s14_universe_config_append.py)
계약 테스트 — 사전등록 슬레이트(50종·섹터 배분·통화 믹스·신규 FX 페어 0·중복 0)
고정. §S13 계약 테스트(test_s13_universe_config_append.py) 패턴 승계."""
import importlib.util
from collections import Counter
from pathlib import Path


def _load_staged():
    path = (Path(__file__).resolve().parents[1] / "outputs"
            / "s14_universe_config_append.py")
    spec = importlib.util.spec_from_file_location("s14_append", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_s14_entries_contract():
    entries = _load_staged().S14_ENTRIES
    assert len(entries) == 50
    for ticker, meta in entries.items():
        assert {"name", "sector", "color"} <= set(meta), ticker
        parts = ticker.rsplit(" ", 2)
        assert len(parts) == 3 and parts[2] == "Equity", ticker
        # 신규 FX 페어 0. IM(밀라노)은 신규 거래소 코드지만 통화는 기지원
        # EUR — 적용 단계에서 MARKET_TO_CURRENCY/FACTSET_MARKET_CODE에
        # "IM" 매핑을 함께 추가한다(스테이징 헤더 3·4단계).
        assert parts[1] in {"US", "NA", "IM", "SW", "JP", "LN", "FP"}, ticker
    simple = [t.rsplit(" ", 2)[0] for t in entries]
    assert len(set(simple)) == 50
    # 사전등록 통화 믹스 (거래소 코드 기준): USD 39·EUR 4(NA1+IM2+FP1)·
    # JPY 4·CHF 2·GBP 1
    assert Counter(t.rsplit(" ", 2)[1] for t in entries) == {
        "US": 39, "JP": 4, "IM": 2, "SW": 2, "NA": 1, "LN": 1, "FP": 1,
    }


def test_s14_no_overlap_and_allocation():
    entries = _load_staged().S14_ENTRIES
    from src.data_loader import TICKERS

    # §S14.2(단계 ⑤) 적용 후 기준: 스테이징 50종이 그대로 TICKERS[200:] 꼬리가
    # 됐다 (§S13 계약 테스트와 동일 패턴).
    assert len(TICKERS) == 250
    simple = {t.rsplit(" ", 2)[0] for t in entries}
    assert simple == set(TICKERS[200:])
    assert not (simple & set(TICKERS[:200]))
    # 사전등록 배분 (MSCI World 비례, GPT 교차리뷰 3회 반영 확정본 —
    # GRMN은 Consumer Discretionary(GICS), Roper 보류로 ROP=로슈 무충돌)
    assert Counter(m["sector"] for m in entries.values()) == {
        "Technology": 11, "Financials": 8, "Industrials": 6,
        "Healthcare": 6, "Consumer Discretionary": 6,
        "Communication Services": 3, "Consumer Staples": 3,
        "Energy": 2, "Materials": 2, "Utilities": 2, "Real Estate": 1,
    }
