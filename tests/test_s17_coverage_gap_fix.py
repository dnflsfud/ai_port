# -*- coding: utf-8 -*-
"""§S17 T-01 — 목표주가 커버리지 갭 임퓨트 수정 (s17_coverage_gap_fix_enabled).

결함(결정 로그 §S17 P2): 로더 `_fill_missing`이 ffill 뒤 남은 NaN(커버리지 개시 전)을
날짜별 횡단면 median = 타 종목의 목표주가 *레벨*로 채운다. 상장 마스크는 상장일만
다루므로 상장 후 커버리지 갭은 어느 패스에서도 재마스킹되지 않고, tg_upside가
VRT 409행 +5.0 상수(클립)로 고정되며 타 249종 z가 2.4× 압축된다.

수정 계약: 가격 레벨 시트(Factset_TG_Price·Factset_Fwd_OpCashflow)는 티커별
첫 관측 이전 셀을 NaN으로 되돌린다(관측 마스크 = 로더 임퓨트 이전
`raw_sheet_observed_mask`, §S15.2 관용구). 첫 관측 이후의 내부 갭은 기존 ffill을
유지한다. NaN이 된 피처 셀은 패널 빌더의 날짜별 median 채움으로 간다(§S13.6
교훈대로 네이티브 NaN 경로가 아님). OFF(기본)는 바이트 동일.
"""

import numpy as np
import pandas as pd

from src.config import PipelineConfig

N_DATES = 12
SPLIT = 6            # BBB 커버리지 개시 행
INTERIOR_GAP = 8     # 커버리지 개시 후 내부 결측(ffill 유지 대상)


class _PlainData:
    """test_sellside._FakeData와 동형 — 관측 마스크 API 없음."""

    def __init__(self, sheets, local_prices):
        self._sheets = dict(sheets)
        self.local_prices = local_prices
        self.prices = local_prices

    def get_sheet(self, name):
        if name not in self._sheets:
            raise KeyError(name)
        return self._sheets[name]


class _MaskedData(_PlainData):
    def __init__(self, sheets, local_prices, observed):
        super().__init__(sheets, local_prices)
        self._observed = dict(observed)

    def raw_sheet_observed_mask(self, name):
        return self._observed.get(name)


def _fixture():
    """AAA 전 구간 커버, BBB는 SPLIT부터 커버. 시트는 로더가 넘겨주는 형태
    (커버리지 전 = 횡단면 median = AAA 값, 내부 갭 = ffill)로 만든다."""
    dates = pd.bdate_range("2021-01-04", periods=N_DATES)
    px = pd.DataFrame(
        {"AAA": np.linspace(100.0, 111.0, N_DATES),
         "BBB": np.linspace(30.0, 41.0, N_DATES)}, index=dates)
    tg_aaa = np.linspace(150.0, 161.0, N_DATES)
    tg_bbb_raw = np.full(N_DATES, np.nan)
    tg_bbb_raw[SPLIT:] = np.linspace(36.0, 41.0, N_DATES - SPLIT)
    tg_bbb_raw[INTERIOR_GAP] = np.nan
    observed = pd.DataFrame(
        {"AAA": True, "BBB": ~np.isnan(tg_bbb_raw)}, index=dates)
    # 로더 재현: ffill -> 잔여 NaN은 횡단면 median (2종이라 AAA 값).
    tg_bbb = pd.Series(tg_bbb_raw, index=dates).ffill()
    tg_bbb = tg_bbb.where(tg_bbb.notna(), pd.Series(tg_aaa, index=dates))
    tg = pd.DataFrame({"AAA": tg_aaa, "BBB": tg_bbb.values}, index=dates)
    opcf = tg * 0.1
    sheets = {"Factset_TG_Price": tg, "Factset_Fwd_OpCashflow": opcf}
    masks = {"Factset_TG_Price": observed, "Factset_Fwd_OpCashflow": observed}
    return sheets, px, masks, dates


def _build(data, flag):
    from src.features.sellside import build_sellside_features
    return build_sellside_features(
        data, config=PipelineConfig(s17_coverage_gap_fix_enabled=flag))


def test_s17_coverage_gap_fix_flag_default_off():
    assert PipelineConfig().s17_coverage_gap_fix_enabled is False


def test_production_variant_pins_s17_1_flip_state():
    """§8/S17.1-A: 사용자 승인 flip(2026-09-03) 이후의 production 상태 핀.

    production variant는 s17_coverage_gap_fix_enabled=True(새 S0′ 1.7330,
    1.7052 은퇴)여야 하고, PipelineConfig 기본값은 여전히 False(§8
    default-OFF 유지)여야 한다."""
    import yaml

    with open("variants/codex_causal_rank_65.yaml", encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh)
    overrides = manifest.get("overrides") or {}
    assert overrides.get("s17_coverage_gap_fix_enabled") is True
    assert PipelineConfig().s17_coverage_gap_fix_enabled is False


def test_off_parity_reproduces_the_plus_five_artifact():
    sheets, px, masks, _ = _fixture()
    off = _build(_MaskedData(sheets, px, masks), False)
    ref = _build(_PlainData(sheets, px), False)
    for key in ("tg_upside", "tg_mom_63d", "tg_upside_z", "fwd_opcf_yield"):
        pd.testing.assert_frame_equal(off[key], ref[key])
    # 결함 재현: 커버리지 전 BBB upside = AAA 목표가 / BBB 가격 - 1 (≈ +4)
    pre = off["tg_upside"]["BBB"].iloc[:SPLIT]
    assert pre.notna().all() and (pre > 3.0).all()


def test_on_masks_pre_coverage_cells_only():
    sheets, px, masks, _ = _fixture()
    on = _build(_MaskedData(sheets, px, masks), True)
    off = _build(_MaskedData(sheets, px, masks), False)
    up = on["tg_upside"]
    assert up["BBB"].iloc[:SPLIT].isna().all()          # 커버리지 전 NaN
    assert up["BBB"].iloc[SPLIT:].notna().all()         # 개시 후 전부 유효 (내부 갭 ffill 유지)
    expected_post = sheets["Factset_TG_Price"]["BBB"].iloc[SPLIT:] / px["BBB"].iloc[SPLIT:] - 1
    assert np.allclose(up["BBB"].iloc[SPLIT:], expected_post)
    pd.testing.assert_series_equal(up["AAA"], off["tg_upside"]["AAA"])  # 완전 커버 종목 불변
    # 파생 모멘텀도 커버리지 전 기준값을 쓰지 않는다.
    assert on["tg_mom_21d"]["BBB"].iloc[:SPLIT].isna().all()


def test_on_applies_to_fwd_opcf_sheet():
    sheets, px, masks, _ = _fixture()
    on = _build(_MaskedData(sheets, px, masks), True)
    y = on["fwd_opcf_yield"]
    assert y["BBB"].iloc[:SPLIT].isna().all()
    assert y["BBB"].iloc[SPLIT:].notna().all()
    assert y["AAA"].notna().all()


def test_on_without_mask_support_is_inert():
    sheets, px, _masks, _ = _fixture()
    on = _build(_PlainData(sheets, px), True)
    off = _build(_PlainData(sheets, px), False)
    pd.testing.assert_frame_equal(on["tg_upside"], off["tg_upside"])


# ---------------------------------------------------------------------------
# 로더 통합 — 실제 UniverseData 경로에서 커버리지 갭이 NaN으로 돌아오는지
# (tests/test_data_loader.py의 관측 마스크 워크북 관용구).
# ---------------------------------------------------------------------------
_ESSENTIAL = {
    "CUR_MKT_CAP", "BEST_EPS", "BEST_SALES", "BEST_PE_RATIO", "OPER_MARGIN",
    "BEST_ROE", "NEWS_SENTIMENT_DAILY_AVG", "EQY_REC_CONS",
    "Factset_EPS_Revision", "Factset_Sales_Revision",
}


def _loader_data(monkeypatch, flag):
    from src.data_loader import UniverseData

    dates = pd.bdate_range("2021-01-04", periods=N_DATES)
    prices = pd.DataFrame(
        {"AAA": np.linspace(100.0, 111.0, N_DATES),
         "BBB": np.linspace(30.0, 41.0, N_DATES)}, index=dates)
    meta = pd.DataFrame(
        {"Ticker": ["AAA", "BBB"], "Name": ["Alpha", "Beta"],
         "Sector": ["Test", "Test"]},
        index=["AAA US Equity", "BBB US Equity"])
    tg = pd.DataFrame(
        {"AAA": np.linspace(150.0, 161.0, N_DATES),
         "BBB": [np.nan] * SPLIT + list(np.linspace(36.0, 41.0, N_DATES - SPLIT))},
        index=dates)
    raw = {"Universe_Meta": meta, "PX_LAST": prices,
           "Daily_Returns": prices.pct_change(fill_method=None).fillna(0.0),
           "Factset_TG_Price": tg}
    for sheet in _ESSENTIAL:
        raw[sheet] = pd.DataFrame(1.0, index=dates, columns=["AAA", "BBB"])
    monkeypatch.setattr(
        "src.data_loader.load_all_sheets",
        lambda _path: {k: v.copy() for k, v in raw.items()})
    cfg = PipelineConfig(fx_source_path="missing.xlsx",
                         s17_coverage_gap_fix_enabled=flag)
    return UniverseData("unused.xlsx", config=cfg), cfg


def test_loader_integration_pre_coverage_gap_becomes_nan(monkeypatch):
    from src.features.sellside import build_sellside_features

    data_off, cfg_off = _loader_data(monkeypatch, False)
    off = build_sellside_features(data_off, config=cfg_off)["tg_upside"]
    assert off["BBB"].iloc[:SPLIT].notna().all()       # 회귀 고정: 로더가 갭을 메운다
    assert (off["BBB"].iloc[:SPLIT] > 3.0).all()       # 타 종목 가격 레벨 → +4 아티팩트

    data_on, cfg_on = _loader_data(monkeypatch, True)
    on = build_sellside_features(data_on, config=cfg_on)["tg_upside"]
    assert on["BBB"].iloc[:SPLIT].isna().all()
    assert on["BBB"].iloc[SPLIT:].notna().all()
    pd.testing.assert_series_equal(on["AAA"], off["AAA"])
