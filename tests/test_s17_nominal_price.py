# -*- coding: utf-8 -*-
"""§S17.3 G1-01b / M1 — 명목가 분모 (nominal_price_source).

결함(결정 로그 §S17 P1): 워크북 PX_LAST 와 PE/PB/PEG 시트가 Bloomberg DPDF 배당 재투자
소급 조정본이라, 명목 목표주가·CF/주·시총을 나누는 tg_upside·fwd_opcf_yield·주식수가
t 이후 배당을 담는다(MO 2014-06-30 19.56 vs 명목 41.94).

수정 계약: nominal_price_source="PX_LAST_UNADJ" 면 로더가 그 시트를 local_prices_nominal
(§S16.1 P1 LN×0.01 동일 적용)·prices_nominal(동일 FX 경로)로 부착하고 PE/PB/PEG 시트를
PX_UNADJ/PX_LAST 로 역산한다. sellside(tg_upside·fwd_opcf_yield)·accounting(주식수)·
growth_tilt TG 레그가 명목 패널을 쓴다. PX_LAST·Daily_Returns 는 불변. OFF(기본 None)는
바이트 동일.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import PipelineConfig
from src.data_loader import NOMINAL_RATIO_SHEETS, UniverseData

N = 30
_ESSENTIAL = {
    "CUR_MKT_CAP", "BEST_EPS", "BEST_SALES", "BEST_PE_RATIO", "OPER_MARGIN",
    "BEST_ROE", "NEWS_SENTIMENT_DAILY_AVG", "EQY_REC_CONS",
    "Factset_EPS_Revision", "Factset_Sales_Revision", "Factset_TG_Price",
}


def _raw_workbook(with_unadj=True, ln=False):
    """AAPL(무배당: UNADJ==PX_LAST) + MO(배당 조정: PX_LAST = 0.5×UNADJ) [+ AZN LN GBp]."""
    dates = pd.bdate_range("2026-01-01", periods=N)
    drift = np.linspace(1.0, 1.05, N)
    unadj = {"AAPL": 300.0 * drift, "MO": 40.0 * drift}
    adj = {"AAPL": 300.0 * drift, "MO": 20.0 * drift}
    tickers = ["AAPL", "MO"]
    meta_index = ["AAPL US Equity", "MO US Equity"]
    if ln:
        unadj["AZN"] = 12000.0 * drift
        adj["AZN"] = 6000.0 * drift
        tickers.append("AZN")
        meta_index.append("AZN LN Equity")
    px = pd.DataFrame(adj, index=dates)
    px_unadj = pd.DataFrame(unadj, index=dates)
    meta = pd.DataFrame(
        {"Ticker": tickers, "Name": tickers, "Sector": ["Test"] * len(tickers),
         "Status": ["Active"] * len(tickers)},
        index=meta_index,
    )
    raw = {
        "Universe_Meta": meta,
        "PX_LAST": px,
        "Daily_Returns": px.pct_change(fill_method=None).fillna(0.0),
        "Factor_PX_LAST": pd.DataFrame({"GBPUSD": np.full(N, 1.30)}, index=dates),
    }
    if with_unadj:
        raw["PX_LAST_UNADJ"] = px_unadj
    for sheet in _ESSENTIAL:
        raw[sheet] = pd.DataFrame(1.0, index=dates, columns=tickers)
    raw["BEST_PE_RATIO"] = pd.DataFrame({t: np.full(N, 10.0) for t in tickers}, index=dates)
    raw["BEST_PX_BPS_RATIO"] = pd.DataFrame({t: np.full(N, 3.0) for t in tickers}, index=dates)
    raw["BEST_PEG_RATIO"] = pd.DataFrame({t: np.full(N, 1.5) for t in tickers}, index=dates)
    raw["BEST_EV_TO_BEST_EBITDA"] = pd.DataFrame({t: np.full(N, 8.0) for t in tickers}, index=dates)
    raw["Factset_TG_Price"] = pd.DataFrame(
        {t: (330.0 if t == "AAPL" else 50.0 if t == "MO" else 159.0) for t in tickers},
        index=dates,
    )
    return raw


def _load(monkeypatch, raw, source=None, unit_fix=False):
    monkeypatch.setattr(
        "src.data_loader.load_all_sheets",
        lambda _path: {name: frame.copy() for name, frame in raw.items()},
    )
    cfg = PipelineConfig(
        fx_source_path="missing.xlsx",
        listing_auto_infer_enabled=False,
        s16_unit_fixpack_enabled=unit_fix,
        nominal_price_source=source,
    )
    return UniverseData("unused.xlsx", config=cfg), cfg


def test_nominal_price_source_default_off():
    assert PipelineConfig().nominal_price_source is None
    assert NOMINAL_RATIO_SHEETS == ("BEST_PE_RATIO", "BEST_PX_BPS_RATIO", "BEST_PEG_RATIO")


def test_off_parity_ignores_the_unadj_sheet(monkeypatch):
    with_sheet, _ = _load(monkeypatch, _raw_workbook(with_unadj=True))
    without, _ = _load(monkeypatch, _raw_workbook(with_unadj=False))
    assert not hasattr(with_sheet, "local_prices_nominal")
    assert not hasattr(with_sheet, "prices_nominal")
    assert "nominal_price" not in with_sheet.data_quality["currency"]
    for sheet in ("PX_LAST", "Daily_Returns", *NOMINAL_RATIO_SHEETS):
        pd.testing.assert_frame_equal(with_sheet.sheets[sheet], without.sheets[sheet])
    pd.testing.assert_frame_equal(with_sheet.local_prices, without.local_prices)


def test_on_attaches_nominal_panel_and_rescales_ratio_sheets(monkeypatch):
    raw = _raw_workbook()
    data, _ = _load(monkeypatch, raw, source="PX_LAST_UNADJ")
    off, _ = _load(monkeypatch, raw)

    pd.testing.assert_frame_equal(data.local_prices_nominal, raw["PX_LAST_UNADJ"])
    # USD 종목: FX 1.0 → prices_nominal == local_prices_nominal
    assert np.allclose(data.prices_nominal.values, raw["PX_LAST_UNADJ"].values)
    # PX_LAST·Daily_Returns·local_prices 는 불변
    pd.testing.assert_frame_equal(data.sheets["PX_LAST"], off.sheets["PX_LAST"])
    pd.testing.assert_frame_equal(data.sheets["Daily_Returns"], off.sheets["Daily_Returns"])
    pd.testing.assert_frame_equal(data.local_prices, off.local_prices)
    # PE/PB/PEG: MO 는 ×2 (UNADJ/PX_LAST = 2), AAPL 은 ×1; EV/EBITDA 는 불변
    for sheet, base in (("BEST_PE_RATIO", 10.0), ("BEST_PX_BPS_RATIO", 3.0), ("BEST_PEG_RATIO", 1.5)):
        assert np.allclose(data.sheets[sheet]["MO"], 2.0 * base)
        assert np.allclose(data.sheets[sheet]["AAPL"], base)
    assert np.allclose(data.sheets["BEST_EV_TO_BEST_EBITDA"]["MO"], 8.0)
    diag = data.data_quality["currency"]["nominal_price"]
    assert diag["source"] == "PX_LAST_UNADJ"
    assert diag["rescaled_sheets"] == list(NOMINAL_RATIO_SHEETS)
    assert diag["ratio_nominal_over_adjusted"]["min"] == pytest.approx(1.0)
    assert diag["ratio_nominal_over_adjusted"]["max"] == pytest.approx(2.0)


def test_on_applies_the_ln_unit_scale_and_fx_to_the_nominal_panel(monkeypatch):
    raw = _raw_workbook(ln=True)
    data, _ = _load(monkeypatch, raw, source="PX_LAST_UNADJ", unit_fix=True)
    # AZN: GBp 12,000 → ×0.01 = GBP 120 (local nominal), USD = ×1.30
    assert np.allclose(data.local_prices_nominal["AZN"], raw["PX_LAST_UNADJ"]["AZN"] * 0.01)
    assert np.allclose(data.prices_nominal["AZN"], raw["PX_LAST_UNADJ"]["AZN"] * 0.01 * 1.30)
    # 비율 시트 계수는 단위 무관: AZN UNADJ/PX_LAST = 2
    assert np.allclose(data.sheets["BEST_PE_RATIO"]["AZN"], 20.0)
    # 조정가 경로(§S16.1 P1)도 그대로
    assert np.allclose(data.local_prices["AZN"], raw["PX_LAST"]["AZN"] * 0.01)


def test_on_without_the_sheet_fails_loud(monkeypatch):
    with pytest.raises(KeyError, match="nominal_price_source"):
        _load(monkeypatch, _raw_workbook(with_unadj=False), source="PX_LAST_UNADJ")


# ---------------------------------------------------------------------------
# 소비 경로: sellside (tg_upside · fwd_opcf_yield)
# ---------------------------------------------------------------------------
class _PriceData:
    def __init__(self, sheets, local, nominal=None):
        self._sheets = dict(sheets)
        self.local_prices = local
        self.prices = local
        if nominal is not None:
            self.local_prices_nominal = nominal

    def get_sheet(self, name):
        if name not in self._sheets:
            raise KeyError(name)
        return self._sheets[name]


def _sellside_fixture():
    dates = pd.bdate_range("2021-01-04", periods=12)
    local = pd.DataFrame({"AAPL": np.linspace(100.0, 111.0, 12), "MO": np.linspace(20.0, 25.0, 12)}, index=dates)
    nominal = local.copy()
    nominal["MO"] = nominal["MO"] * 2.0
    tg = pd.DataFrame({"AAPL": 150.0, "MO": 50.0}, index=dates)
    opcf = tg * 0.1
    sheets = {"Factset_TG_Price": tg, "Factset_Fwd_OpCashflow": opcf}
    return sheets, local, nominal


def test_sellside_off_parity_and_on_uses_nominal_denominator():
    from src.features.sellside import build_sellside_features

    sheets, local, nominal = _sellside_fixture()
    off_plain = build_sellside_features(_PriceData(sheets, local), config=PipelineConfig())
    off_with = build_sellside_features(_PriceData(sheets, local, nominal), config=PipelineConfig())
    for key in ("tg_upside", "tg_upside_z", "fwd_opcf_yield"):
        pd.testing.assert_frame_equal(off_plain[key], off_with[key])

    on = build_sellside_features(
        _PriceData(sheets, local, nominal), config=PipelineConfig(nominal_price_source="PX_LAST_UNADJ"))
    assert np.allclose(on["tg_upside"]["MO"], sheets["Factset_TG_Price"]["MO"] / nominal["MO"] - 1)
    assert np.allclose(on["tg_upside"]["AAPL"], off_plain["tg_upside"]["AAPL"])   # 무배당 종목 불변
    assert np.allclose(on["fwd_opcf_yield"]["MO"], sheets["Factset_Fwd_OpCashflow"]["MO"] / nominal["MO"])
    # 목표주가 모멘텀은 분모와 무관 → 불변
    pd.testing.assert_frame_equal(on["tg_mom_21d"], off_plain["tg_mom_21d"])


def test_sellside_on_without_nominal_panel_fails_loud():
    from src.features.sellside import build_sellside_features

    sheets, local, _ = _sellside_fixture()
    with pytest.raises(AttributeError, match="local_prices_nominal"):
        build_sellside_features(_PriceData(sheets, local), config=PipelineConfig(nominal_price_source="PX_LAST_UNADJ"))


# ---------------------------------------------------------------------------
# 소비 경로: accounting (cash_conversion_z 의 주식수)
# ---------------------------------------------------------------------------
class _AcctData(_PriceData):
    def __init__(self, sheets, prices, market_cap, prices_nominal=None):
        super().__init__(sheets, prices)
        self.market_cap = market_cap
        if prices_nominal is not None:
            self.prices_nominal = prices_nominal


def test_accounting_share_count_uses_nominal_usd_price_when_on():
    from src.features.accounting import build_accounting_features

    dates = pd.bdate_range("2021-01-04", periods=8)
    cols = ["AAPL", "MO", "XOM"]
    prices = pd.DataFrame({"AAPL": 100.0, "MO": 20.0, "XOM": 50.0}, index=dates)
    nominal = prices.copy()
    nominal["MO"] = 40.0
    mcap = pd.DataFrame({"AAPL": 1000.0, "MO": 400.0, "XOM": 500.0}, index=dates)
    eps = pd.DataFrame({"AAPL": 5.0, "MO": 2.0, "XOM": 4.0}, index=dates)
    fcf = pd.DataFrame({"AAPL": 60.0, "MO": 30.0, "XOM": 45.0}, index=dates)
    sheets = {"BEST_EPS": eps, "BEST_CALCULATED_FCF": fcf}

    cfg_off = PipelineConfig(s16_unit_fixpack_enabled=True)
    cfg_on = PipelineConfig(s16_unit_fixpack_enabled=True, nominal_price_source="PX_LAST_UNADJ")
    off = build_accounting_features(_AcctData(sheets, prices, mcap, nominal), config=cfg_off)["cash_conversion_z"]
    off_plain = build_accounting_features(_AcctData(sheets, prices, mcap), config=cfg_off)["cash_conversion_z"]
    on = build_accounting_features(_AcctData(sheets, prices, mcap, nominal), config=cfg_on)["cash_conversion_z"]
    pd.testing.assert_frame_equal(off, off_plain)

    from src.features.utils import cross_sectional_zscore
    expect_off = cross_sectional_zscore(fcf / (eps * (mcap / prices)).abs())
    expect_on = cross_sectional_zscore(fcf / (eps * (mcap / nominal)).abs())
    pd.testing.assert_frame_equal(off.reindex(columns=cols), expect_off.reindex(columns=cols))
    pd.testing.assert_frame_equal(on.reindex(columns=cols), expect_on.reindex(columns=cols))
    assert not np.allclose(on["MO"], off["MO"])


# ---------------------------------------------------------------------------
# 소비 경로: growth tilt TG 레그 (backtest.apply_growth_tilt)
# ---------------------------------------------------------------------------
class _TiltData(_PriceData):
    pass


def test_growth_tilt_tg_leg_uses_nominal_prices_when_on():
    from src.backtest import apply_growth_tilt

    dates = pd.bdate_range("2021-01-04", periods=80)
    cols = ["AAPL", "MO"]
    local = pd.DataFrame({"AAPL": 100.0, "MO": 20.0}, index=dates)
    nominal = local.copy()
    nominal["MO"] = 40.0
    # 목표주가는 상수 → 63d 모멘텀 레그가 두 종목 동률이라 upside 레그만 순위를 정한다.
    tg = pd.DataFrame({"AAPL": 110.0, "MO": 50.0}, index=dates)
    sheets = {"Factset_TG_Price": tg}
    preds = pd.DataFrame(0.0, index=dates, columns=cols)
    base_kwargs = dict(
        growth_tilt_enabled=True, growth_tilt_weight=0.25, growth_tilt_rev_weight=1.0,
        growth_tilt_fundamental_weight=0.0, growth_tilt_rev_eps_share=0.0,
        growth_tilt_rev_sales_share=0.0, growth_tilt_rev_tg_share=1.0,
    )
    off = apply_growth_tilt(preds, _TiltData(sheets, local, nominal), PipelineConfig(**base_kwargs))
    off_plain = apply_growth_tilt(preds, _TiltData(sheets, local), PipelineConfig(**base_kwargs))
    on = apply_growth_tilt(
        preds, _TiltData(sheets, local, nominal),
        PipelineConfig(**base_kwargs, nominal_price_source="PX_LAST_UNADJ"))
    pd.testing.assert_frame_equal(off, off_plain)
    # OFF: MO upside 1.5 > AAPL 0.1 → MO 가 우위; ON: MO upside 0.25 > AAPL 0.1 (순위 유지) 이나
    # 어느 쪽이든 두 경로가 다른 분모를 썼는지는 upside 값이 아니라 순위 기반이라 같을 수 있다.
    # 분모 차이를 직접 드러내기 위해 AAPL 목표주가를 명목 기준 MO 보다 크게 둔 두 번째 케이스.
    tg2 = tg.copy()
    tg2["AAPL"] = 140.0  # AAPL upside 0.4: OFF MO 1.5 > 0.4, ON MO 0.25 < 0.4 → 순위 반전
    sheets2 = {"Factset_TG_Price": tg2}
    off2 = apply_growth_tilt(preds, _TiltData(sheets2, local, nominal), PipelineConfig(**base_kwargs))
    on2 = apply_growth_tilt(
        preds, _TiltData(sheets2, local, nominal),
        PipelineConfig(**base_kwargs, nominal_price_source="PX_LAST_UNADJ"))
    last = dates[-1]
    assert off2.loc[last, "MO"] > off2.loc[last, "AAPL"]
    assert on2.loc[last, "MO"] < on2.loc[last, "AAPL"]
    assert on.shape == off.shape


# ---------------------------------------------------------------------------
# variant 핀: arm = production + 단일 파라미터, production 은 default-OFF 유지
# ---------------------------------------------------------------------------
def test_arm_variant_equals_production_after_the_s17_3_flip():
    """§8/S17.3: flip(2026-09-04) 이후 역사적 arm variant(수정 금지)는 production 과 overrides 가 동일하다."""
    import yaml

    with open("variants/codex_causal_rank_65.yaml", encoding="utf-8") as fh:
        prod = yaml.safe_load(fh)["overrides"]
    with open("variants/s17_5_nominal_price.yaml", encoding="utf-8") as fh:
        arm = yaml.safe_load(fh)["overrides"]
    assert arm["nominal_price_source"] == "PX_LAST_UNADJ"
    # Flags promoted AFTER this historical arm was frozen (S18.2 flip 2026-09-08 ...)
    # are excluded; the arm itself is never edited.
    post_s17_3_flips = {"vol_quality_tilt_negative_equity_mask", "tg_basis_events"}
    assert arm == {k: v for k, v in prod.items() if k not in post_s17_3_flips}


def test_production_variant_pins_s17_3_flip_state():
    """§8/S17.3: 사용자 승인 flip(2026-09-04) 이후의 production 상태 핀.

    production variant는 nominal_price_source=PX_LAST_UNADJ(새 S0′ 1.7633,
    1.7118 은퇴)여야 하고, PipelineConfig 기본값은 여전히 None(§8
    default-OFF 유지)여야 한다."""
    import yaml

    with open("variants/codex_causal_rank_65.yaml", encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh)
    overrides = manifest.get("overrides") or {}
    assert overrides.get("nominal_price_source") == "PX_LAST_UNADJ"
    assert PipelineConfig().nominal_price_source is None
