# -*- coding: utf-8 -*-
"""§S18.1 (2026-09-07) — 구조 리뷰 5차 수정 4종 (결정 로그 §S18 P1·P2·P3).

1. 목표주가 기저 가드: 로더 `tg_px_ratio_suspect`([0.6, 1.7] 밖) + run_variant 직전 런 대비
   |Δlog| > 0.25 점프 → production HOLD 게이트 `tg_px_ratio_ok`(fail-closed). 산출물 불변.
2. `tg_basis_events`: 이벤트 전 TG 행 × factor (빈 dict = 같은 객체·바이트 동일).
3. `vol_quality_tilt_negative_equity_mask`: BEST_ROE<0 ∧ BEST_EPS>0 셀의 품질 입력 NaN →
   해당 셀 바이트 불변(함수 계약). data 없이 ON 이면 ValueError(무음 폴백 금지).
4. `static_execution_enabled`: 신뢰도 ≡ 1 — compute_signal_confidence 미호출.
"""
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from src.backtest import (
    apply_vol_quality_tilt, negative_equity_mask, simulate_portfolio,
)
from src.config import PipelineConfig
from src.data_loader import TG_PX_RATIO_SUSPECT_BAND, UniverseData
from src.features.sellside import apply_tg_basis_events, build_sellside_features

AI_PORT_VARIANTS = "variants"
ARMS = {
    "s18_1_tilt_negative_equity": {"vol_quality_tilt_negative_equity_mask": True},
    "s18_2_tg_basis_events": {"tg_basis_events": {
        "RTX": {"2020-04-03": 1.696}, "T": {"2022-04-11": 1.324},
        "DELL": {"2021-11-02": 0.506}, "DHR": {"2016-07-05": 0.758},
    }},
    "s18_3_static_execution": {"static_execution_enabled": True},
}


# --------------------------------------------------------------------------- 공통 스텁
class _Data:
    """UniverseData 최소 스텁: get_sheet / prices / local_prices / tickers / dates."""

    def __init__(self, sheets, dates, tickers):
        self.sheets = sheets
        self.dates = dates
        self.tickers = list(tickers)
        self.prices = sheets["PX_LAST"]
        self.local_prices = sheets["PX_LAST"]

    def get_sheet(self, name):
        if name not in self.sheets:
            raise KeyError(name)
        return self.sheets[name]


def _stub(n=40, tickers=("AAA", "BBB", "CCC")):
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(3)
    px = pd.DataFrame(100 + rng.normal(0, 1, (n, len(tickers))).cumsum(axis=0), index=dates, columns=tickers)
    tg = px * 1.1
    sheets = {"PX_LAST": px, "Factset_TG_Price": tg}
    return _Data(sheets, dates, tickers), px, tg


# --------------------------------------------------------------------------- 0. 기본값
def test_config_defaults_are_off():
    cfg = PipelineConfig()
    assert cfg.vol_quality_tilt_negative_equity_mask is False
    assert cfg.tg_basis_events == {}
    assert cfg.static_execution_enabled is False
    assert TG_PX_RATIO_SUSPECT_BAND == (0.6, 1.7)


# --------------------------------------------------------------------------- 1. 가드
def _loader_shell(tg, px):
    obj = UniverseData.__new__(UniverseData)
    obj.sheets = {"Factset_TG_Price": tg}
    obj.local_prices = px
    obj.data_quality = {"currency": {}}
    return obj


def test_tg_ratio_guard_flags_vendor_basis_mismatch_only():
    _, px, tg = _stub()
    tg = tg.copy()
    tg["BBB"] = px["BBB"] * 2.33   # APH-type pre-split TG basis
    tg["CCC"] = px["CCC"] * 1.55   # optimistic but plausible (ORCL 1.60)
    shell = _loader_shell(tg, px)
    shell._check_target_price_unit_ratio()
    cur = shell.data_quality["currency"]
    assert set(cur["tg_px_ratio_suspect"]) == {"BBB"}
    assert cur["tg_px_ratio_suspect"]["BBB"] == pytest.approx(2.33, rel=1e-6)
    assert "AAA" in cur["tg_px_ratio_median"] and "CCC" in cur["tg_px_ratio_median"]


def test_tg_ratio_jump_vs_previous_run(tmp_path):
    from run_variant import TG_PX_RATIO_JUMP_MAX, annotate_tg_ratio_jump

    prev = {"data_quality": {"currency": {"tg_px_ratio_median": {"APH": 1.16, "AAPL": 1.05}}}}
    (tmp_path / "metrics.json").write_text(json.dumps(prev), encoding="utf-8")
    dq = {"currency": {"tg_px_ratio_median": {"APH": 2.33, "AAPL": 1.10, "NEW": 1.2}}}
    jumps = annotate_tg_ratio_jump(tmp_path, dq)
    assert set(jumps) == {"APH"}
    assert dq["currency"]["tg_px_ratio_jump_vs_prev"] == jumps
    assert abs(np.log(2.33 / 1.16)) > TG_PX_RATIO_JUMP_MAX > abs(np.log(1.10 / 1.05))
    # no previous run -> empty, never raises
    dq2 = {"currency": {"tg_px_ratio_median": {"APH": 2.33}}}
    assert annotate_tg_ratio_jump(tmp_path / "nowhere", dq2) == {}
    assert annotate_tg_ratio_jump(tmp_path, None) == {}


def test_production_gate_tg_px_ratio_check_is_fail_closed():
    from scripts.validate_portfolio_bundles import evaluate_production

    clean = evaluate_production({"performance": {"data_quality": {
        "currency": {"tg_px_ratio_suspect": {}}}}})
    assert clean["checks"]["tg_px_ratio_ok"] is True
    suspect = evaluate_production({"performance": {"data_quality": {
        "currency": {"tg_px_ratio_suspect": {"APH": 2.33}}}}})
    assert suspect["checks"]["tg_px_ratio_ok"] is False
    assert suspect["status"] == "HOLD"
    jumped = evaluate_production({"performance": {"data_quality": {
        "currency": {"tg_px_ratio_suspect": {},
                     "tg_px_ratio_jump_vs_prev": {"APH": {"previous": 1.16, "now": 2.33}}}}}})
    assert jumped["checks"]["tg_px_ratio_ok"] is False
    missing = evaluate_production({"performance": {"data_quality": {}}})
    assert missing["checks"]["tg_px_ratio_ok"] is None   # missing evidence -> HOLD


# --------------------------------------------------------------------------- 2. TG 기저 이벤트
def test_tg_basis_events_empty_returns_same_object():
    _, _, tg = _stub()
    assert apply_tg_basis_events(tg, {}) is tg
    assert apply_tg_basis_events(tg, None) is tg


def test_tg_basis_events_scales_only_pre_event_rows_of_the_named_ticker():
    _, _, tg = _stub()
    event = str(tg.index[20].date())
    out = apply_tg_basis_events(tg, {"BBB": {event: 0.5}, "ZZZ": {event: 9.0}})
    assert out is not tg
    pd.testing.assert_series_equal(out["AAA"], tg["AAA"])
    pd.testing.assert_series_equal(out["CCC"], tg["CCC"])
    pd.testing.assert_series_equal(out["BBB"].iloc[:20], tg["BBB"].iloc[:20] * 0.5, check_names=False)
    pd.testing.assert_series_equal(out["BBB"].iloc[20:], tg["BBB"].iloc[20:])
    pd.testing.assert_frame_equal(tg, _stub()[2])   # input never mutated


def test_sellside_off_parity_and_on_only_moves_tg_features_of_the_event_ticker():
    data, px, tg = _stub()
    off = build_sellside_features(data, config=PipelineConfig())
    off2 = build_sellside_features(data, config=PipelineConfig(tg_basis_events={}))
    for k in off:
        pd.testing.assert_frame_equal(off[k], off2[k])
    event = str(tg.index[20].date())
    on = build_sellside_features(data, config=PipelineConfig(tg_basis_events={"BBB": {event: 0.5}}))
    assert set(on) == set(off)
    for k in off:
        if k.startswith("tg_"):
            # AAA/CCC raw upside identical; BBB pre-event lower
            if k == "tg_upside":
                pd.testing.assert_series_equal(on[k]["AAA"], off[k]["AAA"])
                assert (on[k]["BBB"].iloc[:20] < off[k]["BBB"].iloc[:20]).all()
                pd.testing.assert_series_equal(on[k]["BBB"].iloc[20:], off[k]["BBB"].iloc[20:])
        else:
            pd.testing.assert_frame_equal(on[k], off[k])


# --------------------------------------------------------------------------- 3. 틸트 음자본 마스크
def _tilt_inputs(n_days=3, n_tk=60):
    idx = pd.bdate_range("2024-01-02", periods=n_days)
    tickers = [f"T{i:02d}" for i in range(n_tk)]
    rng = np.random.default_rng(7)
    preds = pd.DataFrame(rng.normal(size=(n_days, n_tk)), index=idx, columns=tickers)
    vol = np.concatenate([np.full(20, 5.0), np.linspace(0.5, 1.0, n_tk - 20)])  # T00..T19 = top tercile
    vol_panel = pd.DataFrame(np.tile(vol, (n_days, 1)), index=idx, columns=tickers)
    q_panel = pd.DataFrame(rng.normal(size=(n_days, n_tk)), index=idx, columns=tickers)
    q_panel["T00"] = -6.0     # extreme negative "quality" (negative-equity ROE signature)
    panel = pd.concat({"idio_vol_63d": vol_panel.stack(), "best_roe_level_z": q_panel.stack()}, axis=1)
    panel.index.names = ["date", "ticker"]
    roe = pd.DataFrame(10.0, index=idx, columns=tickers); roe["T00"] = -400.0
    eps = pd.DataFrame(2.0, index=idx, columns=tickers)
    data = _Data({"BEST_ROE": roe, "BEST_EPS": eps, "PX_LAST": pd.DataFrame(1.0, index=idx, columns=tickers)},
                 idx, tickers)
    return preds, panel, data


def test_negative_equity_mask_signature():
    preds, panel, data = _tilt_inputs()
    m = negative_equity_mask(data, preds.index, preds.columns)
    assert m["T00"].all() and not m.drop(columns="T00").to_numpy().any()


def test_tilt_mask_off_is_byte_identical_with_or_without_data():
    preds, panel, data = _tilt_inputs()
    cfg = PipelineConfig(vol_quality_tilt_enabled=True)
    pd.testing.assert_frame_equal(apply_vol_quality_tilt(preds, panel, cfg),
                                  apply_vol_quality_tilt(preds, panel, cfg, data=data))


def test_tilt_mask_on_leaves_negative_equity_cell_untouched_and_requires_data():
    preds, panel, data = _tilt_inputs()
    base_cfg = PipelineConfig(vol_quality_tilt_enabled=True)
    cfg = PipelineConfig(vol_quality_tilt_enabled=True, vol_quality_tilt_negative_equity_mask=True)
    off = apply_vol_quality_tilt(preds, panel, base_cfg)
    on = apply_vol_quality_tilt(preds, panel, cfg, data=data)
    # OFF penalised T00 hard; ON leaves T00 byte-unchanged
    assert (off["T00"] < preds["T00"] - 0.5).all()
    pd.testing.assert_series_equal(on["T00"], preds["T00"])
    # other tercile members still tilted; names outside the tercile untouched in both
    assert not on["T05"].equals(preds["T05"])
    pd.testing.assert_frame_equal(on.loc[:, "T20":], preds.loc[:, "T20":])
    with pytest.raises(ValueError):
        apply_vol_quality_tilt(preds, panel, cfg)   # ON without data -> no silent fallback


# --------------------------------------------------------------------------- 4. 정적 집행
def _sim_inputs(n_days=30, n_tk=12):
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    tickers = [f"S{i:02d}" for i in range(n_tk)]
    rng = np.random.default_rng(11)
    rets = pd.DataFrame(rng.normal(0, 0.01, (n_days, n_tk)), index=dates, columns=tickers)
    preds = pd.DataFrame(rng.normal(size=(n_days, n_tk)), index=dates, columns=tickers)
    return preds, rets, tickers, dates


def test_static_execution_skips_confidence_and_uses_base_eta(monkeypatch):
    import src.backtest as bt

    preds, rets, tickers, dates = _sim_inputs()

    def _boom(*a, **k):
        raise AssertionError("compute_signal_confidence must not be consulted")

    monkeypatch.setattr(bt, "compute_signal_confidence", _boom)
    cfg_on = PipelineConfig(use_score_based=True, static_execution_enabled=True)
    res_on = simulate_portfolio(preds, rets, tickers, dates, rebalance_freq=5, config=cfg_on,
                                raw_predictions=preds, track_ic=False)
    assert len(res_on.portfolio_weights) > 1
    cfg_off = PipelineConfig(use_score_based=True)
    with pytest.raises(AssertionError):
        simulate_portfolio(preds, rets, tickers, dates, rebalance_freq=5, config=cfg_off,
                           raw_predictions=preds, track_ic=False)


def test_static_execution_equals_confidence_one(monkeypatch):
    import src.backtest as bt

    preds, rets, tickers, dates = _sim_inputs()
    cfg_on = PipelineConfig(use_score_based=True, static_execution_enabled=True)
    res_on = simulate_portfolio(preds, rets, tickers, dates, rebalance_freq=5, config=cfg_on,
                                raw_predictions=preds, track_ic=False)
    monkeypatch.setattr(bt, "compute_signal_confidence", lambda *a, **k: 1.0)
    cfg_off = PipelineConfig(use_score_based=True)
    res_off = simulate_portfolio(preds, rets, tickers, dates, rebalance_freq=5, config=cfg_off,
                                 raw_predictions=preds, track_ic=False)
    pd.testing.assert_series_equal(res_on.portfolio_returns, res_off.portfolio_returns)


# --------------------------------------------------------------------------- 5. variants / 판정 프레임
# §8 flips adopted AFTER the S18.1 arms were frozen (historical arm variants are
# never edited; the pin compares against the pre-flip production state).
S18_PRODUCTION_FLIPS = {
    "vol_quality_tilt_negative_equity_mask",  # S18.2 flip (2026-09-08)
    "tg_basis_events",  # S18.2 flip (2026-09-08)
}


@pytest.mark.parametrize("label", sorted(ARMS))
def test_arm_variant_is_production_plus_exactly_one_parameter(label):
    prod = yaml.safe_load(open(f"{AI_PORT_VARIANTS}/codex_causal_rank_65.yaml", encoding="utf-8"))["overrides"]
    prod_pre_flip = {k: v for k, v in prod.items() if k not in S18_PRODUCTION_FLIPS}
    arm = yaml.safe_load(open(f"{AI_PORT_VARIANTS}/{label}.yaml", encoding="utf-8"))
    assert arm["out_dir"] == f"outputs/{label}"
    extra = {k: v for k, v in arm["overrides"].items() if k not in prod_pre_flip}
    assert extra == ARMS[label]
    assert {k: v for k, v in arm["overrides"].items() if k in prod_pre_flip} == prod_pre_flip


def test_production_variant_pins_s18_2_flip_state():
    """§8/S18.2: 사용자 승인 flip(2026-09-08) 이후의 production 상태 핀.

    production variant 는 vol_quality_tilt_negative_equity_mask=True(새 S0′ 1.6373,
    1.6106 은퇴)여야 하고, PipelineConfig 기본값은 여전히 False(§8 default-OFF 유지)."""
    prod = yaml.safe_load(open(f"{AI_PORT_VARIANTS}/codex_causal_rank_65.yaml", encoding="utf-8"))["overrides"]
    assert prod.get("vol_quality_tilt_negative_equity_mask") is True
    assert PipelineConfig().vol_quality_tilt_negative_equity_mask is False
    assert prod.get("tg_basis_events") == ARMS["s18_2_tg_basis_events"]["tg_basis_events"]
    assert PipelineConfig().tg_basis_events == {}


def test_s18_4_recert_variant_is_a_byte_copy_of_production():
    """§S18.2 재검증 런: overrides 가 production 과 동일(두 flip 포함), out_dir 만 다름."""
    prod = yaml.safe_load(open(f"{AI_PORT_VARIANTS}/codex_causal_rank_65.yaml", encoding="utf-8"))
    rec = yaml.safe_load(open(f"{AI_PORT_VARIANTS}/s18_4_flip2_recert.yaml", encoding="utf-8"))
    assert rec["overrides"] == prod["overrides"]
    assert rec["out_dir"] == "outputs/s18_4_flip2_recert"
    assert rec["tuning_mode"] == "production" and rec["portfolio_role"] == "diagnostic"


def test_eval_s18_frames():
    from scripts.eval_s18_arm import BASE, FRAMES, NEG_EQUITY_NAMES

    assert set(FRAMES) == set(ARMS)
    assert BASE == "s18_s0recert"
    assert FRAMES["s18_1_tilt_negative_equity"]["frame"] == "correctness"
    assert FRAMES["s18_2_tg_basis_events"]["frame"] == "correctness"
    assert FRAMES["s18_3_static_execution"]["frame"] == "execution"
    assert FRAMES["s18_3_static_execution"]["alpha_identical"] is True
    assert {"ABBV", "ORCL", "DELL", "CL"} <= set(NEG_EQUITY_NAMES)
