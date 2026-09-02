# -*- coding: utf-8 -*-
"""§S17.1 피처 채널 정확성 사전점검 (read-only, 백테스트 0회).

결정 로그 §S17.1 사전등록(2026-09-02)의 E1(정확성 확증) — arm 실행 없이 워크북에서
플래그 OFF/ON 피처를 직접 만들어 비교한다(피처는 데이터+플래그로 결정적).

  T-01 (s17_coverage_gap_fix_enabled):
    A1 결함 재현: VRT 커버리지 갭창에서 OFF tg_upside z 가 +5.0 클립 상수(≥1행)
    A2 수정 확인: 같은 창에서 ON tg_upside 가 전부 NaN(패널 빌더가 날짜별 median 채움)
    A3 압축 해소: 갭창의 타 종목 z 횡단면 std 중앙값 ≥ 0.90 (OFF ≈ 0.4)
    관측: 갭 셀 수(종목별), tg_mom_63d 갭 종료 후 63BD 아티팩트(≤ −4.99 비율)
  M2 피처 (s17_beta_overlap_enabled, K=5):
    B1 아시아 16종 beta_63d 중앙값 ON/OFF ≥ 2.0 (보고서 실측 0.324 → Dimson 0.997)
    B2 미국 종목 beta_63d 중앙값 |ON/OFF − 1| ≤ 0.15 (동시 거래 종목 보존)
    관측: EU 비, idio_vol_63d 지역별 비, vol-quality 틸트 상위 터실 멤버십 변화/일

출력: outputs/s17_prechecks/feature_fixes_accuracy.json
"""

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AI_PORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_PORT))

OUT = AI_PORT / "outputs" / "s17_prechecks" / "feature_fixes_accuracy.json"
VARIANT = AI_PORT / "variants" / "codex_causal_rank_65.yaml"

ASIA_CODES = {"KS", "JP"}
EU_CODES = {"FP", "GR", "NA", "SW", "LN", "DC", "SM", "IM"}
T01_CS_STD_MIN = 0.90
BETA_ASIA_RATIO_MIN = 2.0
BETA_US_BAND = 0.15
GAP_TICKER = "VRT"


def region_of(code: str) -> str:
    if code in ASIA_CODES:
        return "ASIA"
    if code in EU_CODES:
        return "EU"
    return "US"


def evaluate_t01(off_plus5_rows: int, on_nan_rows: int, window_rows: int,
                 cs_std_on: float) -> dict:
    std_ok = bool(np.isfinite(cs_std_on) and cs_std_on >= T01_CS_STD_MIN)
    return {
        "a1_defect_reproduced": bool(off_plus5_rows > 0),
        "a2_gap_window_nan": bool(window_rows > 0 and on_nan_rows == window_rows),
        "a3_cs_std_restored": std_ok,
        "t01_pass": bool(off_plus5_rows > 0 and window_rows > 0
                         and on_nan_rows == window_rows and std_ok),
    }


def evaluate_beta(ratio_asia: float, ratio_us: float) -> dict:
    asia_ok = bool(np.isfinite(ratio_asia) and ratio_asia >= BETA_ASIA_RATIO_MIN)
    us_ok = bool(np.isfinite(ratio_us) and abs(ratio_us - 1.0) <= BETA_US_BAND)
    return {"b1_asia_ratio_ok": asia_ok, "b2_us_preserved": us_ok,
            "beta_pass": bool(asia_ok and us_ok)}


def _region_median(frame: pd.DataFrame, regions: dict, region: str) -> float:
    cols = [t for t in frame.columns if regions.get(t) == region]
    if not cols:
        return float("nan")
    return float(np.nanmedian(frame[cols].values))


def _top_tercile(idio: pd.DataFrame) -> pd.DataFrame:
    return idio.rank(axis=1, pct=True) >= 2.0 / 3.0


def _p95(frame: pd.DataFrame) -> float:
    return float(frame.abs().quantile(0.95, axis=1).median()) if len(frame) else float("nan")


def main() -> None:
    import yaml

    from src.data_loader import UniverseData
    from src.features.price import build_price_features
    from src.features.sellside import build_sellside_features
    from src.features.utils import cross_sectional_zscore
    from src.harness import build_override_config, inject_config

    overrides = dict(yaml.safe_load(VARIANT.read_text(encoding="utf-8"))["overrides"])
    cfg_off = build_override_config(overrides)
    inject_config(cfg_off)
    cfg_t01 = build_override_config({**overrides, "s17_coverage_gap_fix_enabled": True})
    cfg_beta = build_override_config({**overrides, "s17_beta_overlap_enabled": True})

    data = UniverseData(cfg_off.data_path, config=cfg_off)
    tickers = list(data.tickers)
    regions = {t: region_of(str(code)) for t, code in data.meta["exchange_code"].items()}
    n_region = {r: sum(1 for t in tickers if regions.get(t) == r) for r in ("ASIA", "EU", "US")}

    # ------------------------------------------------------------------ T-01
    feats = build_sellside_features(data, config=cfg_off)
    up_off = feats["tg_upside"].reindex(columns=tickers)
    mom_off = feats["tg_mom_63d"].reindex(columns=tickers)
    del feats
    gc.collect()
    feats = build_sellside_features(data, config=cfg_t01)
    up_on = feats["tg_upside"].reindex(columns=tickers)
    mom_on = feats["tg_mom_63d"].reindex(columns=tickers)
    del feats
    gc.collect()

    observed = data.raw_sheet_observed_mask("Factset_TG_Price").reindex(
        index=up_off.index, columns=tickers).fillna(False).astype(bool)
    covered = observed.cummax(axis=0)
    gap = (~covered) & up_off.notna()               # 상장 후·커버리지 전·로더가 메운 셀
    gap_counts = gap.sum().sort_values(ascending=False)
    gap_counts = gap_counts[gap_counts > 0]
    changed = int(((up_off != up_on) & ~(up_off.isna() & up_on.isna())).sum().sum())

    z_off = cross_sectional_zscore(up_off).clip(-5.0, 5.0)
    z_on = cross_sectional_zscore(up_on).clip(-5.0, 5.0)
    win = gap.index[gap[GAP_TICKER].values] if GAP_TICKER in gap.columns else gap.index[:0]
    others = [t for t in tickers if t != GAP_TICKER]
    if len(win):
        off_plus5 = int((z_off.loc[win, GAP_TICKER] >= 4.999).sum())
        on_nan = int(up_on.loc[win, GAP_TICKER].isna().sum())
        cs_std_off = float(z_off.loc[win, others].std(axis=1).median())
        cs_std_on = float(z_on.loc[win, others].std(axis=1).median())
        i0, i1 = up_off.index.get_loc(win[0]), up_off.index.get_loc(win[-1])
        before = up_off.index[max(0, i0 - 63): i0]
        after = up_off.index[i1 + 1: i1 + 64]
        p95 = {"off": _p95(z_off.loc[win, others]), "on": _p95(z_on.loc[win, others]),
               "off_63d_before": _p95(z_off.loc[before, others]),
               "off_63d_after": _p95(z_off.loc[after, others])}
        zm_off = cross_sectional_zscore(mom_off).clip(-5.0, 5.0)
        zm_on = cross_sectional_zscore(mom_on).clip(-5.0, 5.0)
        mom_art = {"off": float((zm_off.loc[after, GAP_TICKER] <= -4.99).mean()) if len(after) else float("nan"),
                   "on": float((zm_on.loc[after, GAP_TICKER] <= -4.99).mean()) if len(after) else float("nan")}
    else:
        off_plus5 = on_nan = 0
        cs_std_off = cs_std_on = float("nan")
        p95 = {"off": float("nan"), "on": float("nan"),
               "off_63d_before": float("nan"), "off_63d_after": float("nan")}
        mom_art = {"off": float("nan"), "on": float("nan")}
    t01 = evaluate_t01(off_plus5, on_nan, int(len(win)), cs_std_on)
    t01_out = {
        "gap_ticker": GAP_TICKER,
        "gap_window": [str(win[0].date()), str(win[-1].date())] if len(win) else None,
        "gap_window_rows": int(len(win)),
        "off_vrt_plus5_rows": off_plus5,
        "on_vrt_nan_rows": on_nan,
        "others_cs_std_median": {"off": round(cs_std_off, 4), "on": round(cs_std_on, 4),
                                 "bar": T01_CS_STD_MIN},
        "others_abs_z_p95_median": {k: round(v, 4) for k, v in p95.items()},
        "tg_mom_63d_vrt_post_gap_share_le_minus4_99": {k: round(v, 4) for k, v in mom_art.items()},
        "coverage_gap_cells_total": int(gap.sum().sum()),
        "changed_cells_total": changed,
        "coverage_gap_cells_by_ticker": {str(t): int(v) for t, v in gap_counts.head(20).items()},
        **t01,
    }
    del up_off, up_on, mom_off, mom_on, z_off, z_on
    gc.collect()

    # ------------------------------------------------------------- M2 feature
    pf = build_price_features(data, config=cfg_off)
    b_off = pf["beta_63d"].reindex(columns=tickers)
    i_off = pf["idio_vol_63d"].reindex(columns=tickers)
    del pf
    gc.collect()
    pf = build_price_features(data, config=cfg_beta)
    b_on = pf["beta_63d"].reindex(columns=tickers)
    i_on = pf["idio_vol_63d"].reindex(columns=tickers)
    del pf
    gc.collect()
    common = b_on.dropna(how="all").index.intersection(b_off.dropna(how="all").index)
    b_off, b_on, i_off, i_on = b_off.loc[common], b_on.loc[common], i_off.loc[common], i_on.loc[common]

    beta_med = {r: {"off": _region_median(b_off, regions, r), "on": _region_median(b_on, regions, r)}
                for r in ("ASIA", "EU", "US")}
    idio_med = {r: {"off": _region_median(i_off, regions, r), "on": _region_median(i_on, regions, r)}
                for r in ("ASIA", "EU", "US")}
    ratio = {r: (v["on"] / v["off"] if np.isfinite(v["off"]) and v["off"] != 0 else float("nan"))
             for r, v in beta_med.items()}
    idio_ratio = {r: (v["on"] / v["off"] if np.isfinite(v["off"]) and v["off"] != 0 else float("nan"))
                  for r, v in idio_med.items()}
    asia_names = [t for t in tickers if regions.get(t) == "ASIA"]
    per_name = {t: {"off": round(float(np.nanmedian(b_off[t])), 3),
                    "on": round(float(np.nanmedian(b_on[t])), 3)} for t in asia_names}
    tercile_change = float((_top_tercile(i_off) ^ _top_tercile(i_on)).sum(axis=1).mean())
    beta = evaluate_beta(ratio["ASIA"], ratio["US"])
    beta_out = {
        "n_region": n_region,
        "beta_63d_median": {r: {k: round(v, 4) for k, v in d.items()} for r, d in beta_med.items()},
        "beta_ratio_on_over_off": {r: round(v, 4) for r, v in ratio.items()},
        "asia_per_name_beta_median": per_name,
        "idio_vol_63d_median": {r: {k: round(v, 4) for k, v in d.items()} for r, d in idio_med.items()},
        "idio_ratio_on_over_off": {r: round(v, 4) for r, v in idio_ratio.items()},
        "vol_quality_top_tercile_membership_change_per_date": round(tercile_change, 3),
        "valid_dates": int(len(common)),
        "bars": {"asia_ratio_min": BETA_ASIA_RATIO_MIN, "us_band": BETA_US_BAND},
        **beta,
    }

    out = {
        "preregistration": "decision log §S17.1 (2026-09-02)",
        "workbook_mtime_kst": pd.Timestamp(Path(cfg_off.data_path).stat().st_mtime, unit="s", tz="UTC")
        .tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S"),
        "t01": t01_out,
        "beta_overlap": beta_out,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nT-01: {'PASS' if t01['t01_pass'] else 'FAIL'}  "
          f"(VRT +5 rows OFF {off_plus5}, ON NaN {on_nan}/{len(win)}, "
          f"others cs-std {cs_std_off:.3f} -> {cs_std_on:.3f})")
    print(f"BETA: {'PASS' if beta['beta_pass'] else 'FAIL'}  "
          f"(ASIA {beta_med['ASIA']['off']:.3f} -> {beta_med['ASIA']['on']:.3f} "
          f"x{ratio['ASIA']:.2f}, US x{ratio['US']:.3f})")


if __name__ == "__main__":
    main()
