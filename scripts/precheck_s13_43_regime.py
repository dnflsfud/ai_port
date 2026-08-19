# -*- coding: utf-8 -*-
"""§S13.43 사전점검 (read-only): 옵션 기반 risk on-off 레짐 신호.

결정 로그 §S13.43 사전등록(2026-08-14)의 P0/P1/P2를 실행한다. arm 미실행.

  합성 신호(고정 6성분 등가중, + = risk-off): vix_tp_z, vix_z, skew_idx_z,
    spx_iv_ts_z, cs_iv_breadth, cs_skew_chg. comp_z = expanding z(번인 252).
  타깃(익일부터 21BD): T1 bm 누적수익률, T2 bm 실현변동성, T3 북 active.
  P0 (분리력): 비중첩 21BD 시점, comp_z 상위 20% vs 나머지 t ≥ 2 AND
     3분할 부호 일관. 방향 고정: T1·T3 하락, T2 상승.
  P1 (증분 선행성): 일별, 직전 21BD bm vol·수익 통제 + NW(lag 21) t ≥ 2.
  P2 (발화율): comp_z > +1 점유율 5~30% AND 평균 지속 ≥ 5BD.

출력: outputs/s13_43_regime_precheck/summary.json
"""

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

AI_PORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_PORT))

PKL = AI_PORT / "outputs" / "arm_s13_41a_optvol_cov" / "backtest_result.pkl"
VARIANT = AI_PORT / "variants" / "arm_s13_41a_optvol_cov.yaml"
OUT_DIR = AI_PORT / "outputs" / "s13_43_regime_precheck"

FWD = 21
BURN_IN = 252
MIN_COMPONENTS = 4
TOP_SHARE = 0.20
STATE_Z = 1.0
# UTC 표기(= KST 13:45:41) — §S13.42 summary.json의 동일 필드와 같은 렌더링
EXPECTED_WORKBOOK_MTIME = "2026-08-12 04:45:41"
ANN = float(np.sqrt(252.0))

# 성분명 → 판정 방향과 무관한 진단 라벨 (사전등록 §S13.43 표와 동일 순서)
COMPONENTS = ("vix_tp_z", "vix_z", "skew_idx_z",
              "spx_iv_ts_z", "cs_iv_breadth", "cs_skew_chg")
# 타깃별 사전 고정 방향 (+ = risk-off일수록 타깃 상승)
DIRECTIONS = {"T1_bm_ret": -1, "T2_bm_vol": +1, "T3_active_ret": -1}


# ---------------------------------------------------------------------------
# 순수 헬퍼 (단위테스트 대상)
# ---------------------------------------------------------------------------

def z252(s: pd.Series, window: int = 252, min_periods: int = 126) -> pd.Series:
    """자기 역사 rolling z-score."""
    m = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std()
    return (s - m) / sd.where(sd > 0)


def expanding_z(s: pd.Series, burn_in: int = BURN_IN) -> pd.Series:
    """point-in-time expanding z (번인 이전 NaN)."""
    m = s.expanding(min_periods=burn_in).mean()
    sd = s.expanding(min_periods=burn_in).std()
    return (s - m) / sd.where(sd > 0)


def build_composite(components: pd.DataFrame,
                    min_components: int = MIN_COMPONENTS) -> pd.Series:
    """등가중 평균 — 가용 성분 < min_components 시점은 NaN."""
    n = components.notna().sum(axis=1)
    return components.mean(axis=1).where(n >= min_components)


def _max_drawdown(rets: np.ndarray) -> float:
    path = np.concatenate([[1.0], np.cumprod(1.0 + rets)])
    runmax = np.maximum.accumulate(path)
    return float(np.min(path / runmax - 1.0))


def forward_targets(bm: pd.Series, active: pd.Series, fwd: int = FWD):
    """신호일 익일부터 fwd BD 타깃 — (T1 bm 누적, T2 bm vol 연율, T3 active
    누적, 진단 창내 최대낙폭). rolling→shift(-fwd)로 t+1..t+fwd만 사용."""
    t1 = ((1.0 + bm).rolling(fwd).apply(np.prod, raw=True) - 1.0).shift(-fwd)
    t2 = (bm.rolling(fwd).std() * ANN).shift(-fwd)
    t3 = ((1.0 + active).rolling(fwd).apply(np.prod, raw=True) - 1.0).shift(-fwd)
    dd = bm.rolling(fwd).apply(_max_drawdown, raw=True).shift(-fwd)
    return t1, t2, t3, dd


def separation_test(signal: pd.Series, target: pd.Series,
                    top_share: float = TOP_SHARE, direction: int = +1) -> dict:
    """상위 top_share vs 나머지 평균차 Welch t + 3분할(각 분할 내 자체 상위
    분위) 부호. t·thirds는 direction 곱으로 부호 정렬 — pass = t≥2 AND 전부 양."""
    pair = pd.concat([signal.rename("s"), target.rename("y")], axis=1).dropna()

    def _diff(chunk: pd.DataFrame) -> float:
        thr = chunk["s"].quantile(1.0 - top_share)
        top, rest = chunk[chunk["s"] >= thr], chunk[chunk["s"] < thr]
        return float(top["y"].mean() - rest["y"].mean())

    thr = pair["s"].quantile(1.0 - top_share)
    top, rest = pair[pair["s"] >= thr], pair[pair["s"] < thr]
    diff = float(top["y"].mean() - rest["y"].mean())
    se = float(np.sqrt(top["y"].var(ddof=1) / len(top)
                       + rest["y"].var(ddof=1) / len(rest)))
    t_adj = direction * diff / se if se > 0 else float("nan")
    cut = np.linspace(0, len(pair), 4, dtype=int)
    thirds = [round(direction * _diff(pair.iloc[cut[i]:cut[i + 1]]), 6)
              for i in range(3)]
    return {"n": int(len(pair)), "n_top": int(len(top)),
            "mean_top": round(float(top["y"].mean()), 6),
            "mean_rest": round(float(rest["y"].mean()), 6),
            "t": round(float(t_adj), 3), "thirds": thirds,
            "pass": bool(t_adj >= 2.0 and all(s > 0 for s in thirds))}


def newey_west_t(y: np.ndarray, x: np.ndarray, lag: int = FWD):
    """절편 포함 OLS + Newey-West(Bartlett) t. 반환 (coef, t) — [절편, ...]."""
    y = np.asarray(y, float)
    design = np.column_stack([np.ones(len(y)), np.asarray(x, float)])
    xtx_inv = np.linalg.inv(design.T @ design)
    coef = xtx_inv @ design.T @ y
    u = y - design @ coef
    g = design * u[:, None]
    s = g.T @ g
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)
        gamma = g[l:].T @ g[:-l]
        s += w * (gamma + gamma.T)
    v = xtx_inv @ s @ xtx_inv
    se = np.sqrt(np.diag(v))
    return coef, coef / np.where(se > 0, se, np.nan)


def state_stats(state: pd.Series) -> dict:
    """bool 상태의 점유율·연속 스펠 통계."""
    state = state.astype(bool)
    runs = (state != state.shift()).cumsum()
    spells = state.groupby(runs).agg(["all", "size"])
    lengths = spells.loc[spells["all"], "size"]
    return {"share": float(state.mean()),
            "n_spells": int(len(lengths)),
            "mean_spell": float(lengths.mean()) if len(lengths) else 0.0}


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

def main() -> None:
    from run_variant import compose_config, load_manifest
    from src.data_loader import UniverseData

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cfg = compose_config(load_manifest(VARIANT))
    workbook_mtime = pd.Timestamp(
        Path(cfg.data_path).stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")
    vintage_match = workbook_mtime == EXPECTED_WORKBOOK_MTIME
    data = UniverseData(cfg.data_path, config=cfg)
    result = pickle.load(open(PKL, "rb"))
    bm = result.benchmark_returns.dropna()
    active = (result.portfolio_returns - result.benchmark_returns).reindex(bm.index)

    # ---------- 데이터 가정 검증 (§9: 어긋나면 중단·보고) ----------
    lvl = data.factor_data.get("Factor_PX_LAST")
    if lvl is None:
        sys.exit("[ABORT] Factor_PX_LAST 시트 없음 — §S13.43 가정 파괴, 보고 요망")
    missing = [c for c in ("VIX", "VIX3M", "SKEW") if c not in lvl.columns]
    if missing:
        sys.exit(f"[ABORT] Factor_PX_LAST에 {missing} 컬럼 없음 — 보고 요망")
    sheets = {}
    for name in ("iv30_z", "iv_term_structure_z", "downside_skew_chg_5d"):
        try:
            sheets[name] = data.get_sheet(name)
        except KeyError:
            sys.exit(f"[ABORT] 시트 {name} 없음 — §S13.37 계층 가정 파괴, 보고 요망")
    # SPX 열은 비유니버스 열 — 로더 정렬 시트가 버리므로 raw에서 취득
    ts_raw = data.raw["iv_term_structure_z"]
    spx_cols = [c for c in ts_raw.columns if "SPX" in str(c)]
    if not spx_cols:
        sys.exit("[ABORT] iv_term_structure_z(raw)에 SPX 열 없음 — 보고 요망")
    spx_col = spx_cols[0]
    spx_ts = ts_raw[spx_col]
    spx_ts.index = pd.to_datetime(spx_ts.index)

    dates = data.returns.index
    universe = [t for t in data.tickers if t in sheets["iv30_z"].columns]

    comp_frame = pd.DataFrame({
        "vix_tp_z": z252(lvl["VIX"] - lvl["VIX3M"]),
        "vix_z": z252(lvl["VIX"]),
        "skew_idx_z": z252(lvl["SKEW"]),
        "spx_iv_ts_z": spx_ts,
        "cs_iv_breadth": sheets["iv30_z"][universe].median(axis=1),
        "cs_skew_chg": z252(sheets["downside_skew_chg_5d"][universe].mean(axis=1)),
    }).reindex(dates)

    coverage = {}
    for c in COMPONENTS:
        col = comp_frame[c].dropna()
        coverage[c] = {
            "nonnan_share": round(float(comp_frame[c].notna().mean()), 3),
            "first_valid": str(col.index[0].date()) if len(col) else None,
        }
    print(f"[load] {time.time()-t0:.0f}s  workbook {workbook_mtime}"
          f" (기대 {EXPECTED_WORKBOOK_MTIME}, 일치 {vintage_match})")
    print(f"[coverage] {json.dumps(coverage, ensure_ascii=False)}")

    composite = build_composite(comp_frame)
    comp_z = expanding_z(composite)

    t1, t2, t3, dd = forward_targets(bm, active)
    trail_vol = bm.rolling(FWD).std() * ANN
    trail_ret = (1.0 + bm).rolling(FWD).apply(np.prod, raw=True) - 1.0

    panel = pd.DataFrame({
        "comp_z": comp_z, "T1_bm_ret": t1, "T2_bm_vol": t2,
        "T3_active_ret": t3, "fwd_dd": dd,
        "trail_vol": trail_vol, "trail_ret": trail_ret,
    }).dropna()
    print(f"[sample] 일별 유효 {len(panel)}개"
          f"  ({panel.index[0].date()} ~ {panel.index[-1].date()})")

    # ---------------- P0: 비중첩 21BD 분리력 ----------------
    nonoverlap = panel.iloc[::FWD]
    p0 = {tgt: separation_test(nonoverlap["comp_z"], nonoverlap[tgt],
                               direction=DIRECTIONS[tgt])
          for tgt in DIRECTIONS}
    for tgt, r in p0.items():
        print(f"[P0] {tgt}: n={r['n']} top={r['mean_top']:+.4f}"
              f" rest={r['mean_rest']:+.4f} t={r['t']:+.2f}"
              f" thirds={r['thirds']} -> {'PASS' if r['pass'] else 'FAIL'}")

    # ---------------- P1: 일별 증분 선행성 (NW lag21) ----------------
    x = panel[["trail_vol", "trail_ret", "comp_z"]].values
    p1 = {}
    for tgt, direction in DIRECTIONS.items():
        coef, tstats = newey_west_t(panel[tgt].values, x)
        t_adj = float(direction * tstats[3])
        p1[tgt] = {"coef_comp_z": round(float(coef[3]), 6),
                   "t_nw": round(t_adj, 3), "pass": bool(t_adj >= 2.0)}
        print(f"[P1] {tgt}: coef={coef[3]:+.5f} t_adj={t_adj:+.2f}"
              f" -> {'PASS' if t_adj >= 2.0 else 'FAIL'}")

    # ---------------- P2: 발화율·지속 ----------------
    st = state_stats(panel["comp_z"] > STATE_Z)
    p2_pass = bool(0.05 <= st["share"] <= 0.30 and st["mean_spell"] >= 5.0)
    print(f"[P2] share={st['share']:.1%} spells={st['n_spells']}"
          f" mean_spell={st['mean_spell']:.1f}BD"
          f" -> {'PASS' if p2_pass else 'FAIL'}")

    # ---------------- 성분별 개별 리드 (진단, 비액션) ----------------
    comp_diag = {}
    for c in COMPONENTS:
        row = {}
        sub = pd.concat([panel[["trail_vol", "trail_ret"]],
                         comp_frame[c].rename("sig"), panel[list(DIRECTIONS)]],
                        axis=1).dropna()
        for tgt, direction in DIRECTIONS.items():
            _, ts = newey_west_t(sub[tgt].values,
                                 sub[["trail_vol", "trail_ret", "sig"]].values)
            row[tgt] = round(float(direction * ts[3]), 2)
        comp_diag[c] = row

    passed = [tgt for tgt in DIRECTIONS if p0[tgt]["pass"] and p1[tgt]["pass"]]
    verdict = "SHELVE" if (not passed or not p2_pass) \
        else "PASS: " + ",".join(passed)

    summary = {
        "preregistration": "decision log §S13.43 (2026-08-14)",
        "workbook_mtime": workbook_mtime,
        "vintage_match_0812": vintage_match,
        "book_series": str(PKL.relative_to(AI_PORT)),
        "coverage": coverage,
        "n_daily": int(len(panel)),
        "n_nonoverlap": int(len(nonoverlap)),
        "p0_separation": p0,
        "p1_incremental_lead": p1,
        "p2_state": {**{k: round(v, 4) if isinstance(v, float) else v
                        for k, v in st.items()}, "pass": p2_pass},
        "component_diagnostics_t_nw": comp_diag,
        "fwd_dd_mean_riskoff": round(float(
            panel.loc[panel["comp_z"] > STATE_Z, "fwd_dd"].mean()), 4),
        "fwd_dd_mean_all": round(float(panel["fwd_dd"].mean()), 4),
        "verdict": verdict,
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nVERDICT: {verdict}  ({time.time()-t0:.0f}s)  saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
