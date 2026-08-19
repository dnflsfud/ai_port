# -*- coding: utf-8 -*-
"""§S13.44 사전점검 (read-only): 고유변동성 spike × 리비전 방향 조건부 틸트.

결정 로그 §S13.44 사전등록(2026-08-14)의 P0/P1/P2를 실행한다. arm 미실행.

  spike: idio_vol_21d(EW mkt, rolling 63BD β — price.py 관용 미러)의
    자기 역사 rolling 252BD z ≥ +1.0.
  rev 방향(판정 proxy): 패널 eps_rev_ma_63d 부호. tg_mom_63d는 진단 비액션.
  P0 (커버리지): 리밸런싱당 평균 spike ≥ 10 AND 상향·하향 각 평균 ≥ 5.
  P1 (조건부 스프레드): spike∩상향 − spike∩하향 전진 21BD 평균 수익 차,
     mean > 0 AND t ≥ 2 AND 3분할 양 ≥ 2/3.
  P2 (잔차 직교성): s = spike×sign(rev)를 최종 score winsor-z에 잔차화한
     전진 21BD Spearman IC, mean > 0 AND t ≥ 2 AND 3분할 양 ≥ 2/3.

출력: outputs/s13_44_volspike_rev_precheck/summary.json
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

from scripts.preflight_s13_30_vol_quality import (  # noqa: E402
    _fwd_return,
    _summarise,
    _zscore_row,
)
from scripts.preflight_s13_36_turnover_tilt import (  # noqa: E402
    _spearman,
    residualize,
)

PKL = AI_PORT / "outputs" / "arm_s13_41a_optvol_cov" / "backtest_result.pkl"
VARIANT = AI_PORT / "variants" / "arm_s13_41a_optvol_cov.yaml"
OUT_DIR = AI_PORT / "outputs" / "s13_44_volspike_rev_precheck"

FWD = 21
BETA_W = 63
VOL_W = 21
Z_W, Z_MIN = 252, 126
SPIKE_Z = 1.0
MIN_SPIKE_AVG = 10.0
MIN_GROUP_AVG = 5.0
# UTC 표기(= KST 13:45:41) — §S13.43 summary.json의 동일 필드와 같은 렌더링
EXPECTED_WORKBOOK_MTIME = "2026-08-12 04:45:41"

REV_COLS = {"eps_rev": "eps_rev_ma_63d", "tg_mom": "tg_mom_63d"}
JUDGE_PROXY = "eps_rev"  # 사전지정 — tg_mom은 진단 비액션


# ---------------------------------------------------------------------------
# 순수 헬퍼 (단위테스트 대상)
# ---------------------------------------------------------------------------

def idio_vol_spike_z(returns: pd.DataFrame,
                     beta_w: int = BETA_W, vol_w: int = VOL_W,
                     z_w: int = Z_W, z_min: int = Z_MIN) -> pd.DataFrame:
    """idio_vol_21d의 자기 역사 z — price.py beta_63d/idio_vol 관용 미러.

    mkt = 유니버스 EW 평균, β = rolling cov/var, resid = r − β·mkt,
    idio_vol = resid rolling(vol_w) std × √252, z = rolling(z_w, min z_min).
    """
    mkt = returns.mean(axis=1)
    xy = returns.mul(mkt, axis=0)
    e_xy = xy.rolling(beta_w, min_periods=beta_w).mean()
    e_x = returns.rolling(beta_w, min_periods=beta_w).mean()
    e_y = mkt.rolling(beta_w, min_periods=beta_w).mean()
    cov_xy = e_xy - e_x.mul(e_y, axis=0)
    var_y = mkt.rolling(beta_w, min_periods=beta_w).var().replace(0, np.nan)
    beta = cov_xy.div(var_y, axis=0)
    resid = returns - beta.mul(mkt, axis=0)
    idio = resid.rolling(vol_w, min_periods=vol_w).std() * np.sqrt(252)
    m = idio.rolling(z_w, min_periods=z_min).mean()
    sd = idio.rolling(z_w, min_periods=z_min).std()
    return (idio - m) / sd.where(sd > 0)


def sparse_signal_row(spike_row: pd.Series, rev_row: pd.Series,
                      scored: pd.Index) -> pd.Series:
    """s = spike × sign(rev) ∈ {−1,0,+1} — scored 전 종목 위에 정의."""
    s = pd.Series(0.0, index=scored)
    spike = spike_row.reindex(scored).ge(SPIKE_Z).fillna(False)
    rev = rev_row.reindex(scored)
    s[spike & rev.gt(0)] = 1.0
    s[spike & rev.lt(0)] = -1.0
    return s


def group_spread_row(signal_row: pd.Series, fwd: pd.Series) -> dict:
    """spike∩상향 vs spike∩하향의 전진수익 평균 차 + 그룹 크기."""
    common = signal_row.index.intersection(fwd.dropna().index)
    s, y = signal_row.loc[common], fwd.loc[common]
    up, down = y[s > 0], y[s < 0]
    spread = (float(up.mean() - down.mean())
              if len(up) and len(down) else float("nan"))
    return {"n_spike": int((s != 0).sum()), "n_up": int(len(up)),
            "n_down": int(len(down)),
            "mean_up": float(up.mean()) if len(up) else float("nan"),
            "mean_down": float(down.mean()) if len(down) else float("nan"),
            "spread": spread}


def residual_ic_row(signal_row: pd.Series, score_row: pd.Series,
                    fwd: pd.Series) -> dict:
    """partial Spearman: s·fwd 양측을 score winsor-z 랭크에 잔차화한 상관.

    §S13.36의 단측 잔차화는 희소(±1/0) 신호에서 zero-block 잔차가
    −b·z_score로 score 단조가 되는 아티팩트가 있어(단위테스트로 확인,
    측정 전 정정 — 결정 로그 §S13.44) 양측 잔차화를 쓴다.
    """
    z_score = _zscore_row(score_row)
    common = (signal_row.index.intersection(z_score.index)
              .intersection(fwd.dropna().index))
    s, y, z = signal_row.loc[common], fwd.loc[common], z_score.loc[common]
    resid_s = residualize(s.rank(), z.rank())
    resid_y = residualize(y.rank(), z.rank())
    idx = resid_s.index.intersection(resid_y.index)
    if len(idx) < 3 or resid_s.loc[idx].std() == 0 or resid_y.loc[idx].std() == 0:
        resid_ic = float("nan")
    else:
        resid_ic = float(np.corrcoef(resid_s.loc[idx], resid_y.loc[idx])[0, 1])
    return {"raw_ic": _spearman(s, y), "resid_ic": resid_ic}


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
    base = pickle.load(open(PKL, "rb"))

    # ---------- 데이터 가정 검증 (§9: 어긋나면 중단·보고) ----------
    if getattr(base, "predictions", None) is None:
        sys.exit("[ABORT] harvest에 predictions 없음 — §S13.44 가정 파괴, 보고 요망")
    panel_cols = set(base.panel.columns)
    missing = [c for c in REV_COLS.values() if c not in panel_cols]
    if missing:
        sys.exit(f"[ABORT] harvest 패널에 {missing} 없음 — 보고 요망")
    if getattr(data, "returns_masked", None) is None:
        sys.exit("[ABORT] returns_masked 없음 — masked USD 수익률 가정 파괴, 보고 요망")

    returns = data.returns_masked
    preds = base.predictions
    rev_panels = {k: base.panel[c].unstack("ticker") for k, c in REV_COLS.items()}
    dates = sorted(base.portfolio_weights.keys())

    print(f"[load] {time.time()-t0:.0f}s  workbook {workbook_mtime}"
          f" (기대 {EXPECTED_WORKBOOK_MTIME}, 일치 {vintage_match})")
    print(f"[sample] rebalances={len(dates)}"
          f" ({pd.Timestamp(dates[0]).date()} ~ {pd.Timestamp(dates[-1]).date()})"
          f"  returns={returns.shape}")

    spike_z = idio_vol_spike_z(returns)
    spike_share = float(spike_z.ge(SPIKE_Z).sum().sum()
                        / spike_z.notna().sum().sum())
    print(f"[spike] 전 종목-일 spike 점유율 {spike_share:.1%}")

    rows = {k: [] for k in REV_COLS}
    skipped = 0
    for dt in dates:
        if dt not in spike_z.index or dt not in preds.index:
            skipped += 1
            continue
        score_row = preds.loc[dt]
        scored = score_row.index[score_row.notna()]
        fwd = _fwd_return(returns, dt, FWD)
        if fwd is None or len(scored) < 30:
            skipped += 1
            continue
        for k, rev_panel in rev_panels.items():
            if dt not in rev_panel.index:
                continue
            s = sparse_signal_row(spike_z.loc[dt], rev_panel.loc[dt], scored)
            row = {"date": str(pd.Timestamp(dt).date()),
                   **group_spread_row(s, fwd),
                   **residual_ic_row(s, score_row.loc[scored], fwd)}
            rows[k].append(row)

    summary_by_proxy = {}
    for k, rlist in rows.items():
        cov = {f: float(np.mean([r[f] for r in rlist]))
               for f in ("n_spike", "n_up", "n_down")}
        summary_by_proxy[k] = {
            "windows": len(rlist),
            "coverage_avg": {f: round(v, 2) for f, v in cov.items()},
            "spread": _summarise(rlist, "spread"),
            "raw_ic": _summarise(rlist, "raw_ic"),
            "residual_ic": _summarise(rlist, "resid_ic"),
            "mean_up": _summarise(rlist, "mean_up"),
            "mean_down": _summarise(rlist, "mean_down"),
        }

    j = summary_by_proxy[JUDGE_PROXY]
    p0 = bool(j["coverage_avg"]["n_spike"] >= MIN_SPIKE_AVG
              and j["coverage_avg"]["n_up"] >= MIN_GROUP_AVG
              and j["coverage_avg"]["n_down"] >= MIN_GROUP_AVG)
    sp, ric = j["spread"], j["residual_ic"]
    p1 = bool(np.isfinite(sp["mean"]) and sp["mean"] > 0
              and sp["t"] >= 2.0 and sp["subperiod_pos"] >= 2)
    p2 = bool(np.isfinite(ric["mean"]) and ric["mean"] > 0
              and ric["t"] >= 2.0 and ric["subperiod_pos"] >= 2)
    verdict = "PROCEED" if (p0 and p1 and p2) else "SHELVE"

    for k, s in summary_by_proxy.items():
        tag = "판정" if k == JUDGE_PROXY else "진단"
        print(f"[{k}/{tag}] cover={s['coverage_avg']}"
              f" spread mean={s['spread']['mean']:+.4f} t={s['spread']['t']:+.2f}"
              f" thirds+={s['spread']['subperiod_pos']}"
              f" | resid_ic mean={s['residual_ic']['mean']:+.4f}"
              f" t={s['residual_ic']['t']:+.2f}"
              f" thirds+={s['residual_ic']['subperiod_pos']}")
    print(f"[gates] P0={p0} P1={p1} P2={p2}")

    summary = {
        "preregistration": "decision log §S13.44 (2026-08-14)",
        "workbook_mtime": workbook_mtime,
        "vintage_match_0812": vintage_match,
        "harvest": str(PKL.relative_to(AI_PORT)),
        "judge_proxy": JUDGE_PROXY,
        "spike_z_threshold": SPIKE_Z,
        "spike_share_all_cells": round(spike_share, 4),
        "skipped_dates": skipped,
        "by_proxy": summary_by_proxy,
        "gates": {"P0_coverage": p0, "P1_spread": p1,
                  "P2_residual_ic": p2, "verdict": verdict},
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    for k, rlist in rows.items():
        pd.DataFrame(rlist).to_csv(OUT_DIR / f"windows_{k}.csv", index=False)
    print(f"\nVERDICT: {verdict}  ({time.time()-t0:.0f}s)  saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
