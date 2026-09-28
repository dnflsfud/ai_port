# -*- coding: utf-8 -*-
"""§S21 사전점검 (read-only): FY2 매출 리비전의 조건부·비선형·기간 형태 4종.

결정 로그 §S21 (2026-09-28, 사전등록 커밋 fe38c35). arm 없음 — 백테스트 0회.
데이터는 §S20 과 동일 파일의 스냅샷(`--snapshot-dir` 의 backtest_result.pkl ·
ai_signal_data.xlsx, SHA-256 고정). 클리닝·피처족은 §S20 경로 그대로이고, 모든 후보
피처에 1영업일 지연(shift(1))을 적용한다(production predictions = t−1 정보, targets[t]
= t+1..t+20).

후보(단일 정의):
  K1  NTM 블렌드 (1−p)·FY1_ma63 + p·FY2_ma63, p = 회계연도 경과율(롤 월 검출)
  K2  FY2_ma63 의 G2 증분 — Technology + Communication Services 부분집합 안
  K3  FY2_ma63 의 G2 증분 — 60BD 타깃(T20 가법 체인), NW lag 60
  K4  FY1_ma63 · 1[sign(FY1_ma63) = sign(FY2_ma63)]
결정 게이트 G2*: score + FY1_ma63 동시 잔차 IC NW|t| ≥ 2.81 AND 3분할 부호 일관.

출력: outputs/s21_precheck.json
"""
import argparse
import hashlib
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

AI_PORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_PORT))

from scripts.precheck_s20_fy2_sales_revision import (  # noqa: E402
    _Shim,
    median_cs_corr,
    nw_t,
    read_sheet,
    residualize,
    rowwise_rank_corr,
    thirds,
)

OUT = AI_PORT / "outputs" / "s21_precheck.json"
T_BAR = 2.81          # Bonferroni: 양측 0.05 / (§S20 6 + §S21 4)
T_NOMINAL = 2.0
LONG_DURATION_SECTORS = ("Technology", "Communication Services")
EXPECTED_SHA = {
    "backtest_result.pkl": "4fa3df6854d5905c81f59f80bebc34e937df1a7812a8cd2d9fb653dac123765d",
    "ai_signal_data.xlsx": "7b330ca7c8b9044c96a9d8e49e9699092eef1495b099b64c84e4ed81b9223d32",
}
# 결산월~보고월 ±1 (사전등록 S2)
ROLL_MONTH_ALLOWED = {
    "AAPL": {9, 10, 11, 12}, "MSFT": {6, 7, 8}, "NVDA": {1, 2, 3}, "WMT": {1, 2, 3},
    "ORCL": {5, 6, 7}, "CSCO": {7, 8, 9}, "COST": {8, 9, 10}, "AVGO": {10, 11, 12, 1},
    "NKE": {5, 6, 7}, "ADBE": {11, 12, 1}, "JPM": {12, 1, 2}, "GOOGL": {12, 1, 2},
}
DEFAULT_ROLL_MONTH = 2
warnings.filterwarnings("ignore", category=RuntimeWarning)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_roll_events(s1: pd.DataFrame, s2: pd.DataFrame, dedupe_days: int = 300) -> dict:
    """1FY 가 직전 2FY 로 넘어가는 날(회계연도 롤)을 종목별로 검출(사전등록 임계값)."""
    l1 = np.log(s1.where(s1 > 0))
    l2 = np.log(s2.where(s2 > 0))
    ev = (
        ((l1 - l2.shift(1)).abs() < 0.03)
        & ((l1 - l1.shift(1)).abs() > 0.05)
        & ((l2.shift(1) - l1.shift(1)).abs() > 0.04)
    )
    events = {}
    for col in ev.columns:
        kept = []
        for d in ev.index[ev[col].to_numpy()]:
            if not kept or (d - kept[-1]).days > dedupe_days:
                kept.append(d)
        events[col] = kept
    return events


def roll_month(dates: list, min_events: int = 2):
    """검출 롤일의 최빈 월(동률은 작은 월). 검출 수 미달이면 None."""
    if len(dates) < min_events:
        return None
    counts = pd.Series([d.month for d in dates]).value_counts()
    top = counts[counts == counts.max()].index
    return int(min(top))


def fiscal_progress(index: pd.DatetimeIndex, columns, m_roll: dict) -> pd.DataFrame:
    """p = ((month(t) − m_roll) mod 12) / 12 — 롤 직후 0, 다음 롤 직전 11/12."""
    months = np.asarray(index.month)[:, None]
    m = np.array([m_roll[c] for c in columns])[None, :]
    return pd.DataFrame(((months - m) % 12) / 12.0, index=index, columns=columns)


def chain_target(t20: pd.DataFrame, n_blocks: int, step: int = 20) -> pd.DataFrame:
    """T20 의 비중첩 가법 체인: T(20·n) ≈ Σ_k T20(t + k·step). 한 블록이라도 NaN 이면 NaN."""
    out = t20.copy()
    for k in range(1, n_blocks):
        out = out + t20.shift(-k * step)
    return out


def agreement_gate(f1: pd.DataFrame, f2: pd.DataFrame) -> pd.DataFrame:
    """FY1 신호를 FY2 부호가 같을 때만 유지(불일치 0). 어느 한쪽 NaN 이면 NaN."""
    agree = np.sign(f1) == np.sign(f2)
    return f1.where(agree, 0.0).where(f1.notna() & f2.notna())


def ic_row(ic: pd.Series, lag: int) -> dict:
    t = nw_t(ic, lag=lag)
    th = thirds(ic)
    mean = float(ic.mean())
    consistent = bool(np.isfinite(t) and all(np.sign(x) == np.sign(mean) and x != 0 for x in th))
    return {
        "n_days": int(len(ic)), "mean": round(mean, 5), "t_nw": round(t, 2),
        "thirds": th, "thirds_sign_consistent": consistent,
    }


def gate_verdict(row: dict) -> str:
    t = row["t_nw"]
    if np.isfinite(t) and abs(t) >= T_BAR and row["thirds_sign_consistent"]:
        return "PASS"
    if np.isfinite(t) and abs(t) >= T_NOMINAL:
        return "FAIL (nominal |t|>=2 only, non-actionable)"
    return "FAIL"


def bucket_ic(resid: pd.DataFrame, target: pd.DataFrame, labels: pd.DataFrame, lag: int) -> dict:
    """잔차 IC 를 셀 라벨(버킷)별로 분리 — 날짜별 버킷 내 ≥10 종목일 때만."""
    out = {}
    lab = labels.reindex(index=resid.index, columns=resid.columns)
    flat = lab.to_numpy().ravel()
    for b in sorted(pd.unique(flat[pd.notna(flat)])):
        ic = rowwise_rank_corr(resid.where(lab == b), target)
        out[str(b)] = ic_row(ic, lag)
    return out


def sample_series(s: pd.Series, every: int = 21) -> list:
    s = s.dropna()
    return [[d.strftime("%Y-%m-%d"), round(float(v), 5)] for d, v in s.iloc[::every].items()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-dir", required=True, type=Path)
    args = ap.parse_args()
    snap = args.snapshot_dir
    pkl_path, bundle = snap / "backtest_result.pkl", snap / "ai_signal_data.xlsx"

    from src.features.sellside import build_bounded_revision_features, get_cleaned_revision
    from src.harness import build_override_config

    t0 = time.time()
    shas = {name: sha256(snap / name) for name in EXPECTED_SHA}
    s3 = all(shas[k] == v for k, v in EXPECTED_SHA.items())
    print(f"S3 sha match: {s3}")
    if not s3:
        raise SystemExit(f"S3 FAIL: snapshot SHA mismatch {shas}")

    manifest = yaml.safe_load(
        (AI_PORT / "variants" / "codex_causal_rank_65.yaml").read_text(encoding="utf-8"))
    cfg = build_override_config(dict(manifest["overrides"]))
    r = pickle.load(open(pkl_path, "rb"))
    t20, score = r.targets, r.predictions
    panel_fy1 = r.panel["sales_rev_ma_63d"].unstack("ticker")
    lag_days = int(getattr(r, "execution_signal_lag_days", 0))
    del r
    print(f"pkl ok {time.time()-t0:.0f}s targets {t20.shape} lag {lag_days}")
    idx, cols = t20.index, t20.columns

    fy1_raw = read_sheet(bundle, "Factset_Sales_Revision").reindex(index=idx, columns=cols)
    fy2_raw = read_sheet(bundle, "Factset_Sales_Revision_FY2").reindex(index=idx, columns=cols)
    s1_raw = read_sheet(bundle, "BEST_SALES_1FY")
    s2_raw = read_sheet(bundle, "BEST_SALES_2FY")
    import openpyxl
    wb = openpyxl.load_workbook(bundle, read_only=True, data_only=True)
    meta_rows = list(wb["Universe_Meta"].iter_rows(values_only=True))
    wb.close()
    sector = {str(t).split()[0]: s for t, _, s, _ in meta_rows[1:]}
    print(f"sheets ok {time.time()-t0:.0f}s")

    shim = _Shim({"Factset_Sales_Revision": fy1_raw, "Factset_Sales_Revision_FY2": fy2_raw})
    f1 = build_bounded_revision_features(
        get_cleaned_revision(shim, "Factset_Sales_Revision", config=cfg), "sales_rev")
    f2 = build_bounded_revision_features(
        get_cleaned_revision(shim, "Factset_Sales_Revision_FY2", config=cfg), "sales_rev_fy2")
    fy1_ma, fy2_ma = f1["sales_rev_ma_63d"], f2["sales_rev_fy2_ma_63d"]

    # ---- S1: FY1 재구성 자기검증(무지연, §S20 동일) ----
    sample_dates = list(idx[::21])
    s1_rho = median_cs_corr(fy1_ma, panel_fy1.reindex(index=idx, columns=cols), sample_dates)
    s1_ok = bool(s1_rho >= 0.98)
    print(f"S1 FY1 reconstruction median spearman {s1_rho:.4f} -> {s1_ok}")

    # ---- K1 전제: 롤 월 검출 + S2 검증 ----
    events = detect_roll_events(s1_raw, s2_raw)
    m_roll, n_events, defaulted = {}, {}, []
    for c in cols:
        n_events[c] = len(events.get(c, []))
        m = roll_month(events.get(c, []))
        if m is None:
            defaulted.append(c)
            m = DEFAULT_ROLL_MONTH
        m_roll[c] = m
    validation = {
        t: {"detected": (None if t in defaulted else m_roll[t]), "allowed": sorted(a),
            "n_events": n_events[t], "ok": (t not in defaulted) and m_roll[t] in a}
        for t, a in ROLL_MONTH_ALLOWED.items() if t in m_roll
    }
    val_share = float(np.mean([v["ok"] for v in validation.values()])) if validation else 0.0
    default_share = len(defaulted) / len(cols)
    s2_ok = bool(val_share >= 0.8 and default_share <= 0.25)
    print(f"S2 roll-month validation {val_share:.2f} default share {default_share:.2f} -> {s2_ok}")
    p = fiscal_progress(idx, cols, m_roll)

    # ---- 후보 피처(1영업일 지연) ----
    L = 1
    fy1_l, fy2_l = fy1_ma.shift(L), fy2_ma.shift(L)
    k1 = ((1 - p) * fy1_ma + p * fy2_ma).shift(L)
    k4 = agreement_gate(fy1_ma, fy2_ma).shift(L)
    t40, t60 = chain_target(t20, 2), chain_target(t20, 3)
    sub = [c for c in cols if sector.get(c) in LONG_DURATION_SECTORS]

    def g2(feat, target, columns=None):
        c = list(columns) if columns is not None else list(cols)
        res = residualize(feat[c], [score[c], fy1_l[c]])
        return res, rowwise_rank_corr(res, target[c])

    res_fy2, ic_ref1 = g2(fy2_l, t20)
    res_fy1 = residualize(fy1_l, [score])
    ic_ref2 = rowwise_rank_corr(res_fy1, t20)
    ic_k1 = g2(k1, t20)[1]
    ic_k2 = g2(fy2_l, t20, sub)[1]
    ic_k3 = g2(fy2_l, t60)[1]
    ic_k4 = g2(k4, t20)[1]

    cands = {
        "K1_ntm_blend": (ic_k1, 20, "(1-p)*FY1_ma63 + p*FY2_ma63, G2 vs score+FY1_ma63, T20"),
        "K2_long_duration_fy2": (ic_k2, 20, f"FY2_ma63 G2 within {'+'.join(LONG_DURATION_SECTORS)} "
                                            f"({len(sub)} names), T20"),
        "K3_slow_horizon_fy2": (ic_k3, 60, "FY2_ma63 G2 vs T60 (T20 additive chain), NW lag 60"),
        "K4_agreement_gate": (ic_k4, 20, "FY1_ma63*1[sign FY1 = sign FY2], G2 vs score+FY1_ma63, T20"),
    }
    result = {
        "preregistration": "decision log §S21 (2026-09-28, commit fe38c35)",
        "snapshot_sha256": shas,
        "target": "PCA-residual 20BD fwd (production); T40/T60 = additive chain",
        "feature_lag_days": L, "score_execution_lag_days": lag_days,
        "t_bar": T_BAR, "t_nominal": T_NOMINAL,
        "gate": "G2*: resid IC vs score + FY1_ma63, NW|t| >= 2.81 AND thirds sign-consistent",
        "selfchecks": {
            "S1_fy1_reconstruction_median_spearman": round(s1_rho, 4), "S1_ok": s1_ok,
            "S2_roll_validation_share": round(val_share, 3),
            "S2_default_share": round(default_share, 3), "S2_ok": s2_ok,
            "S2_validation": validation, "S3_sha_ok": s3,
        },
        "roll_month_hist": {str(k): int(v) for k, v in
                            pd.Series(m_roll).value_counts().sort_index().items()},
        "n_long_duration_names": len(sub),
        "references": {
            "REF1_fy2_ma63_G2_lagged": ic_row(ic_ref1, 20),
            "REF2_fy1_ma63_G1_lagged": ic_row(ic_ref2, 20),
        },
        "candidates": {},
    }
    for name, (ic, lag, desc) in cands.items():
        row = {"definition": desc, **ic_row(ic, lag)}
        if not s1_ok:
            row["verdict"] = "INVALID (S1 reconstruction self-check failed)"
        elif name == "K1_ntm_blend" and not s2_ok:
            row["verdict"] = "INVALID (S2 roll-month self-check failed)"
        else:
            row["verdict"] = gate_verdict(row)
        result["candidates"][name] = row
        print(f"{name:24s} n={row['n_days']} IC={row['mean']:+.5f} t={row['t_nw']:+.2f} "
              f"thirds={row['thirds']} -> {row['verdict']}")

    # ---- 진단(비게이트) ----
    pq = np.floor(p * 4).clip(upper=3).where(fy2_l.notna())
    grp = pd.DataFrame({c: ("long_duration" if c in sub else "other") for c in cols}, index=idx)
    horizons = (("T20", t20, 20), ("T40", t40, 40), ("T60", t60, 60))
    result["diagnostics"] = {
        "fy2_G2_by_fiscal_progress_quartile": bucket_ic(res_fy2, t20, pq, 20),
        "fy2_G2_by_sector_group": bucket_ic(res_fy2, t20, grp, 20),
        "horizon_profile": {
            "fy1_G1_vs_score": {h: ic_row(rowwise_rank_corr(res_fy1, tg), lag)
                                for h, tg, lag in horizons},
            "fy2_G2_vs_score_fy1": {h: ic_row(rowwise_rank_corr(res_fy2, tg), lag)
                                    for h, tg, lag in horizons},
        },
        "fy1_fy2_cs_spearman_series": sample_series(rowwise_rank_corr(fy1_ma, fy2_ma)),
        "cum_resid_ic_series": {
            **{k: sample_series(v[0].cumsum()) for k, v in cands.items()},
            "REF1_fy2_ma63_G2_lagged": sample_series(ic_ref1.cumsum()),
            "REF2_fy1_ma63_G1_lagged": sample_series(ic_ref2.cumsum()),
        },
    }
    passed = [k for k, v in result["candidates"].items() if v["verdict"] == "PASS"]
    result["verdict"] = (
        "PROCEED: " + ", ".join(passed) + " (single pre-registered arm, user decision)"
        if passed else "SHELVE (no candidate passes G2*) - FY2 dataset closed across 10 forms")
    result["elapsed_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nverdict: {result['verdict']}\ndone {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
