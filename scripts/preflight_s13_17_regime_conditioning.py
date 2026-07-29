# -*- coding: utf-8 -*-
"""§S13.17 Phase 0 사전점검: S0 일별 액티브 수익의 레짐 조건부 귀속.

신규 백테스트 없음 — S0 production(codex_causal_rank_65)의 기존
backtest_result.pkl 액티브 수익을, regime_v2(10피처 워크포워드 diag GaussianHMM,
new_ai_port에서 검증된 설계와 동일 수식·시드)의 모달 상태로 조건화한다.

게이트 (결정 로그 §S13.17 사전등록):
  P0-G1: stress-모달일 연환산 액티브 < -1%/yr AND calm+mid-모달일 > +1%/yr
  P0-G2: 백테스트 구간 반분(전반/후반) 모두 stress-버킷 액티브 부호 음
  P0-G3: stress-모달일 점유율 10~50%
하나라도 실패 → arm 미구현 shelve.

실행: <PY> scripts/preflight_s13_17_regime_conditioning.py  (WD=ai_port, PYTHONPATH=.)
산출물: outputs/s13_17_regime_preflight.csv (상태별 통계)
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import PipelineConfig  # noqa: E402

# ---- regime_v2 사전등록 하이퍼파라미터 (new_ai_port V2_REGIME 동일) ----
N_STATES = 3
INITIAL_TRAIN_BD = 504
REFIT_EVERY_BD = 252
N_RANDOM_STARTS = 5
MIN_COVAR = 1e-4
MIN_STATE_OCCUPANCY = 0.02
SEED = 42
SORT_DIM = 1   # spx_rv21
STATE_NAMES = np.array(["calm", "mid", "stress"])


def build_market_features(px: pd.DataFrame) -> pd.DataFrame:
    """regime_v2.build_market_features와 동일 수식 (Factor_PX_LAST 직접 입력)."""
    vix = px["VIX"]
    eps = {k: np.log(px[f"{k}_FWD_EPS"]) for k in ("SPX", "NDX", "MXWD")}
    spx_ret = px["SPX"].pct_change()
    return pd.DataFrame({
        "mxwd_ret21": np.log(px["MXWD"]).diff().rolling(21).sum(),
        "spx_rv21": spx_ret.rolling(21).std() * np.sqrt(252.0),
        "vix_log": np.log(vix),
        "vix_chg21": vix - vix.shift(21),
        "ust_slope": px["UST_10Y"] - px["UST_3M"],
        "risk_appetite": px["F_HiBeta"] / px["F_HiBeta"].shift(21)
                         - px["F_MinVol"] / px["F_MinVol"].shift(21),
        "eps_g63": eps["MXWD"].diff(63),
        "eps_us_lead63": eps["SPX"].diff(63) - eps["MXWD"].diff(63),
        "eps_us_lead252": eps["SPX"].diff(252) - eps["MXWD"].diff(252),
        "eps_tech_lead63": eps["NDX"].diff(63) - eps["SPX"].diff(63),
    }, index=px.index)


def fit_hmm(X: np.ndarray, warm: dict | None, seed: int) -> dict:
    """diag GaussianHMM: N_RANDOM_STARTS 랜덤 + warm, 최고 LL 승자.
    점유율 < MIN_STATE_OCCUPANCY 또는 비유한 후보는 기각(warm fallback)."""
    mu = X.mean(axis=0)
    sd = np.where(X.std(axis=0) < 1e-12, 1.0, X.std(axis=0))
    Xs = (X - mu) / sd
    candidates = [GaussianHMM(n_components=N_STATES, covariance_type="diag",
                              n_iter=200, tol=1e-4, min_covar=MIN_COVAR,
                              random_state=seed + j)
                  for j in range(N_RANDOM_STARTS)]
    if warm is not None:
        m = GaussianHMM(n_components=N_STATES, covariance_type="diag",
                        n_iter=200, tol=1e-4, min_covar=MIN_COVAR,
                        init_params="", params="stmc")
        m.startprob_ = warm["startprob"].copy()
        m.transmat_ = warm["transmat"].copy()
        raw_means = warm["mu"] + warm["sd"] * warm["means"]
        raw_vars = warm["covars"] * warm["sd"] ** 2
        m.means_ = (raw_means - mu) / sd
        m.covars_ = np.maximum(raw_vars / sd ** 2, MIN_COVAR)
        candidates.append(m)

    best = best_unrejected = None
    for m in candidates:
        try:
            m.fit(Xs)
            ll = float(m.score(Xs))
            occ = np.bincount(m.predict(Xs), minlength=N_STATES) / len(Xs)
            arrays = (m.startprob_.copy(), m.transmat_.copy(), m.means_.copy(),
                      np.diagonal(m.covars_, axis1=1, axis2=2).copy())
        except Exception:
            continue
        if not (np.isfinite(ll) and all(np.isfinite(a).all() for a in arrays)):
            continue
        if best_unrejected is None or ll > best_unrejected[0]:
            best_unrejected = (ll, *arrays)
        if occ.min() < MIN_STATE_OCCUPANCY:
            continue
        if best is None or ll > best[0]:
            best = (ll, *arrays)
    if best is None:
        if warm is not None:
            return warm
        best = best_unrejected
    ll, startprob, transmat, means, covars = best
    order = np.argsort(means[:, SORT_DIM], kind="stable")
    return {"startprob": startprob[order],
            "transmat": transmat[np.ix_(order, order)],
            "means": means[order], "covars": covars[order],
            "mu": mu, "sd": sd}


def filter_probs(p: dict, X: np.ndarray) -> np.ndarray:
    """필터드 P(s_t | x_{<=t}) — 자체 forward recursion (스무딩 금지)."""
    Xs = (X - p["mu"]) / p["sd"]
    T, d = Xs.shape
    var = p["covars"]
    log_norm = -0.5 * (d * np.log(2.0 * np.pi) + np.log(var).sum(axis=1))
    diff = Xs[:, None, :] - p["means"][None, :, :]
    log_b = log_norm[None, :] - 0.5 * (diff ** 2 / var[None, :, :]).sum(axis=2)
    with np.errstate(divide="ignore"):
        log_start = np.log(p["startprob"])
        log_a = np.log(p["transmat"])
    out = np.empty((T, N_STATES))
    la = log_start + log_b[0]
    la -= logsumexp(la)
    out[0] = np.exp(la) / np.exp(la).sum()
    for t in range(1, T):
        la = log_b[t] + logsumexp(la[:, None] + log_a, axis=0)
        la -= logsumexp(la)
        pr = np.exp(la)
        out[t] = pr / pr.sum()
    return out


def walkforward_probs(F: pd.DataFrame, bd: pd.DatetimeIndex) -> pd.DataFrame:
    clean = F.dropna()
    X = clean.to_numpy(dtype=np.float64)
    fit_ends = [bd[i] for i in range(INITIAL_TRAIN_BD, len(bd), REFIT_EVERY_BD)]
    params_list = []
    warm = None
    for k, fe in enumerate(fit_ends):
        hi = clean.index.searchsorted(fe, side="right")
        warm = fit_hmm(X[:hi], warm, seed=SEED + 1000 * k)
        params_list.append((fe, warm))
    bounds = [clean.index.searchsorted(fe) for fe, _ in params_list] + [len(clean)]
    chunks = [filter_probs(p, X[:bounds[k + 1]])[bounds[k]:bounds[k + 1]]
              for k, (_, p) in enumerate(params_list)]
    return pd.DataFrame(np.vstack(chunks), index=clean.index[bounds[0]:],
                        columns=["p_calm", "p_mid", "p_stress"])


def bucket_stats(active: pd.Series, modal: pd.Series) -> pd.DataFrame:
    rows = []
    for name in STATE_NAMES:
        a = active[modal == name]
        rows.append({"state": name, "n_days": len(a),
                     "share": len(a) / len(active),
                     "ann_active": float(a.mean()) * 252.0 if len(a) else np.nan,
                     "t_stat": (float(a.mean() / a.std() * np.sqrt(len(a)))
                                if len(a) > 1 and a.std() > 0 else np.nan)})
    return pd.DataFrame(rows).set_index("state")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    with open(root / "outputs/codex_causal_rank_65/backtest_result.pkl", "rb") as f:
        result = pickle.load(f)
    active = result.active_returns.dropna()
    print(f"S0 active_returns: {len(active)} days "
          f"{active.index[0].date()} ~ {active.index[-1].date()} "
          f"(full ann active {active.mean() * 252:.4f})")

    px = pd.read_excel(PipelineConfig().data_path, sheet_name="Factor_PX_LAST",
                       index_col=0, engine="openpyxl")
    px.index = pd.DatetimeIndex(px.index)
    probs = walkforward_probs(build_market_features(px), px.index)
    print(f"regime probs: {probs.index[0].date()} ~ {probs.index[-1].date()} "
          f"({len(probs)} rows)")

    modal = pd.Series(STATE_NAMES[probs.to_numpy().argmax(axis=1)],
                      index=probs.index)
    common = active.index.intersection(modal.index)
    uncovered = len(active) - len(common)
    active, modal = active.loc[common], modal.loc[common]
    print(f"joined {len(common)} days (uncovered {uncovered})\n")

    stats = bucket_stats(active, modal)
    print("[full period]")
    print(stats.round(4).to_string())

    half = len(active) // 2
    halves = {}
    for tag, seg in (("H1", slice(None, half)), ("H2", slice(half, None))):
        s = bucket_stats(active.iloc[seg], modal.iloc[seg])
        halves[tag] = s
        print(f"\n[{tag}: {active.index[seg][0].date()} ~ "
              f"{active.index[seg][-1].date()}]")
        print(s.round(4).to_string())

    calm_mid = active[modal != "stress"]
    cm_ann = float(calm_mid.mean()) * 252.0
    stress_ann = float(stats.loc["stress", "ann_active"])
    stress_share = float(stats.loc["stress", "share"])

    g1 = stress_ann < -0.01 and cm_ann > 0.01
    g2 = all(halves[h].loc["stress", "ann_active"] < 0.0 for h in ("H1", "H2"))
    g3 = 0.10 <= stress_share <= 0.50

    print("\n==== §S13.17 P0 gates ====")
    print(f"P0-G1 stress {stress_ann:+.4f} < -0.01 AND calm+mid {cm_ann:+.4f} "
          f"> +0.01  -> {'PASS' if g1 else 'FAIL'}")
    print(f"P0-G2 halves stress sign: "
          f"H1 {halves['H1'].loc['stress', 'ann_active']:+.4f} / "
          f"H2 {halves['H2'].loc['stress', 'ann_active']:+.4f}  "
          f"-> {'PASS' if g2 else 'FAIL'}")
    print(f"P0-G3 stress share {stress_share:.3f} in [0.10, 0.50]  "
          f"-> {'PASS' if g3 else 'FAIL'}")
    verdict = "PROCEED (arm 구현 진행)" if g1 and g2 and g3 else "SHELVE (arm 미구현)"
    print(f"\nP0 verdict: {verdict}")

    out = stats.copy()
    for tag, s in halves.items():
        out[f"ann_active_{tag}"] = s["ann_active"]
    out.to_csv(root / "outputs/s13_17_regime_preflight.csv",
               encoding="utf-8-sig")
    print("saved: outputs/s13_17_regime_preflight.csv")
    return 0 if (g1 and g2 and g3) else 1


if __name__ == "__main__":
    sys.exit(main())
