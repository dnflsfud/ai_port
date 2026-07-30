# -*- coding: utf-8 -*-
"""§S13.23 사전점검: 국가-매핑 지수 리비전 레벨의 횡단면 신호 (백테스트 0회).

bcast 형태는 §S13.18/21에서 gain 0으로 폐쇄 — 유일한 실행 가능 형태인
국가 매핑(종목→소속시장 지수 리비전)의 날짜내 횡단면 변동·신호를 점검한다.
게이트(결정 로그 §S13.23 사전등록):
  G1(변동): 200종 전부 매핑 성공 & 비앵커(SPX_REV 외) 비중 >= 10%
  G2(신호): 날짜별 횡단면 Spearman IC(레벨 vs 21BD 선행 USD 수익,
            유효쌍 >= 100) — |mean IC| > 0.01 & 전·후반 부호 일관(방향 무관)
실행: ai_port에서 `<PY> scripts/preflight_s13_23_index_rev_mapping.py`
산출물: outputs/s13_23_index_rev_preflight.csv
"""

from typing import Dict

import numpy as np
import pandas as pd

# 사전약정 매핑(결정 로그 §S13.23) — 스윕 금지.
EXCHANGE_TO_REV: Dict[str, str] = {
    "US": "SPX_REV",
    "FP": "CAC_REV",
    "GR": "DAX_REV",
    "JP": "JPN_REV",
    "NA": "SX5E_REV",
    "SM": "SX5E_REV",
    "LN": "SX5E_REV",  # 유럽 프록시(비유로존 근사, 선언된 부정확)
    "SW": "SX5E_REV",
    "DC": "SX5E_REV",
    "KS": "SPX_REV",   # 글로벌 앵커 fallback(2/200, 선언된 부정확)
}
ANCHOR = "SPX_REV"
HORIZON = 21
MIN_PAIRS = 100


def map_universe(exchange_codes: pd.Series) -> pd.Series:
    """ticker→리비전 컬럼. 미지 거래소 코드는 ValueError."""
    unknown = sorted(set(exchange_codes.dropna()) - set(EXCHANGE_TO_REV))
    if unknown:
        raise ValueError(f"unknown exchange codes: {unknown}")
    return exchange_codes.map(EXCHANGE_TO_REV)


def build_mapped_panel(factor_px: pd.DataFrame, mapping: pd.Series,
                       dates: pd.DatetimeIndex, tickers: list) -> pd.DataFrame:
    """dates×tickers 패널 — 각 종목 열 = 소속 지수 리비전 시계열."""
    cols = {t: factor_px[mapping.loc[t]].reindex(dates) for t in tickers}
    return pd.DataFrame(cols, index=dates)


def forward_returns(returns: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """t 시점 값 = t+1..t+horizon 일간 수익 합(근사 누적)."""
    return returns.rolling(horizon).sum().shift(-horizon)


def daily_spearman_ic(feature: pd.DataFrame, fwd: pd.DataFrame,
                      min_pairs: int = MIN_PAIRS) -> pd.Series:
    """날짜별 횡단면 Spearman IC. 유효쌍<min_pairs 또는 피처 단일값 날짜 제외."""
    out = {}
    for date in feature.index.intersection(fwd.index):
        f, r = feature.loc[date], fwd.loc[date]
        valid = f.notna() & r.notna()
        if valid.sum() < min_pairs or f[valid].nunique() < 2:
            continue
        out[date] = f[valid].corr(r[valid], method="spearman")
    return pd.Series(out).sort_index()


def judge_gates(nonanchor_share: float, mean_ic: float,
                h1: float, h2: float) -> dict:
    g1 = nonanchor_share >= 0.10
    g2 = abs(mean_ic) > 0.01 and np.sign(h1) == np.sign(h2) and h1 != 0
    return {"G1": bool(g1), "G2": bool(g2),
            "verdict": "PROCEED" if (g1 and g2) else "SHELVE"}


def main() -> None:
    from src.config import PipelineConfig
    from src.data_loader import UniverseData

    config = PipelineConfig()
    data = UniverseData(config.data_path, config=config)
    tickers = list(data.tickers)
    exchange = data.meta.loc[tickers, "exchange_code"]
    mapping = map_universe(exchange)
    n_nonanchor = int((mapping != ANCHOR).sum())
    share = n_nonanchor / len(tickers)
    print(f"[G1] mapped {len(mapping)}/{len(tickers)} tickers, "
          f"non-anchor {n_nonanchor} ({share:.1%})")
    print(mapping.value_counts().to_string())

    factor_px = data.factor_prices
    missing = [c for c in set(mapping) if c not in factor_px.columns]
    if missing:
        raise ValueError(f"factor_prices missing revision columns: {missing}")

    returns = data.sheets["Daily_Returns"]
    dates = returns.index
    feature = build_mapped_panel(factor_px, mapping, dates, tickers)
    fwd = forward_returns(returns[tickers])
    ic = daily_spearman_ic(feature, fwd)

    half = len(ic) // 2
    h1, h2 = float(ic.iloc[:half].mean()), float(ic.iloc[half:].mean())
    mean_ic = float(ic.mean())
    print(f"\n[G2] daily CS Spearman IC: n={len(ic)} mean={mean_ic:+.4f} "
          f"H1={h1:+.4f} H2={h2:+.4f}")
    thirds = np.array_split(ic, 3)
    print("  서술: 3분할 IC = "
          + " / ".join(f"{t.mean():+.4f}" for t in thirds))
    by_group = {}
    for col in sorted(set(mapping)):
        members = mapping.index[mapping == col]
        by_group[col] = float(fwd[members].stack().mean() * (252 / HORIZON))
    print("  서술: 그룹별 평균 선행수익(연율) = "
          + ", ".join(f"{k} {v:+.3%}" for k, v in by_group.items()))

    gates = judge_gates(share, mean_ic, h1, h2)
    print(f"\nG1 {'PASS' if gates['G1'] else 'FAIL'} | "
          f"G2 {'PASS' if gates['G2'] else 'FAIL'} | verdict {gates['verdict']}")

    ic.rename("spearman_ic").to_frame().to_csv(
        "outputs/s13_23_index_rev_preflight.csv")
    print("saved: outputs/s13_23_index_rev_preflight.csv")


if __name__ == "__main__":
    main()
