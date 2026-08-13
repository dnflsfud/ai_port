# Selection Bias Analysis Report

Generated: 2026-08-13 14:57:20

## 1. Summary Verdict
- **FAIL** -- DSR p=0.1319, Adjusted SR=0.39, MinTRL=1.0yr

## 2. Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
- Observed SR: 1.634
- Number of trials (N): 465
- Expected max SR under null: 1.239
- sigma(SR): 0.3534
- Deflated SR: 1.118 (p-value: 0.1319)
- Skewness: 0.145, Kurtosis: 5.419
- Observations: 2011 trading days
- Verdict: **FAIL -- 다중 비교 보정 후 유의하지 않음**

## 3. Minimum Track Record Length
- Required: 1.0 years (256 trading days)
- Available: 8.0 years (2011 trading days)
- Verdict: **SUFFICIENT**

## 4. Grid Search Bias (Haircut)
- Combinations tested: 465
- Observed SR: 1.634
- Haircut: 1.239
- Adjusted SR: 0.395
- Verdict: **PASS**

## 5. Universe Survivorship
- Backtest start: 2018-11-27
- Late entrants (data starts >30d after backtest): PLTR (from 2020-10-01), GEV (from 2024-04-03), SNDK (from 2025-02-25), CEG (from 2022-02-03), ARM (from 2023-09-15), 285A (from 2024-12-19), DELL (from 2018-12-29), BN (from 2022-12-13), UMG (from 2021-09-22), ABNB (from 2020-12-11), GE (from 2024-04-03), TT (from 2020-03-03), CRWD (from 2019-06-13), WDC (from 2025-02-25), DDOG (from 2019-09-20), COF (from 2025-05-20), UBER (from 2019-05-11), DASH (from 2020-12-10), RBLX (from 2021-03-11)
- Verdict: **WARN -- 19개 종목 생존 편향 의심**

## 6. Sub-period Stability
- Period 1 (2018-11-27 ~ 2021-06-21): IR = 1.418 [PASS]
- Period 2 (2021-06-22 ~ 2024-01-15): IR = 1.324 [PASS]
- Period 3 (2024-01-16 ~ 2026-08-11): IR = 2.112 [PASS]
- Verdict: **STABLE**

## References
- Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio"
- Harvey, C. R., & Liu, Y. (2015). "Backtesting" (Haircut Sharpe Ratio)
- Bailey et al. (2014). "Pseudo-Mathematics and Financial Charlatanism"
