# Selection Bias Analysis Report

Generated: 2026-08-04 09:10:26

## 1. Summary Verdict
- **FAIL** -- DSR p=0.1918, Adjusted SR=0.31, MinTRL=1.1yr

## 2. Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
- Observed SR: 1.548
- Number of trials (N): 451
- Expected max SR under null: 1.239
- sigma(SR): 0.3545
- Deflated SR: 0.871 (p-value: 0.1918)
- Skewness: 0.119, Kurtosis: 5.472
- Observations: 2003 trading days
- Verdict: **FAIL -- 다중 비교 보정 후 유의하지 않음**

## 3. Minimum Track Record Length
- Required: 1.1 years (285 trading days)
- Available: 7.9 years (2003 trading days)
- Verdict: **SUFFICIENT**

## 4. Grid Search Bias (Haircut)
- Combinations tested: 451
- Observed SR: 1.548
- Haircut: 1.239
- Adjusted SR: 0.309
- Verdict: **PASS**

## 5. Universe Survivorship
- Backtest start: 2018-11-27
- Late entrants (data starts >30d after backtest): PLTR (from 2020-10-01), GEV (from 2024-04-03), SNDK (from 2025-02-25), CEG (from 2022-02-03), ARM (from 2023-09-15), 285A (from 2024-12-19), DELL (from 2018-12-29), BN (from 2022-12-13), UMG (from 2021-09-22), ABNB (from 2020-12-11), GE (from 2024-04-03), TT (from 2020-03-03), CRWD (from 2019-06-13), WDC (from 2025-02-25), DDOG (from 2019-09-20), COF (from 2025-05-20), UBER (from 2019-05-11), DASH (from 2020-12-10), RBLX (from 2021-03-11)
- Verdict: **WARN -- 19개 종목 생존 편향 의심**

## 6. Sub-period Stability
- Period 1 (2018-11-27 ~ 2021-06-16): IR = 1.341 [PASS]
- Period 2 (2021-06-17 ~ 2024-01-05): IR = 1.235 [PASS]
- Period 3 (2024-01-08 ~ 2026-07-30): IR = 2.003 [PASS]
- Verdict: **STABLE**

## References
- Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio"
- Harvey, C. R., & Liu, Y. (2015). "Backtesting" (Haircut Sharpe Ratio)
- Bailey et al. (2014). "Pseudo-Mathematics and Financial Charlatanism"
