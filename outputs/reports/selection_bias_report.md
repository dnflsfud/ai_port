# Selection Bias Analysis Report

Generated: 2026-08-20 15:22:55

## 1. Summary Verdict
- **FAIL** -- DSR p=0.0543, Adjusted SR=0.56, MinTRL=0.8yr

## 2. Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
- Observed SR: 1.794
- Number of trials (N): 467
- Expected max SR under null: 1.231
- sigma(SR): 0.3510
- Deflated SR: 1.605 (p-value: 0.0543)
- Skewness: 0.243, Kurtosis: 5.196
- Observations: 2017 trading days
- Verdict: **FAIL -- 다중 비교 보정 후 유의하지 않음**

## 3. Minimum Track Record Length
- Required: 0.8 years (210 trading days)
- Available: 8.0 years (2017 trading days)
- Verdict: **SUFFICIENT**

## 4. Grid Search Bias (Haircut)
- Combinations tested: 467
- Observed SR: 1.794
- Haircut: 1.231
- Adjusted SR: 0.563
- Verdict: **PASS**

## 5. Universe Survivorship
- Backtest start: 2018-11-27
- Late entrants (data starts >30d after backtest): PLTR (from 2020-10-01), GEV (from 2024-04-03), SNDK (from 2025-02-25), CEG (from 2022-02-03), ARM (from 2023-09-15), 285A (from 2024-12-19), DELL (from 2018-12-29), BN (from 2022-12-13), UMG (from 2021-09-22), ABNB (from 2020-12-11), GE (from 2024-04-03), TT (from 2020-03-03), CRWD (from 2019-06-13), WDC (from 2025-02-25), DDOG (from 2019-09-20), COF (from 2025-05-20), UBER (from 2019-05-11), DASH (from 2020-12-10), RBLX (from 2021-03-11)
- Verdict: **WARN -- 19개 종목 생존 편향 의심**

## 6. Sub-period Stability
- Period 1 (2018-11-27 ~ 2021-06-23): IR = 1.387 [PASS]
- Period 2 (2021-06-24 ~ 2024-01-19): IR = 0.957 [PASS]
- Period 3 (2024-01-22 ~ 2026-08-19): IR = 2.924 [PASS]
- Verdict: **STABLE**

## References
- Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio"
- Harvey, C. R., & Liu, Y. (2015). "Backtesting" (Haircut Sharpe Ratio)
- Bailey et al. (2014). "Pseudo-Mathematics and Financial Charlatanism"
