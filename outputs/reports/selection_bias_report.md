# Selection Bias Analysis Report

Generated: 2026-07-28 18:16:23

## 1. Summary Verdict
- **FAIL** -- DSR p=0.3309, Adjusted SR=0.16, MinTRL=1.4yr

## 2. Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
- Observed SR: 1.404
- Number of trials (N): 443
- Expected max SR under null: 1.248
- sigma(SR): 0.3575
- Deflated SR: 0.438 (p-value: 0.3309)
- Skewness: -0.053, Kurtosis: 5.817
- Observations: 2000 trading days
- Verdict: **FAIL -- 다중 비교 보정 후 유의하지 않음**

## 3. Minimum Track Record Length
- Required: 1.4 years (352 trading days)
- Available: 7.9 years (2000 trading days)
- Verdict: **SUFFICIENT**

## 4. Grid Search Bias (Haircut)
- Combinations tested: 443
- Observed SR: 1.404
- Haircut: 1.248
- Adjusted SR: 0.156
- Verdict: **PASS**

## 5. Universe Survivorship
- Backtest start: 2018-11-27
- Late entrants (data starts >30d after backtest): PLTR (from 2020-10-01), GEV (from 2024-04-03), SNDK (from 2025-02-25), CEG (from 2022-02-03), ARM (from 2023-09-15), 285A (from 2024-12-19), DELL (from 2018-12-29), BN (from 2022-12-13), UMG (from 2021-09-22), ABNB (from 2020-12-11), GE (from 2024-04-03), TT (from 2020-03-03), CRWD (from 2019-06-13), WDC (from 2025-02-25), DDOG (from 2019-09-20), COF (from 2025-05-20), UBER (from 2019-05-11), DASH (from 2020-12-10), RBLX (from 2021-03-11)
- Verdict: **WARN -- 19개 종목 생존 편향 의심**

## 6. Sub-period Stability
- Period 1 (2018-11-27 ~ 2021-06-15): IR = 1.546 [PASS]
- Period 2 (2021-06-16 ~ 2024-01-03): IR = 0.767 [PASS]
- Period 3 (2024-01-04 ~ 2026-07-27): IR = 1.842 [PASS]
- Verdict: **STABLE**

## References
- Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio"
- Harvey, C. R., & Liu, Y. (2015). "Backtesting" (Haircut Sharpe Ratio)
- Bailey et al. (2014). "Pseudo-Mathematics and Financial Charlatanism"
