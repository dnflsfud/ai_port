# Selection Bias Analysis Report

Generated: 2026-07-18 14:09:36

## 1. Summary Verdict
- **FAIL** -- DSR p=0.1693, Adjusted SR=0.33, MinTRL=1.1yr

## 2. Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
- Observed SR: 1.545
- Number of trials (N): 413
- Expected max SR under null: 1.211
- sigma(SR): 0.3488
- Deflated SR: 0.957 (p-value: 0.1693)
- Skewness: 0.554, Kurtosis: 7.907
- Observations: 1993 trading days
- Verdict: **FAIL -- 다중 비교 보정 후 유의하지 않음**

## 3. Minimum Track Record Length
- Required: 1.1 years (276 trading days)
- Available: 7.9 years (1993 trading days)
- Verdict: **SUFFICIENT**

## 4. Grid Search Bias (Haircut)
- Combinations tested: 413
- Observed SR: 1.545
- Haircut: 1.211
- Adjusted SR: 0.334
- Verdict: **PASS**

## 5. Universe Survivorship
- Backtest start: 2018-11-27
- Late entrants (data starts >30d after backtest): PLTR (from 2020-10-01), GEV (from 2024-04-03), SNDK (from 2025-02-25), CEG (from 2022-02-03), ARM (from 2023-09-15), 285A (from 2024-12-19)
- Verdict: **WARN -- 6개 종목 생존 편향 의심**

## 6. Sub-period Stability
- Period 1 (2018-11-27 ~ 2021-06-11): IR = 0.706 [PASS]
- Period 2 (2021-06-14 ~ 2023-12-28): IR = 0.992 [PASS]
- Period 3 (2023-12-29 ~ 2026-07-16): IR = 2.721 [PASS]
- Verdict: **STABLE**

## References
- Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio"
- Harvey, C. R., & Liu, Y. (2015). "Backtesting" (Haircut Sharpe Ratio)
- Bailey et al. (2014). "Pseudo-Mathematics and Financial Charlatanism"
