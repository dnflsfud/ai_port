# Selection Bias Analysis Report

Generated: 2026-06-23 16:19:02

## 1. Summary Verdict
- **FAIL** -- DSR p=0.2272, Adjusted SR=0.26, MinTRL=1.2yr

## 2. Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
- Observed SR: 1.464
- Number of trials (N): 403
- Expected max SR under null: 1.204
- sigma(SR): 0.3476
- Deflated SR: 0.748 (p-value: 0.2272)
- Skewness: 0.807, Kurtosis: 10.535
- Observations: 1973 trading days
- Verdict: **FAIL -- 다중 비교 보정 후 유의하지 않음**

## 3. Minimum Track Record Length
- Required: 1.2 years (302 trading days)
- Available: 7.8 years (1973 trading days)
- Verdict: **SUFFICIENT**

## 4. Grid Search Bias (Haircut)
- Combinations tested: 403
- Observed SR: 1.464
- Haircut: 1.204
- Adjusted SR: 0.260
- Verdict: **PASS**

## 5. Universe Survivorship
- Backtest start: 2018-11-26
- Late entrants (data starts >30d after backtest): None
- Verdict: **CLEAN**

## 6. Sub-period Stability
- Period 1 (2018-11-26 ~ 2021-06-01): IR = 1.629 [PASS]
- Period 2 (2021-06-02 ~ 2023-12-07): IR = 0.536 [PASS]
- Period 3 (2023-12-08 ~ 2026-06-11): IR = 1.975 [PASS]
- Verdict: **STABLE**

## References
- Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio"
- Harvey, C. R., & Liu, Y. (2015). "Backtesting" (Haircut Sharpe Ratio)
- Bailey et al. (2014). "Pseudo-Mathematics and Financial Charlatanism"
