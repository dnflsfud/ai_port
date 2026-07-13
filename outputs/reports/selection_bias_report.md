# Selection Bias Analysis Report

Generated: 2026-07-11 16:28:19

## 1. Summary Verdict
- **FAIL** -- DSR p=0.1128, Adjusted SR=0.43, MinTRL=1.0yr

## 2. Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
- Observed SR: 1.661
- Number of trials (N): 412
- Expected max SR under null: 1.231
- sigma(SR): 0.3547
- Deflated SR: 1.212 (p-value: 0.1128)
- Skewness: 0.288, Kurtosis: 6.296
- Observations: 1972 trading days
- Verdict: **FAIL -- 다중 비교 보정 후 유의하지 않음**

## 3. Minimum Track Record Length
- Required: 1.0 years (244 trading days)
- Available: 7.8 years (1972 trading days)
- Verdict: **SUFFICIENT**

## 4. Grid Search Bias (Haircut)
- Combinations tested: 412
- Observed SR: 1.661
- Haircut: 1.231
- Adjusted SR: 0.430
- Verdict: **PASS**

## 5. Universe Survivorship
- Backtest start: 2018-11-27
- Late entrants (data starts >30d after backtest): None
- Verdict: **CLEAN**

## 6. Sub-period Stability
- Period 1 (2018-11-27 ~ 2021-06-02): IR = 0.721 [PASS]
- Period 2 (2021-06-03 ~ 2023-12-08): IR = 0.947 [PASS]
- Period 3 (2023-12-11 ~ 2026-06-11): IR = 2.961 [PASS]
- Verdict: **STABLE**

## References
- Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio"
- Harvey, C. R., & Liu, Y. (2015). "Backtesting" (Haircut Sharpe Ratio)
- Bailey et al. (2014). "Pseudo-Mathematics and Financial Charlatanism"
