# ai_signal_data stability audit

- Source: `C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\ai_signal_data.xlsx`
- Universe: 200 tickers
- Sheets: 33
- Business-date range: 2014-01-02 to 2026-07-23
- Eligibility dates resolved: 200/200
- Issues: Critical 0, High 0, Medium 5, Low 7

## Severity interpretation

Pre-listing cells are excluded. Weekend forward-fills are excluded by evaluating continuity only on BusinessDays.
Optional fundamental-sheet gaps are reported as Low unless another validity rule fails.

## Top issues

| Severity | Check | Sheet | Ticker | Evidence |
|---|---|---|---|---|
| Medium | extreme_daily_return | Daily_Returns | AMD | 1 daily return(s) exceed 50% in absolute value |
| Medium | extreme_daily_return | Daily_Returns | BE | 1 daily return(s) exceed 50% in absolute value |
| Medium | post_listing_essential_coverage | Factset_EPS_Revision | VRT | coverage=79.94%, longest gap=402, trailing gap=0 business day(s) |
| Medium | post_listing_essential_coverage | Factset_Sales_Revision | VRT | coverage=79.94%, longest gap=402, trailing gap=0 business day(s) |
| Medium | post_listing_essential_coverage | Factset_TG_Price | VRT | coverage=80.39%, longest gap=393, trailing gap=0 business day(s) |
| Low | missing_ticker_columns | BEST_CALCULATED_FCF |  | 13 Universe_Meta ticker(s) absent: JPM, GS, WFC, BAC, TSM, HSBA, BN, ZURN, 8306, BRK/B, CB, COF |
| Low | missing_ticker_columns | BEST_CAPEX |  | 9 Universe_Meta ticker(s) absent: JPM, GS, WFC, BAC, HSBA, ZURN, 8306, CB, COF |
| Low | missing_ticker_columns | BEST_EV_TO_BEST_EBITDA |  | 7 Universe_Meta ticker(s) absent: JPM, WFC, BAC, HSBA, MS, 8306, CB |
| Low | missing_ticker_columns | BEST_GROSS_MARGIN |  | 9 Universe_Meta ticker(s) absent: WFC, BAC, AXP, PGR, C, 8306, CB, COF, MUV2 |
| Low | missing_ticker_columns | BEST_PEG_RATIO |  | 1 Universe_Meta ticker(s) absent: FN |
| Low | missing_ticker_columns | BEST_PX_BPS_RATIO |  | 1 Universe_Meta ticker(s) absent: PM |
| Low | missing_ticker_columns | SHORT_INT_RATIO |  | 40 Universe_Meta ticker(s) absent: 000660, 005930, 285A, SU, SIE, RHM, ALV, MC, NESN, RR/, SAP, ASML |

## Sheet coverage

| Sheet | Rows | Columns | Last date | Median coverage | Minimum coverage |
|---|---:|---:|---|---:|---:|
| PX_LAST | 4587 | 200 | 2026-07-24 | 100.00% | 100.00% |
| BEST_EPS | 4588 | 200 | 2026-07-24 | 100.00% | 100.00% |
| BEST_SALES | 4588 | 200 | 2026-07-24 | 100.00% | 100.00% |
| BEST_PE_RATIO | 4587 | 200 | 2026-07-24 | 100.00% | 100.00% |
| BEST_PEG_RATIO | 4587 | 199 | 2026-07-24 | 100.00% | 100.00% |
| BEST_CALCULATED_FCF | 4588 | 187 | 2026-07-24 | 100.00% | 100.00% |
| BEST_GROSS_MARGIN | 4588 | 191 | 2026-07-24 | 100.00% | 100.00% |
| CUR_MKT_CAP | 4587 | 200 | 2026-07-24 | 100.00% | 100.00% |
| OPER_MARGIN | 4564 | 200 | 2026-07-24 | 100.00% | 100.00% |
| BEST_CAPEX | 4588 | 191 | 2026-07-24 | 100.00% | 100.00% |
| BEST_ROE | 4588 | 200 | 2026-07-24 | 100.00% | 100.00% |
| BEST_PX_BPS_RATIO | 4587 | 199 | 2026-07-24 | 100.00% | 100.00% |
| BEST_EV_TO_BEST_EBITDA | 4587 | 193 | 2026-07-24 | 100.00% | 100.00% |
| NEWS_SENTIMENT_DAILY_AVG | 4588 | 200 | 2026-07-24 | 100.00% | 100.00% |
| EQY_REC_CONS | 4588 | 200 | 2026-07-24 | 100.00% | 100.00% |
| SHORT_INT_RATIO | 4574 | 160 | 2026-07-24 | 100.00% | 100.00% |
| Earnings_Date | 3278 | 200 | 2026-07-24 | 100.00% | 100.00% |
| Daily_Returns | 4587 | 200 | 2026-07-24 | 100.00% | 100.00% |
| Sent_Trend_Momentum_Timeseries | 4241 | 200 | 2026-07-24 | 100.00% | 74.70% |
| Sent_Trend_21d_Timeseries | 4242 | 200 | 2026-07-24 | 100.00% | 74.75% |
| Factset_EPS_Revision | 4952 | 200 | 2026-07-23 | 100.00% | 79.94% |
| Factset_Sales_Revision | 4952 | 200 | 2026-07-23 | 100.00% | 79.94% |
| Factset_TG_Price | 4952 | 200 | 2026-07-23 | 100.00% | 80.39% |
| Factset_Fwd_OpCashflow | 4952 | 200 | 2026-07-23 | 100.00% | 0.00% |
| Factset_EPS_Surprise | 4952 | 200 | 2026-07-23 | 100.00% | 0.00% |
| Factset_Sales_Surprise | 4952 | 200 | 2026-07-23 | 100.00% | 0.00% |
| Earnings_Timeline | 3278 | 200 | 2026-07-24 | 100.00% | 100.00% |
