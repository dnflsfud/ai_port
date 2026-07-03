"""RL layer for ai_signal_cc2_rl — CS-DR-Alpha (the production alpha engine).

------------------------------------------------------------------------------
CS-DR-Alpha — Cross-Sectional Direct-Reinforcement Alpha   [PRODUCTION]
------------------------------------------------------------------------------
A torch policy trained by gradient ascent on a differentiable,
turnover-penalised cross-sectional Sharpe (Moody-Saffell Direct Reinforcement),
run true walk-forward so every emitted score is OOS-by-construction. Its scores
feed the production MVO via run_backtest(precomputed_predictions=...).

When config.dr_alpha_enabled is set (production variant
iter15_65tkr_reb21_vtg), run_variant.py and daily_update.py harvest the
LightGBM baseline, train the DR walk-forward, then re-run the MVO on the DR
scores. Modules:
  - dr_alpha       : DRAlphaPolicy + train_fold + xs_zscore
  - dr_walkforward : run_walkforward (true walk-forward driver)
  - driver         : scripts/run_dr_alpha.py (grid / DSR harness)
Config: dr_alpha_* (off by default; the production variant turns it on).
See docs/RL_DR_ALPHA_RESULTS.md.

History: the earlier PPO Meta-Controller + Signal Overlay (Gen-1) was retired
on 2026-06-05 — it overfit (in-sample +0.70 / OOS -0.56) and was superseded by
this engine.
"""
