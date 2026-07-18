"""
Centralized pipeline configuration.

All module-level constants are consolidated here in a single dataclass.
Each module re-exports backwards-compatible aliases that point to
DEFAULT_CONFIG, so existing callers continue to work unchanged.
New callers can instantiate PipelineConfig with custom values and
pass it via the ``config`` parameter of key functions.

This file is the single source of truth (SSOT) for the pipeline.
Docs/AGENTS.md describe the design intent; this file describes what
actually runs. When the two differ, this file wins.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class PipelineConfig:
    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data_path: str = r"C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\ai_signal_data.xlsx"
    output_dir: str = "./outputs"

    # Portfolio accounting currency. Equity prices/returns are supplied in
    # each listing's local currency; the live portfolio and dashboard are USD
    # based. CUR_MKT_CAP is already converted upstream and must not be touched.
    base_currency: str = "USD"
    fx_source_path: str = r"C:\Users\westl\PycharmProjects\pythonProject\Data\Index.xlsx"
    max_fx_staleness_days: int = 7
    fail_on_missing_fx: bool = True
    convert_returns_to_usd: bool = True

    # Pre-listing backfill masking (2026-07-02 structure review Critical #1).
    # NaN-mask the pre-listing constant backfill in the
    # source xlsx — PLTR/GEV/BE plus new members 285A/SNDK/ARM/CEG carry
    # pre-listing history that can pollute the cap-weighted BM, open a zero-vol
    # free-OW path, and contaminate the training panel. VRT is excluded — it
    # has genuine SPAC-predecessor trading history.
    listing_mask_enabled: bool = True
    listing_dates: Dict[str, str] = field(default_factory=lambda: {
        "PLTR": "2020-09-30", "GEV": "2024-04-02", "BE": "2018-07-25",
        "285A": "2024-12-18", "SNDK": "2025-02-24",
        "ARM": "2023-09-14", "CEG": "2022-02-02",
    })

    # ------------------------------------------------------------------
    # Target (PCA)
    # ------------------------------------------------------------------
    # REDESIGN L (2026-04-13): n_remove=1 was tested but degraded P1 IR
    # (+1.34 -> -0.26) because the 2018-2021 regime needs aggressive factor
    # scrubbing. Reverted to 2. PE clipping + peg_growth_spread cleanup kept.
    pca_components: int = 5
    pca_n_remove: int = 2
    pca_lookback: int = 252
    forward_horizon: int = 20

    # ------------------------------------------------------------------
    # P2 signal-layer infrastructure (2026-04-20, INFRA, OFF by default)
    # ------------------------------------------------------------------
    # Regime-aware PCA lookback: switch `pca_lookback` between a short and
    # long window based on realised vol regime. In high-vol periods a
    # shorter lookback adapts faster to regime breaks (P2 candidate fix);
    # in low-vol periods a longer lookback reduces PCA noise.
    #
    #   effective_lookback = pca_lookback_short if vol_regime_high else pca_lookback_long
    #
    # OFF when regime_aware_pca_lookback=False (default). Needs validation
    # vs iter15_FINAL baseline before promoting. Companion work to Item 3
    # of the P2 roadmap — see docs/ROADMAP.md.
    regime_aware_pca_lookback: bool = False
    pca_lookback_short: int = 126        # used when vol regime HIGH
    pca_lookback_long: int = 504         # used when vol regime LOW (2y)
    pca_regime_vol_col: str = "VIX"      # factor_prices column driving toggle
    pca_regime_vol_threshold: float = 1.0  # z-score above which = HIGH

    # Regime-aware PCA weighted fit (2026-04-22, Phase 2 signal-layer P2 fix).
    # When enabled, PCA fit uses sample weights conditional on the target
    # date's regime (stress vs normal). Same-regime historical observations
    # get weight 1.0; different-regime observations get `regime_pca_offreg_weight`
    # (default 0.3). This makes the PCA extract the covariance structure that
    # dominates the *current* regime, instead of averaging across regimes and
    # missing the rate-shock-specific common factor that plagued P2.
    # Independent of regime_aware_pca_lookback — they can be combined.
    regime_pca_weighted_enabled: bool = False
    regime_pca_vix_threshold: float = 0.5   # VIX z-score above which = stress regime
    regime_pca_offreg_weight: float = 0.3   # weight for observations in the *other* regime
    regime_pca_min_effective_n: int = 30    # fallback to unweighted PCA if same-regime count too low

    # Macro × ticker cross features (Phase 2, 2026-04-22). Five features:
    # rate×rev, slope×rev, VIX×mom252, vol×mom63, DXY×rev.
    #
    # D1 resolution (2026-07-02): this is a baseline_v2 COMPONENT, intentionally
    # ON by default. The certified S0 (ECOS IR 1.481, certified 2026-06-18,
    # re-confirmed 2026-07-02) was measured with this ON — it is part of the
    # baseline, not a new candidate arm. Disabling it is for ablations only (to
    # isolate the regime-PCA contribution); flipping it OFF re-baselines S0 (§2-2).
    # The OFF-default invariant (CLAUDE.md §2-1) applies to NEW Pictet-adoption
    # arms (2026-06-18+), not to this pre-existing baseline component.
    macro_cross_enabled: bool = True

    # Multi-horizon target ensemble: weighted blend of specific returns
    # computed at multiple forward horizons. Intuition — short horizons
    # (5d) catch fast regime rotations; long horizons (63d) catch
    # persistent trends. OFF when multi_horizon_weights is empty dict.
    multi_horizon_targets_enabled: bool = False
    # Map of horizon (days) -> weight. Must sum to ~1 if enabled. Example:
    #   {20: 0.6, 5: 0.2, 63: 0.2}
    multi_horizon_weights: Dict[int, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Revision spike cleaning mode (2026-04-20, INFRA)
    # ------------------------------------------------------------------
    # Current baseline (`down_only`) only masks `daily_diff < -threshold`.
    # But Factset window rollover can produce UP-spikes too: a stock with
    # score -72 can jump to +5 on rollover day purely as data artifact,
    # making a fundamentally-poor stock look "neutral" to the model.
    #
    # Modes:
    #   "down_only"      — iter15 baseline: only negative spikes masked.
    #   "symmetric"      — abs(daily_diff) > threshold masked (both sides).
    #                      Risk: genuine analyst upgrades get filtered too.
    #   "reversion_gated" — mask ONLY if prev_level was extreme and today's
    #                      level collapsed toward neutral (both directions).
    #                      Preserves genuine moves into extremes.
    #
    # The reversion-gated detector:
    #   prev = rev.shift(1).abs()
    #   is_rollover = (|daily_diff| > threshold) & (prev > extreme_thr) &
    #                 (|rev| < prev * reversion_ratio)
    #
    # See docs/ROADMAP.md § Phase 2.4 for rollout plan + gate criteria.
    #
    # 2026-04-21: reversion_gated promoted to baseline_v2. In the
    # apples-to-apples A/B (exp_revision_reversion_gated vs
    # iter15_FINAL_reproduce under the cleaned codebase), reversion_gated
    # delivers IR 1.024 vs 0.903 (+0.121) and P3 IR 1.640 vs 1.152 (+0.488)
    # — largely from correctly filtering Factset rollover UP-spikes that
    # inflated revision momentum features for fundamentally weak names.
    # See outputs/exp_revision_reversion_gated/metrics.json and
    # docs/BASELINE.md for the authoritative baseline_v2 artifacts.
    revision_clean_mode: str = "reversion_gated"     # {"down_only","symmetric","reversion_gated"}
    revision_clean_threshold: float = 15.0           # daily-diff magnitude trigger
    revision_clean_extreme_threshold: float = 50.0   # prev-level "extreme" for reversion mode
    revision_clean_reversion_ratio: float = 0.5      # today's |level| < prev × this → collapse

    # ------------------------------------------------------------------
    # Model (LightGBM)  — REDESIGN D (2026-04)
    # ------------------------------------------------------------------
    # Previous settings produced degenerate models (mean trees 64 in P3)
    # because low lr + high min_child_samples + ~3y heterogeneous window
    # triggered early stopping on almost every retrain. Rebalanced for
    # stable convergence at the cost of slight per-model capacity.
    train_window: int = 1260          # 5 years (was 756 / 3 years)
    retrain_freq: int = 63
    val_window: int = 126
    # Alternative cross-sectional ranking model.  All switches are OFF by
    # default so the canonical regression portfolio remains byte-for-byte on
    # its historical execution path.
    model_objective: str = "regression"  # regression | cross_sectional_rank
    causal_validation_enabled: bool = False
    execution_signal_lag_days: int = 0
    rank_relevance_levels: int = 10
    rank_eval_at: List[int] = field(default_factory=lambda: [5, 10])
    # REDESIGN R-9 (2026-04-14): 0.8 → 0.5 — P2 IR -0.343 회복 시도.
    # 2021-2023 금리급등기에서 model이 regime shift에 느리게 적응 (ema가 옛 signal 끌어안음).
    # alpha 0.5는 50% 새 signal + 50% prior. Forensics 권장.
    prediction_ema_alpha: float = 0.5  # was 0.8
    lgbm_params: Dict = field(default_factory=lambda: {
        "objective": "regression",
        "metric": "mse",
        "learning_rate": 0.02,         # was 0.03 — V2 pattern: slightly lower for stability
        "num_leaves": 31,
        "max_depth": 5,
        "min_child_samples": 60,       # was 20 — V2 value: stronger regularization, less noise splits
        "subsample": 0.8,             # was 0.7 — V2 value
        "colsample_bytree": 0.8,       # was 0.5 — V2 value: use more features per tree
        "reg_alpha": 0.3,
        "reg_lambda": 2.0,             # was 1.5 — V2 value
        "n_estimators": 800,           # was 500 — V2 value: more trees with lower lr
        "verbose": -1,
        "random_state": 42,
    })
    early_stopping_rounds: int = 100   # NEW — explicit patience control
    min_model_trees: int = 10           # below this, treat retrain as degenerate
    max_degenerate_model_rate: float = 0.25
    fail_on_degenerate_model_rate: bool = False

    # ------------------------------------------------------------------
    # EWMA Feature Importance
    # ------------------------------------------------------------------
    ewma_enabled: bool = True
    ewma_alpha: float = 0.3          # decay factor (0.3 = 70% weight to history)
    ewma_min_retrains: int = 2       # cold start: uniform until N retrains
    ewma_drop_pct: float = 0.05     # drop bottom 5% features by EWMA importance
    ewma_min_features: int = 60      # never drop below this many features

    # ------------------------------------------------------------------
    # Features  — REDESIGN C++ (2026-04-11 PM)
    # ------------------------------------------------------------------
    # Feature mode evolution:
    #   "full" (345): original panel, ~70% Accounting+Sellside (Quality/Value)
    #   "lean" (239): drops redundant accounting horizons + adds momentum
    #   "core" (~85): hand-picked whitelist, explicit style balance
    #
    # The core whitelist was built from the A+C+D+E run's feature_importance
    # ranking, preserving top N per style axis (Quality/Growth/Value/Momentum
    # /Price/Sellside/Macro/Regime). Smaller panel reduces noise and training
    # instability; C+ growth balance experiment proved adding MORE features
    # without stronger base signal hurts IR.
    feature_mode: str = "core"         # {"full", "lean", "core"}

    # ------------------------------------------------------------------
    # Benchmark  — REDESIGN A (2026-04)
    # ------------------------------------------------------------------
    # Previous EW(1/n) benchmark fought mega-cap concentration: model had
    # virtually no room to deviate on NVDA/MSFT/AMZN because each name
    # maxed out at ~2% BM weight. Switched default to cap-weighted (using
    # CUR_MKT_CAP) so the optimizer's active budget aligns with the
    # market's actual concentration.
    benchmark_type: str = "cap_weighted"  # {"cap_weighted", "equal_weight"}

    # ------------------------------------------------------------------
    # Optimizer  — REDESIGN E (2026-04)
    # ------------------------------------------------------------------
    # Previous settings were extremely constrained (TP 0.3 = 60x Pictet
    # baseline), producing ~1% active share and IC↔PnL correlation near
    # zero. Relaxed to let the signal actually reach the book.
    # REDESIGN R-FINAL (2026-04-14): risk_aversion 1.0 — iter 11에서 0.5 시도했으나
    # MVO에서 binding 안 됨 (TE quadratic 제약이 risk를 이미 잡음). 효과 없어서 iter 9로 롤백.
    risk_aversion: float = 1.0             # iter 9 baseline
    turnover_penalty: float = 0.03         # back to original
    # REDESIGN P (2026-04-14): TRUE codex_v2 baseline 발견 (TE 4.5%, IR 1.046).
    # 사용자 옵션 A: TE 풀기. 0.020 → 0.045 (codex level). 이전 iter 3은
    # TE 3% strict로 alpha 공간이 좁아 active 3.42%에 머물렀음. 이제 IR/Active를
    # codex 수준으로 끌어올릴 공간 확보.
    max_te_annual: float = 0.045
    max_single_turnover: float = 0.15
    sector_deviation: float = 0.10         # was 0.20 — tighter neutrality
    cov_lookback: int = 126
    # Mega-cap volatility shrinkage in estimate_covariance (D@S@D on high-BM
    # names). Default ON preserves current behavior; exposed as a flag for
    # ablation (2026-07-02 structure review #5). False skips the shrinkage.
    cov_megacap_vol_shrink_enabled: bool = True
    bm_weight_floor: float = 0.02          # was 0.10 — free UW space
    max_active_share: float = 0.50
    # Optional hard ceiling on the runtime-derived L1 active-share cap used by
    # Hook B's two-way satellite_budget propagation (derive_max_active_share).
    # None = inert; off-path config-build stays bit-identical.
    max_active_share_ceiling: Optional[float] = None
    max_weight: float = 0.15               # was 0.125
    max_active_per_stock: float = 0.12     # was 0.105
    use_score_based: bool = False

    # ------------------------------------------------------------------
    # Portfolio style  — REDESIGN F (2026-04-11 PM)
    # ------------------------------------------------------------------
    # Core-satellite philosophy: hold a BM-tracking "core" (70-80%) and make
    # concentrated active bets ("satellite", 20-30%). Implemented by
    # tightening max_active_share and max_active_per_stock to force the
    # optimizer away from big-bet mode. Previous single-name OWs of +10-12%
    # were too concentrated for a core-satellite book.
    #
    # Active share conventions (this code uses L1 form cp.norm1(w - bm)):
    #   one-way active share = L1 / 2   (industry convention)
    #   satellite_budget is the target one-way active share
    #   core_weight = 1 - satellite_budget  (the "passive" share)
    portfolio_style: str = "core_satellite"   # {"unconstrained", "core_satellite"}
    # REDESIGN R-FINAL (2026-04-14): satellite_budget 0.225 (iter 9 baseline).
    # iter 10에서 0.30 시도 → 무효과 (active share 10% 평균으로 cap 1/3만 사용).
    # Cap이 binding 아님. 롤백해서 iter 9 production 베이스라인 복원.
    satellite_budget: float = 0.225           # one-way active share target
    # REDESIGN iter17 (2026-04-17): 0.06 → 0.035. OW 분산화.
    # iter15에서 top-3 OW가 active budget의 88% 차지 (과도한 집중).
    # 0.035로 낮추면 optimizer가 4~6번째 종목으로 alpha 분산.
    # REDESIGN iter19 (2026-04-17): 0.06 → 0.04. Single-stock active 리스크 제한.
    # UNH +6% OW → 7월 -97bp 단일월 손실 사례. 4% cap으로 max loss ~-65bp 수준.
    satellite_max_per_stock: float = 0.04     # was 0.06

    # ------------------------------------------------------------------
    # Score-gated OW  — REDESIGN G (2026-04-11 PM)
    # ------------------------------------------------------------------
    # Classic MVO quirk: the covariance (risk) term rewards diversification
    # of active bets, so the optimizer sometimes OW'd stocks with NEGATIVE
    # scores just because they had low correlation with other active bets.
    # The monthly OW explanations explicitly flagged this case:
    #   "Optimizer risk-adjusted OW despite negative signal (z=-0.23)"
    #
    # Fix: hard-constrain w_i <= bm_i for any stock whose score is below
    # score_threshold_for_ow. In other words, only stocks the model likes
    # (score > threshold) can be overweighted. Stocks below the threshold
    # can still be HELD (up to their BM weight) or UW'd, but never OW'd.
    #
    # score_threshold_for_ow = 0.0 means "require strictly positive z-score
    # for OW". Raise to 0.25 or 0.50 for stricter conviction gates.
    enforce_score_gated_ow: bool = True
    score_threshold_for_ow: float = 0.0

    # ------------------------------------------------------------------
    # Turnover control  — REDESIGN J (2026-04-12, from codex_v2)
    # ------------------------------------------------------------------
    # Two complementary mechanisms to reduce churn without changing the
    # underlying alpha model:
    #
    # 1. No-trade band: if a position's weight change |delta_i| is below
    #    this threshold, skip that name entirely. Eliminates micro-trades
    #    caused by z-score noise between rebalances.
    # 2. Partial execution (eta): execute only eta% of the desired weight
    #    change per rebalance. eta=1.0 = full step (old behavior),
    #    eta=0.5 = half-step (codex_v2 default).
    #
    # Together these typically cut annual two-way turnover by 50-70%.
    # REDESIGN O-d (2026-04-14): 0.007 시도 → 함께 P3 망가짐. 롤백.
    no_trade_band: float = 0.003           # skip trades < 30 bps (back)
    # REDESIGN iter17 (2026-04-17): 0.50 → 0.65. IC→PnL 전환 효율 개선.
    # P2 IC=0.061 양호, P2 IR=0.30 저조 → eta가 병목. 더 빠른 수렴으로 시그널 반영.
    partial_rebalance_eta: float = 0.50    # iter15 baseline restored

    # ------------------------------------------------------------------
    # PEAD post-process boost — REDESIGN U (2026-04-14)
    # ------------------------------------------------------------------
    # Post-process score adjustment for Post-Earnings Announcement Drift.
    # Computes per (date, ticker):
    #   days_since_earnings (from data.earnings_timeline)
    #   revision_quality = clip(eps_rev_ma_63d / 100, 0, 1)
    #   pead_signal = exp(-days_since / decay_days) × revision_quality
    #   prediction += boost_weight × pead_signal
    # Only positive boost (no penalty side). Stocks with positive analyst
    # revision pre-earnings continue to drift up for ~3 weeks post-earnings.
    pead_boost_enabled: bool = True
    pead_boost_weight: float = 0.30          # max boost in z-score units
    pead_decay_days: float = 7.0             # exponential decay rate
    pead_max_days: int = 21                  # cutoff window after announcement

    # ------------------------------------------------------------------
    # Growth/Revision tilt — REDESIGN iter19 (2026-04-17)
    # ------------------------------------------------------------------
    # Post-prediction score adjustment favoring stocks with strong EPS/Sales
    # growth momentum and positive analyst revisions. Addresses the 0.61x
    # importance ratio of growth/revision vs margin/quality features.
    #
    # Mechanism: for each (date, ticker), compute a composite growth signal
    # from revision MA + fundamental growth. Add weight * cs_rank(composite)
    # to model prediction. Tilts OW toward "growing AND revised up" names.
    growth_tilt_enabled: bool = True
    # iter19 PRODUCTION: tilt 0.25, rev 50/fund 50, EPS:Sales 50:50.
    # iter19e (Sales40+TG30) IR 1.298 확인 후 원복: TG는 후행지표라 역효과.
    growth_tilt_weight: float = 0.25         # boost in z-score units
    growth_tilt_rev_weight: float = 0.50     # revision 50%
    growth_tilt_fundamental_weight: float = 0.50  # growth 50%
    growth_tilt_eps_skew: float = 0.50       # EPS:Sales within fundamental (50:50)
    # revision composite: EPS 50 + Sales 50 (TG share=0 disables TG component)
    growth_tilt_rev_eps_share: float = 0.50       # EPS revision
    growth_tilt_rev_sales_share: float = 0.50     # Sales revision
    growth_tilt_rev_tg_share: float = 0.00        # TG price OFF (후행지표)

    # ------------------------------------------------------------------
    # Mega-cap asymmetric protection — REDESIGN V (2026-04-14, iter16)
    # ------------------------------------------------------------------
    # Fixes structural UW of large-BM names even when the model scores
    # them positive. Diagnosis: on 15 dates with MSFT z>0.25, portfolio
    # was UW on 12 (mean -1.86%). Root cause: quad TE penalty makes
    # active OW on high-vol / correlated mega caps expensive, so the
    # optimizer routes the active budget to smaller names.
    #
    # Asymmetric per-name active bounds for names with bm_i >= threshold:
    #   score >= no_uw_score  →  w_i >= bm_i          (no UW allowed)
    #   score <= wide_uw_score →  bm_i - w_i <= wide_uw_cap
    #                             (deeper UW allowed than default)
    #   middle zone           →  standard max_active_per_stock applies
    #
    # Interacts with enforce_score_gated_ow (score<0 → w<=bm) — a mega
    # cap with score in [no_uw_score, 0) is pinned to exactly bm_i.
    #
    # NOTE: the current implementation only wires the funding-mode branch;
    # with funding_mode=False the protection is a no-op (generates no
    # constraints). See _build_mvo_constraints, which warns in that case.
    mega_cap_protection_enabled: bool = True
    mega_cap_bm_threshold: float = 0.04      # only names with bm_i >= 4%
    mega_cap_wide_uw_cap: float = 0.10       # max |UW| for selected funding set

    # Concentrated funding mode (iter19): instead of spreading UW across all
    # mega caps, pick the K worst-scoring mega caps and concentrate UW on
    # them. All other mega caps are pinned to w_i >= bm_i (no UW). This
    # prevents the optimizer from routinely UW'ing high-scoring mega caps
    # (the MSFT/AVGO problem) while still providing a real funding source.
    # Only mega caps with score < funding_score_max are eligible; if fewer
    # than K qualify, we use whatever's available.
    mega_cap_funding_mode: bool = True
    mega_cap_funding_k: int = 4              # number of mega caps used as funding
    mega_cap_funding_score_max: float = 0.0  # only UW mega caps with score below this

    # ------------------------------------------------------------------
    # BM-proportional active cap (2026-04-20, INFRA)
    # ------------------------------------------------------------------
    # Generalises mega_cap_protection as a continuous function of BM weight
    # and realised volatility, rather than a hard threshold + funding set.
    # Formula (when bm_proportional_cap_enabled=True):
    #
    #   cap_i = base_cap × bm_scale(bm_i) × vol_scale(vol_i)
    #   bm_scale  = 1 + (bm_scale_at_top - 1) × (bm_i / max(bm))
    #   vol_scale = clip(median_vol / vol_i, vol_scale_floor, 1.0)
    #
    # Effect: mega caps get LARGER active room proportional to their BM
    # weight (room to express conviction without dominating the active
    # share), while high-vol names get TIGHTER caps (single-stock risk).
    # OFF by default — needs validation against baseline before promoting.
    bm_proportional_cap_enabled: bool = False
    bm_proportional_cap_bm_scale_at_top: float = 1.5   # mega cap gets 1.5× base cap
    bm_proportional_cap_vol_scale_floor: float = 0.5   # high-vol cap floored at 0.5×
    bm_proportional_cap_vol_lookback: int = 63         # days for vol estimate

    # ------------------------------------------------------------------
    # Optimizer / risk guardrails
    # ------------------------------------------------------------------
    allow_scs_on_ecos_exception: bool = False
    max_name_active_risk_share: float = 0.35
    # 0.75 (uncalibrated vendored value) sat below the last-2y median (0.769)
    # of the 100-name book — a permanent alarm, not a control. 0.85 = full-period
    # P90 (0.827) rounded up to 0.05; flags only top-decile concentration.
    # Calibration: decision log §S10.1 (2026-07-18, 94-rebalance reconstruction).
    max_sector_active_risk_share: float = 0.85

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------
    # REDESIGN R (2026-04-14): 10 → 21. iter 6 baseline turnover 455% (codex 157%).
    # rebal_freq 두 배로 늘려 trade 횟수 직접 감소. 예상 turnover ~225%.
    # P2 IR도 약간 도움 가능 (less churn). P3 (signal-following 강한 시기) 슬로우다운 리스크.
    rebalance_freq: int = 21
    one_way_tc: float = 0.0010

    # ------------------------------------------------------------------
    # Signal stability penalty (2026-04-20, INFRA)
    # ------------------------------------------------------------------
    # Post-prediction shrinkage toward the previous retrain's predictions.
    #   z_adj_t = z_t - λ × (z_t - z_{t_prev_retrain})
    # Rationale: the 20d horizon signal is noisy between retrains; each
    # retrain produces a sudden jump in scores → turnover spike. Shrinking
    # toward the previous retrain's score in a controlled way damps that
    # without altering the model. Set lambda to 0 to disable.
    # OFF by default — needs validation vs baseline 115.6% turnover.
    signal_stability_lambda: float = 0.0       # 0.0 = disabled

    # ------------------------------------------------------------------
    # Value-trap gate (2026-04-24) — Phase 3 guard
    # ------------------------------------------------------------------
    # Empirical finding: "cheap (fin_pe_level_z < -0.5) + bad momentum
    # (momentum_252d < -0.5) + margin accel (oper_margin_accel > +0.5)"
    # pattern has -0.25%/20d fwd specific return (hit 47.3%) vs +0.82%
    # for the same cheap+bad_mom profile WITHOUT the accel leg. In P3
    # regime (2023-): -1.99%/20d. Margin_accel acts as bear-trap bounce
    # indicator in dying industries, not a reversal signal.
    # Gate: if all three conditions met, multiply prediction by `scale`.
    # OFF by default.
    value_trap_gate_enabled: bool = False
    vtg_pe_z_threshold: float = -0.5
    vtg_momentum_threshold: float = -0.5
    vtg_accel_threshold: float = 0.5
    vtg_scale: float = 0.0        # 0.0 = zero score, 1.0 = no effect

    # ------------------------------------------------------------------
    # A1 (2026-07-06) — z→mu volatility scaling (Grinold α = σ·z)
    # ------------------------------------------------------------------
    # Final overlay step: rescales the post-overlay unit-free CS z into an
    # expected-return (mu) panel by each ticker's trailing realized vol,
    # median-normalised so the median-σ name is unchanged (parameter-free).
    # Uses the same non-interpolated raw returns / trailing `cov_lookback`
    # window as the covariance estimator. OFF by default (byte-identical).
    mu_vol_scaling_enabled: bool = False

    # ------------------------------------------------------------------
    # S8 (2026-07-07) — news_trend sentiment feature arm
    # ------------------------------------------------------------------
    # Single pre-registered feature `news_trend` conditionally added to the
    # core whitelist (assembly.apply_core_filter). OFF by default: byte-
    # identical panel. Do NOT flip before the gate passes
    # (ΔIR > +0.36 & sub-period sign consistency).
    news_trend_feature_enabled: bool = False

    # ------------------------------------------------------------------
    # Data freshness guardrail
    # ------------------------------------------------------------------
    max_tail_ffill_days: int = 10
    fail_on_stale_tail_ffill: bool = False

    # ------------------------------------------------------------------
    # OOS hold-out (2026-04-20)
    # ------------------------------------------------------------------
    # Enforces a split between the "research window" (≤ train_cutoff_date)
    # and a strictly reserved "final OOS evaluation window" (> cutoff).
    #
    # Usage:
    #   - During tuning/search: set train_cutoff_date to e.g. "2024-12-31"
    #     and `enforce_oos_holdout=True`. walk_forward_train will refuse to
    #     generate predictions past the cutoff, so any backtest metric you
    #     compute is mechanically OOS-safe for the reserved window.
    #   - For final verification run: leave cutoff None (or set
    #     enforce_oos_holdout=False). Only do this ONCE per baseline
    #     candidate.
    #
    # Rationale: N_trials already = 402 per experiment_inventory.json. Every
    # additional run that "peeks" at post-cutoff data inflates selection
    # bias and erodes the haircut-adjusted Sharpe. Without a hard gate the
    # discipline tends to slip — this makes it mechanical.
    enforce_oos_holdout: bool = False        # off by default so existing
                                              # iter15_FINAL reproduction is
                                              # bitwise unchanged
    train_cutoff_date: Optional[str] = None   # e.g. "2024-12-31" (inclusive)

    # --- Shared derivations (SSOT for __post_init__) -------------------------
    @staticmethod
    def derive_max_active_share(satellite_budget, portfolio_style,
                                current_max_active_share, ceiling=None) -> float:
        """L1 active-share cap implied by a (possibly runtime) satellite_budget.

        cp.norm1(w - bm) is the L1 form == 2 * one-way active share, and
        satellite_budget is the one-way target, so the implied L1 cap is
        2 * satellite_budget.
          - core_satellite: cap = 2*satellite_budget, TWO-WAY (loosen + tighten),
            optionally min()'d with ``ceiling``. This is the SSOT Hook B uses so a
            runtime satellite_budget override actually reaches the MVO.
          - any other style: unchanged (returns current_max_active_share).
        """
        if portfolio_style != "core_satellite":
            return float(current_max_active_share)
        cap = 2.0 * float(satellite_budget)
        if ceiling is not None:
            cap = min(cap, float(ceiling))
        return float(cap)

    def __post_init__(self):
        """Apply portfolio_style overrides after dataclass init."""
        if self.portfolio_style == "core_satellite":
            # L1 cap implied by satellite_budget (SSOT: derive_max_active_share).
            # __post_init__ stays TIGHTEN-ONLY for off-path bit-parity; two-way
            # tracking under a runtime RL override is Hook B's job.
            cs_l1 = self.derive_max_active_share(
                self.satellite_budget, self.portfolio_style,
                self.max_active_share, ceiling=self.max_active_share_ceiling)
            if self.max_active_share > cs_l1:
                self.max_active_share = cs_l1
            if self.max_active_per_stock > self.satellite_max_per_stock:
                self.max_active_per_stock = self.satellite_max_per_stock

        if (self.max_active_share_ceiling is not None
                and not (0.0 < self.max_active_share_ceiling <= 2.0)):
            raise ValueError(
                "max_active_share_ceiling must be None or in (0, 2], got "
                f"{self.max_active_share_ceiling}"
            )
        if self.min_model_trees < 1:
            raise ValueError("min_model_trees must be >= 1")
        if not (0.0 <= self.max_degenerate_model_rate <= 1.0):
            raise ValueError("max_degenerate_model_rate must be in [0, 1]")
        if self.model_objective not in ("regression", "cross_sectional_rank"):
            raise ValueError(
                "model_objective must be 'regression' or "
                f"'cross_sectional_rank', got {self.model_objective!r}"
            )
        if self.execution_signal_lag_days < 0:
            raise ValueError("execution_signal_lag_days must be >= 0")
        if not (2 <= self.rank_relevance_levels <= 31):
            raise ValueError("rank_relevance_levels must be in [2, 31]")
        if not self.rank_eval_at or any(int(k) <= 0 for k in self.rank_eval_at):
            raise ValueError("rank_eval_at must contain positive integers")
        if self.max_tail_ffill_days < 0:
            raise ValueError("max_tail_ffill_days must be >= 0")
        self.base_currency = str(self.base_currency).upper()
        if self.base_currency != "USD":
            raise ValueError(
                "base_currency must be 'USD'; the configured FX quote map "
                f"does not support {self.base_currency!r}"
            )
        if self.max_fx_staleness_days < 0:
            raise ValueError("max_fx_staleness_days must be >= 0")
        if self.convert_returns_to_usd and not str(self.fx_source_path).strip():
            raise ValueError(
                "fx_source_path must be non-empty when convert_returns_to_usd=True"
            )
        if self.max_name_active_risk_share <= 0:
            raise ValueError("max_name_active_risk_share must be > 0")
        if self.max_sector_active_risk_share <= 0:
            raise ValueError("max_sector_active_risk_share must be > 0")
        if self.projection_fallback_mode not in ("target", "prev"):
            raise ValueError(
                "projection_fallback_mode must be 'target' or 'prev', got "
                f"{self.projection_fallback_mode!r}"
            )

    # ------------------------------------------------------------------
    # Attribution
    # ------------------------------------------------------------------
    n_grid_points: int = 30
    # M6: marginal-prediction sampling cap. Previously hardcoded to 200
    # inside attribution._compute_marginal_prediction. For larger
    # universes (50+ tickers) 500 reduces grid variance noticeably.
    marginal_prediction_n_samples: int = 200

    # ------------------------------------------------------------------
    # Alpha-source attribution — Pictet crystal-box (2026-06-18)
    # ------------------------------------------------------------------
    # Wires the existing (dead) src/attribution.run_attribution into the
    # run_variant / CLI metrics path. Read-only: changes no weights. When
    # enabled, attaches metrics['alpha_attribution'] (linear / marginal_nl /
    # interaction-upper-bound group shares). EXPENSIVE (SHAP over n_dates
    # model dates) — OFF by default, turn on only in research/ablation runs.
    alpha_attribution_enabled: bool = False
    alpha_attribution_n_dates: int = 8     # model dates subsampled for SHAP/decomp

    # ------------------------------------------------------------------
    # Style-factor active-exposure neutralization — Pictet (2026-06-18)
    # ------------------------------------------------------------------
    # SOFT penalty pulling active style exposures (loadings @ active) toward 0.
    # Loadings are existing cross-sectional feature z-scores. GROWTH+MOMENTUM
    # axes excluded a priori (they conflict with growth_tilt/PEAD overlays).
    # SIZE axis dropped: no size/market-cap column exists in the feature panel
    # (assembly.py CORE_FEATURE_WHITELIST) — decision-log §S5 §4.3 pre-check.
    # Likely NON-BINDING (TE-var already absorbs systematic style variance) —
    # judge by exposure drop, NOT IR. OFF by default; single pre-committed
    # penalty (no sweep-by-IR selection = p-hacking).
    factor_neutral_enabled: bool = False
    factor_neutral_penalty: float = 5.0    # single pre-committed weight
    factor_neutral_axes: List[str] = field(
        default_factory=lambda: ["value", "quality", "lowvol"]
    )
    # axis -> feature column used as the loading (must exist in the panel)
    factor_neutral_loadings: Dict[str, str] = field(default_factory=lambda: {
        "value": "best_peg_ratio_level_z",
        "quality": "best_roe_level_z",
        "lowvol": "idio_vol_63d",
    })

    # ------------------------------------------------------------------
    # Dynamic execution (REDESIGN K)
    # ------------------------------------------------------------------
    # Item 12: trailing IC window for confidence-based execution. 6
    # rebalances × rebalance_freq days ≈ 60 calendar days of recent model
    # accuracy. Tune upward for smoother confidence, downward for more
    # regime-responsive behaviour.
    trailing_ic_window: int = 6

    # Confidence spread normalization scale (2026-07-02 structure review #2).
    # compute_signal_confidence divides the top-bottom prediction spread by this
    # before clipping to [0.20, 1.00]. Predictions are cross-sectional z-scores,
    # so the spread is ≈3+ and 0.20 keeps the spread term perpetually saturated
    # at 1.00 (confidence collapses to the IC term). Default keeps current
    # behaviour; recalibrate only via a variant override after ablation.
    confidence_spread_scale: float = 0.20

    # Projection-failure execution fallback (2026-07-02 structure review).
    # When the MVO hard-constraint projection is infeasible, which book to
    # execute: "target"=current (full-step MVO target) or "prev"=no trade
    # (hold prev_weights, matching the partial-execution intent). Default
    # "target" reproduces current behaviour bit-for-bit.
    projection_fallback_mode: str = "target"    # {"target", "prev"}

    # ------------------------------------------------------------------
    # CS-DR-Alpha (cross-sectional direct-reinforcement alpha) — PRODUCTION
    # ------------------------------------------------------------------
    # The production alpha engine. A torch policy is trained by gradient ascent
    # on a differentiable, turnover-penalised cross-sectional Sharpe, run true
    # walk-forward, and its scores feed the production MVO via
    # run_backtest(precomputed_predictions=...). When dr_alpha_enabled is set
    # (production variant iter15_65tkr_reb21_vtg), run_variant.py and
    # daily_update.py harvest the LightGBM baseline, train the DR walk-forward,
    # then re-run the MVO on the DR scores. Engine: src/rl/dr_alpha.py,
    # src/rl/dr_walkforward.py (driver: scripts/run_dr_alpha.py).
    # Default False so a bare PipelineConfig() / plain backtest stays pure
    # LightGBM (the comparison baseline); the production variant turns it on.
    dr_alpha_enabled: bool = False
    dr_alpha_arch: str = "linear"            # "linear" | "tiny"
    dr_alpha_hidden: int = 16                # tiny-MLP hidden units
    dr_alpha_lr: float = 1e-2
    dr_alpha_epochs: int = 300
    dr_alpha_l2: float = 1e-3                # Adam weight decay
    dr_alpha_turnover_lambda: float = 0.10   # λ on the turnover term in reward
    dr_alpha_residual: bool = True           # combined = LGBM_z + γ·zscore(π)
    dr_alpha_gamma: float = 1.0              # residual strength (0 ⇒ baseline)
    dr_alpha_use_lgbm_feature: bool = False  # append LGBM z as a policy input
                                             # feature (lets standalone anchor on
                                             # LGBM and learn a smoothed alpha)
    dr_alpha_embargo: int = 20               # trading-day embargo (= H horizon)
    dr_alpha_warm_start: bool = True         # warm-start each fold from previous
    dr_alpha_seed: int = 42
    dr_alpha_val_months: int = 6             # validation slice (early stop)
    dr_alpha_min_train_rebal: int = 12       # min sampled rebal dates per fold.
                                             # Folds whose (prior-clipped) pool
                                             # yields fewer samples keep the
                                             # LGBM prior (12 ≈ 1y of history).
    dr_alpha_apply_ema: bool = True          # blend DR scores with
                                             # prediction_ema_alpha before the
                                             # MVO re-run — parity with the
                                             # LightGBM path, whose predictions
                                             # are EMA-blended inside
                                             # walk_forward_train. False trades
                                             # the un-smoothed DR signal.


# Global default instance -- every module references this by default.
DEFAULT_CONFIG = PipelineConfig()


# =============================================================================
# Experiment manifest helpers
# -----------------------------------------------------------------------------
# Every pipeline run should dump the active config to
# outputs/experiment_manifest.json alongside the git hash and timestamp.
# This prevents "I can't reproduce last week's backtest because config moved"
# and gives selection-bias accounting a stable record of what was tried.
# =============================================================================


def _git_hash(repo: Optional[Path] = None) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo or Path(__file__).resolve().parents[1],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def _git_dirty(repo: Optional[Path] = None) -> Optional[bool]:
    # Exclude outputs/ so a backtest that regenerates artefacts before the
    # manifest is written is not reported as a dirty working tree.
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--", ".", ":(exclude)outputs"],
            cwd=repo or Path(__file__).resolve().parents[1],
            stderr=subprocess.DEVNULL,
        )
        return bool(out.decode().strip())
    except Exception:
        return None


def dump_experiment_manifest(
    config: PipelineConfig = DEFAULT_CONFIG,
    output_dir: Optional[str] = None,
    extra: Optional[Dict] = None,
) -> Path:
    """Write an experiment manifest JSON snapshot.

    The manifest records:
      - timestamp (UTC)
      - git HEAD hash (if available) and whether working tree was dirty
      - full PipelineConfig as a dict
      - optional `extra` dict for run-specific metadata (e.g. run label,
        dataset version)

    Returns the path to the written manifest file.
    """
    out_dir = Path(output_dir or config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "experiment_manifest.json"

    manifest = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "git_hash": _git_hash(),
        "git_dirty": _git_dirty(),
        "config": asdict(config),
    }
    if extra:
        manifest["extra"] = extra

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return manifest_path
