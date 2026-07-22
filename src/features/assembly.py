"""Feature assembly — imports from all category modules and builds the final panel.

This is the main entry point: build_all_features().
"""

import gc
import logging
import re

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

from src.config import DEFAULT_CONFIG, PipelineConfig
from src.data_loader import UniverseData, TICKERS
from src.features.utils import cross_sectional_zscore, clip_outliers, cs_rank, safe_pct_change, rolling_tsz
from src.features.accounting import build_accounting_features
from src.features.price import build_price_features
from src.features.sellside import build_sellside_features
from src.features.conditioning import build_conditioning_features
from src.features.factor import build_factor_features
from src.features.interaction import build_sector_interaction_features
from src.features.regime import build_regime_features
from src.features.short_interest import build_short_interest_features
from src.features.macro_cross import build_macro_cross_features

# EPS/Sales/Margin: 변화율·가속도만 중요, level 자체는 제거
LEVEL_SKIP_SHEETS = {"BEST_EPS", "BEST_SALES", "BEST_GROSS_MARGIN", "OPER_MARGIN"}


# =============================================================================
# REDESIGN C (2026-04) — "lean" feature mode
# -----------------------------------------------------------------------------
# Previous 345-feature panel was ~70% Accounting+Sellside (Quality/Value
# composite) and the model was unable to catch AI/narrative momentum in P3.
# Lean mode prunes the redundant low-importance horizons in each category
# and appends a few multi-horizon momentum composites so the model has more
# short-to-medium-term momentum signal to work with. Target ~150 features.
# =============================================================================

# Regex patterns (case-insensitive) to drop when feature_mode == "lean".
# Every pattern is a safety guard against the bloat that dominated the 345
# panel. Hand-picked from feature_importance.csv and ad-hoc inspection.
#
# ROLLBACK 2026-04-11 PM: the C+ negative-lookahead experiment (preserve
# EPS/Sales/FCF short-window growth + add 17 composites) produced a worse
# backtest (IR 0.62 → 0.23) despite raising Growth importance share from
# 5.7% to 19.5%. Diagnosis: analyst-estimate data is noisy at 21d/126d
# horizons, composites were redundant, and the model ended up concentrating
# on contrarian names (000660 +12%, UNH +5.5%) instead of Mag7 Growth.
# Restoring the simpler A+C+D+E baseline filter here.
LEAN_DROP_PATTERNS: List[str] = [
    # ---- Accounting: drop noisy short-window changes and change-vol ----
    r"_chg_21d$",          # keep 63/126/252 for trend
    r"_chg_126d$",          # keep 63/252 for trend
    r"_chg_vol$",           # vol of changes: noisy, low importance
    r"_rank$",              # raw percentile rank: we already have _level_z
    r"_vs_median$",         # level_z already covers relative position
    # ---- Valuation: keep accel and level_z only ----
    r"^best_pe_ratio_chg_21d$", r"^best_pe_ratio_chg_63d$", r"^best_pe_ratio_vol$",
    r"^best_peg_ratio_chg_21d$", r"^best_peg_ratio_chg_63d$", r"^best_peg_ratio_vol$",
    r"^best_px_bps_ratio_chg_21d$", r"^best_px_bps_ratio_chg_63d$", r"^best_px_bps_ratio_vol$",
    r"^best_ev_to_best_ebitda_chg_21d$", r"^best_ev_to_best_ebitda_chg_63d$",
    r"^best_ev_to_best_ebitda_vol$",
    # ---- Conditioning: drop sector one-hot (low importance) ----
    r"^sector_",            # sector one-hot: 2% importance in SHAP
    r"^cal_is_Q[234]$",     # keep only Q1 (January effect)
    r"^cal_is_monday$",
    r"^cal_is_mid_qtr$",
    r"^cal_is_first_half$",
    r"^regime_avg_vol_63d$",
    r"^regime_mkt_ret_63d$",
    # ---- Factor: drop vol and accel of each factor (keep momentum only) ----
    r"^fac_.*_vol_21d$",
    r"^fac_.*_vol_63d$",
    r"^fac_.*_accel$",
    r"^fac_VIX_panic$",    # binary duplicates
    r"^fac_curve_(inverted|steep)$",
    # ---- Sellside: drop short-window revision noise ----
    r"_rev_time_(low|high)$",
    r"_rev_ma_21d$",
    # ---- Price: drop highly correlated duplicate distributions ----
    r"^ret_kurt_",          # skew is more robust
    r"^pos_ret_ratio_",
    r"^downside_vol_63d$",
]

_LEAN_RE = re.compile("|".join(f"(?:{p})" for p in LEAN_DROP_PATTERNS), re.IGNORECASE)


def apply_lean_filter(features: Dict[str, pd.DataFrame],
                      feature_groups: Dict[str, List[str]]) -> None:
    """In-place prune features whose names match LEAN_DROP_PATTERNS.

    Also removes dropped names from the feature_groups mapping so downstream
    attribution / group contribution calculations see the same universe.
    """
    dropped = [name for name in list(features.keys()) if _LEAN_RE.search(name)]
    for name in dropped:
        features.pop(name, None)
    for grp, names in list(feature_groups.items()):
        feature_groups[grp] = [n for n in names if n not in dropped]
    print(f"[FeatureEngine] lean mode: dropped {len(dropped)} features, "
          f"kept {len(features)}")


# =============================================================================
# REDESIGN C++ (2026-04-11 PM) — "core" feature whitelist
# -----------------------------------------------------------------------------
# After lean mode's 239-feature panel, we further prune to ~85 core features
# using an explicit whitelist derived from the A+C+D+E run's feature
# importance ranking. Style proportions are balanced so no single axis
# dominates:
#   Quality    : 18 features (21%)
#   Growth     : 6  features  (7%)   ← just the multi-horizon 252d/63d base
#   Value      : 6  features  (7%)
#   Momentum   : 14 features (16%)   ← boosted vs lean 20% share
#   Price/Risk :  6 features  (7%)
#   Sellside   : 14 features (16%)
#   Macro      : 10 features (12%)
#   Regime/Cal :  8 features (9%)
#   Growth-composite extras: 3 (4%)  ← mom_accel_63_252 which actually worked
# Total target: ~85
# =============================================================================

CORE_FEATURE_WHITELIST: set = {
    # REDESIGN H (2026-04-12): further pruned from 81 -> 46 features based
    # on the Core-Satellite + Score-Gate run's feature_importance ranking.
    # Removed 35 features that had low or ZERO importance in production:
    #   Conditioning: earn_days_since / cal_is_jan / is_mega_cap had 0 imp
    #   Factor:       fac_VIX_level had 0 imp, dropped 21d variants
    #   Price:        dropped short-horizon momentum duplicates + reversal_5d
    #   Accounting:   dropped accel variants of roe/pe/peg, bottom chg_63d
    #   Sellside:     dropped bottom 6 by importance (tg_upside_z, rev_combined,
    #                 analyst_rec_accel, eps_rev_level_persist_63d, etc.)
    # Target proportions preserved: Acct 39%, Price 26%, Sellside 17%,
    # Factor 11%, Conditioning 7%.

    # --- Quality (12) — top margins / cash / capex / ROE / op leverage ---
    "oper_margin_chg_63d", "oper_margin_chg_252d", "oper_margin_accel",
    "best_gross_margin_chg_63d", "best_gross_margin_chg_252d",
    "cash_conversion_z",
    "capex_intensity_z",
    "op_leverage_63d",
    "best_roe_level_z",
    "earnings_quality_252d",
    "best_calculated_fcf_level_z",
    "best_capex_level_z",

    # --- Growth (3) — pure 252d multi-base growth ---
    # REDESIGN R (2026-04-14): best_eps_chg_63d 최종 제거. iter 7 실험 결과
    # tg_upside와 함께 추가해도 IR 1.091 → 0.703 destructive. 사용자도 옵션 C
    # (변수 포기) 결정. iter 6 (IR 1.091)을 production 베이스라인으로 채택.
    "best_sales_chg_252d", "best_sales_accel",
    "best_eps_chg_252d",

    # --- Value (3) — valuation levels only, drop accel variants ---
    "best_peg_ratio_level_z",
    "best_ev_to_best_ebitda_level_z",
    "best_px_bps_ratio_level_z",

    # --- Momentum (7) — keep top trend + acceleration only ---
    "momentum_252d",
    "risk_adj_mom_252d",
    "ma_cross_21_50", "ma_cross_50_200",
    "max_ret_63d", "min_ret_63d",
    "mom_accel_63_252",        # lean composite that actually worked

    # --- Price / Risk (5) ---
    "beta_63d", "idio_vol_63d",
    "realized_vol_21d", "realized_vol_126d",
    "dist_52w_high",

    # --- Sellside (8) — analyst + revision + target ---
    # iter21 실험: tg_upside 제거 → IR 1.119 (-0.48 vs iter19). 원복.
    # Feature PnL은 marginal attribution, counterfactual PnL 아님 — interaction 효과로 제거 시 손실.
    "analyst_rec_level", "analyst_rec_stability",
    "tg_mom_63d", "tg_upside",
    # 2026-04-21 (Phase 2.4 final): revision MAs reverted to 63d after window
    # sweep (10d/21d/dual) all underperformed. Whitelist is 56 (baseline_v2).
    # Post-process overlays (PEAD, growth_tilt) go through get_cleaned_revision
    # so they use the same reversion_gated cleaned stream as these features.
    "eps_rev_ma_63d", "eps_rev_trend", "eps_rev",
    "sales_rev_ma_63d",

    # --- Macro/Factor (5) — 63d horizon only, drop 21d + VIX_level ---
    "fac_yield_slope",
    "fac_F_Quality_mom_63d",
    "fac_F_Growth_mom_63d",
    "fac_F_Value_mom_63d",
    "fac_value_growth_63d",

    # --- Regime / Conditioning (2) — REDESIGN U (2026-04-14) ---
    # earn_cycle_pos 제거: iter 14에서 timeline activation으로 panel에 생성된 후
    # whitelist 매칭으로 model에 들어가서 IR -0.290 손실. iter 15는 panel은
    # 그대로 두되 whitelist에서만 제거해 model 56-feature 상태 유지.
    "regime_mkt_ret_21d",
    "cal_is_Q1",
    # REDESIGN iter16 시리즈 (2026-04-16): regime 5개 + SI 3~4개 총 9회 시도,
    # 전부 destructive. 56-feature가 이 유니버스/아키텍처의 local optimum.
    # 코드는 regime.py / short_interest.py에 보존. whitelist에서 제외.

    # --- Financials (11) — V2 bank-specific feature block ---
    "fin_roe_level_z", "fin_roe_chg_63d", "fin_roe_chg_252d",
    "fin_pb_level_z", "fin_pb_chg_63d",
    "fin_pe_level_z", "fin_pe_chg_63d",
    # iter21: fin_eps_chg_63d 제거 시도 → P1 IR -0.637 붕괴. 원복.
    "fin_eps_chg_63d", "fin_sales_chg_63d",
    "fin_roe_pb_gap", "fin_roe_pe_gap",

    # --- Macro Cross (5) — Phase 2, 2026-04-22, target P2 rate-shock regime ---
    # Each term is macro_scalar(date) × ticker_specific_signal(date, ticker)
    # so it has real cross-sectional variation (unlike factor.py's bcast-only
    # macro features). See src/features/macro_cross.py for the full rationale.
    "mc_rate_x_eps_rev",      # UST_10Y_zscore × eps_rev_cs_z
    "mc_slope_x_eps_rev",     # (UST_10Y - UST_2Y) × eps_rev_cs_z
    "mc_vix_x_mom252",        # VIX_zscore × mom252_cs_z
    "mc_vol_x_mom63",         # realized_vol_21d_cs_z × mom63_cs_z
    "mc_dxy_x_eps_rev",       # DXY_zscore × eps_rev_cs_z

    # NOTE: REDESIGN S/S2 quality_gated_value_z 시도되었으나 두 번 다 destructive.
    #   iter 12 (composite + post-process): IR 1.183 → 1.005 (P1/P3 손실)
    #   iter 13 (composite only):           IR 1.183 → 0.621 (P3 catastrophic -1.021)
    # 모델이 새 feature 추가에 매우 민감 (iter 4, 5, 7, 12, 13 5회 반복 검증).
    # 추측: LightGBM column space 변경이 split selection 분배를 교란시키고,
    # EWMA importance cold-start로 안정화 전 손실이 marginal value를 초과.
    # composite 컴퓨테이션 코드는 accounting.py에 보존 (orphan, but harmless).
    # Quality gate post-process는 2026-04-20 삭제 (rollback_log.md 참고).
}


def apply_core_filter(features: Dict[str, pd.DataFrame],
                      feature_groups: Dict[str, List[str]],
                      extra_whitelist: set | None = None) -> None:
    """In-place prune features to CORE_FEATURE_WHITELIST only.

    Features in the whitelist that don't actually exist in the panel are
    silently ignored (e.g. if a sheet was missing). After the filter, each
    feature_group entry is updated to reflect the survivors. If a group ends
    up empty it is removed from feature_groups entirely.

    `extra_whitelist` (S8) conditionally admits additional feature keys on top
    of CORE_FEATURE_WHITELIST; None (default) is inert and byte-identical to
    the legacy filter.
    """
    before = len(features)
    survivors = set(features.keys()) & (CORE_FEATURE_WHITELIST | (extra_whitelist or set()))
    dropped_here = [n for n in list(features.keys()) if n not in survivors]
    for name in dropped_here:
        features.pop(name, None)

    # Warn on any whitelist misses so we can spot stale entries
    missing = sorted(CORE_FEATURE_WHITELIST - survivors)
    if missing:
        print(f"[FeatureEngine] core mode: whitelist misses (will be ignored): {missing[:8]}"
              + ("..." if len(missing) > 8 else ""))

    for grp, names in list(feature_groups.items()):
        kept = [n for n in names if n in survivors]
        if kept:
            feature_groups[grp] = kept
        else:
            feature_groups.pop(grp, None)

    print(f"[FeatureEngine] core mode: {before} -> {len(features)} features "
          f"({len(missing)} whitelist entries missing from panel)")


def build_lean_momentum_composites(
    data: UniverseData,
    tickers: List[str],
) -> Dict[str, pd.DataFrame]:
    """REDESIGN C: multi-horizon momentum composite + residual momentum.

    These are added ON TOP of the existing price.py momentum features to
    give the model cleaner cross-sectional ranked momentum (less sensitive
    to absolute magnitude, more aligned with the "narrative momentum"
    that dominated 2023-24).
    """
    feats: Dict[str, pd.DataFrame] = {}
    # §S11.7: PIT 뷰(상장 전 NaN) — 모멘텀 횡단면 순위에서 유령 제외.
    returns = data.returns_masked.reindex(columns=tickers)

    # Cross-sectional momentum rank composite (5 horizons, then average).
    # Skip-1 momentum to avoid short-term reversal bias.
    horizons = [10, 21, 63, 126, 252]
    ranks = []
    for w in horizons:
        rolling = returns.rolling(w, min_periods=w).sum()
        ranks.append(cs_rank(rolling))
    feats["mom_rank_composite"] = sum(ranks) / len(ranks)

    # Short-mid momentum spread (1m vs 3m): captures acceleration
    mom21 = returns.rolling(21, min_periods=21).sum()
    mom63 = returns.rolling(63, min_periods=63).sum()
    feats["mom_accel_21_63"] = cs_rank(mom21) - cs_rank(mom63)

    # Mid-long momentum spread (3m vs 12m): trend vs long-term value
    mom252 = returns.rolling(252, min_periods=126).sum()
    feats["mom_accel_63_252"] = cs_rank(mom63) - cs_rank(mom252)

    # Price breakout: (price / 20-day high) − 1
    prices = data.prices.reindex(columns=tickers)
    max20 = prices.rolling(20, min_periods=10).max().replace(0, np.nan)
    feats["price_breakout_20d"] = (prices / max20) - 1.0

    # RSI(14) approximation via simple up/down smoothing
    delta = prices.diff()
    up = delta.clip(lower=0).rolling(14, min_periods=7).mean()
    down = (-delta.clip(upper=0)).rolling(14, min_periods=7).mean().replace(0, np.nan)
    rs = up / down
    feats["price_rsi_14"] = 100 - (100 / (1 + rs))

    return feats


def build_growth_composites(
    data: UniverseData,
    tickers: List[str],
    config=None,
) -> Dict[str, pd.DataFrame]:
    """REDESIGN C+ (2026-04-11): dedicated Growth-style feature block.

    Context: style-axis analysis showed Quality = 23.4% of feature importance
    but direct Growth (*_chg on EPS/Sales/FCF) was only 5.7%, a ~4x imbalance.
    This helper adds ~17 features drawn purely from raw fundamental sheets so
    Growth signal can stand on its own without being crowded out by margin /
    ROE / capex Quality variables.

    Feature catalogue:
      1. Growth acceleration (5) — cross-horizon differences of EPS/Sales/FCF
      2. Growth stability (2)    — rolling std of short-window growth
      3. Rank composites (2)     — avg cross-sectional rank across bases
      4. Growth momentum spread (3) — short-vs-long growth rank spread
      5. Growth-Value/Quality interactions (3) — PEG-implied growth, etc.
      6. Revision composites (2) — forward-looking growth proxy from Factset

    All features are raw (pre-zscore). cross_sectional_zscore is applied
    later in build_all_features, keeping the semantics consistent with
    other Accounting/Sellside features.
    """
    feats: Dict[str, pd.DataFrame] = {}

    def _get(sheet: str):
        try:
            raw = data.get_sheet(sheet)
            return raw.reindex(columns=tickers) if raw is not None else None
        except KeyError:
            logger.debug("assembly.composites: sheet %s missing", sheet)
            return None

    eps = _get("BEST_EPS")
    sales = _get("BEST_SALES")
    fcf = _get("BEST_CALCULATED_FCF")
    oper_margin = _get("OPER_MARGIN")
    roe = _get("BEST_ROE")
    pe = _get("BEST_PE_RATIO")
    # Revision sheets: route through Phase 2.4 cleaner for consistency with
    # the sellside feature panel + backtest post-process overlays.
    from src.features.sellside import get_cleaned_revision
    eps_rev_cleaned = get_cleaned_revision(data, "Factset_EPS_Revision", config=config)
    sales_rev_cleaned = get_cleaned_revision(data, "Factset_Sales_Revision", config=config)
    eps_rev = eps_rev_cleaned.reindex(columns=tickers) if eps_rev_cleaned is not None else None
    sales_rev = sales_rev_cleaned.reindex(columns=tickers) if sales_rev_cleaned is not None else None

    # ---- 1. Growth acceleration (short vs medium, medium vs long) ----
    if eps is not None:
        eps_21 = safe_pct_change(eps, 21)
        eps_63 = safe_pct_change(eps, 63)
        eps_252 = safe_pct_change(eps, 252)
        feats["eps_accel_21_63"] = eps_21 - eps_63
        feats["eps_accel_63_252"] = eps_63 - eps_252
        feats["eps_growth_stability_63"] = -eps_21.rolling(63, min_periods=21).std()

    if sales is not None:
        sales_21 = safe_pct_change(sales, 21)
        sales_63 = safe_pct_change(sales, 63)
        sales_252 = safe_pct_change(sales, 252)
        feats["sales_accel_21_63"] = sales_21 - sales_63
        feats["sales_accel_63_252"] = sales_63 - sales_252
        feats["sales_growth_stability_63"] = -sales_21.rolling(63, min_periods=21).std()

    if fcf is not None:
        fcf_63 = safe_pct_change(fcf, 63)
        fcf_252 = safe_pct_change(fcf, 252)
        feats["fcf_accel_63_252"] = fcf_63 - fcf_252

    # ---- 2. Growth rank composites (average across EPS/Sales/FCF bases) ----
    ranks_252 = []
    ranks_63 = []
    if eps is not None:
        ranks_252.append(cs_rank(safe_pct_change(eps, 252)))
        ranks_63.append(cs_rank(safe_pct_change(eps, 63)))
    if sales is not None:
        ranks_252.append(cs_rank(safe_pct_change(sales, 252)))
        ranks_63.append(cs_rank(safe_pct_change(sales, 63)))
    if fcf is not None:
        ranks_252.append(cs_rank(safe_pct_change(fcf, 252)))
        ranks_63.append(cs_rank(safe_pct_change(fcf, 63)))
    if ranks_252:
        feats["growth_rank_composite_252"] = sum(ranks_252) / len(ranks_252)
    if ranks_63:
        feats["growth_rank_composite_63"] = sum(ranks_63) / len(ranks_63)

    # ---- 3. Growth momentum spreads (short-term catching trend) ----
    if eps is not None:
        feats["eps_growth_spread_21_252"] = (
            cs_rank(safe_pct_change(eps, 21)) - cs_rank(safe_pct_change(eps, 252))
        )
    if sales is not None:
        feats["sales_growth_spread_21_252"] = (
            cs_rank(safe_pct_change(sales, 21)) - cs_rank(safe_pct_change(sales, 252))
        )
    if fcf is not None:
        feats["fcf_growth_spread_63_252"] = (
            cs_rank(safe_pct_change(fcf, 63)) - cs_rank(safe_pct_change(fcf, 252))
        )

    # ---- 4. Growth vs Value / Quality interactions ----
    # NOTE (2026-04-13): peg_growth_spread was removed. It used forward PE as
    # the "value" leg, which double-penalized high-growth names already paying
    # up on PE even when PEG itself was reasonable. The raw PEG-level feature
    # (best_peg_ratio_level_z, built in accounting.py) is the active valuation
    # proxy now — its denominator already contains growth, so it doesn't
    # penalize earners like NVDA whose PE is elevated by genuine expected growth.
    if eps is not None and oper_margin is not None:
        # Growth backed by margin level ("quality growth")
        feats["growth_at_quality_252"] = (
            cs_rank(safe_pct_change(eps, 252)) + cs_rank(oper_margin)
        ) / 2.0
    if sales is not None and roe is not None:
        # ROE-weighted growth (Buffett-ish growth)
        feats["roe_growth_252"] = (
            cs_rank(roe) + cs_rank(safe_pct_change(sales, 252))
        ) / 2.0

    # ---- 5. Forward-looking revision composites (Factset) ----
    # 2026-04-21: composite MA kept at 63d (long-horizon, matching post-process
    # overlays). The shorter 10d horizon is exposed to the model through the
    # dual-MA feature split, not through this composite.
    if eps_rev is not None:
        eps_rev_ma = eps_rev.rolling(63, min_periods=21).mean()
        feats["eps_rev_growth_composite"] = cs_rank(eps_rev_ma)
    if sales_rev is not None:
        sales_rev_ma = sales_rev.rolling(63, min_periods=21).mean()
        feats["sales_rev_growth_composite"] = cs_rank(sales_rev_ma)

    return feats


def build_financials_features(
    data: UniverseData,
    tickers: List[str],
) -> Dict[str, pd.DataFrame]:
    """REDESIGN K (2026-04-12): Financials-specific feature block.

    Ported from codex_v2 COMPACT_FEATURE_CATALOG["Financials"].
    Banks (JPM, GS, BLK, etc.) have no traditional FCF / CAPEX / EBITDA.
    Instead, the key drivers are ROE, P/B, P/E, and the gaps between them.

    These features are computed for ALL tickers (not just banks) so the
    model can learn "bank-like" patterns that may also appear in non-bank
    financials. For non-financial tickers the values simply represent
    standard ROE/PB/PE characteristics which are already useful.
    """
    feats: Dict[str, pd.DataFrame] = {}

    def _get(sheet: str):
        try:
            return data.get_sheet(sheet).reindex(columns=tickers)
        except KeyError:
            logger.debug("assembly.bank_features: sheet %s missing", sheet)
            return None

    roe = _get("BEST_ROE")
    pb = _get("BEST_PX_BPS_RATIO")
    pe = _get("BEST_PE_RATIO")
    eps = _get("BEST_EPS")
    sales = _get("BEST_SALES")

    if roe is not None:
        feats["fin_roe_level_z"] = cross_sectional_zscore(roe)
        feats["fin_roe_chg_63d"] = safe_pct_change(roe, 63)
        feats["fin_roe_chg_252d"] = safe_pct_change(roe, 252)

    # REDESIGN M (2026-04-13): Financials valuation level features use
    # per-ticker rolling 3Y self-normalization followed by cross-section.
    # Same rationale as accounting.py valuation block — raw PE/PB level
    # cross-section over-penalized NVDA/LLY-style structurally high-multiple
    # names; TSZ answers "vs own recent history" first.
    if pb is not None:
        pb_tsz = rolling_tsz(pb, window=756, min_periods=252)
        feats["fin_pb_level_z"] = cross_sectional_zscore(pb_tsz)
        feats["fin_pb_chg_63d"] = safe_pct_change(pb, 63)

    if pe is not None:
        # Upper-tail clip at z=1.5 still applied: even after TSZ->CS,
        # extreme outliers during regime shifts should not dominate the
        # signal (standard normal P90 ~1.28, so 1.5 is a soft cap).
        pe_tsz = rolling_tsz(pe, window=756, min_periods=252)
        feats["fin_pe_level_z"] = cross_sectional_zscore(pe_tsz).clip(upper=1.5)
        feats["fin_pe_chg_63d"] = safe_pct_change(pe, 63)

    if eps is not None:
        feats["fin_eps_chg_63d"] = safe_pct_change(eps, 63)

    if sales is not None:
        feats["fin_sales_chg_63d"] = safe_pct_change(sales, 63)

    # Cross-ratio gaps (bank valuation drivers)
    if roe is not None and pb is not None:
        # ROE-PB gap: high ROE + low PB = undervalued bank
        feats["fin_roe_pb_gap"] = cs_rank(roe) - cs_rank(pb)

    if roe is not None and pe is not None:
        # ROE-PE gap: high ROE + low PE = quality at discount
        feats["fin_roe_pe_gap"] = cs_rank(roe) - cs_rank(pe)

    return feats


def build_all_features(
    data: UniverseData,
    include_sector_interactions: bool = False,
    config: PipelineConfig = None,
) -> Tuple[pd.DataFrame, List[str], Dict[str, List[str]]]:
    config = config or DEFAULT_CONFIG
    feature_mode = getattr(config, "feature_mode", "full")

    accounting = build_accounting_features(data)
    price = build_price_features(data)
    sellside = build_sellside_features(data, config=config)
    conditioning = build_conditioning_features(data, config=config)
    factor = build_factor_features(data)
    regime = build_regime_features(data)
    short_interest = build_short_interest_features(data)
    # Phase 2 (2026-04-22): Macro × ticker cross features for P2 rate-shock fix.
    macro_cross = build_macro_cross_features(data, config=config)
    gc.collect()

    feature_groups = {
        "Accounting": list(accounting.keys()),
        "Price": list(price.keys()),
        "Sellside": list(sellside.keys()),
        "Conditioning": list(conditioning.keys()),
        "Factor": list(factor.keys()),
        "Regime": list(regime.keys()),
        "ShortInterest": list(short_interest.keys()),
        "MacroCross": list(macro_cross.keys()),
    }

    all_features: Dict[str, pd.DataFrame] = {}
    all_features.update(accounting)
    all_features.update(price)
    all_features.update(sellside)
    all_features.update(conditioning)
    all_features.update(factor)
    all_features.update(regime)
    all_features.update(short_interest)
    all_features.update(macro_cross)

    # Use the actually-loaded universe (intersection across all sheets)
    # stored on data.tickers.
    tickers = list(data.tickers)

    # REDESIGN C: lean feature mode prunes low-importance variants and
    # appends multi-horizon momentum composites to the Price group.
    if feature_mode in ("lean", "core"):
        print(f"[FeatureEngine] feature_mode={feature_mode}, starting from {len(all_features)} features")
        apply_lean_filter(all_features, feature_groups)

        mom_extras = build_lean_momentum_composites(data, tickers)
        for name, df in mom_extras.items():
            all_features[name] = df
        feature_groups["Price"] = list(feature_groups.get("Price", [])) + list(mom_extras.keys())
        print(f"[FeatureEngine] lean mode: added {len(mom_extras)} momentum composites, "
              f"total now {len(all_features)}")

        # NOTE: build_growth_composites is available but DISABLED by default
        # after the C+ experiment showed it hurt IR (0.62 -> 0.23). Kept as
        # reference implementation for future regime-conditional work.

        # REDESIGN K (2026-04-12): Financials feature block (V2 port)
        fin_extras = build_financials_features(data, tickers)
        for name, df in fin_extras.items():
            all_features[name] = df
        feature_groups["Financials"] = list(fin_extras.keys())
        print(f"[FeatureEngine] added {len(fin_extras)} financials features, "
              f"total now {len(all_features)}")

    # REDESIGN C++ (2026-04-11 PM): "core" mode further prunes to a hand-picked
    # whitelist (~85 features) with explicit style balance based on the
    # A+C+D+E run's feature importance ranking. This drops 239 -> 85 while
    # preserving the top features per style axis.
    if feature_mode == "core":
        extra = {"news_trend"} if getattr(config, "news_trend_feature_enabled", False) else None
        apply_core_filter(all_features, feature_groups, extra_whitelist=extra)

    # CS Z-score: conditioning / factor / regime(broadcast)는 제외
    skip_zscore = (set(feature_groups.get("Conditioning", []))
                   | set(feature_groups.get("Factor", []))
                   | set(feature_groups.get("Regime", [])))
    for name, df in list(all_features.items()):
        if name not in skip_zscore:
            all_features[name] = cross_sectional_zscore(df)

    # Sector Interaction Features (선택적)
    if include_sector_interactions:
        # Z-score 적용된 핵심 피처들로 interaction 생성
        sector_ix = build_sector_interaction_features(data, all_features)
        if sector_ix:
            feature_groups["SectorInteraction"] = list(sector_ix.keys())
            all_features.update(sector_ix)
            # interaction 피처도 skip_zscore에 추가 (이미 Z-score된 값의 interaction이므로)
            skip_zscore.update(sector_ix.keys())

    # 극단값 클리핑
    for name, df in all_features.items():
        all_features[name] = clip_outliers(df)

    # 개별 카테고리 dict 메모리 해제
    del accounting, price, sellside, conditioning, factor, regime, short_interest
    gc.collect()

    # 3D -> 2D panel (메모리 최적화: feature를 하나씩 변환 후 즉시 삭제)
    gc.collect()

    dates = data.dates
    common_idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    feature_names = list(all_features.keys())
    n_rows = len(common_idx)
    n_features = len(feature_names)

    # feature를 pop으로 하나씩 꺼내면서 메모리 해제
    # NaN 보존: reindex에서 fill_value=0을 쓰지 않고, 최종 단계에서
    # per-feature median으로 채워 raw broadcast 피처(conditioning, factor)의
    # 0-bias를 방지한다. z-score 피처의 median은 ~0이라 영향이 거의 없다.
    panel_values = np.full((n_rows, n_features), np.nan, dtype=np.float32)
    for i, feat_name in enumerate(feature_names):
        df = all_features.pop(feat_name)
        aligned = df.reindex(columns=tickers, index=dates)
        panel_values[:, i] = aligned.values.ravel().astype(np.float32)
        del df, aligned
        if i % 50 == 0:
            gc.collect()
    del all_features
    gc.collect()

    panel = pd.DataFrame(panel_values, index=common_idx, columns=feature_names)
    del panel_values
    gc.collect()

    # -------------------------------------------------------------------------
    # Per-date cross-sectional median fill (look-ahead free).
    #
    # Earlier drafts of this file used a GLOBAL median per feature, i.e.
    #     panel.median(axis=0).
    # That leaks future data into early missing cells because the median is
    # computed across the full date range. We now compute the median PER DATE
    # (across tickers only) via groupby(level="date").transform("median") and
    # fill NaN only with values from that same date. This matches the intent
    # of "ffill -> cross-sectional median" without reaching forward in time.
    #
    # Remaining NaN (whole row NaN for a feature on some date) falls back to
    # 0. This is only hit when no ticker has a value for that (date, feature)
    # pair, which is the same degenerate case the old code had to handle.
    # -------------------------------------------------------------------------
    per_date_median = panel.groupby(level="date").transform("median")
    panel = panel.fillna(per_date_median)
    panel = panel.fillna(0.0)

    print(f"[FeatureEngine] 총 피처 수: {len(feature_names)}")
    for group, names in feature_groups.items():
        print(f"  {group:15s}: {len(names)}개")

    return panel, feature_names, feature_groups
