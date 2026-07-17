"""Category 3: Sellside / Sentiment (~60 features).

Includes revision spike cleaning and bounded revision feature builders.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict

from src.data_loader import UniverseData
from src.features.utils import cross_sectional_zscore, safe_pct_change, cs_rank

logger = logging.getLogger(__name__)


# ===========================================================================
# Revision 전처리 유틸리티
# ===========================================================================

def clean_revision_spikes(
    rev: pd.DataFrame,
    threshold: float = 30,
    earnings_timeline: "pd.DataFrame | None" = None,
    mode: str = "down_only",
    extreme_threshold: float = 50.0,
    reversion_ratio: float = 0.5,
) -> pd.DataFrame:
    """실적발표 전후 컨센서스 기간 전환으로 인한 급변 스무딩.

    Modes
    -----
    "down_only" (default, baseline): only mask single-day drops greater than
        threshold. Preserves up-side moves. This is the original iter15
        behaviour — see docs/rollback_log.md for context.

    "symmetric": mask ``|daily_diff| > threshold`` in both directions. Use
        this mode to hedge against BOTH Factset rollover drop AND the
        symmetric rollover-induced jump (a bad-fundamentals stock whose
        score goes from -80 to +5 on rollover day). Risk: genuine analyst
        up-moves are also filtered.

    "reversion_gated": most targeted. Only mask large moves that (a) exceed
        the daily threshold AND (b) start from an extreme previous level
        (|prev| > extreme_threshold) AND (c) collapse toward neutral
        (|today| < |prev| * reversion_ratio). Catches rollover artifacts
        in both directions while preserving genuine moves INTO extremes.

    Pattern 1 handling is mode-dependent (see above). Pattern 2 (gradual
    drops near earnings) always operates on the down-side only — it
    represents a different phenomenon (pre-earnings leak of negative
    sentiment) that is directional by nature.

    처리: 스파이크/급변 구간을 NaN 마킹 후 ffill로 직전 정상값 복원.
    """
    if mode not in ("down_only", "symmetric", "reversion_gated"):
        raise ValueError(
            f"revision_clean_mode must be one of 'down_only', 'symmetric', "
            f"'reversion_gated', got {mode!r}"
        )

    cleaned = rev.copy()
    daily_diff = cleaned.diff()
    prev = rev.shift(1)

    # -- Pattern 1: single-day extreme moves ---------------------------------
    if mode == "down_only":
        spike_mask = daily_diff < -threshold
        spike_up_mask = pd.DataFrame(False, index=cleaned.index, columns=cleaned.columns)
    elif mode == "symmetric":
        spike_mask = daily_diff < -threshold
        spike_up_mask = daily_diff > threshold
    elif mode == "reversion_gated":
        large_move = daily_diff.abs() > threshold
        extreme_prev = prev.abs() > extreme_threshold
        collapse = cleaned.abs() < (prev.abs() * reversion_ratio)
        is_rollover = large_move & extreme_prev & collapse
        # Split into down (prev > 0 → drop) and up (prev < 0 → jump) for
        # reporting symmetry, but both are treated the same way.
        spike_mask = is_rollover & (prev > 0)
        spike_up_mask = is_rollover & (prev < 0)

    # -- Pattern 2: gradual pre-earnings drops (down-only, all modes) --------
    if earnings_timeline is not None:
        common_cols = [c for c in rev.columns if c in earnings_timeline.columns]
        common_dates = rev.index.intersection(earnings_timeline.index)
        earn_aligned = earnings_timeline.reindex(
            index=rev.index, columns=rev.columns, fill_value=0
        )
        pre_earn_window = earn_aligned.copy()
        for shift in range(1, 6):
            pre_earn_window = pre_earn_window | earn_aligned.shift(-shift, fill_value=0)
        cum_3d = daily_diff.rolling(3, min_periods=2).sum()
        neg_streak = (daily_diff < -3).astype(float).rolling(3, min_periods=2).sum()
        gradual_mask = (cum_3d < -threshold) & (neg_streak >= 2) & (pre_earn_window > 0)
        n_earn_clean = int(gradual_mask.sum().sum())
        print(f"[RevisionClean] 실적발표일 기반 보정: {n_earn_clean}개 구간 마스킹")
    else:
        cum_3d = daily_diff.rolling(3, min_periods=2).sum()
        neg_streak = (daily_diff < -3).astype(float).rolling(3, min_periods=2).sum()
        is_earnings = np.array(cleaned.index.month.isin([1, 2, 4, 5, 7, 8, 10, 11]))
        earnings_bcast = pd.DataFrame(
            np.tile(is_earnings.reshape(-1, 1), (1, cleaned.shape[1])),
            index=cleaned.index, columns=cleaned.columns,
        )
        gradual_mask = (cum_3d < -threshold) & (neg_streak >= 2) & earnings_bcast

    combined = spike_mask | spike_up_mask | gradual_mask
    cleaned[combined] = np.nan
    cleaned = cleaned.ffill()

    n_down = int(spike_mask.sum().sum())
    n_up = int(spike_up_mask.sum().sum())
    n_grad = int((gradual_mask & ~(spike_mask | spike_up_mask)).sum().sum())
    print(
        f"[RevisionClean] mode={mode} threshold={threshold} "
        f"down-spikes={n_down} up-spikes={n_up} gradual-down={n_grad}"
    )
    return cleaned


def get_cleaned_revision(
    data,
    sheet_name: str,
    config=None,
    earnings_timeline=None,
) -> "pd.DataFrame | None":
    """Return the revision sheet after config-driven spike cleaning.

    Single source of truth for ALL revision data the model consumes —
    feature panel (sellside, conditioning) AND post-process overlays
    (PEAD boost, growth tilt). Before this helper existed, three
    different call sites used three different hardcoded cleaner configs,
    so post-process overlays were feeding raw rollover artifacts back
    into the model's effective prediction.

    Parameters
    ----------
    data : UniverseData
        The loaded universe. The sheet is fetched via `data.get_sheet`.
    sheet_name : str
        e.g. "Factset_EPS_Revision" or "Factset_Sales_Revision".
    config : PipelineConfig, optional
        Supplies `revision_clean_mode`, `revision_clean_threshold`, etc.
        If None, falls back to DEFAULT_CONFIG.
    earnings_timeline : pd.DataFrame, optional
        Usually None — the iter15 calendar-fallback branch is retained
        by default for baseline reproducibility. Callers that want the
        precise timeline-based pattern-2 mask can pass this explicitly.

    Returns
    -------
    cleaned DataFrame, or None if the sheet is missing.
    """
    from src.config import DEFAULT_CONFIG
    cfg = config if config is not None else DEFAULT_CONFIG
    try:
        raw = data.get_sheet(sheet_name)
    except KeyError:
        return None
    return clean_revision_spikes(
        raw,
        threshold=float(getattr(cfg, "revision_clean_threshold", 15.0)),
        earnings_timeline=earnings_timeline,
        mode=getattr(cfg, "revision_clean_mode", "down_only"),
        extreme_threshold=float(getattr(cfg, "revision_clean_extreme_threshold", 50.0)),
        reversion_ratio=float(getattr(cfg, "revision_clean_reversion_ratio", 0.5)),
    )


def build_bounded_revision_features(
    rev: pd.DataFrame, prefix: str
) -> Dict[str, pd.DataFrame]:
    """Bounded 지표(-100~100)에 특화된 Revision feature 생성.

    문제: 단순 diff 사용 시, 100 유지(diff=0) → CS z-score 중립/하위
         반면 50→80(diff=+30) → z-score 상위. 100 유지가 더 좋은데 저평가.

    해결:
      1. 기존 diff는 정제된 데이터에서 계산 (스파이크 제거)
      2. level_persist: rolling mean / 100 → 100 유지 시 +1.0 (GBT interaction)
      3. time_at_extreme: 최근 N일 중 극단값 비율 (persistence 포착)
      4. level_dir: level × 변화방향 복합 (상단 유지 = 강한 양수)
    """
    features: Dict[str, pd.DataFrame] = {}

    # 2026-04-21 (Phase 2.4 final): revision MA reverted to 63d single window
    # after window-sweep experiments (10d / 21d / dual) all failed to beat
    # baseline_v2 (IR 1.024) on the overall gate:
    #   10d: IR 1.010 (P3 best +1.984 but P1 collapses to +0.902)
    #   21d: IR 0.906 (non-monotonic, worst of sweep)
    #   dual: IR 0.966 (noisy midpoint, not max — LightGBM can't do regime selection)
    # What's kept from Phase 2.4: the shared get_cleaned_revision helper so
    # the feature panel AND post-process overlays use the same cleaned
    # (reversion_gated) revision stream. Window tuning as a P2-fix route is
    # a dead end — P2 needs signal-layer work (multi-horizon target,
    # regime-aware PCA, macro cross features).

    # ── 기존 feature (정제 데이터 기반) ──
    features[f"{prefix}"] = rev
    features[f"{prefix}_diff_5d"] = rev - rev.shift(5)
    features[f"{prefix}_diff_21d"] = rev - rev.shift(21)
    features[f"{prefix}_diff_63d"] = rev - rev.shift(63)
    features[f"{prefix}_ma_63d"] = rev.rolling(63, min_periods=21).mean()
    features[f"{prefix}_accel"] = (rev - rev.shift(5)) - (rev - rev.shift(21))
    features[f"{prefix}_rank"] = cs_rank(rev)
    features[f"{prefix}_rel_strength"] = cs_rank(rev - rev.shift(21))
    features[f"{prefix}_vol"] = rev.rolling(63, min_periods=21).std()
    features[f"{prefix}_trend"] = (
        rev.rolling(21, min_periods=10).mean()
        - rev.rolling(63, min_periods=21).mean()
    )
    features[f"{prefix}_momentum"] = cs_rank(rev - rev.shift(5))
    features[f"{prefix}_vs_median"] = rev - rev.rolling(252, min_periods=126).median()

    # ── NEW: Bounded 보정 feature ──

    # 1. Level persistence: 63d rolling mean normalised to [0, 1].
    features[f"{prefix}_level_persist_63d"] = (
        rev.rolling(63, min_periods=21).mean() / 100
    )

    # 2. Time at extreme: 극단 유지 비율 (GBT가 "100유지" 패턴 학습)
    features[f"{prefix}_time_high"] = (
        (rev > 70).astype(float).rolling(21, min_periods=5).mean()
    )
    features[f"{prefix}_time_low"] = (
        (rev < -70).astype(float).rolling(21, min_periods=5).mean()
    )

    # 3. Level-direction composite: 레벨 × 변화방향
    #    100유지(dir=0) → 1.0, 100↑(dir=+1) → 1.3, 50↑(dir=+1) → 0.65
    diff_5d = rev - rev.shift(5)
    direction = np.sign(diff_5d)
    features[f"{prefix}_level_dir"] = (rev / 100) * (1 + direction * 0.3)

    # ── NEW: Ceiling-adjusted features (100 유지 종목 저평가 해결) ──

    # 4. Ceiling-adjusted diff: 남은 여유 공간 대비 변화율
    #    level=95에서 +3 → 3/(100-95)=0.6 (상한 근접에서의 개선은 인상적)
    #    level=50에서 +3 → 3/(100-50)=0.06 (중립)
    #    level=100에서  0 → 0/5=0.0 (변화 없음 → 다른 피처에서 높은 수준 반영)
    shifted_5 = rev.shift(5)
    room_5 = (100 - shifted_5.clip(lower=0)).clip(lower=5)
    features[f"{prefix}_adj_diff_5d"] = (rev - shifted_5) / room_5

    shifted_21 = rev.shift(21)
    room_21 = (100 - shifted_21.clip(lower=0)).clip(lower=5)
    features[f"{prefix}_adj_diff_21d"] = (rev - shifted_21) / room_21

    # 5. Stability-at-level score: 높은 수준의 안정적 유지를 보상
    #    level/100 × (1 - normalized_vol) → 100 유지 + 변동 없음 = 최고 점수
    #    diff=0이어도 level이 높고 변동이 적으면 최상위 z-score 획득
    rev_vol_21 = rev.rolling(21, min_periods=5).std().clip(lower=0.01)
    norm_vol = (rev_vol_21 / 50).clip(upper=1)  # 50 ≈ bounded 변수의 이론적 최대 std
    features[f"{prefix}_stability_score"] = (rev / 100) * (1 - norm_vol)

    # 6. Quality composite: 수준 × 추세방향 복합
    #    높은 수준 유지 or 상승 = 최고, 높은 수준 하락 = 경고
    #    낮은 수준 상승 = 양호, 낮은 수준 하락 = 최하
    avg_21 = rev.rolling(21, min_periods=5).mean() / 100
    trend_dir = np.sign(rev - rev.shift(10))  # 10일 방향
    features[f"{prefix}_quality_composite"] = avg_21 * (1 + trend_dir * 0.2)

    return features


def build_sellside_features(data: UniverseData, config=None) -> Dict[str, pd.DataFrame]:
    """Build the sellside / sentiment feature block.

    `config` threads through PipelineConfig so revision cleaning can switch
    between `down_only` (baseline), `symmetric`, and `reversion_gated` modes
    without modifying feature-engine callers. When `config=None` the
    baseline `down_only` + threshold=15 behaviour is preserved exactly.
    """
    from src.config import DEFAULT_CONFIG
    if config is None:
        config = DEFAULT_CONFIG

    clean_mode = getattr(config, "revision_clean_mode", "down_only")
    clean_thr = float(getattr(config, "revision_clean_threshold", 15.0))
    extreme_thr = float(getattr(config, "revision_clean_extreme_threshold", 50.0))
    reversion_ratio = float(getattr(config, "revision_clean_reversion_ratio", 0.5))

    features: Dict[str, pd.DataFrame] = {}

    # --- Analyst Recommendation (~8) ---
    # EQY_REC_CONS: 1~5 scale, 5=Strong Buy (best)
    try:
        rec = data.get_sheet("EQY_REC_CONS")
        features["analyst_rec_level"] = rec
        features["analyst_rec_diff_5d"] = rec - rec.shift(5)
        features["analyst_rec_diff_21d"] = rec - rec.shift(21)
        features["analyst_rec_diff_63d"] = rec - rec.shift(63)
        features["analyst_rec_accel"] = (rec - rec.shift(21)) - (rec - rec.shift(63))
        features["analyst_rec_rank"] = cs_rank(rec)
        features["analyst_rec_stability"] = rec.rolling(63, min_periods=21).std()
        med = rec.rolling(252, min_periods=126).median()
        features["analyst_rec_vs_median"] = rec - med
    except KeyError:
        logger.debug("sellside: EQY_REC_CONS missing — skipping analyst_rec_* features")

    # --- Target Price (~10) ---
    try:
        tg = data.get_sheet("Factset_TG_Price")
        # Vendor target prices are quoted in each listing's local currency.
        # UniverseData.prices is USD-normalized for return/momentum features,
        # so target-price upside must retain the matching local price unit.
        px = getattr(data, "local_prices", data.prices).replace(0, np.nan)
        upside = (tg / px) - 1
        features["tg_upside"] = upside
        features["tg_upside_diff_5d"] = upside - upside.shift(5)
        features["tg_upside_diff_21d"] = upside - upside.shift(21)
        features["tg_upside_rank"] = cs_rank(upside)
        features["tg_upside_z"] = cross_sectional_zscore(upside)
        med = upside.rolling(126, min_periods=63).median()
        features["tg_upside_vs_median"] = upside - med
        features["tg_upside_vol"] = upside.rolling(63, min_periods=21).std()
        features["tg_mom_21d"] = safe_pct_change(tg, 21)
        features["tg_mom_63d"] = safe_pct_change(tg, 63)
        # REDESIGN O-a (2026-04-14): tg_price의 1년 변화율 — tg_upside 대체.
        # 절대 upside 수준보다 TG 자체의 추세가 sellside 방향성을 더 잘 포착.
        features["tg_mom_252d"] = safe_pct_change(tg, 252)
        features["tg_conviction"] = safe_pct_change(tg, 21).rolling(63, min_periods=21).std()
    except KeyError:
        logger.debug("sellside: Factset_TG_Price missing — skipping tg_* features")

    # --- EPS / Sales Revision (~34, spike-cleaned + bounded-adjusted) ---
    # REDESIGN U (2026-04-14): earnings_timeline left at None keeps the iter9
    # calendar-fallback mask. The cleaner-mode switch (Phase 2.4, 2026-04-21)
    # is config-driven via get_cleaned_revision below.
    # The same helper is reused from backtest post-process (PEAD, growth_tilt)
    # and conditioning.py, so every downstream consumer sees identical
    # cleaned data rather than three separate ad-hoc cleanings.
    eps_rev_cleaned = get_cleaned_revision(data, "Factset_EPS_Revision", config=config)
    if eps_rev_cleaned is not None:
        features.update(build_bounded_revision_features(eps_rev_cleaned, "eps_rev"))
    else:
        logger.debug("sellside: Factset_EPS_Revision missing — skipping eps_rev_* features")

    sales_rev_cleaned = get_cleaned_revision(data, "Factset_Sales_Revision", config=config)
    if sales_rev_cleaned is not None:
        features.update(build_bounded_revision_features(sales_rev_cleaned, "sales_rev"))
    else:
        logger.debug("sellside: Factset_Sales_Revision missing — skipping sales_rev_* features")

    # --- Cross-revision (~4, reuse cleaned data) ---
    if eps_rev_cleaned is not None and sales_rev_cleaned is not None:
        features["rev_divergence"] = eps_rev_cleaned - sales_rev_cleaned
        features["rev_rank_divergence"] = cs_rank(eps_rev_cleaned) - cs_rank(sales_rev_cleaned)
        features["rev_combined"] = (eps_rev_cleaned + sales_rev_cleaned) / 2
        features["rev_breadth"] = ((eps_rev_cleaned > 0) & (sales_rev_cleaned > 0)).astype(float)

    # --- News Sentiment (~10) ---
    try:
        news = data.get_sheet("NEWS_SENTIMENT_DAILY_AVG")
        features["news_raw"] = news
        features["news_ma5"] = news.rolling(5, min_periods=1).mean()
        features["news_ma21"] = news.rolling(21, min_periods=5).mean()
        features["news_ma63"] = news.rolling(63, min_periods=21).mean()
        features["news_trend"] = news.rolling(5, min_periods=1).mean() - news.rolling(21, min_periods=5).mean()
        features["news_accel"] = (news - news.shift(5)) - (news - news.shift(21))
        features["news_vol"] = news.rolling(21, min_periods=10).std()
        features["news_surprise"] = news - news.rolling(21, min_periods=5).mean()
        features["news_rank"] = cs_rank(news)
        rng = news.rolling(126, min_periods=63)
        features["news_range_pos"] = (news - rng.min()) / (rng.max() - rng.min()).replace(0, np.nan)
    except KeyError:
        logger.debug("sellside: NEWS_SENTIMENT_DAILY_AVG missing — skipping news_* features")

    # --- Sent Trend (~5) ---
    try:
        sent_mom = data.get_sheet("Sent_Trend_Momentum_Timeseries")
        features["sent_momentum"] = sent_mom
        features["sent_momentum_diff"] = sent_mom - sent_mom.shift(21)
        features["sent_momentum_rank"] = cs_rank(sent_mom)
    except KeyError:
        logger.debug("sellside: Sent_Trend_Momentum_Timeseries missing — skipping sent_momentum_* features")
    try:
        sent_21 = data.get_sheet("Sent_Trend_21d_Timeseries")
        features["sent_21d"] = sent_21
        features["sent_21d_accel"] = sent_21 - sent_21.shift(21)
    except KeyError:
        logger.debug("sellside: Sent_Trend_21d_Timeseries missing — skipping sent_21d_* features")

    return features
