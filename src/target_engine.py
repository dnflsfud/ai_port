"""
Phase 3: 타겟 변수 생성
20일 Specific Return = PCA 잔차 수익률

각 시점 t에서:
1. 과거 252일 일간 수익률로 PCA fitting (n_components=5)
2. t~t+20 영업일 forward cumulative return
3. PCA common component 제거
4. 잔차 = Specific Return = 타겟

look-ahead bias 방지: PCA fitting은 반드시 과거 데이터만 사용.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from src.config import PipelineConfig, DEFAULT_CONFIG
from src.data_loader import UniverseData, TICKERS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backwards-compatible module-level aliases (read from DEFAULT_CONFIG)
# ---------------------------------------------------------------------------
PCA_COMPONENTS = DEFAULT_CONFIG.pca_components
PCA_N_REMOVE = DEFAULT_CONFIG.pca_n_remove       # 제거할 PCA 성분 수
PCA_LOOKBACK = DEFAULT_CONFIG.pca_lookback
FORWARD_HORIZON = DEFAULT_CONFIG.forward_horizon


def compute_forward_returns(returns: pd.DataFrame, horizon: int = FORWARD_HORIZON) -> pd.DataFrame:
    """t~t+horizon 영업일 forward cumulative return 계산 (vectorized)."""
    cum = (1 + returns).cumprod()
    # forward return at t = cum[t+horizon] / cum[t] - 1
    fwd = cum.shift(-horizon) / cum - 1
    # 마지막 horizon일은 NaN (미래 데이터 없음)
    return fwd


def compute_specific_returns(
    returns: pd.DataFrame,
    n_components: int = PCA_COMPONENTS,
    n_remove: int = None,
    lookback: int = PCA_LOOKBACK,
    horizon: int = FORWARD_HORIZON,
    config: PipelineConfig = None,
) -> pd.DataFrame:
    """
    각 시점에서 PCA 잔차 기반 20일 Specific Return 계산.

    Args:
        n_components: PCA fitting에 사용할 성분 수 (기본 5)
        n_remove: 제거할 성분 수 (기본 None=n_components, 즉 전부 제거)
                  2로 설정하면 PC1(시장)+PC2(대형/소형)만 제거,
                  PC3~PC5(섹터 로테이션 등)는 specific return에 포함됨.
        config: PipelineConfig (overrides module-level defaults when provided)

    Returns:
        specific_ret: DataFrame (dates x tickers), 값 = 20일 specific return
    """
    config = config or DEFAULT_CONFIG
    if n_components == PCA_COMPONENTS:
        n_components = config.pca_components
    if lookback == PCA_LOOKBACK:
        lookback = config.pca_lookback
    if horizon == FORWARD_HORIZON:
        horizon = config.forward_horizon
    if n_remove is None:
        n_remove = n_components  # 기본: 전부 제거 (기존 동작)

    dates = returns.index
    tickers = returns.columns
    n_dates = len(dates)

    # Forward cumulative returns
    fwd_ret = compute_forward_returns(returns, horizon)

    specific_ret = pd.DataFrame(np.nan, index=dates, columns=tickers)

    # C7: track why specific returns are NaN for diagnostic reporting
    n_attempted = 0
    n_skipped_sparse = 0
    n_pca_failed = 0
    n_skipped_fwd_nan = 0

    for t in range(lookback, n_dates - horizon):
        n_attempted += 1
        # 과거 lookback일 일간 수익률로 PCA fitting
        hist_returns = returns.iloc[t - lookback: t].copy()

        # 결측치가 너무 많으면 스킵
        valid_mask = hist_returns.notna().all(axis=1)
        hist_clean = hist_returns.loc[valid_mask]

        if len(hist_clean) < lookback // 2:
            n_skipped_sparse += 1
            continue

        # PCA fitting
        try:
            actual_n = min(n_components, len(tickers) - 1)
            pca = PCA(n_components=actual_n)
            pca.fit(hist_clean.values)
        except (ValueError, np.linalg.LinAlgError) as e:
            n_pca_failed += 1
            logger.warning("[TargetEngine] PCA failed at t=%d (%s): %s", t, dates[t].strftime('%Y-%m-%d'), e)
            continue

        # 시점 t의 forward return
        fwd_t = fwd_ret.iloc[t].values.reshape(1, -1)

        if np.any(np.isnan(fwd_t)):
            n_skipped_fwd_nan += 1
            continue

        # Common component 계산
        factors = pca.transform(fwd_t)  # shape (1, actual_n)

        if n_remove < actual_n:
            # Partial PCA: PC1~PC(n_remove)만 제거, 나머지는 유지
            factors_partial = factors.copy()
            factors_partial[:, n_remove:] = 0  # n_remove 이후 성분은 0으로
            common = pca.inverse_transform(factors_partial)
        else:
            # 전체 제거 (기존 동작)
            common = pca.inverse_transform(factors)

        # Specific return = forward return - common component
        spec = fwd_t - common
        specific_ret.iloc[t] = spec.flatten()

    # C7: emit a summary so silent failures surface in the log
    if n_attempted > 0:
        fail_frac = n_pca_failed / n_attempted
        level_fn = logger.warning if fail_frac > 0.01 else logger.info
        level_fn(
            "[TargetEngine] Summary: attempted=%d, sparse_skip=%d, pca_fail=%d (%.2f%%), fwd_nan_skip=%d",
            n_attempted, n_skipped_sparse, n_pca_failed, fail_frac * 100, n_skipped_fwd_nan,
        )

    return specific_ret


def compute_specific_returns_regime_weighted(
    returns: pd.DataFrame,
    vix_series: pd.Series,
    n_components: int,
    n_remove: int,
    lookback: int,
    horizon: int,
    config: PipelineConfig,
) -> pd.DataFrame:
    """Regime-aware (weighted) PCA variant of compute_specific_returns.

    The core idea is:

        - At each target date t we determine a regime label (0 = normal,
          1 = stress) from the VIX z-score.
        - Historical observations in the same regime as t get weight 1.0,
          observations in the *other* regime get `regime_pca_offreg_weight`
          (default 0.3).
        - A weighted mean / weighted covariance is computed from these
          weights, and the eigendecomposition of the weighted covariance
          matrix gives the regime-conditional principal components.
        - Everything after that (project forward return, subtract first
          n_remove components) is identical to the unweighted branch.

    This way, during the rate-shock regime (P2), the PCA captures the
    specific covariance structure that dominated P2 rather than averaging
    it with the bull-market regime that dominates the rest of the window.

    Falls back to unweighted PCA (straight numpy.linalg.eigh) if the
    same-regime sample count is below `regime_pca_min_effective_n`.
    """
    thr = float(getattr(config, "regime_pca_vix_threshold", 0.5))
    offreg_w = float(getattr(config, "regime_pca_offreg_weight", 0.3))
    min_eff = int(getattr(config, "regime_pca_min_effective_n", 30))

    dates = returns.index
    tickers = returns.columns
    n_dates = len(dates)

    # Align VIX to return index, ffill short gaps.
    vix_aligned = vix_series.reindex(dates).ffill()
    # 63d rolling z-score for regime classification.
    vix_rm = vix_aligned.rolling(63, min_periods=21).mean()
    vix_rs = vix_aligned.rolling(63, min_periods=21).std().replace(0, np.nan)
    vix_z = (vix_aligned - vix_rm) / vix_rs
    # Binary regime label. NaN early periods fall into "normal" bucket (0).
    regime = (vix_z > thr).astype(float).fillna(0.0)

    fwd_ret = compute_forward_returns(returns, horizon)
    specific_ret = pd.DataFrame(np.nan, index=dates, columns=tickers)

    n_attempted = n_pca_failed = n_skipped = n_fallback_unweighted = 0
    n_stress_days = 0

    for t in range(lookback, n_dates - horizon):
        n_attempted += 1

        hist = returns.iloc[t - lookback: t]
        valid_mask = hist.notna().all(axis=1)
        hist_clean = hist.loc[valid_mask]
        if len(hist_clean) < lookback // 2:
            n_skipped += 1
            continue

        fwd_t = fwd_ret.iloc[t].values
        if np.any(np.isnan(fwd_t)):
            n_skipped += 1
            continue

        cur_reg = float(regime.iloc[t])
        hist_reg = regime.iloc[t - lookback: t].loc[valid_mask].values
        weights = np.where(hist_reg == cur_reg, 1.0, offreg_w)

        # Guard: if effective same-regime count is too low, fall back to equal
        # weights so the PCA matches the unweighted baseline.
        n_same = int((hist_reg == cur_reg).sum())
        if n_same < min_eff:
            weights = np.ones(len(hist_clean))
            n_fallback_unweighted += 1
        else:
            if cur_reg == 1.0:
                n_stress_days += 1

        W = float(weights.sum())
        X = hist_clean.values  # (lookback, K)
        w_col = weights.reshape(-1, 1)

        mean_w = (X * w_col).sum(axis=0) / W
        centered = X - mean_w

        # Weighted covariance: (X-μ)^T diag(w) (X-μ) / W
        cov_w = (centered.T * weights) @ centered / W

        try:
            # symmetric eigendecomposition, ascending eigenvalues
            eigvals, eigvecs = np.linalg.eigh(cov_w)
        except np.linalg.LinAlgError as e:
            n_pca_failed += 1
            logger.warning(
                "[TargetEngine-Regime] eigh failed at t=%d (%s): %s",
                t, dates[t].strftime('%Y-%m-%d'), e,
            )
            continue

        # sort descending and take top n_components
        order = np.argsort(eigvals)[::-1]
        V = eigvecs[:, order]
        actual_n = min(n_components, V.shape[1], len(tickers) - 1)
        V = V[:, :actual_n]

        centered_fwd = fwd_t - mean_w
        factors = centered_fwd @ V        # (actual_n,)
        factors_partial = factors.copy()
        # Remove the first n_remove components only (partial PCA semantics).
        if n_remove < actual_n:
            factors_partial[n_remove:] = 0
            common = factors_partial @ V.T
        else:
            common = factors @ V.T

        specific_ret.iloc[t] = centered_fwd - common

    level_fn = logger.warning if n_pca_failed > 0 else logger.info
    level_fn(
        "[TargetEngine-Regime] attempted=%d, skipped=%d, pca_fail=%d, "
        "fallback_unweighted=%d, stress_weighted_days=%d "
        "(thr=%.2f, offreg_w=%.2f, min_eff=%d)",
        n_attempted, n_skipped, n_pca_failed, n_fallback_unweighted, n_stress_days,
        thr, offreg_w, min_eff,
    )
    return specific_ret


def build_targets(data: UniverseData, n_remove: int = None, config: PipelineConfig = None) -> pd.DataFrame:
    """
    UniverseData에서 타겟 변수(20일 Specific Return) 생성.

    Args:
        data: UniverseData
        n_remove: 제거할 PCA 성분 수 (None=기본값 PCA_N_REMOVE)
        config: PipelineConfig (overrides module-level defaults when provided)

    Returns:
        targets: DataFrame (dates x tickers)
    """
    config = config or DEFAULT_CONFIG
    if n_remove is None:
        n_remove = config.pca_n_remove
    # Restrict to the authoritative universe (data.tickers). If some sheets
    # dropped tickers, Daily_Returns may still contain extra columns that
    # assembly/backtest will never use — we skip them here to keep the
    # PCA / specific-return pipeline shape-aligned with the rest.
    returns = data.returns.loc[:, list(data.tickers)]

    # P2 infrastructure hook: multi-horizon target blend (OFF by default).
    mh_enabled = getattr(config, "multi_horizon_targets_enabled", False)
    mh_weights = getattr(config, "multi_horizon_weights", None) or {}

    # Phase 2 (2026-04-22): regime-weighted PCA (OFF by default). When enabled,
    # the PCA is fit with sample weights that emphasise observations matching
    # the target date's VIX regime, so the specific-return residual captures
    # the correct common factor in rate-shock windows. Requires Factor_PX_LAST
    # to have a VIX column; silently falls through to standard PCA if missing.
    regime_w_enabled = getattr(config, "regime_pca_weighted_enabled", False)

    if mh_enabled and mh_weights:
        targets = _build_multi_horizon_targets(returns, n_remove, mh_weights, config)
    elif regime_w_enabled and data.has_factor_data() and \
            "VIX" in data.factor_prices.columns:
        logger.info("TargetEngine: regime-weighted PCA ENABLED (VIX-driven).")
        targets = compute_specific_returns_regime_weighted(
            returns=returns,
            vix_series=data.factor_prices["VIX"],
            n_components=config.pca_components,
            n_remove=n_remove,
            lookback=config.pca_lookback,
            horizon=config.forward_horizon,
            config=config,
        )
    else:
        if regime_w_enabled:
            logger.warning(
                "TargetEngine: regime_pca_weighted_enabled=True but VIX series "
                "missing — falling back to unweighted PCA."
            )
        targets = compute_specific_returns(returns, n_remove=n_remove, config=config)

    valid_count = targets.notna().sum().sum()
    total = targets.size
    remove_label = f"n_remove={n_remove}" if n_remove < config.pca_components else "Full"
    if mh_enabled and mh_weights:
        remove_label += f" multi-horizon={dict(mh_weights)}"
    logger.info("TargetEngine: 타겟 생성 완료 (PCA Residual, %s)", remove_label)
    logger.info(
        "TargetEngine: 기간 %s ~ %s, 유효 관측치 %d / %d (%.1f%%)",
        targets.index[0].strftime('%Y-%m-%d'),
        targets.index[-1].strftime('%Y-%m-%d'),
        valid_count, total, valid_count / total * 100,
    )

    return targets


def _build_multi_horizon_targets(
    returns: pd.DataFrame,
    n_remove: int,
    weights_map: "dict[int, float]",
    config: PipelineConfig,
) -> pd.DataFrame:
    """Weighted blend of specific returns over multiple forward horizons.

    Each horizon is normalised to annualised scale (div by sqrt(H/252))
    before blending so the blended target stays in a comparable numeric
    range to the single-horizon baseline. Disabled by default via
    config.multi_horizon_targets_enabled.
    """
    import copy as _copy
    blended = None
    total_w = sum(float(v) for v in weights_map.values())
    if total_w <= 0:
        raise ValueError(f"multi_horizon_weights sum must be > 0, got {total_w}")

    for h, w in weights_map.items():
        h = int(h)
        w = float(w) / total_w
        sub_cfg = _copy.copy(config)
        sub_cfg.forward_horizon = h
        t_h = compute_specific_returns(returns, n_remove=n_remove, config=sub_cfg)
        # Annualisation scale: horizon H returns ~ sqrt(H/252) × annual vol
        scale = np.sqrt(252.0 / max(h, 1))
        piece = t_h * scale * w
        blended = piece if blended is None else blended.add(piece, fill_value=np.nan)

    logger.info(
        "TargetEngine: multi-horizon blend horizons=%s weights=%s",
        list(weights_map.keys()),
        list(weights_map.values()),
    )
    return blended
