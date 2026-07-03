"""
Phase 7: Advanced Attribution System (Pictet 논문 방식)

1. Shapley Value Decomposition
   - 피처별 포트폴리오 노출도 (SHAP × weight → portfolio contribution)
   - 종목별/피처그룹별 수익 기여도 분해

2. Li et al. 3-Component Marginal Prediction Attribution
   - Linear: OLS fit on marginal prediction curve
   - Marginal Non-linear: Marginal - Linear
   - Interaction: Full prediction - sum of all marginals
   → 선형/비선형/상호작용 3분류

3. Market Explainer
   - 모델 피처 + 파생 매크로 피처로 특정 기간 시장 성과 설명
"""

import pandas as pd
import numpy as np
import shap
import lightgbm as lgb
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional

from src.config import PipelineConfig, DEFAULT_CONFIG


# ============================================================
# 1. Shapley Value Decomposition
# ============================================================

def compute_shap_values(
    model: lgb.LGBMRegressor,
    X: np.ndarray,
    feature_names: List[str],
) -> np.ndarray:
    """SHAP TreeExplainer로 feature-level SHAP values 계산."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values


def portfolio_shap_decomposition(
    shap_values: np.ndarray,
    feature_names: List[str],
    feature_groups: Dict[str, List[str]],
    tickers: List[str],
    weights: np.ndarray,
) -> Dict:
    """포트폴리오 비중 가중 SHAP 분해."""
    weighted_shap = shap_values * weights.reshape(-1, 1)

    feature_contrib = {}
    for j, fname in enumerate(feature_names):
        feature_contrib[fname] = weighted_shap[:, j].sum()

    group_contrib = {}
    for group_name, group_features in feature_groups.items():
        indices = [j for j, f in enumerate(feature_names) if f in group_features]
        if indices:
            group_contrib[group_name] = weighted_shap[:, indices].sum()
        else:
            group_contrib[group_name] = 0.0

    ticker_contrib = {}
    for i, ticker in enumerate(tickers):
        ticker_contrib[ticker] = {}
        for group_name, group_features in feature_groups.items():
            indices = [j for j, f in enumerate(feature_names) if f in group_features]
            if indices:
                ticker_contrib[ticker][group_name] = weighted_shap[i, indices].sum()
            else:
                ticker_contrib[ticker][group_name] = 0.0

    return {
        "feature_contrib": feature_contrib,
        "group_contrib": group_contrib,
        "ticker_contrib": ticker_contrib,
    }


def compute_feature_exposure(
    panel: pd.DataFrame,
    feature_names: List[str],
    feature_groups: Dict[str, List[str]],
    tickers: List[str],
    weights: np.ndarray,
    date: pd.Timestamp,
) -> Dict[str, float]:
    """포트폴리오의 피처 그룹별 노출도 (Active Exposure)."""
    mask = panel.index.get_level_values("date") == date
    X = panel.loc[mask, feature_names]
    if len(X) == 0:
        return {}

    n = len(tickers)
    bm_weights = np.ones(n) / n
    active_weights = weights - bm_weights

    exposures = {}
    for group_name, group_features in feature_groups.items():
        indices = [j for j, f in enumerate(feature_names) if f in group_features]
        if indices:
            group_vals = X.iloc[:, indices].values
            exposure = (active_weights @ group_vals).mean()
            exposures[group_name] = float(exposure)
        else:
            exposures[group_name] = 0.0

    return exposures


# ============================================================
# 2. Li et al. 3-Component Marginal Prediction Attribution
# ============================================================

# ---------------------------------------------------------------------------
# Backwards-compatible module-level alias (read from DEFAULT_CONFIG)
# ---------------------------------------------------------------------------
N_GRID_POINTS = DEFAULT_CONFIG.n_grid_points  # marginal prediction 계산용 grid point 수


def _compute_marginal_prediction(
    model: lgb.LGBMRegressor,
    X: np.ndarray,
    feature_idx: int,
    n_grid: int = N_GRID_POINTS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    단일 피처의 marginal prediction 곡선 계산.

    Φ_marginal(z_ℓ) = (1/N) Σ_i Φ(z_ℓ, z_{-ℓ,i})

    i.e., feature ℓ을 고정하고 나머지 피처는 data에서 샘플링하여 평균.

    Args:
        model: 학습된 모델
        X: 데이터 행렬 (n_samples, n_features)
        feature_idx: 분석할 피처 인덱스
        n_grid: grid point 수

    Returns:
        grid_values: 피처 값 grid (n_grid,)
        marginal_pred: 각 grid point에서의 marginal prediction (n_grid,)
    """
    feature_values = X[:, feature_idx]
    finite_values = feature_values[np.isfinite(feature_values)]
    if finite_values.size == 0:
        return np.array([0.0]), np.array([0.0])
    lo, hi = np.percentile(finite_values, [2, 98])

    if lo == hi:
        lo = feature_values.min()
        hi = feature_values.max()
    if lo == hi:
        return np.array([lo]), np.array([0.0])

    grid_values = np.linspace(lo, hi, n_grid)

    # M6: 샘플 수 제한 (속도). Default (200) preserves backward-compat; larger
    # universes can lift the cap via config.marginal_prediction_n_samples.
    cap = getattr(DEFAULT_CONFIG, "marginal_prediction_n_samples", 200)
    n_samples = min(len(X), int(cap))
    if n_samples < len(X):
        _rng = np.random.RandomState(42)
        idx = _rng.choice(len(X), n_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    marginal_pred = np.zeros(n_grid)
    for g, val in enumerate(grid_values):
        X_modified = X_sample.copy()
        X_modified[:, feature_idx] = val
        preds = model.predict(X_modified)
        marginal_pred[g] = preds.mean()

    return grid_values, marginal_pred


def _decompose_marginal(
    grid_values: np.ndarray,
    marginal_pred: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Marginal prediction 곡선을 Linear / Marginal Non-linear로 분해.

    Linear: OLS fit (slope × z)
    Marginal Non-linear: marginal - linear

    Returns:
        var_linear: Linear component의 분산
        var_marginal_nl: Marginal Non-linear component의 분산
        slope: OLS 기울기
    """
    if len(grid_values) < 3:
        return 0.0, 0.0, 0.0

    # Center
    z = grid_values - grid_values.mean()
    y = marginal_pred - marginal_pred.mean()

    # OLS: y = slope * z
    z_sq = (z ** 2).sum()
    if z_sq < 1e-10:
        return 0.0, np.var(y), 0.0

    slope = (z * y).sum() / z_sq
    linear_fit = slope * z
    nonlinear_residual = y - linear_fit

    var_linear = np.var(linear_fit)
    var_marginal_nl = np.var(nonlinear_residual)

    return var_linear, var_marginal_nl, slope


def li_three_component_attribution(
    model: lgb.LGBMRegressor,
    X: np.ndarray,
    feature_names: List[str],
    feature_groups: Dict[str, List[str]],
    n_grid: int = N_GRID_POINTS,
    max_features: int = 50,
) -> Dict:
    """
    Li et al. 3-component marginal prediction attribution.

    1. 전체 예측의 분산 = Total variance
    2. 각 피처별 marginal prediction 계산 → Linear/Marginal NL 분해
    3. Interaction = Total - Σ(marginal variances)

    Returns:
        dict with:
        - linear_ratio: 전체 중 Linear 비율
        - marginal_nl_ratio: 전체 중 Marginal Non-linear 비율
        - interaction_ratio: 전체 중 Interaction 비율
        - group_linear: {group: linear 비율}
        - group_marginal_nl: {group: marginal NL 비율}
        - group_interaction: {group: interaction 비율}
        - feature_slopes: {feature: OLS slope}
    """
    # 전체 예측
    y_hat = model.predict(X)
    total_var = np.var(y_hat)

    if total_var < 1e-10:
        return {
            "linear_ratio": np.nan, "marginal_nl_ratio": np.nan, "interaction_ratio": np.nan,
            "group_linear": {g: 0.0 for g in feature_groups},
            "group_marginal_nl": {g: 0.0 for g in feature_groups},
            "group_interaction": {g: 0.0 for g in feature_groups},
            "feature_slopes": {},
        }

    # 피처 중요도 기반으로 상위 피처만 분석 (속도)
    importances = model.feature_importances_
    top_indices = np.argsort(-importances)[:max_features]

    # 각 피처별 marginal prediction 분석
    feature_linear_var = np.zeros(len(feature_names))
    feature_marginal_nl_var = np.zeros(len(feature_names))
    feature_total_marginal_var = np.zeros(len(feature_names))
    feature_slopes = {}

    for idx in top_indices:
        grid_vals, marg_pred = _compute_marginal_prediction(
            model, X, idx, n_grid
        )
        var_lin, var_nl, slope = _decompose_marginal(grid_vals, marg_pred)

        feature_linear_var[idx] = var_lin
        feature_marginal_nl_var[idx] = var_nl
        feature_total_marginal_var[idx] = var_lin + var_nl
        feature_slopes[feature_names[idx]] = slope

    # 전체 분해
    total_linear = feature_linear_var.sum()
    total_marginal_nl = feature_marginal_nl_var.sum()
    total_marginal = total_linear + total_marginal_nl
    total_interaction = max(0, total_var - total_marginal)

    # 비율 (total_var 기준)
    linear_ratio = total_linear / total_var if total_var > 0 else 0
    marginal_nl_ratio = total_marginal_nl / total_var if total_var > 0 else 0
    interaction_ratio = total_interaction / total_var if total_var > 0 else 0

    # 정규화 (합 = 1)
    total_ratio = linear_ratio + marginal_nl_ratio + interaction_ratio
    if total_ratio > 0:
        linear_ratio /= total_ratio
        marginal_nl_ratio /= total_ratio
        interaction_ratio /= total_ratio

    # 그룹별 분해
    group_linear = {}
    group_marginal_nl = {}
    group_interaction = {}

    for group_name, group_features in feature_groups.items():
        indices = [j for j, f in enumerate(feature_names) if f in group_features]
        if not indices:
            group_linear[group_name] = 0.0
            group_marginal_nl[group_name] = 0.0
            group_interaction[group_name] = 0.0
            continue

        g_linear = feature_linear_var[indices].sum()
        g_nl = feature_marginal_nl_var[indices].sum()
        g_total_marginal = g_linear + g_nl

        if total_var > 0:
            group_linear[group_name] = g_linear / total_var
            group_marginal_nl[group_name] = g_nl / total_var
            # Interaction은 각 그룹의 marginal 비중에 비례하여 배분
            if total_marginal > 0:
                group_interaction[group_name] = total_interaction / total_var * (g_total_marginal / total_marginal)
            else:
                group_interaction[group_name] = total_interaction / total_var / len(feature_groups)
        else:
            group_linear[group_name] = 0.0
            group_marginal_nl[group_name] = 0.0
            group_interaction[group_name] = 0.0

    return {
        "linear_ratio": linear_ratio,
        "marginal_nl_ratio": marginal_nl_ratio,
        "interaction_ratio": interaction_ratio,
        "nonlinear_ratio": marginal_nl_ratio + interaction_ratio,  # 호환성
        "group_linear": group_linear,
        "group_marginal_nl": group_marginal_nl,
        "group_interaction": group_interaction,
        "feature_slopes": feature_slopes,
    }


# ============================================================
# 3. Market Explainer
# ============================================================

def build_macro_features(returns: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """매크로/시장 구조 피처 생성."""
    macro = pd.DataFrame(index=returns.index)
    ew_ret = returns.mean(axis=1)

    for w in [5, 21, 63, 126]:
        macro[f"mkt_ret_{w}d"] = ew_ret.rolling(w, min_periods=w).sum()
        macro[f"mkt_vol_{w}d"] = ew_ret.rolling(w, min_periods=w).std() * np.sqrt(252)

    macro["cs_dispersion_21d"] = returns.rolling(21, min_periods=21).mean().std(axis=1)
    macro["cs_dispersion_63d"] = returns.rolling(63, min_periods=63).mean().std(axis=1)

    ma50 = prices.rolling(50, min_periods=50).mean()
    macro["breadth_50d"] = (prices > ma50).sum(axis=1) / prices.shape[1]

    for w in [63, 126]:
        # 메모리 효율적 avg correlation: 월별 샘플링
        avg_corrs = []
        sample_dates = returns.index[w::21]  # 21일마다 샘플
        for date in sample_dates:
            loc = returns.index.get_loc(date)
            if loc < w:
                continue
            window_ret = returns.iloc[loc-w:loc].dropna(axis=1, how='all')
            if window_ret.shape[1] < 3:
                continue
            corr_mat = window_ret.corr().values
            mask = np.ones(corr_mat.shape, dtype=bool)
            np.fill_diagonal(mask, False)
            finite_vals = corr_mat[mask]
            finite_vals = finite_vals[np.isfinite(finite_vals)]
            if len(finite_vals) > 0:
                avg_corrs.append((date, finite_vals.mean()))
        if avg_corrs:
            corr_series = pd.Series(dict(avg_corrs))
            # 샘플 간 보간
            macro[f"avg_corr_{w}d"] = corr_series.reindex(returns.index).interpolate(method='linear')

    asset_light = [
        "MSFT", "GOOGL", "META", "PLTR", "CRM", "NFLX",
        "V", "MA", "SPGI", "BLK", "GS",
    ]
    asset_heavy = [
        "NVDA", "AVGO", "MU", "AMD",
        "GEV", "VRT", "BE", "LITE", "000660", "005930",
        "XOM", "LNG", "FCX", "LIN", "NEE",
        "CAT", "HON", "DE", "UNP", "ETN", "LMT",
    ]

    al_cols = [c for c in asset_light if c in returns.columns]
    ah_cols = [c for c in asset_heavy if c in returns.columns]

    if al_cols and ah_cols:
        al_ret = returns[al_cols].mean(axis=1)
        ah_ret = returns[ah_cols].mean(axis=1)
        rotation = ah_ret - al_ret
        for w in [21, 63, 126]:
            macro[f"rotation_ah_al_{w}d"] = rotation.rolling(w, min_periods=w).sum()
        macro["rotation_regime"] = macro.get("rotation_ah_al_63d", pd.Series(0, index=returns.index))

    macro["vol_regime_21d"] = macro.get("mkt_vol_21d", pd.Series(0, index=returns.index))
    macro["vol_regime_change"] = macro.get("mkt_vol_21d", pd.Series(0, index=returns.index)) - \
                                  macro.get("mkt_vol_63d", pd.Series(0, index=returns.index))
    macro["mom_regime_short"] = macro.get("mkt_ret_21d", pd.Series(0, index=returns.index))
    macro["mom_regime_long"] = macro.get("mkt_ret_126d", pd.Series(0, index=returns.index))

    macro = macro.fillna(0)
    return macro


def explain_period(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    start_date: str,
    end_date: str,
    feature_groups_hint: Optional[Dict] = None,
) -> Dict:
    """Market Explainer: 특정 기간의 시장 성과를 매크로 피처로 설명."""
    macro = build_macro_features(returns, prices)

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    period_mask = (returns.index >= start) & (returns.index <= end)
    period_returns = returns.loc[period_mask]

    if len(period_returns) == 0:
        return {"error": "No data in specified period"}

    cum_ret = (1 + period_returns).prod() - 1
    period_macro = macro.loc[period_mask].mean()

    asset_light = [
        "MSFT", "GOOGL", "META", "PLTR", "CRM", "NFLX",
        "V", "MA", "SPGI", "BLK", "GS",
    ]
    asset_heavy = [
        "NVDA", "AVGO", "MU", "AMD",
        "GEV", "VRT", "BE", "LITE", "000660", "005930",
        "XOM", "LNG", "FCX", "LIN", "NEE",
        "CAT", "HON", "DE", "UNP", "ETN", "LMT",
    ]
    mixed = [
        "AAPL", "AMZN", "TSLA",
        "UNH", "LLY", "ISRG", "ABBV", "REGN",
        "JPM", "COST", "HD", "PG", "WMT", "MCD",
        "AMT", "EQIX", "TMUS", "PLD",
    ]

    al_cols = [c for c in asset_light if c in cum_ret.index]
    ah_cols = [c for c in asset_heavy if c in cum_ret.index]
    mx_cols = [c for c in mixed if c in cum_ret.index]

    al_avg = cum_ret[al_cols].mean() if al_cols else 0
    ah_avg = cum_ret[ah_cols].mean() if ah_cols else 0
    mx_avg = cum_ret[mx_cols].mean() if mx_cols else 0

    rotation_spread = ah_avg - al_avg

    if rotation_spread > 0.05:
        rotation_desc = "Asset-Heavy Outperformance (HW/Energy/Infra > Software/Platform)"
    elif rotation_spread < -0.05:
        rotation_desc = "Asset-Light Outperformance (Software/Platform > HW/Energy/Infra)"
    else:
        rotation_desc = "Neutral (No clear rotation)"

    avg_vol = period_macro.get("mkt_vol_21d", 0)
    if avg_vol > 0.25:
        vol_regime = "High Volatility"
    elif avg_vol > 0.15:
        vol_regime = "Normal Volatility"
    else:
        vol_regime = "Low Volatility"

    mkt_ret = period_macro.get("mkt_ret_21d", 0)
    if mkt_ret > 0.02:
        mkt_direction = "Bullish"
    elif mkt_ret < -0.02:
        mkt_direction = "Bearish"
    else:
        mkt_direction = "Sideways"

    period_macro_daily = macro.loc[period_mask]
    ew_daily = returns.loc[period_mask].mean(axis=1)

    top_drivers = {}
    if len(period_macro_daily) > 5:
        X_macro = period_macro_daily.values
        y_mkt = ew_daily.values
        valid = ~np.isnan(y_mkt) & np.all(~np.isnan(X_macro), axis=1)
        if valid.sum() > 5:
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_macro[valid], y_mkt[valid])
            importance = np.abs(ridge.coef_)
            for idx in np.argsort(-importance)[:10]:
                fname = period_macro_daily.columns[idx]
                top_drivers[fname] = {
                    "coefficient": float(ridge.coef_[idx]),
                    "importance": float(importance[idx]),
                    "mean_value": float(period_macro_daily.iloc[:, idx].mean()),
                }

    dispersion = period_macro.get("cs_dispersion_21d", 0)
    breadth = period_macro.get("breadth_50d", 0)

    return {
        "period": f"{start_date} ~ {end_date}",
        "trading_days": len(period_returns),
        "period_returns": cum_ret.sort_values(ascending=False).to_dict(),
        "group_returns": {
            "Asset-Light (SW/Platform)": float(al_avg),
            "Asset-Heavy (HW/Energy/Infra)": float(ah_avg),
            "Mixed (AAPL/AMZN/TSLA)": float(mx_avg),
        },
        "rotation_analysis": {
            "spread (Heavy - Light)": float(rotation_spread),
            "description": rotation_desc,
        },
        "regime": {
            "market_direction": mkt_direction,
            "volatility": vol_regime,
            "dispersion": float(dispersion),
            "breadth_50d": float(breadth),
        },
        "top_drivers": top_drivers,
    }


# ============================================================
# Legacy / Convenience
# ============================================================

def feature_group_attribution(
    shap_values: np.ndarray,
    feature_names: List[str],
    feature_groups: Dict[str, List[str]],
) -> Dict[str, float]:
    """Feature group별 평균 절대 SHAP 기여도."""
    abs_shap = np.abs(shap_values).mean(axis=0)
    total = abs_shap.sum()

    if total == 0:
        return {g: 0.0 for g in feature_groups}

    group_contrib = {}
    for group_name, group_features in feature_groups.items():
        indices = [i for i, f in enumerate(feature_names) if f in group_features]
        if indices:
            group_contrib[group_name] = abs_shap[indices].sum() / total
        else:
            group_contrib[group_name] = 0.0

    return group_contrib


def compute_feature_importance(
    model: lgb.LGBMRegressor,
    feature_names: List[str],
) -> pd.Series:
    """LightGBM built-in feature importance (gain)."""
    importance = model.feature_importances_
    return pd.Series(importance, index=feature_names, name="importance").sort_values(ascending=False)


# ============================================================
# Main Entry Point
# ============================================================

def run_attribution(
    models: Dict[pd.Timestamp, lgb.LGBMRegressor],
    panel: pd.DataFrame,
    feature_names: List[str],
    feature_groups: Dict[str, List[str]],
    weights_history: Optional[Dict[pd.Timestamp, pd.Series]] = None,
    n_sample_dates: int = 8,
) -> Dict:
    """
    전체 attribution 분석 (Pictet 논문 방식 - Li et al. 3-component).

    Returns:
        dict with keys:
        - group_contributions: {date: {group: ratio}}
        - linear_ratios: {date: (linear, nonlinear)}
        - linear_nonlinear_detail: {date: Li et al. 3-component decomposition}
        - feature_importance: 평균 feature importance
        - portfolio_decomposition: {date: portfolio-weighted SHAP decomposition}
        - shap_values_sample: last model's SHAP data for visualization
    """
    results = {
        "group_contributions": {},
        "linear_ratios": {},
        "linear_nonlinear_detail": {},
        "feature_importance": None,
        "portfolio_decomposition": {},
        "stock_shap_breakdown": {},
        "shap_values_sample": None,
    }

    model_dates = sorted(models.keys())
    if len(model_dates) > n_sample_dates:
        step = len(model_dates) // n_sample_dates
        sample_dates = model_dates[::step][:n_sample_dates]
    else:
        sample_dates = model_dates

    all_importance = []

    for m_date in sample_dates:
        model = models[m_date]
        print(f"[Attribution] 분석 중: {m_date.strftime('%Y-%m-%d')}")

        # Each model may have been trained on a narrower EWMA-pruned feature
        # set than `feature_names`. Respect the subset the model was actually
        # fit on (stored as `_active_features` by walk_forward_train) so the
        # SHAP / LGBM predict call sees the right shape.
        model_features = getattr(model, "_active_features", None) or feature_names
        model_fw = getattr(model, "_active_fw", None)

        mask = panel.index.get_level_values("date") == m_date
        X_df = panel.loc[mask, model_features]
        X = X_df.values
        if model_fw is not None:
            X = X * model_fw[np.newaxis, :]
        tickers = X_df.index.get_level_values("ticker").tolist()

        if len(X) == 0:
            continue

        # --- SHAP values (on the model's own feature subset) ---
        shap_vals = compute_shap_values(model, X, model_features)

        # NOTE: shap_vals has len(model_features) columns, not feature_names.
        # All downstream SHAP-consuming code must index into model_features.
        # --- Group attribution (absolute SHAP) ---
        group_contrib = feature_group_attribution(shap_vals, model_features, feature_groups)
        results["group_contributions"][m_date] = group_contrib

        # --- Per-stock raw SHAP group breakdown (unweighted) ---
        stock_shap = {}
        for i, ticker in enumerate(tickers):
            stock_shap[ticker] = {}
            for group_name, group_features in feature_groups.items():
                indices = [j for j, f in enumerate(model_features) if f in group_features]
                if indices:
                    stock_shap[ticker][group_name] = float(shap_vals[i, indices].sum())
                else:
                    stock_shap[ticker][group_name] = 0.0
            stock_shap[ticker]["total"] = float(shap_vals[i].sum())
        results["stock_shap_breakdown"][m_date] = stock_shap

        # --- Li et al. 3-Component Attribution ---
        try:
            li_detail = li_three_component_attribution(
                model, X, model_features, feature_groups,
            )
            # 호환성: linear_ratios에 (linear, nonlinear) 형태로 저장
            results["linear_ratios"][m_date] = (
                li_detail["linear_ratio"],
                li_detail["nonlinear_ratio"],
            )
            results["linear_nonlinear_detail"][m_date] = li_detail
        except Exception as e:
            print(f"    Li et al. attribution error: {e}")
            # Store NaN (not a magic 0.33/0.67) so the headline averaging in
            # compute_alpha_attribution skips this date instead of blending in
            # a fabricated constant.
            results["linear_ratios"][m_date] = (np.nan, np.nan)

        # --- Portfolio-weighted SHAP decomposition ---
        if weights_history:
            w_dates = sorted(weights_history.keys())
            closest = max([d for d in w_dates if d <= m_date], default=None)
            if closest is not None:
                w_series = weights_history[closest]
                w_array = np.array([w_series.get(t, 1.0/len(tickers)) for t in tickers])
                port_decomp = portfolio_shap_decomposition(
                    shap_vals, model_features, feature_groups, tickers, w_array
                )
                results["portfolio_decomposition"][m_date] = port_decomp

        # --- Feature importance ---
        imp = compute_feature_importance(model, model_features)
        all_importance.append(imp)

    # Average importance across sampled dates.
    # Different retrains may have different active feature sets (EWMA pruning),
    # so we outer-join the per-date importance Series; missing features become
    # 0 before averaging.
    if all_importance:
        imp_df = pd.concat(all_importance, axis=1)
        results["feature_importance"] = imp_df.fillna(0.0).mean(axis=1).sort_values(ascending=False)

    # Save last model's SHAP for visualization (using ITS feature subset)
    if sample_dates:
        last_m_date = sample_dates[-1]
        last_model = models[last_m_date]
        last_features = getattr(last_model, "_active_features", None) or feature_names
        last_fw = getattr(last_model, "_active_fw", None)
        mask = panel.index.get_level_values("date") == last_m_date
        X_last = panel.loc[mask, last_features].values
        if last_fw is not None:
            X_last = X_last * last_fw[np.newaxis, :]
        if len(X_last) > 0:
            results["shap_values_sample"] = {
                "values": compute_shap_values(last_model, X_last, last_features),
                "data": X_last,
                "feature_names": last_features,
            }

    return results
