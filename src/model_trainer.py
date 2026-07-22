"""
Phase 4: LightGBM 모델 학습
- objective: regression (연속값)
- 출력: cross-sectional Z-score -> expected_return 변환
- 훈련: 3년(756일) rolling window
- 재훈련: 3개월(63일)마다
- Validation: 훈련 마지막 6개월(126일)
- EWMA Feature Importance: 재훈련마다 feature importance를 EWMA로 추적,
  시간 감쇠 평균으로 feature selection/weighting에 활용 (selection bias 감소)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import List, Dict, Tuple, Optional

from src.config import PipelineConfig, DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# EWMA Feature Importance Tracker
# ---------------------------------------------------------------------------
class EWMAFeatureTracker:
    """재훈련마다 feature importance를 EWMA로 누적 추적.

    - alpha=0.3: 새 importance 30%, 기존 history 70%
    - cold start: ewma_min_retrains 미만이면 uniform 반환
    - drop_pct: 하위 N% feature를 제거 (최소 min_features 유지)
    - weighting: sqrt(ewma / mean) 로 feature scaling
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.ewma_importance: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None
        self.n_updates: int = 0
        self.history: List[Dict] = []  # [{date, importances}] for export
        self.raw_importance_history: List[np.ndarray] = []

    def init_full_features(self, all_feature_names: List[str]) -> None:
        """전체 feature 목록으로 초기화. walk_forward_train 시작 시 1회 호출."""
        self.feature_names = list(all_feature_names)
        self.ewma_importance = np.ones(len(all_feature_names)) / len(all_feature_names)

    def update(self, model: lgb.LGBMRegressor, active_features: List[str],
               retrain_date: pd.Timestamp) -> None:
        """새 모델의 feature importance로 EWMA 업데이트.

        active_features가 전체 feature_names의 부분집합일 수 있으므로,
        전체 배열에서 해당 위치만 업데이트한다.
        """
        importance_type = getattr(self.config, "ewma_importance_type", "split")
        booster = getattr(model, "booster_", None)
        permutation = getattr(model, "_ewma_permutation_importance", None)
        if importance_type == "permutation" and permutation is not None:
            raw_imp = np.asarray(permutation, dtype=float)
        elif booster is not None:
            raw_imp = booster.feature_importance(
                importance_type=(
                    "gain" if importance_type in ("permutation", "stability")
                    else importance_type
                )
            ).astype(float)
        else:
            raw_imp = model.feature_importances_.astype(float)

        # Normalize to sum=1
        total = raw_imp.sum()
        if total > 0:
            norm_imp = raw_imp / total
        else:
            norm_imp = np.ones(len(raw_imp)) / len(raw_imp)

        # 전체 feature 배열에 매핑
        full_imp = np.zeros(len(self.feature_names))
        for i, fname in enumerate(active_features):
            full_idx = self.feature_names.index(fname)
            full_imp[full_idx] = norm_imp[i]

        self.raw_importance_history.append(full_imp.copy())
        if importance_type == "stability":
            window = int(getattr(self.config, "ewma_stability_window", 4))
            recent = np.vstack(self.raw_importance_history[-window:])
            mean_imp = recent.mean(axis=0)
            std_imp = recent.std(axis=0, ddof=0)
            coefficient_of_variation = std_imp / np.maximum(mean_imp, 1e-12)
            full_imp = mean_imp / (1.0 + coefficient_of_variation)
            stable_total = full_imp.sum()
            if stable_total > 0:
                full_imp = full_imp / stable_total

        alpha = self.config.ewma_alpha
        self.ewma_importance = alpha * full_imp + (1 - alpha) * self.ewma_importance

        # Re-normalize
        total = self.ewma_importance.sum()
        if total > 0:
            self.ewma_importance = self.ewma_importance / total

        self.n_updates += 1
        self.history.append({
            "date": retrain_date,
            "importances": self.ewma_importance.copy(),
        })

    def is_ready(self) -> bool:
        """Cold start 기간이 지났는지 확인."""
        return self.n_updates >= self.config.ewma_min_retrains

    def get_active_features(self, feature_names: List[str]) -> List[str]:
        """EWMA importance 기반으로 활성 feature 목록 반환.

        하위 drop_pct feature를 제거하되 min_features 이상 유지.
        Cold start 기간에는 전체 feature 반환.
        """
        if not self.config.ewma_enabled or not self.is_ready():
            return list(feature_names)

        refresh_interval = int(
            getattr(self.config, "ewma_full_refresh_interval", 0)
        )
        if refresh_interval > 0 and self.n_updates % refresh_interval == 0:
            return list(feature_names)

        n_total = len(feature_names)
        n_drop = int(n_total * self.config.ewma_drop_pct)
        n_keep = max(n_total - n_drop, self.config.ewma_min_features)
        n_keep = min(n_keep, n_total)

        # EWMA importance 기준 상위 n_keep개 선택
        sorted_idx = np.argsort(-self.ewma_importance)[:n_keep]
        active = [feature_names[i] for i in sorted(sorted_idx)]
        return active

    def get_feature_weights(self, feature_names: List[str]) -> Optional[np.ndarray]:
        """EWMA importance 기반 feature scaling weights 반환.

        sqrt(ewma / mean) 으로 scaling → 중요 feature 증폭, 약한 feature 감쇠.
        Cold start 기간에는 None 반환 (uniform).
        """
        if (
            not self.config.ewma_enabled
            or not self.is_ready()
            or not getattr(self.config, "ewma_feature_scaling_enabled", True)
        ):
            return None

        mean_imp = self.ewma_importance.mean()
        if mean_imp <= 0:
            return None

        weights = np.sqrt(self.ewma_importance / mean_imp)
        # Clip to [0.5, 2.0] to prevent extreme scaling
        weights = np.clip(weights, 0.5, 2.0)
        return weights

    def export_history(self) -> pd.DataFrame:
        """EWMA importance 이력을 DataFrame으로 반환."""
        if not self.history or self.feature_names is None:
            return pd.DataFrame()

        rows = []
        for h in self.history:
            row = {"date": h["date"]}
            for i, fname in enumerate(self.feature_names):
                row[fname] = h["importances"][i]
            rows.append(row)
        return pd.DataFrame(rows).set_index("date")

# ---------------------------------------------------------------------------
# Backwards-compatible module-level aliases (read from DEFAULT_CONFIG)
# ---------------------------------------------------------------------------
LGBM_PARAMS = DEFAULT_CONFIG.lgbm_params

# 예측 스무딩: 새 예측 = alpha * new + (1-alpha) * old
# 낮을수록 예측이 천천히 변함 → turnover 감소, 재훈련 상관 증가
PREDICTION_EMA_ALPHA = DEFAULT_CONFIG.prediction_ema_alpha

TRAIN_WINDOW = DEFAULT_CONFIG.train_window       # 3년
RETRAIN_FREQ = DEFAULT_CONFIG.retrain_freq       # 3개월
VAL_WINDOW = DEFAULT_CONFIG.val_window           # 6개월


def _prepare_train_data(
    panel: pd.DataFrame,
    targets: pd.DataFrame,
    feature_names: List[str],
    train_dates: pd.DatetimeIndex,
) -> Tuple[np.ndarray, np.ndarray]:
    """훈련 데이터(X, y) 준비. NaN 행 제거."""
    # panel은 MultiIndex (date, ticker)
    mask = panel.index.get_level_values("date").isin(train_dates)
    X_panel = panel.loc[mask, feature_names]

    # targets를 동일한 MultiIndex로 변환
    target_stacked = targets.stack()
    target_stacked.index.names = ["date", "ticker"]
    y_panel = target_stacked.reindex(X_panel.index)

    # NaN 제거
    valid = y_panel.notna() & X_panel.notna().all(axis=1)
    X = X_panel.loc[valid].values
    y = y_panel.loc[valid].values

    return X, y


def prepare_rank_data(
    panel: pd.DataFrame,
    targets: pd.DataFrame,
    feature_names: List[str],
    dates: pd.DatetimeIndex,
    relevance_levels: int = 10,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Build deterministic date-grouped data for ``LGBMRanker``.

    Each date is one ranking query.  Continuous forward returns are converted
    to integer cross-sectional relevance labels in ``[0, levels - 1]``.  The
    stable date/ticker ordering makes the group vector auditable and keeps
    repeated runs deterministic.
    """
    mask = panel.index.get_level_values("date").isin(dates)
    X_panel = panel.loc[mask, feature_names].sort_index()
    target_stacked = targets.stack()
    target_stacked.index.names = ["date", "ticker"]
    y_panel = target_stacked.reindex(X_panel.index)
    valid = y_panel.notna() & X_panel.notna().all(axis=1)
    X_valid = X_panel.loc[valid]
    y_valid = y_panel.loc[valid]

    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    groups: List[int] = []
    for _date, x_group in X_valid.groupby(level="date", sort=True):
        if len(x_group) < 2:
            continue
        y_group = y_valid.reindex(x_group.index).astype(float)
        # ``method='first'`` is deterministic because X_valid is ticker-sorted.
        ranks = y_group.rank(method="first", ascending=True).to_numpy() - 1.0
        relevance = np.floor(ranks * relevance_levels / len(x_group))
        relevance = np.clip(relevance, 0, relevance_levels - 1).astype(np.int32)
        x_parts.append(x_group.to_numpy())
        y_parts.append(relevance)
        groups.append(int(len(x_group)))

    if not x_parts:
        return (
            np.empty((0, len(feature_names)), dtype=float),
            np.empty((0,), dtype=np.int32),
            [],
        )
    return np.vstack(x_parts), np.concatenate(y_parts), groups


def prepare_symmetric_rank_data(
    panel: pd.DataFrame,
    targets: pd.DataFrame,
    feature_names: List[str],
    dates: pd.DatetimeIndex,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build continuous top/bottom-symmetric labels in ``[-1, 1]``.

    Within each date the best/worst observations are equidistant from zero.
    Unlike NDCG, the lower tail is part of the direct training loss, matching
    long/underweight portfolio construction.
    """
    mask = panel.index.get_level_values("date").isin(dates)
    X_panel = panel.loc[mask, feature_names].sort_index()
    target_stacked = targets.stack()
    target_stacked.index.names = ["date", "ticker"]
    y_panel = target_stacked.reindex(X_panel.index)
    valid = y_panel.notna() & X_panel.notna().all(axis=1)
    X_valid = X_panel.loc[valid]
    y_valid = y_panel.loc[valid]

    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    for _date, x_group in X_valid.groupby(level="date", sort=True):
        if len(x_group) < 2:
            continue
        y_group = y_valid.reindex(x_group.index).astype(float)
        ranks = y_group.rank(method="average", ascending=True).to_numpy()
        symmetric = 2.0 * (ranks - 1.0) / (len(x_group) - 1.0) - 1.0
        x_parts.append(x_group.to_numpy())
        y_parts.append(symmetric.astype(float))

    if not x_parts:
        return (
            np.empty((0, len(feature_names)), dtype=float),
            np.empty((0,), dtype=float),
        )
    return np.vstack(x_parts), np.concatenate(y_parts)


def effective_label_horizon(config) -> int:
    """§S11.8(b): purge/embargo가 커버해야 하는 라벨 실현창(거래일).

    multi-horizon 블렌드 타깃이 켜져 있으면 라벨에 최장 horizon의 미래
    수익률이 섞이므로, causal split은 `forward_horizon`이 아니라 블렌드
    최장 horizon으로 purge해야 한다(아니면 train/val 라벨 중첩 = 누수).
    OFF(기본)면 `forward_horizon` 그대로 — 기존 분할과 동일(파리티).
    """
    horizon = int(config.forward_horizon)
    if getattr(config, "multi_horizon_targets_enabled", False):
        weights = getattr(config, "multi_horizon_weights", None) or {}
        if weights:
            horizon = max(horizon, max(int(h) for h in weights))
    return horizon


def build_walk_forward_split(
    all_dates: pd.DatetimeIndex,
    prediction_idx: int,
    train_window: int,
    val_window: int,
    forward_horizon: int,
) -> dict:
    """Return a purged, embargoed train/validation split for one prediction.

    The last validation label may finish on the prediction date, never after
    it.  The last training label finishes strictly before validation features
    begin, eliminating overlapping realization windows.
    """
    val_end = prediction_idx - forward_horizon + 1
    val_start = val_end - val_window
    train_start = max(0, prediction_idx - train_window)
    train_end = val_start - forward_horizon
    if min(val_start, train_end) < 0 or train_end <= train_start:
        raise ValueError(
            "insufficient history for causal validation split: "
            f"t={prediction_idx}, train=[{train_start},{train_end}), "
            f"val=[{val_start},{val_end}), H={forward_horizon}"
        )

    latest_val_realization_idx = val_end - 1 + forward_horizon
    latest_train_realization_idx = train_end - 1 + forward_horizon
    causal_ok = (
        latest_val_realization_idx <= prediction_idx
        and latest_train_realization_idx < val_start
    )
    return {
        "train_dates": all_dates[train_start:train_end],
        "val_dates": all_dates[val_start:val_end],
        "audit": {
            "prediction_date": str(all_dates[prediction_idx].date()),
            "train_start": str(all_dates[train_start].date()),
            "train_end": str(all_dates[train_end - 1].date()),
            "embargo_start": str(all_dates[train_end].date()),
            "embargo_end": str(all_dates[val_start - 1].date()),
            "validation_start": str(all_dates[val_start].date()),
            "validation_end": str(all_dates[val_end - 1].date()),
            "latest_train_label_realization": str(
                all_dates[latest_train_realization_idx].date()
            ),
            "latest_validation_label_realization": str(
                all_dates[latest_val_realization_idx].date()
            ),
            "embargo_days": int(val_start - train_end),
            "forward_horizon": int(forward_horizon),
            "causal_validation_ok": bool(causal_ok),
        },
    }


def train_model(
    panel: pd.DataFrame,
    targets: pd.DataFrame,
    feature_names: List[str],
    train_dates: pd.DatetimeIndex,
    val_dates: pd.DatetimeIndex,
    config: PipelineConfig = None,
    feature_scale: Optional[np.ndarray] = None,
) -> lgb.LGBMModel:
    """단일 모델 훈련 (early stopping with validation)."""
    config = config or DEFAULT_CONFIG
    objective = getattr(config, "model_objective", "regression")
    if objective == "cross_sectional_rank":
        levels = int(getattr(config, "rank_relevance_levels", 10))
        X_train, y_train, train_groups = prepare_rank_data(
            panel, targets, feature_names, train_dates, levels
        )
        X_val, y_val, val_groups = prepare_rank_data(
            panel, targets, feature_names, val_dates, levels
        )
    elif objective == "symmetric_rank":
        X_train, y_train = prepare_symmetric_rank_data(
            panel, targets, feature_names, train_dates
        )
        X_val, y_val = prepare_symmetric_rank_data(
            panel, targets, feature_names, val_dates
        )
        train_groups = val_groups = None
    else:
        X_train, y_train = _prepare_train_data(panel, targets, feature_names, train_dates)
        X_val, y_val = _prepare_train_data(panel, targets, feature_names, val_dates)
        train_groups = val_groups = None

    if len(X_train) == 0:
        raise ValueError("no valid training observations after filtering")

    # EWMA feature scaling (numpy 레벨 — panel 복사 불필요)
    if feature_scale is not None:
        X_train = X_train * feature_scale[np.newaxis, :]
        if len(X_val) > 0:
            X_val = X_val * feature_scale[np.newaxis, :]

    if objective == "cross_sectional_rank":
        params = dict(config.lgbm_params)
        params["objective"] = "rank_xendcg"
        params["metric"] = "ndcg"
        model = lgb.LGBMRanker(**params)
    elif objective == "symmetric_rank":
        params = dict(config.lgbm_params)
        params["objective"] = getattr(config, "symmetric_rank_loss", "huber")
        params["metric"] = getattr(config, "symmetric_rank_metric", "l2")
        model = lgb.LGBMRegressor(**params)
    else:
        # Keep the canonical construction unchanged on the default path.
        model = lgb.LGBMRegressor(**config.lgbm_params)

    # REDESIGN D: looser early stopping (default 100 rounds, configurable)
    # to avoid degenerate 10-20 tree models when val loss plateaus briefly.
    es_rounds = getattr(config, "early_stopping_rounds", 100)

    if len(X_val) > 0 and objective == "cross_sectional_rank":
        model.fit(
            X_train, y_train,
            group=train_groups,
            eval_set=[(X_val, y_val)],
            eval_group=[val_groups],
            eval_at=tuple(int(k) for k in config.rank_eval_at),
            callbacks=[lgb.early_stopping(es_rounds, verbose=False),
                       lgb.log_evaluation(0)],
        )
    elif objective == "cross_sectional_rank":
        model.fit(X_train, y_train, group=train_groups)
    elif len(X_val) > 0:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(es_rounds, verbose=False),
                       lgb.log_evaluation(0)],
        )
    else:
        model.fit(X_train, y_train)

    if (
        getattr(config, "ewma_importance_type", "split") == "permutation"
        and len(X_val) >= 2
    ):
        try:
            from sklearn.inspection import permutation_importance

            max_samples = int(
                getattr(config, "ewma_permutation_max_samples", 10000)
            )
            if len(X_val) > max_samples:
                rng = np.random.default_rng(
                    int(config.lgbm_params.get("random_state", 42))
                )
                sample_idx = np.sort(
                    rng.choice(len(X_val), size=max_samples, replace=False)
                )
                permutation_x = X_val[sample_idx]
                permutation_y = y_val[sample_idx]
            else:
                permutation_x = X_val
                permutation_y = y_val
            permutation_result = permutation_importance(
                model,
                permutation_x,
                permutation_y,
                scoring="neg_mean_squared_error",
                n_repeats=int(getattr(config, "ewma_permutation_repeats", 3)),
                random_state=int(config.lgbm_params.get("random_state", 42)),
                n_jobs=1,
            )
            model._ewma_permutation_importance = np.maximum(
                permutation_result.importances_mean, 0.0
            )
        except Exception as exc:
            print(
                "[ModelTrainer] permutation importance failed; "
                f"falling back to gain ({exc})"
            )

    return model


MIN_TREES = 10  # 최소 트리 수 (이하면 degenerate로 판단)


def predict_cross_sectional(
    model: lgb.LGBMRegressor,
    panel: pd.DataFrame,
    feature_names: List[str],
    pred_date: pd.Timestamp,
    feature_scale: Optional[np.ndarray] = None,
) -> pd.Series:
    """
    단일 날짜의 cross-sectional 예측.
    raw 예측값을 cross-sectional Z-score로 변환.
    """
    mask = panel.index.get_level_values("date") == pred_date
    X = panel.loc[mask, feature_names]

    if len(X) == 0:
        return pd.Series(dtype=float)

    X_vals = X.values
    if feature_scale is not None:
        X_vals = X_vals * feature_scale[np.newaxis, :]

    raw_pred = model.predict(X_vals)
    tickers = X.index.get_level_values("ticker")

    pred = pd.Series(raw_pred, index=tickers, name="expected_return")

    # Cross-sectional Z-score 변환
    mean = pred.mean()
    std = pred.std()
    if std > 0:
        pred = (pred - mean) / std

    return pred


def apply_prediction_ema(pred: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Frame version of walk_forward_train's in-loop prediction EMA.

    Replicates the recursion at the blending step inside walk_forward_train
    (`blended = alpha * pred + (1 - alpha) * prev_pred` over the ticker
    intersection, where prev_pred is the PREVIOUS prediction date's blended
    row): rows with no values are skipped; tickers absent from the previous
    row keep their raw value. Needed by the CS-DR-Alpha 2-pass path, whose
    precomputed_predictions bypass walk_forward_train — without this the DR
    re-run trades an un-smoothed signal while the baseline trades the EMA'd
    one (not an apples-to-apples comparison).
    """
    out = pred.copy()
    prev: Optional[pd.Series] = None
    for d in out.index:
        cur = out.loc[d].dropna()
        if len(cur) == 0:
            continue
        if prev is not None:
            common = cur.index.intersection(prev.index)
            if len(common) > 0:
                blended = alpha * cur[common] + (1 - alpha) * prev[common]
                cur.loc[common] = blended
                out.loc[d, common] = blended.values
        prev = cur
    return out


def walk_forward_train(
    panel: pd.DataFrame,
    targets: pd.DataFrame,
    feature_names: List[str],
    all_dates: pd.DatetimeIndex,
    train_window: int = TRAIN_WINDOW,
    retrain_freq: int = RETRAIN_FREQ,
    val_window: int = VAL_WINDOW,
    config: PipelineConfig = None,
) -> Tuple[Dict[pd.Timestamp, lgb.LGBMRegressor], pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward 방식으로 모델 학습 및 예측 생성.

    Returns:
        models: {재훈련 시점: 모델}
        predictions: DataFrame (date x ticker) EMA 블렌딩된 예측값
        raw_predictions: DataFrame (date x ticker) 블렌딩 전 순수 모델 예측값 (IC 계산용)
    """
    config = config or DEFAULT_CONFIG
    # When explicit window args differ from module defaults, honour them;
    # otherwise fall back to config values.
    if train_window == TRAIN_WINDOW:
        train_window = config.train_window
    if retrain_freq == RETRAIN_FREQ:
        retrain_freq = config.retrain_freq
    if val_window == VAL_WINDOW:
        val_window = config.val_window
    min_trees = int(getattr(config, "min_model_trees", MIN_TREES))

    # OOS hold-out enforcement: during tuning, prevent predictions beyond
    # the reserved OOS window so accidental peeking can't contaminate the
    # selection-bias accounting. See config.enforce_oos_holdout for rationale.
    oos_cutoff_idx: Optional[int] = None
    if getattr(config, "enforce_oos_holdout", False):
        cutoff_str = getattr(config, "train_cutoff_date", None)
        if cutoff_str:
            cutoff_ts = pd.Timestamp(cutoff_str)
            # Find last index position where date <= cutoff.
            mask_le_cutoff = all_dates <= cutoff_ts
            if mask_le_cutoff.any():
                oos_cutoff_idx = int(np.where(mask_le_cutoff)[0][-1]) + 1
                print(f"[WalkForward] OOS hold-out ACTIVE — predicting only "
                      f"through {cutoff_str} (idx {oos_cutoff_idx}/{len(all_dates)}, "
                      f"reserving {len(all_dates) - oos_cutoff_idx} days)")
            else:
                raise ValueError(
                    f"train_cutoff_date={cutoff_str} is before all_dates start "
                    f"({all_dates[0]}). Cannot enforce OOS hold-out."
                )

    predictions = pd.DataFrame(index=all_dates, columns=targets.columns, dtype=float)
    raw_predictions = pd.DataFrame(index=all_dates, columns=targets.columns, dtype=float)
    models = {}
    current_model = None
    prev_model = None
    # Track the (active_features, active_fw) tuple that each retained model
    # was actually trained on, so that when a degenerate model triggers a
    # "reuse prev_model" fallback we also restore the matching feature set.
    # Without this, EWMA keeps dropping features independently and the next
    # predict() call feeds a narrower X into a wider model -> ValueError
    # ("X has 328 features, LGBMRegressor expects 345").
    current_features: Optional[List[str]] = None
    current_fw: Optional[np.ndarray] = None
    prev_features: Optional[List[str]] = None
    prev_fw: Optional[np.ndarray] = None
    last_train_idx = -retrain_freq  # 첫 루프에서 바로 훈련
    prev_pred = None  # EMA 스무딩용
    total_retrains = 0
    degenerate_retrains = 0
    degenerate_events = []
    split_audit = []
    # S12.3: full-refresh / re-entry evidence (recorded only when the
    # periodic refresh is enabled — keys stay absent on the parity path).
    refresh_interval = int(getattr(config, "ewma_full_refresh_interval", 0) or 0)
    refresh_dates: List[str] = []
    reentry_events: List[Dict] = []
    prev_selected: Optional[set] = None

    # EWMA Feature Importance Tracker
    ewma_tracker = EWMAFeatureTracker(config)
    ewma_tracker.init_full_features(feature_names)
    active_features = list(feature_names)  # 초기: 전체 feature 사용
    active_fw = None  # 초기: uniform

    # OOS hold-out: if cutoff is set, iterate only up to the cutoff index.
    last_idx_exclusive = oos_cutoff_idx if oos_cutoff_idx is not None else len(all_dates)

    for t_idx in range(train_window, last_idx_exclusive):
        t_date = all_dates[t_idx]

        # 재훈련 시점인지 확인
        if t_idx - last_train_idx >= retrain_freq or current_model is None:
            if getattr(config, "causal_validation_enabled", False):
                split = build_walk_forward_split(
                    all_dates=all_dates,
                    prediction_idx=t_idx,
                    train_window=train_window,
                    val_window=val_window,
                    forward_horizon=effective_label_horizon(config),
                )
                train_dates = split["train_dates"]
                val_dates = split["val_dates"]
                split_audit.append(split["audit"])
            else:
                train_start = max(0, t_idx - train_window)
                train_end = t_idx - val_window
                val_start = t_idx - val_window
                val_end = t_idx

                if train_end <= train_start:
                    train_end = t_idx
                    val_start = t_idx
                    val_end = t_idx

                train_dates = all_dates[train_start:train_end]
                val_dates = all_dates[val_start:val_end]

            # EWMA: 이전 재훈련 결과 기반으로 active feature 선택 (candidate)
            candidate_features = ewma_tracker.get_active_features(feature_names)
            n_dropped = len(feature_names) - len(candidate_features)

            # S12.3: classify this selection as a periodic full refresh or a
            # normal EWMA selection; a normal selection regaining a feature
            # absent from the previous normal selection is a re-entry (the
            # mechanism the refresh exists to enable).
            if refresh_interval > 0 and ewma_tracker.is_ready():
                if ewma_tracker.n_updates % refresh_interval == 0:
                    refresh_dates.append(t_date.strftime("%Y-%m-%d"))
                else:
                    selected = set(candidate_features)
                    if prev_selected is not None:
                        reentered = sorted(selected - prev_selected)
                        if reentered:
                            reentry_events.append({
                                "date": t_date.strftime("%Y-%m-%d"),
                                "features": reentered,
                            })
                    prev_selected = selected

            # EWMA: candidate feature 의 weight 벡터 (numpy 레벨 적용)
            fw = ewma_tracker.get_feature_weights(feature_names)
            if fw is not None:
                candidate_fw = np.array(
                    [fw[feature_names.index(f)] for f in candidate_features]
                )
            else:
                candidate_fw = None

            # Snapshot current (feature_set, weights) before swapping models.
            # If the new model turns out to be degenerate we will rewind
            # BOTH the model AND the feature set to this snapshot.
            prev_model = current_model
            prev_features = current_features
            prev_fw = current_fw

            new_model = train_model(panel, targets, candidate_features, train_dates, val_dates,
                                    config=config, feature_scale=candidate_fw)

            # Degenerate model fallback: too few trees means the retrain did
            # not materially learn. Reuse the prior model/features if possible.
            total_retrains += 1
            n_trees = int(getattr(new_model, "n_estimators_", 0) or 0)
            is_degenerate = n_trees < min_trees
            if is_degenerate:
                degenerate_retrains += 1
                degenerate_events.append({
                    "date": t_date.strftime("%Y-%m-%d"),
                    "n_trees": n_trees,
                    "min_model_trees": min_trees,
                    "candidate_features": len(candidate_features),
                    "reused_prev_model": prev_model is not None,
                })
            if is_degenerate and prev_model is not None:
                print(f"[ModelTrainer] WARNING: Degenerate model ({n_trees} trees) "
                      f"-> reuse prev model AND prev feature set ({len(prev_features)} features)")
                current_model = prev_model
                active_features = prev_features
                active_fw = prev_fw
                current_features = prev_features
                current_fw = prev_fw
            else:
                current_model = new_model
                active_features = candidate_features
                active_fw = candidate_fw
                current_features = candidate_features
                current_fw = candidate_fw
                # EWMA: 새 모델의 importance로 tracker 업데이트
                ewma_tracker.update(current_model, active_features, t_date)

            # Attach the active feature set (and weights) to the model so
            # downstream code (attribution.py SHAP, re-prediction) can slice
            # the panel to the exact columns this model was trained on.
            # Without this, any subsequent LGBM predict() with the full panel
            # crashes with "number of features in data != training data".
            try:
                current_model._active_features = list(active_features)
                current_model._active_fw = (
                    None if active_fw is None else np.array(active_fw, copy=True)
                )
            except Exception:
                pass

            models[t_date] = current_model
            last_train_idx = t_idx

            ewma_status = ""
            if ewma_tracker.is_ready():
                ewma_status = f", EWMA: {len(active_features)}/{len(feature_names)} features"
                if n_dropped > 0:
                    ewma_status += f" (-{n_dropped})"

            print(f"[ModelTrainer] 재훈련 @ {t_date.strftime('%Y-%m-%d')} "
                  f"(train: {len(train_dates)}d, val: {len(val_dates)}d, trees: {n_trees}{ewma_status})")

        # 예측: EWMA feature weight를 numpy 레벨에서 적용
        pred = predict_cross_sectional(current_model, panel, active_features, t_date,
                                       feature_scale=active_fw)

        # raw 예측 저장 (EMA 블렌딩 전, IC 계산용) — vectorized
        common_raw = pred.index.intersection(raw_predictions.columns)
        if len(common_raw) > 0:
            raw_predictions.loc[t_date, common_raw] = pred[common_raw].values

        # EMA 스무딩: α=0.5로 완화
        if prev_pred is not None and len(pred) > 0:
            alpha = config.prediction_ema_alpha
            common = pred.index.intersection(prev_pred.index)
            blended = alpha * pred[common] + (1 - alpha) * prev_pred[common]
            pred[common] = blended

        if len(pred) > 0:
            prev_pred = pred.copy()
            # vectorized 저장
            common_pred = pred.index.intersection(predictions.columns)
            if len(common_pred) > 0:
                predictions.loc[t_date, common_pred] = pred[common_pred].values

    valid_count = predictions.notna().sum().sum()
    print(f"[ModelTrainer] 예측 완료: {valid_count}개 관측치")

    if ewma_tracker.is_ready():
        print(f"[ModelTrainer] EWMA Feature Tracker: {ewma_tracker.n_updates} updates, "
              f"final active features: {len(active_features)}/{len(feature_names)}")

    degenerate_rate = (
        degenerate_retrains / total_retrains if total_retrains > 0 else 0.0
    )
    ewma_tracker.model_quality = {
        "source": "walk_forward_train",
        "total_retrains": int(total_retrains),
        "degenerate_retrains": int(degenerate_retrains),
        "degenerate_rate": float(degenerate_rate),
        "min_model_trees": int(min_trees),
        "max_degenerate_model_rate": float(
            getattr(config, "max_degenerate_model_rate", 0.25)
        ),
        "fail_on_degenerate_model_rate": bool(
            getattr(config, "fail_on_degenerate_model_rate", False)
        ),
        "events": degenerate_events,
    }
    if getattr(config, "causal_validation_enabled", False):
        ewma_tracker.model_quality.update({
            "model_objective": getattr(config, "model_objective", "regression"),
            "causal_validation_enabled": True,
            "causal_validation_ok": bool(split_audit) and all(
                row["causal_validation_ok"] for row in split_audit
            ),
            "split_audit": split_audit,
        })
    if refresh_interval > 0:
        ewma_tracker.model_quality["ewma_full_refresh"] = {
            "interval": refresh_interval,
            "refresh_dates": refresh_dates,
            "reentry_events": reentry_events,
            "reentry_feature_count": int(
                len({f for e in reentry_events for f in e["features"]})
            ),
        }
    max_rate = float(getattr(config, "max_degenerate_model_rate", 0.25))
    if degenerate_rate > max_rate:
        msg = (
            f"[ModelTrainer] Degenerate model rate {degenerate_rate:.1%} "
            f"exceeds max_degenerate_model_rate={max_rate:.1%} "
            f"({degenerate_retrains}/{total_retrains})."
        )
        print("WARNING: " + msg)
        if getattr(config, "fail_on_degenerate_model_rate", False):
            raise RuntimeError(msg)

    return models, predictions, raw_predictions, ewma_tracker
