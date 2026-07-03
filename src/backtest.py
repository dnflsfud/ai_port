"""
Phase 5: Walk-Forward 백테스트

SSOT for all knobs is ``src/config.DEFAULT_CONFIG`` (this docstring is a
non-authoritative snapshot). Current production defaults:
- train_window: 1260일(5년)
- retrain_freq: 63일(3개월)
- prediction_horizon: 20일
- rebalance_freq: 21일(월간)
"""

import datetime
import hashlib
import hmac
import inspect
import json
import logging
import os
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from src.config import PipelineConfig, DEFAULT_CONFIG
from src.data_loader import UniverseData, TICKERS, mask_pre_listing
from src.feature_engine import build_all_features
from src.target_engine import build_targets
from src.model_trainer import walk_forward_train, TRAIN_WINDOW
from src.portfolio_optimizer import (
    estimate_covariance,
    optimize_portfolio,
    print_optimizer_config,
    project_capped_weights,
    project_portfolio_weights,
)
from src.utils import annualise_return, compute_performance_metrics, compute_beta
from src.features.utils import cross_sectional_zscore, cs_rank


def apply_value_trap_gate(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    """Post-prediction gate that zeroes scores for CRM-like value-trap profiles.

    Empirically validated pattern (2015-2026, 2924 days):
        cheap  (fin_pe_level_z < -0.5)
      & bad_mom (momentum_252d < -0.5)
      & accel  (oper_margin_accel > +0.5)
      → mean 20d fwd specific return -0.25%, hit rate 47.3%
      → in P3 regime (2023-): -1.99%/20d
      → same cheap+bad_mom WITHOUT accel: +0.82%, hit 55.3%

    So adding the accel leg actively destroys alpha in the value-trap
    regime. This gate multiplies prediction by `vtg_scale` (default 0)
    when all three conditions match, leaving everything else untouched.
    """
    if not getattr(config, "value_trap_gate_enabled", False):
        return predictions

    pe_th = float(getattr(config, "vtg_pe_z_threshold", -0.5))
    mom_th = float(getattr(config, "vtg_momentum_threshold", -0.5))
    ac_th = float(getattr(config, "vtg_accel_threshold", 0.5))
    scale = float(getattr(config, "vtg_scale", 0.0))

    required = {"fin_pe_level_z", "momentum_252d", "oper_margin_accel"}
    missing = required - set(panel.columns)
    if missing:
        print(f"[ValueTrapGate] SKIPPED — missing panel columns: {sorted(missing)}")
        return predictions

    # Unstack to (date × ticker) wide frames aligned to predictions grid
    pe = panel["fin_pe_level_z"].unstack("ticker").reindex(
        index=predictions.index, columns=predictions.columns
    )
    mom = panel["momentum_252d"].unstack("ticker").reindex(
        index=predictions.index, columns=predictions.columns
    )
    accel = panel["oper_margin_accel"].unstack("ticker").reindex(
        index=predictions.index, columns=predictions.columns
    )

    trap_mask = (pe < pe_th) & (mom < mom_th) & (accel > ac_th)
    trap_mask = trap_mask.fillna(False)

    adjusted = predictions.copy()
    # Preserve NaN structure; only modify where both pred and mask defined
    valid = predictions.notna() & trap_mask
    adjusted = adjusted.where(~valid, predictions * scale)

    n_hits = int(valid.values.sum())
    total = int(predictions.notna().values.sum())
    share = 100.0 * n_hits / total if total else 0.0
    print(f"[ValueTrapGate] applied (pe_z<{pe_th}, mom<{mom_th}, accel>{ac_th}, scale={scale}): "
          f"{n_hits}/{total} cells ({share:.2f}%) gated")
    return adjusted


def apply_growth_tilt(
    predictions: pd.DataFrame,
    data: "UniverseData",
    config: PipelineConfig,
) -> pd.DataFrame:
    """REDESIGN iter19 (2026-04-17): Growth/Revision prediction tilt.

    Post-prediction boost that tilts the OW book toward stocks with:
      1. Strong EPS/Sales revision momentum (forward-looking sell-side signal)
      2. Strong fundamental EPS/Sales growth (backward-looking realized growth)

    Mechanism:
      rev_composite   = cs_rank(eps_rev_ma_63d) * 0.5 + cs_rank(sales_rev_ma_63d) * 0.5
      growth_composite = cs_rank(eps_chg_252d) * 0.5 + cs_rank(sales_chg_252d) * 0.5
      tilt_signal     = rev_weight * rev_composite + fund_weight * growth_composite
      adjusted_z      = z + boost_weight * (tilt_signal - 0.5) * 2   # center at 0

    The (tilt_signal - 0.5) * 2 centering ensures:
      - Top-growth/revision stocks: +boost (up to +weight)
      - Bottom-growth/revision stocks: -boost (penalty)
      - Median stocks: no change

    2026-04-21 changes:
      - MA window for revision composite: kept at 63d (long-horizon, stable
        overlay). Dual-MA feature panel exposes a 10d horizon to the model;
        post-process stays at 63d because it's a slow boost layer and
        switching it to 10d earlier tanked P1 IR. Post-process no
        longer operates on a different horizon than the model).
      - Revision data source: raw Factset sheets → `get_cleaned_revision`
        (Phase 2.4 spike cleaning applied here too, so post-process boost
        is consistent with the feature panel rather than operating on
        rollover-artifact-contaminated data).

    Model (56-feature LightGBM) is completely untouched.
    """
    if not getattr(config, "growth_tilt_enabled", False):
        return predictions

    boost_w = float(getattr(config, "growth_tilt_weight", 0.0))
    rev_w = float(getattr(config, "growth_tilt_rev_weight", 0.50))
    fund_w = float(getattr(config, "growth_tilt_fundamental_weight", 0.50))
    eps_skew = float(getattr(config, "growth_tilt_eps_skew", 0.50))
    sales_skew = 1.0 - eps_skew
    # iter19e: revision composite sub-weights
    rev_eps_share = float(getattr(config, "growth_tilt_rev_eps_share", 0.50))
    rev_sales_share = float(getattr(config, "growth_tilt_rev_sales_share", 0.50))
    rev_tg_share = float(getattr(config, "growth_tilt_rev_tg_share", 0.0))
    if boost_w == 0.0:
        return predictions

    pred_idx = predictions.index
    pred_cols = list(predictions.columns)

    # Load raw (non-revision) sheets
    def _get(sheet_name):
        try:
            df = data.get_sheet(sheet_name)
            return df.reindex(index=pred_idx, columns=pred_cols).ffill()
        except KeyError:
            return None

    # Revision sheets: go through Phase 2.4 cleaner (shared with feature panel)
    from src.features.sellside import get_cleaned_revision
    _eps_rev_cleaned = get_cleaned_revision(data, "Factset_EPS_Revision", config=config)
    _sales_rev_cleaned = get_cleaned_revision(data, "Factset_Sales_Revision", config=config)
    eps_rev = (
        _eps_rev_cleaned.reindex(index=pred_idx, columns=pred_cols).ffill()
        if _eps_rev_cleaned is not None else None
    )
    sales_rev = (
        _sales_rev_cleaned.reindex(index=pred_idx, columns=pred_cols).ffill()
        if _sales_rev_cleaned is not None else None
    )

    eps_raw = _get("BEST_EPS")
    sales_raw = _get("BEST_SALES")
    tg_price = _get("Factset_TG_Price")
    px_last = _get("PX_LAST")

    # Revision MA window for the post-process composite — matched to the
    # feature panel exposes DUAL horizons (10d + 63d); this composite uses
    # the long 63d horizon for a stable overlay.
    _REV_MA_WINDOW = 63
    _REV_MA_MIN = 21

    # --- Revision composite (forward-looking) ---
    # iter19e: now includes EPS rev + Sales rev + TG price upside
    # Each component independently contributes by its sub-share weight.
    rev_parts = []   # list of (share, rank_df)
    if eps_rev is not None and rev_eps_share > 0:
        eps_rev_ma = eps_rev.rolling(_REV_MA_WINDOW, min_periods=_REV_MA_MIN).mean()
        rev_parts.append((rev_eps_share, eps_rev_ma.rank(axis=1, pct=True)))
    if sales_rev is not None and rev_sales_share > 0:
        sales_rev_ma = sales_rev.rolling(_REV_MA_WINDOW, min_periods=_REV_MA_MIN).mean()
        rev_parts.append((rev_sales_share, sales_rev_ma.rank(axis=1, pct=True)))
    if tg_price is not None and px_last is not None and rev_tg_share > 0:
        # TG upside = (target price / current price) - 1
        tg_upside = (tg_price / px_last.replace(0, np.nan)) - 1.0
        # Also use 63d momentum of TG to capture revision direction
        tg_mom_63d = tg_price.diff(63) / tg_price.shift(63).abs().replace(0, np.nan)
        tg_combined = (tg_upside.rank(axis=1, pct=True)
                       + tg_mom_63d.rank(axis=1, pct=True)) / 2.0
        rev_parts.append((rev_tg_share, tg_combined))

    if rev_parts:
        total_share = sum(s for s, _ in rev_parts)
        if total_share > 0:
            rev_composite = sum((s / total_share) * r for s, r in rev_parts)
        else:
            rev_composite = pd.DataFrame(0.5, index=pred_idx, columns=pred_cols)
    else:
        rev_composite = pd.DataFrame(0.5, index=pred_idx, columns=pred_cols)

    # --- Fundamental growth composite (backward-looking) ---
    fund_eps_part = None
    fund_sales_part = None
    if eps_raw is not None:
        eps_shifted = eps_raw.shift(252)
        eps_growth = (eps_raw - eps_shifted) / eps_shifted.abs().replace(0, np.nan)
        fund_eps_part = eps_growth.rank(axis=1, pct=True)
    if sales_raw is not None:
        sales_shifted = sales_raw.shift(252)
        sales_growth = (sales_raw - sales_shifted) / sales_shifted.abs().replace(0, np.nan)
        fund_sales_part = sales_growth.rank(axis=1, pct=True)

    if fund_eps_part is not None and fund_sales_part is not None:
        fund_composite = eps_skew * fund_eps_part + sales_skew * fund_sales_part
    elif fund_eps_part is not None:
        fund_composite = fund_eps_part
    elif fund_sales_part is not None:
        fund_composite = fund_sales_part
    else:
        fund_composite = pd.DataFrame(0.5, index=pred_idx, columns=pred_cols)

    # --- Combined tilt signal ---
    tilt_signal = rev_w * rev_composite + fund_w * fund_composite
    # Center: rank 0.5 → no boost, rank 1.0 → +boost, rank 0.0 → -boost
    tilt_centered = (tilt_signal - 0.5) * 2.0  # range: [-1, +1]

    # Apply to predictions (preserve NaN structure)
    adjusted = predictions.copy()
    valid_mask = predictions.notna()
    boost_values = boost_w * tilt_centered
    adjusted[valid_mask] = predictions[valid_mask] + boost_values[valid_mask]

    n_boosted = (boost_values.abs() > 0.01).sum().sum()
    print(f"[GrowthTilt] Applied: weight={boost_w}, "
          f"rev_w={rev_w}, fund_w={fund_w}, eps_skew={eps_skew}, "
          f"rev_shares=(EPS={rev_eps_share}, Sales={rev_sales_share}, TG={rev_tg_share}), "
          f"affected={n_boosted} cells")

    return adjusted


def apply_signal_stability_shrinkage(
    predictions: pd.DataFrame,
    raw_predictions: Optional[pd.DataFrame],
    config: PipelineConfig,
    retrain_freq: int,
) -> pd.DataFrame:
    """OPTIONAL (2026-04-20): shrink today's prediction toward the score
    at the start of the current retrain window.

        z_adj_t = z_t - λ × (z_t - z_t_anchor)
                = (1 - λ) × z_t + λ × z_t_anchor

    where z_t_anchor is the raw prediction from (t - (t mod retrain_freq)).
    Damps the sudden score jumps at retrain boundaries that drive turnover
    spikes without altering the model or the alpha signal's direction.

    OFF when ``config.signal_stability_lambda == 0.0`` (default). Requires
    ``raw_predictions`` to compute the anchor — falls back to predictions
    if the raw panel is unavailable.
    """
    lam = float(getattr(config, "signal_stability_lambda", 0.0))
    if lam <= 0.0:
        return predictions

    src = raw_predictions if raw_predictions is not None else predictions
    # Anchor each row to the most recent retrain-boundary row.
    # Simple implementation: use a rolling forward-fill on a "stamp" series
    # that marks retrain boundaries; for OFF-by-default infra we use a
    # conservative approximation — the value from `retrain_freq` days ago.
    anchor = src.shift(retrain_freq)

    adjusted = predictions.copy()
    mask = predictions.notna() & anchor.notna()
    adjusted[mask] = (1.0 - lam) * predictions[mask] + lam * anchor[mask]
    return adjusted


def apply_pead_boost(
    predictions: pd.DataFrame,
    data: "UniverseData",
    config: PipelineConfig,
) -> pd.DataFrame:
    """REDESIGN U (2026-04-14): Post-Earnings Announcement Drift boost.

    The PEAD anomaly: stocks with positive pre-earnings analyst revisions tend
    to continue outperforming for several weeks after the earnings event.
    This is a well-documented persistent anomaly the model could in principle
    learn, but it requires earnings event timing in the panel — adding which
    has been verified destructive 6 times across iter 4/5/7/12/13/14.

    Workaround: post-process boost that injects PEAD signal AFTER model
    prediction, leaving the panel untouched. Per (date, ticker):

        days_since = trading days since last earnings announcement
        decay      = exp(−days_since / decay_days)  for days_since ≤ max_days
        rev_quality = clip(eps_rev_ma_63d / 100, 0, 1)    # Phase 2.4 cleaned, 63d (stable overlay)
        pead_signal = decay × rev_quality   ∈ [0, 1]
        adjusted_z = z + boost_weight × pead_signal

    No penalty side (PEAD is one-sided positive drift). NaN predictions are
    preserved so the backtest start date is unchanged.

    2026-04-21: rev_quality is a cleaned (reversion_gated) 63d MA. The MA
    window experiment (10d/21d) showed that shortening this overlay
    destabilises P1 IR; kept at 63d for post-process overlay stability while
    the model's feature panel now carries both 10d and 63d horizons.
    """
    if not getattr(config, "pead_boost_enabled", False):
        return predictions

    boost_w = float(getattr(config, "pead_boost_weight", 0.0))
    decay_days = float(getattr(config, "pead_decay_days", 7.0))
    max_days = int(getattr(config, "pead_max_days", 21))
    if boost_w == 0.0:
        return predictions

    earn_tl = getattr(data, "earnings_timeline", None)
    if earn_tl is None:
        return predictions

    pred_idx = predictions.index
    pred_cols = list(predictions.columns)

    # Align earnings timeline to prediction grid
    earn_aligned = (
        earn_tl.reindex(index=pred_idx, columns=pred_cols)
        .fillna(0)
        .astype(int)
    )

    # Vectorized days_since per ticker via searchsorted on event dates
    dates_ts = pred_idx.values.astype("datetime64[ns]")
    days_since = pd.DataFrame(np.nan, index=pred_idx, columns=pred_cols)
    for col in pred_cols:
        if col not in earn_aligned.columns:
            continue
        earn_dates_col = earn_aligned.index[earn_aligned[col] == 1]
        if len(earn_dates_col) == 0:
            continue
        earn_ts = earn_dates_col.values.astype("datetime64[ns]")
        idx_arr = np.searchsorted(earn_ts, dates_ts, side="right") - 1
        valid = idx_arr >= 0
        deltas = np.where(
            valid,
            (dates_ts - earn_ts[np.clip(idx_arr, 0, len(earn_ts) - 1)])
            .astype("timedelta64[D]")
            .astype(float),
            np.nan,
        )
        days_since[col] = deltas

    # Decay only within window: NaN outside, zero after fillna
    in_window = (days_since >= 0) & (days_since <= max_days)
    days_clean = days_since.where(in_window, np.nan)
    decay = np.exp(-days_clean / decay_days).fillna(0.0)

    # Revision quality: 63d MA of cleaned EPS revision, normalised to [0, 1].
    # 2026-04-21 Dual-MA variant: feature panel carries both 10d + 63d, but
    # this post-process overlay stays at 63d (stable, slow boost layer).
    # Still goes through get_cleaned_revision so rollover artifacts are
    # removed consistently with the feature panel.
    from src.features.sellside import get_cleaned_revision
    eps_rev_cleaned = get_cleaned_revision(data, "Factset_EPS_Revision", config=config)
    if eps_rev_cleaned is None:
        return predictions
    eps_rev_ma = eps_rev_cleaned.rolling(63, min_periods=21).mean()
    rev_aligned = eps_rev_ma.reindex(index=pred_idx, columns=pred_cols)
    rev_quality = (rev_aligned / 100.0).clip(lower=0.0, upper=1.0).fillna(0.0)

    pead_signal = decay * rev_quality           # in [0, 1]
    boost_term = (boost_w * pead_signal).fillna(0.0)

    # NaN-preserving add (do NOT pad pre-training dates)
    return predictions + boost_term


def summarize_cached_model_quality(models: Dict, config: PipelineConfig) -> Dict:
    """Best-effort model-quality telemetry when Phase 4 was loaded from cache."""
    min_trees = int(getattr(config, "min_model_trees", 10))
    tree_counts = []
    for model in (models or {}).values():
        n_trees = int(getattr(model, "n_estimators_", 0) or 0)
        tree_counts.append(n_trees)
    total = len(tree_counts)
    degenerate = sum(1 for n in tree_counts if n < min_trees)
    return {
        "source": "cached_models",
        "total_retrains": int(total),
        "degenerate_retrains": int(degenerate),
        "degenerate_rate": float(degenerate / total) if total else 0.0,
        "min_model_trees": int(min_trees),
        "max_degenerate_model_rate": float(
            getattr(config, "max_degenerate_model_rate", 0.25)
        ),
        "fail_on_degenerate_model_rate": bool(
            getattr(config, "fail_on_degenerate_model_rate", False)
        ),
        "tree_count_min": int(min(tree_counts)) if tree_counts else None,
        "tree_count_median": float(np.median(tree_counts)) if tree_counts else None,
    }

# ---------------------------------------------------------------------------
# Backwards-compatible module-level aliases (read from DEFAULT_CONFIG)
# ---------------------------------------------------------------------------
REBALANCE_FREQ = DEFAULT_CONFIG.rebalance_freq  # 격주 리밸런싱
ONE_WAY_TC = DEFAULT_CONFIG.one_way_tc          # 편도 거래비용 10bps (대형주 기준)

# ---------------------------------------------------------------------------
# HMAC key for pickle checkpoint signing
# ---------------------------------------------------------------------------
# Security: pickle.load executes arbitrary code on untrusted input, so we
# HMAC-sign every checkpoint. PICKLE_HMAC_KEY should be set to a secret; the
# fallback exists only for single-user local research and emits a warning.
# PICKLE_HMAC_STRICT=1 requires signature files to exist (recommended).
_PICKLE_KEY_ENV = os.environ.get("PICKLE_HMAC_KEY")
_PICKLE_STRICT = os.environ.get("PICKLE_HMAC_STRICT", "0") == "1"
_DEFAULT_KEY_WARNED = False

if _PICKLE_KEY_ENV:
    _PICKLE_KEY = _PICKLE_KEY_ENV.encode()
else:
    if _PICKLE_STRICT:
        raise RuntimeError(
            "PICKLE_HMAC_STRICT=1 requires PICKLE_HMAC_KEY to be set. "
            "Set a secret in the environment before running."
        )
    _PICKLE_KEY = b"ai_signal_default_key"


def _warn_default_key_once() -> None:
    global _DEFAULT_KEY_WARNED
    if _PICKLE_KEY_ENV is None and not _DEFAULT_KEY_WARNED:
        logger.warning(
            "Checkpoint HMAC: PICKLE_HMAC_KEY env var is not set; "
            "using default key. Do not share checkpoints across machines."
        )
        _DEFAULT_KEY_WARNED = True


def _sign_file(path: Path) -> str:
    """Compute HMAC-SHA256 hex digest for a file."""
    h = hmac.new(_PICKLE_KEY, digestmod=hashlib.sha256)
    h.update(path.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# C-1: Checkpointing (HMAC signed)
# ---------------------------------------------------------------------------
def save_checkpoint(phase: str, data: dict, output_dir: str = "./outputs"):
    """Phase별 체크포인트 저장 (HMAC signed)."""
    _warn_default_key_once()
    cp_dir = Path(output_dir) / "checkpoints"
    cp_dir.mkdir(parents=True, exist_ok=True)
    path = cp_dir / f"checkpoint_{phase}.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)
    # Sign
    sig_path = path.with_suffix(".pkl.sig")
    sig_path.write_text(_sign_file(path))
    print(f"[Checkpoint] {phase} 저장 → {path}")

def load_checkpoint(phase: str, output_dir: str = "./outputs"):
    """Phase 체크포인트 로드 (HMAC verified).

    Strict mode (PICKLE_HMAC_STRICT=1): sig file must exist, signature must match.
    Default mode: sig file missing → WARN and refuse (never silently loads).
    """
    _warn_default_key_once()
    path = Path(output_dir) / "checkpoints" / f"checkpoint_{phase}.pkl"
    if not path.exists():
        return None
    # Verify signature — always required. Deleting the .sig file does NOT bypass.
    sig_path = path.with_suffix(".pkl.sig")
    if not sig_path.exists():
        raise RuntimeError(
            f"Checkpoint signature missing for {phase}: {sig_path}. "
            "Refusing to load unsigned pickle. Re-run the producing phase to regenerate."
        )
    expected = sig_path.read_text().strip()
    actual = _sign_file(path)
    if actual != expected:
        raise RuntimeError(f"Checkpoint signature mismatch for {phase}! File may be tampered.")
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"[Checkpoint] {phase} 로드 ← {path}")
    return data


# ---------------------------------------------------------------------------
# C-3: Structured Progress Logger
# ---------------------------------------------------------------------------
class ProgressLogger:
    """구조화된 진행 로그."""

    def __init__(self, output_dir: str = "./outputs"):
        self.path = Path(output_dir) / "progress.md"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries = []

    def log(self, phase: str, status: str, details: str = ""):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"- Phase {phase}: {status} ({ts}) {details}"
        self.entries.append(entry)
        # 파일에 즉시 기록
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("# Backtest Progress\n\n")
            f.write("\n".join(self.entries))
            f.write("\n")

    def __str__(self):
        return "\n".join(self.entries)


class BacktestResult:
    """백테스트 결과 컨테이너."""

    def __init__(self):
        self.portfolio_weights: Dict[pd.Timestamp, pd.Series] = {}  # 리밸런싱일 목표비중
        self.daily_weights: Dict[pd.Timestamp, pd.Series] = {}      # 매일 drift 반영 비중
        self.portfolio_returns: pd.Series = pd.Series(dtype=float)
        self.benchmark_returns: pd.Series = pd.Series(dtype=float)
        self.spx_returns: pd.Series = pd.Series(dtype=float)
        self.turnover: pd.Series = pd.Series(dtype=float)
        self.predictions: Optional[pd.DataFrame] = None
        self.raw_predictions: Optional[pd.DataFrame] = None
        self.targets: Optional[pd.DataFrame] = None
        self.models: Dict = {}
        self.ic_series: pd.Series = pd.Series(dtype=float)
        self.panel: Optional[pd.DataFrame] = None
        self.feature_names: Optional[List[str]] = None
        self.feature_groups: Optional[Dict] = None
        self.optimizer_failures: int = 0
        self.optimizer_rebalances: int = 0
        self.optimizer_failure_rate: float = 0.0
        self.optimizer_solver_counts: Dict[str, int] = {}
        self.optimizer_solver_solves: int = 0
        self.optimizer_solver_fallbacks: int = 0
        self.optimizer_solver_fallback_rate: float = 0.0
        self.optimizer_fallback_reason_counts: Dict[str, int] = {}
        self.model_quality: Optional[Dict] = None
        self.data_quality: Optional[Dict] = None
        # Factor-neutral (P3) live loading-coverage telemetry. None unless
        # factor_neutral_enabled — then run_backtest attaches per-date impute /
        # inert counts so a thin-coverage (effectively-off) penalty can't be
        # misread as "TE already neutralises style" (§4.3 / review M1).
        self.factor_neutral_telemetry: Optional[Dict] = None
        # Benchmark label used by summary() output. Set by run_backtest from
        # config.benchmark_type so the printed summary matches the actual BM
        # used in calculations. Default reflects production setting.
        self.benchmark_type: str = "cap_weighted"

    @property
    def cumulative_returns(self) -> pd.Series:
        return (1 + self.portfolio_returns).cumprod()

    @property
    def cumulative_benchmark(self) -> pd.Series:
        return (1 + self.benchmark_returns).cumprod()

    @property
    def cumulative_spx(self) -> pd.Series:
        if len(self.spx_returns) > 0:
            return (1 + self.spx_returns).cumprod()
        return pd.Series(dtype=float)

    @property
    def active_returns(self) -> pd.Series:
        return self.portfolio_returns - self.benchmark_returns

    def compute_metrics(self) -> Dict[str, float]:
        """주요 성과 지표 계산 (geometric annualisation, ddof=1).

        Turnover convention (IMPORTANT)
        -------------------------------
        ``self.turnover`` 는 리밸런싱마다 ``sum(|new_w - old_w|)`` (L1 norm)
        으로 기록된다. 이건 **two-way turnover** (매수+매도 합) 이다.

            two_way_L1 = sum(|new - old|)
            one_way    = 0.5 * two_way_L1   # 업계 표준 ("half-turnover")

        본 클래스가 노출하는 ``avg_annual_turnover`` 와 ``annual_tc`` 는
        two-way 기준이다. 벤치마크 논문/리포트의 one-way 수치와 비교할
        때는 **2배 차이**가 난다는 점에 주의할 것.

        거래비용은 two-way * one_way_tc 로 계산되어 있으므로 실제 지출
        비용과는 일치한다 (one-way 거래비용 단가 * 전체 거래 회전율).
        """
        port = self.portfolio_returns.dropna()
        bm = self.benchmark_returns.dropna()

        ann_factor = 252

        # Canonical metrics (geometric return, ddof=1)
        base = compute_performance_metrics(port, bm, ann_factor)

        # Turnover: 리밸런싱당 평균 turnover * 연간 리밸런싱 횟수
        # (self.turnover 는 two-way L1. one-way 비교시 0.5배 할 것)
        if len(self.turnover) > 0:
            n_years = len(port) / ann_factor
            total_turnover = self.turnover.sum()
            avg_turnover_two_way = total_turnover / n_years if n_years > 0 else 0
        else:
            avg_turnover_two_way = 0
        avg_turnover_one_way = 0.5 * avg_turnover_two_way

        # IC
        avg_ic = self.ic_series.mean() if len(self.ic_series) > 0 else 0

        # 연간 거래비용: two-way * one-way-tc 단가 (실제 지출 비용)
        annual_tc = avg_turnover_two_way * ONE_WAY_TC

        result = {
            "annual_return": base.get("annual_return", 0),
            "annual_vol": base.get("annual_vol", 0),
            "sharpe_ratio": base.get("sharpe_ratio", 0),
            "active_return": base.get("active_return", 0),
            "tracking_error": base.get("tracking_error", 0),
            "information_ratio": base.get("information_ratio", 0),
            "max_drawdown": base.get("max_drawdown", 0),
            "avg_annual_turnover": avg_turnover_two_way,       # two-way (L1)
            "avg_annual_turnover_one_way": avg_turnover_one_way,  # industry convention
            "avg_ic": avg_ic,
            "annual_tc": annual_tc,
        }

        # S&P 500 metrics
        if len(self.spx_returns) > 0:
            spx_metrics = compute_performance_metrics(
                self.spx_returns.reindex(port.index, fill_value=0),
                ann_factor=ann_factor,
            )
            result["spx_annual_return"] = spx_metrics.get("annual_return", 0)
            result["spx_annual_vol"] = spx_metrics.get("annual_vol", 0)
            result["spx_sharpe"] = spx_metrics.get("sharpe_ratio", 0)

        # Realized regression beta diagnostic (Pictet beta=1.0 규율 판정용).
        # 가중치/IR/TE에 영향 없는 read-only 진단. portfolio vs benchmark.
        # NOTE (review L1): full-sample OLS beta cov(p,bm)/var(bm) over ALL
        # overlapping rows — NOT a trailing-252d window. realized_active_beta is
        # the algebraic identity realized_beta-1 (same bm in both), not an
        # independent neutrality measure (review GAP5).
        _bm = self.benchmark_returns.reindex(port.index).ffill().fillna(0.0)
        result["realized_beta"] = compute_beta(port, _bm)
        result["realized_active_beta"] = compute_beta(port - _bm, _bm)

        return result

    def summary(self) -> str:
        m = self.compute_metrics()
        # Render BM label from the actual benchmark_type used in run_backtest,
        # not a hardcoded "Equal Weight". Production default is cap_weighted
        # (set in src/config.py:benchmark_type).
        bm_label_map = {
            "cap_weighted": ("Cap-Weighted", "vs cap-weighted BM"),
            "equal_weight": ("Equal Weight", "vs EW"),
        }
        bm_full, vs_label = bm_label_map.get(
            self.benchmark_type, (self.benchmark_type, f"vs {self.benchmark_type}")
        )
        lines = [
            "=" * 50,
            f"백테스트 결과 요약 (BM={bm_full})",
            "=" * 50,
            f"  연간 수익률:      {m['annual_return']:.2%}",
            f"  연간 변동성:      {m['annual_vol']:.2%}",
            f"  Sharpe Ratio:     {m['sharpe_ratio']:.2f}",
            f"  Active Return:    {m['active_return']:.2%}  ({vs_label})",
            f"  Tracking Error:   {m['tracking_error']:.2%}",
            f"  Information Ratio:{m['information_ratio']:.2f}",
            f"  Max Drawdown:     {m['max_drawdown']:.2%}",
            f"  연간 Turnover:    {m['avg_annual_turnover']:.0%}  (two-way L1)",
            f"                    {m.get('avg_annual_turnover_one_way', 0):.0%}  (one-way, 업계 기준)",
            f"  연간 거래비용:    {m['annual_tc']:.2%} (편도 {ONE_WAY_TC*10000:.0f}bps)",
            f"  평균 IC:          {m['avg_ic']:.4f}",
        ]
        if "spx_annual_return" in m:
            lines.extend([
                "",
                f"  S&P 500 연간:     {m['spx_annual_return']:.2%}",
                f"  S&P 500 변동성:   {m['spx_annual_vol']:.2%}",
                f"  S&P 500 Sharpe:   {m['spx_sharpe']:.2f}",
            ])
        lines.append("=" * 50)
        return "\n".join(lines)


def get_sector_map(data: UniverseData) -> Dict[str, str]:
    """종목별 섹터 매핑 추출 (UniverseData.meta 기반, 런타임 권위 소스)."""
    meta = data.meta
    sector_map = {}

    if isinstance(meta, pd.DataFrame):
        if "sector" in meta.columns:
            for ticker in TICKERS:
                if ticker in meta.index:
                    sector_map[ticker] = str(meta.loc[ticker, "sector"])
        elif len(meta.columns) > 0:
            col = meta.columns[0]
            for ticker in TICKERS:
                if ticker in meta.index:
                    sector_map[ticker] = str(meta.loc[ticker, col])

    return sector_map


# Backwards-compatible alias (deprecated -- use get_sector_map instead)
_get_sector_map = get_sector_map


# ---------------------------------------------------------------------------
# Benchmark weight helpers  — REDESIGN A (2026-04)
# ---------------------------------------------------------------------------
# Equal-weight(1/n) was the previous default but it fights the real market
# structure: the universe has NVDA/MSFT/AMZN/META/GOOGL as 30-50% of SPX cap,
# yet EW forces each name to ~2% BM weight. The optimizer then couldn't
# express meaningful UW on mega-caps (active maxed at -2%) and couldn't track
# the real index. Cap-weighted restores the economically meaningful baseline.
# ---------------------------------------------------------------------------


def make_ew_bm_fn(tickers: list):
    """Equal-weight benchmark function: each name = 1/n."""
    n = len(tickers)
    ew = np.ones(n) / n

    def _fn(t_date, tickers_, n_):
        return ew.copy() if n_ == n else np.ones(n_) / n_
    return _fn


def make_capweight_bm_fn(data: UniverseData, tickers: list,
                         config: PipelineConfig = None):
    """Cap-weighted benchmark function using CUR_MKT_CAP.

    Reads ``data.market_cap`` (same index as returns) and returns a closure
    that yields normalised cap weights for each rebalance date, falling back
    to the previous valid row on missing data. If market cap data is absent
    for a date or an entire ticker, that name defaults to uniform.

    When ``config.listing_mask_enabled`` is set, NaN / non-positive caps
    (pre-listing masked names) get ZERO weight instead of a historical-median
    substitution, so a phantom pre-listing name never gets a free BM holding.
    OFF / config omitted preserves the legacy median-fill path bit-for-bit.
    """
    mask_enabled = bool(
        config is not None and getattr(config, "listing_mask_enabled", False)
    )
    mc = data.market_cap  # DataFrame dates x tickers
    # Make sure we have a forward-filled view aligned to tickers
    mc_aligned = mc.reindex(columns=tickers).ffill()
    n = len(tickers)
    uniform = np.ones(n) / n

    def _fn(t_date, tickers_, n_):
        if t_date in mc_aligned.index:
            row = mc_aligned.loc[t_date].values.astype(float)
        else:
            # As-of lookup for dates not in market_cap index
            asof = mc_aligned.index[mc_aligned.index <= t_date]
            if len(asof) == 0:
                return uniform.copy()
            row = mc_aligned.loc[asof[-1]].values.astype(float)

        if not mask_enabled:
            # Replace NaN / non-positive with ticker's historical median
            nan_mask = ~np.isfinite(row) | (row <= 0)
            if nan_mask.any():
                medians = mc_aligned.median(axis=0).values
                row[nan_mask] = medians[nan_mask]
        # Masked names (mask_enabled) skip median-fill → drop to 0 below.
        # If still NaN (no history at all), fall back to uniform
        row = np.where(np.isfinite(row) & (row > 0), row, 0.0)
        s = row.sum()
        if s <= 0:
            return uniform.copy()
        return row / s

    return _fn


def get_benchmark_fn(data: UniverseData, tickers: list,
                     config: PipelineConfig = None) -> Callable:
    """Return the benchmark-weight closure specified by ``config.benchmark_type``.

    Falls back to equal-weight if cap-weighted is requested but the
    underlying market-cap data is unavailable.
    """
    config = config or DEFAULT_CONFIG
    bm_type = getattr(config, "benchmark_type", "equal_weight")
    if bm_type == "cap_weighted":
        try:
            mc = data.market_cap
            if mc is not None and not mc.empty:
                return make_capweight_bm_fn(data, tickers, config=config)
            # C6: cap-weight requested but data is unusable → explicit warning
            # (previously silent EW fallback masked upstream data problems)
            logger.warning(
                "benchmark_type='cap_weighted' but market_cap is %s. "
                "Falling back to equal-weight — INVESTIGATE the data feed before "
                "trusting these results.",
                "None" if mc is None else "empty",
            )
        except Exception as exc:
            logger.warning(
                "cap-weighted BM construction failed (%r); falling back to equal-weight. "
                "Active-return comparisons will be biased.",
                exc,
            )
    return make_ew_bm_fn(tickers)


# ---------------------------------------------------------------------------
# REDESIGN K (2026-04-12): Dynamic confidence-based execution
# ---------------------------------------------------------------------------
# Ported from codex_v2/redesign_pipeline.py. Replaces the static eta=0.5
# execution with an adaptive system that scales trading intensity based on:
#   1. Signal sharpness (top-bottom spread of raw predictions)
#   2. Trailing realized IC (recent model accuracy)
# When confidence is low, alpha is shrunk and turnover cap is tightened.
# ---------------------------------------------------------------------------


def compute_signal_confidence(
    pred_row: pd.Series,
    raw_row: Optional[pd.Series],
    trailing_ic_mean: float,
    spread_scale: float = 0.20,
) -> float:
    """Convert signal sharpness + recent realized IC into a 0-1 confidence.

    Ported from codex_v2 redesign_pipeline._compute_signal_confidence.
    """
    pred_valid = pred_row.dropna()
    if len(pred_valid) < 5:
        return 0.0

    raw_valid = raw_row.dropna() if raw_row is not None and len(raw_row.dropna()) >= 5 else pred_valid

    tail_n = max(3, len(raw_valid) // 10)
    top_mean = float(raw_valid.sort_values(ascending=False).head(tail_n).mean())
    bot_mean = float(raw_valid.sort_values().head(tail_n).mean())
    raw_spread = top_mean - bot_mean

    spread_score = float(np.clip(raw_spread / spread_scale, 0.20, 1.00))
    ic_score = float(np.clip((trailing_ic_mean + 0.01) / 0.04, 0.20, 1.00))
    return float(np.clip(spread_score * ic_score, 0.10, 1.00))


def apply_dynamic_execution(
    prev_weights: np.ndarray,
    target_weights: np.ndarray,
    confidence: float,
    config: "PipelineConfig",
) -> np.ndarray:
    """Build a smoothed candidate book before hard-constraint projection.

    Ported from codex_v2 redesign_pipeline._apply_execution_controls, extended
    with confidence-adaptive eta and no-trade band scaling.

    Dynamic behavior:
      - High confidence (0.8-1.0): eta ≈ 0.7-0.9, full signal
      - Medium confidence (0.4-0.7): eta ≈ 0.3-0.6, partial signal
      - Low confidence (0.1-0.3): eta ≈ 0.1-0.25, mostly hold
    """
    base_eta = getattr(config, "partial_rebalance_eta", 0.50)
    base_ntb = getattr(config, "no_trade_band", 0.003)

    # Scale eta by confidence: eta_effective = base_eta * confidence^0.5
    # (sqrt gives moderate scaling — not too aggressive at low confidence)
    eta = base_eta * (confidence ** 0.5)
    eta = float(np.clip(eta, 0.05, 0.95))

    # Scale no-trade band inversely: wider band when confidence is low.
    ntb = base_ntb / max(confidence, 0.15)
    ntb = float(np.clip(ntb, base_ntb, base_ntb * 6))

    delta = target_weights - prev_weights
    delta[np.abs(delta) < ntb] = 0.0

    candidate = prev_weights + eta * delta
    candidate = np.maximum(candidate, 0.0)
    if not np.all(np.isfinite(candidate)) or candidate.sum() <= 0:
        return prev_weights.copy()
    return candidate


def compute_ic(predictions: pd.Series, realized: pd.Series) -> float:
    """Information Coefficient (rank correlation)."""
    valid = predictions.notna() & realized.notna()
    if valid.sum() < 3:
        return np.nan
    return predictions[valid].corr(realized[valid], method="spearman")


# ---------------------------------------------------------------------------
# Reusable portfolio simulation loop
# ---------------------------------------------------------------------------


def _default_rebal_check(t_idx: int, t_date, start_idx: int, rebalance_freq: int,
                         state: dict) -> bool:
    """Default rebalance check: every *rebalance_freq* trading days or first day."""
    if state.get("first_rebal", True):
        return True
    return (t_idx - start_idx) % rebalance_freq == 0


def _sanitize_daily_ret(daily_ret: np.ndarray) -> np.ndarray:
    """Replace NaN in a daily return vector with 0.

    Extracted from simulate_portfolio so the null-handling rule is visible
    and unit-testable. Returns a new array; does not mutate input.
    """
    if np.any(np.isnan(daily_ret)):
        return np.nan_to_num(daily_ret, nan=0.0)
    return daily_ret


def _drift_weights(weights: np.ndarray, daily_ret: np.ndarray) -> np.ndarray:
    """Apply a single day's return drift to a weight vector and renormalize.

    Matches the original inline logic: w_t+1 = w_t * (1 + r_t) / sum(...).
    If the sum is non-positive (degenerate — every name went to zero), the
    weights are returned unchanged so callers don't divide by zero.
    """
    new = weights * (1.0 + daily_ret)
    s = new.sum()
    if s > 0:
        new = new / s
    return new


def simulate_portfolio(
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
    tickers: list,
    all_dates: pd.DatetimeIndex,
    sector_map: dict = None,
    rebalance_freq: int = 10,
    one_way_tc: float = 0.001,
    optimizer_fn: Callable = None,
    targets: pd.DataFrame = None,
    bm_weights_fn: Callable = None,
    rebal_check_fn: Callable = None,
    weight_drift: bool = True,
    bm_drift: bool = True,
    track_ic: bool = True,
    track_spx: bool = False,
    raw_predictions: pd.DataFrame = None,
    config: "PipelineConfig | None" = None,
    spx_series: pd.Series = None,
    track_daily_weights: bool = True,
    risk_returns: pd.DataFrame = None,
) -> BacktestResult:
    """Reusable portfolio simulation loop.

    This function encapsulates the inner backtest loop that was previously
    duplicated across run_backtest, grid_search.run_optimization_only,
    test_monthly_rebal.run_monthly_backtest, and
    test_te_sensitivity.run_backtest_with_te.

    Args:
        predictions: DataFrame of model predictions (date x ticker).
        returns: DataFrame of daily returns (date x ticker).
        tickers: List of ticker strings.
        all_dates: DatetimeIndex of all trading dates.
        sector_map: Ticker -> sector mapping (optional).
        rebalance_freq: Rebalancing frequency in trading days (default 10).
        one_way_tc: One-way transaction cost (default 10 bps).
        optimizer_fn: Callable(pred_row, hist_returns, prev_weights,
                      sector_map, bm_w) -> np.ndarray of new weights.
                      If None, uses the default optimize_portfolio.
        targets: DataFrame of target values for IC computation (optional).
        bm_weights_fn: Callable(t_date, tickers, n_tickers) -> np.ndarray
                       of benchmark weights. If None, uses equal-weight.
        rebal_check_fn: Callable(t_idx, t_date, start_idx, rebalance_freq,
                        state) -> bool.  If None, uses default fixed-period
                        rebalancing.
        weight_drift: Whether to drift portfolio weights with daily returns
                      (default True).
        bm_drift: Whether to drift benchmark weights between rebalances
                  (default True).
        track_ic: Whether to compute IC at rebalance dates (default True).
        track_spx: Whether to track S&P 500 returns (default False).
        spx_series: S&P 500 daily return series (required if track_spx=True).
        track_daily_weights: Whether to store daily weight snapshots
                             (default True).

    Returns:
        BacktestResult with portfolio_returns, benchmark_returns, turnover,
        portfolio_weights, daily_weights, ic_series, and spx_returns filled.
    """
    config = config or DEFAULT_CONFIG
    n_tickers = len(tickers)
    result = BacktestResult()
    result.benchmark_type = getattr(config, "benchmark_type", "cap_weighted")

    # Default optimizer: standard MVO
    if optimizer_fn is None:
        def optimizer_fn(pred_row, hist_returns, prev_w, s_map, bm_w, diagnostics=None):
            cov_matrix = estimate_covariance(hist_returns, bm_weights=bm_w, config=config)
            if diagnostics is not None:
                diagnostics["cov_matrix"] = cov_matrix
                diagnostics["max_te_annual"] = config.max_te_annual
                diagnostics["sector_deviation"] = config.sector_deviation
            return optimize_portfolio(
                expected_returns=pred_row,
                cov_matrix=cov_matrix,
                prev_weights=prev_w,
                sector_map=s_map if s_map else None,
                bm_weights=bm_w,
                config=config,
                diagnostics=diagnostics,
            )

    try:
        optimizer_accepts_diagnostics = "diagnostics" in inspect.signature(optimizer_fn).parameters
    except (TypeError, ValueError):
        optimizer_accepts_diagnostics = False
    risk_source = risk_returns if risk_returns is not None else returns

    # Default benchmark: equal-weight
    if bm_weights_fn is None:
        def bm_weights_fn(t_date, tickers_, n_):
            return np.ones(n_) / n_

    # Default rebalance check
    if rebal_check_fn is None:
        rebal_check_fn = _default_rebal_check

    port_rets = []
    bm_rets = []
    spx_rets = []
    turnovers = []
    ic_values = []
    weight_history = {}
    weight_history_daily = {}
    optimizer_failures = 0
    solver_counts: Dict[str, int] = {}
    solver_fallbacks = 0
    fallback_reason_counts: Dict[str, int] = {}

    # Find valid prediction start
    pred_valid = predictions.dropna(how="all")
    if len(pred_valid) == 0:
        return result

    start_idx = all_dates.get_loc(pred_valid.index[0])

    # REDESIGN A: initialise portfolio and BM book with the actual benchmark
    # weights at the start date. Previously both were hardcoded to 1/n which
    # produced a spurious "first-day rebalance from EW" even when the target
    # benchmark was cap-weighted. Now they start aligned with the benchmark.
    init_bm = bm_weights_fn(all_dates[start_idx], tickers, n_tickers)
    prev_weights = np.asarray(init_bm, dtype=float).copy()
    bm_curr_weights = np.asarray(init_bm, dtype=float).copy()

    # State dict for rebal_check_fn
    state = {
        "first_rebal": True,
        "_raw_predictions": raw_predictions,
    }

    # =========================================================================
    # Execution timing (look-ahead bias fix)
    # -------------------------------------------------------------------------
    # The previous version rebalanced at the TOP of the loop and then booked
    # today's return using the NEW weights. That is look-ahead: today's close
    # return was already known before the rebalance and should be earned by the
    # weights coming INTO the day, not by the new weights.
    #
    # Corrected order per bar t:
    #   1. Book today's PnL with the weights that entered the day
    #      (prev_weights, possibly drifted from prior days)
    #   2. Apply weight drift over today's return
    #   3. Rebalance at close_t  →  new_weights take effect NEXT bar (t+1)
    #   4. Transaction cost from the rebalance is charged to today's PnL
    #      (trade executed at close_t; cash leaves the book today)
    # =========================================================================
    for t_idx in range(start_idx, len(all_dates)):
        t_date = all_dates[t_idx]

        # --- Step 1: Today's PnL with weights ENTERING the day ----------------
        daily_ret = _sanitize_daily_ret(returns.loc[t_date, tickers].values)

        port_ret = np.dot(prev_weights, daily_ret)

        # Benchmark return (entering BM weights, pre-drift)
        if bm_drift:
            bm_ret = np.dot(bm_curr_weights, daily_ret)
        else:
            bm_w_today = bm_weights_fn(t_date, tickers, n_tickers)
            bm_ret = np.dot(bm_w_today, daily_ret)

        # --- Step 2: Weight drift from today's return -------------------------
        if weight_drift:
            prev_weights = _drift_weights(prev_weights, daily_ret)
        if bm_drift:
            bm_curr_weights = _drift_weights(bm_curr_weights, daily_ret)

        # --- Step 3: Rebalance at close (effective NEXT bar) ------------------
        is_rebal = rebal_check_fn(t_idx, t_date, start_idx, rebalance_freq, state)
        if is_rebal:
            pred_row = predictions.loc[t_date, tickers]

            # C5: inf/-inf in predictions would corrupt cvxpy silently. Replace
            # non-finite values with NaN so the coverage check below treats them
            # the same as missing.
            non_finite_mask = ~np.isfinite(pred_row.astype(float))
            if non_finite_mask.any():
                n_bad = int(non_finite_mask.sum())
                logger.warning(
                    "non-finite prediction(s) at %s: n=%d — treating as NaN. "
                    "Check upstream model output for inf/-inf.",
                    t_date.date(), n_bad,
                )
                pred_row = pred_row.mask(non_finite_mask)

            if pred_row.notna().sum() >= 10:
                bm_w = bm_weights_fn(t_date, tickers, n_tickers)

                # M10: respect config.cov_lookback (was hardcoded to 126).
                cov_lookback = int(getattr(config, "cov_lookback", 126))
                hist_start = max(0, t_idx - cov_lookback)
                hist_returns = risk_source[tickers].iloc[hist_start:t_idx]

                optimizer_diagnostics = {}
                if optimizer_accepts_diagnostics:
                    target_weights = optimizer_fn(
                        pred_row,
                        hist_returns,
                        prev_weights,
                        sector_map,
                        bm_w,
                        diagnostics=optimizer_diagnostics,
                    )
                else:
                    target_weights = optimizer_fn(
                        pred_row,
                        hist_returns,
                        prev_weights,
                        sector_map,
                        bm_w,
                    )

                rebal_had_fallback = False
                if optimizer_diagnostics.get("used_fallback", False):
                    rebal_had_fallback = True
                elif (not optimizer_accepts_diagnostics) and np.allclose(target_weights, bm_w, atol=1e-6):
                    rebal_had_fallback = True

                # -----------------------------------------------------------
                # REDESIGN K: Dynamic confidence-based execution (V2 port)
                # -----------------------------------------------------------
                # Compute signal confidence from spread + trailing IC, then
                # scale eta and no-trade band adaptively. When model is
                # uncertain, trade less. When model is sharp, trade more.
                raw_pred_row = None
                rp = state.get("_raw_predictions")
                if rp is not None and t_date in rp.index:
                    raw_pred_row = rp.loc[t_date, tickers]

                # Trailing IC from recent realized ICs
                # Item 12: promoted to config so experiments can tune without
                # editing backtest.py. Default 6 matches prior hardcoded value.
                trailing_ic_window = int(getattr(config, "trailing_ic_window", 6))
                if len(ic_values) >= 2:
                    recent_ics = [v for _, v in ic_values[-trailing_ic_window:]]
                    trailing_ic_mean = float(np.nanmean(recent_ics))
                else:
                    trailing_ic_mean = 0.0

                confidence = compute_signal_confidence(
                    pred_row, raw_pred_row, trailing_ic_mean,
                    spread_scale=float(getattr(config, "confidence_spread_scale", 0.20)),
                )

                candidate_weights = apply_dynamic_execution(
                    prev_weights, target_weights, confidence, config,
                )

                # Projection-failure fallback book (2026-07-02 structure review).
                # "prev" holds the previous weights (no-trade); "target" (default)
                # keeps the full-step MVO target — bit-identical to prior behaviour.
                if getattr(config, "projection_fallback_mode", "target") == "prev":
                    projection_fallback = prev_weights.copy()
                else:
                    projection_fallback = target_weights

                projection_diagnostics = {}
                if getattr(config, "use_score_based", False):
                    new_weights = project_capped_weights(
                        candidate_weights=candidate_weights,
                        max_weight=config.max_weight,
                        fallback_weights=projection_fallback,
                        config=config,
                        diagnostics=projection_diagnostics,
                    )
                else:
                    cov_matrix = optimizer_diagnostics.get("cov_matrix")
                    if cov_matrix is None:
                        cov_matrix = estimate_covariance(
                            hist_returns,
                            bm_weights=bm_w,
                            config=config,
                        )
                    new_weights = project_portfolio_weights(
                        candidate_weights=candidate_weights,
                        expected_returns=pred_row,
                        cov_matrix=cov_matrix,
                        prev_weights=prev_weights,
                        sector_map=sector_map,
                        bm_weights=bm_w,
                        max_te_annual=optimizer_diagnostics.get(
                            "max_te_annual",
                            config.max_te_annual,
                        ),
                        sector_deviation=optimizer_diagnostics.get(
                            "sector_deviation",
                            config.sector_deviation,
                        ),
                        config=config,
                        fallback_weights=projection_fallback,
                        diagnostics=projection_diagnostics,
                    )

                for diag in (optimizer_diagnostics, projection_diagnostics):
                    solver_name = diag.get("solver")
                    if solver_name:
                        solver_counts[solver_name] = solver_counts.get(solver_name, 0) + 1
                    if diag.get("solver_fallback", False):
                        solver_fallbacks += 1
                    if diag.get("used_fallback", False):
                        rebal_had_fallback = True
                        mode = diag.get("mode") or "unknown"
                        reason = diag.get("fallback_reason") or "unknown"
                        key = f"{mode}:{reason}"
                        fallback_reason_counts[key] = fallback_reason_counts.get(key, 0) + 1

                if rebal_had_fallback:
                    optimizer_failures += 1

                # Two-way L1 turnover: sum(|new - old|).
                # NOTE: industry one-way convention = 0.5 * L1 (halve this).
                turnover = np.abs(new_weights - prev_weights).sum()
                turnovers.append((t_date, turnover))
                weight_history[t_date] = pd.Series(new_weights, index=tickers)

                # Step 4: TC charged to today's PnL (paid at close_t)
                tc_cost = turnover * one_way_tc
                port_ret -= tc_cost

                prev_weights = new_weights
                state["first_rebal"] = False

                # Benchmark rebalance on the same dates (only when BM drifts)
                if bm_drift:
                    bm_curr_weights = bm_weights_fn(t_date, tickers, n_tickers)

        # --- Record ------------------------------------------------------------
        port_rets.append((t_date, port_ret))
        bm_rets.append((t_date, bm_ret))

        if track_daily_weights:
            weight_history_daily[t_date] = pd.Series(prev_weights.copy(), index=tickers)

        # S&P 500
        if track_spx and spx_series is not None and t_date in spx_series.index:
            spx_rets.append((t_date, spx_series.loc[t_date]))

        # IC computation at rebalance dates — single definition (targets convention
        # only), D2 resolution 2026-07-02. The former raw 20d forward-sum fallback
        # silently mixed two incomparable IC definitions into the avg_ic gate metric.
        # Production probe: 0 firings (94/94 rebal dates covered by targets) →
        # removing the fallback is byte-identical on the certified run.
        if track_ic and targets is not None and turnovers and turnovers[-1][0] == t_date:
            pred_row = predictions.loc[t_date, tickers]
            if t_date in targets.index:
                realized = targets.loc[t_date, tickers]
            else:
                realized = None
            if realized is not None:
                ic = compute_ic(pred_row, realized)
                if not np.isnan(ic):
                    ic_values.append((t_date, ic))

    # --- Assemble result ---
    result.portfolio_returns = pd.Series(
        dict(port_rets), name="portfolio"
    ).sort_index()
    result.benchmark_returns = pd.Series(
        dict(bm_rets), name="benchmark"
    ).sort_index()
    result.turnover = pd.Series(
        dict(turnovers), name="turnover"
    ).sort_index()
    result.portfolio_weights = weight_history
    result.daily_weights = weight_history_daily
    result.ic_series = pd.Series(
        dict(ic_values), name="IC"
    ).sort_index()
    if spx_rets:
        result.spx_returns = pd.Series(
            dict(spx_rets), name="SPX"
        ).sort_index()

    # Log optimizer failures
    total_rebals = len(weight_history)
    result.optimizer_failures = optimizer_failures
    result.optimizer_rebalances = total_rebals
    result.optimizer_failure_rate = (
        optimizer_failures / total_rebals if total_rebals > 0 else 0.0
    )
    result.optimizer_solver_counts = dict(solver_counts)
    result.optimizer_solver_solves = int(sum(solver_counts.values()))
    result.optimizer_solver_fallbacks = int(solver_fallbacks)
    result.optimizer_solver_fallback_rate = (
        solver_fallbacks / result.optimizer_solver_solves
        if result.optimizer_solver_solves > 0 else 0.0
    )
    result.optimizer_fallback_reason_counts = dict(fallback_reason_counts)
    if total_rebals > 0:
        fail_rate = optimizer_failures / total_rebals
        print(f"[simulate_portfolio] Optimizer fallback: "
              f"{optimizer_failures}/{total_rebals} ({fail_rate:.1%})")
        print(f"[simulate_portfolio] Solver counts: {result.optimizer_solver_counts}, "
              f"ECOS->SCS fallback rate: {result.optimizer_solver_fallback_rate:.1%}")

    return result


def run_backtest(
    data: UniverseData,
    rebalance_freq: int = REBALANCE_FREQ,
    pca_n_remove: int = None,
    include_sector_interactions: bool = False,
    precomputed_panel: Optional[pd.DataFrame] = None,
    precomputed_feature_names: Optional[List[str]] = None,
    precomputed_feature_groups: Optional[Dict] = None,
    precomputed_targets: Optional[pd.DataFrame] = None,
    precomputed_models: Optional[Dict] = None,
    precomputed_predictions: Optional[pd.DataFrame] = None,
    precomputed_raw_predictions: Optional[pd.DataFrame] = None,
    config: PipelineConfig = None,
) -> BacktestResult:
    """전체 백테스트 실행.

    Args:
        data: UniverseData
        rebalance_freq: 리밸런싱 주기 (영업일)
        pca_n_remove: PCA 제거 성분 수 (None=기본값, 2=Partial PCA)
        include_sector_interactions: Sector×Feature interaction 피처 포함 여부
        precomputed_panel: 사전 계산된 피처 패널 (None이면 내부에서 생성)
        precomputed_feature_names: 사전 계산된 피처 이름 목록
        precomputed_feature_groups: 사전 계산된 피처 그룹 매핑
        precomputed_targets: 사전 계산된 타겟
        precomputed_models: 사전 훈련된 모델 dict
        precomputed_predictions: 사전 계산된 예측값
        precomputed_raw_predictions: 사전 계산된 원시 예측값
        config: PipelineConfig (overrides module-level defaults when provided)
    """
    config = config or DEFAULT_CONFIG
    if rebalance_freq == REBALANCE_FREQ:
        rebalance_freq = config.rebalance_freq

    result = BacktestResult()
    # Capture the benchmark identity used in this run so summary() and
    # downstream report builders render correct labels (no hardcoded "EW").
    result.benchmark_type = getattr(config, "benchmark_type", "cap_weighted")

    # C-5: 구조화된 진행 로그
    progress = ProgressLogger()

    # Phase 2: 피처 생성 (사전 계산된 경우 스킵)
    if precomputed_panel is not None and precomputed_feature_names is not None:
        panel = precomputed_panel
        feature_names = precomputed_feature_names
        feature_groups = precomputed_feature_groups or {}
        print("[Backtest] Phase 2: 사전 계산된 피처 사용")
    else:
        progress.log("2", "START", "피처 생성")
        print("[Backtest] Phase 2: 피처 생성 중...")
        panel, feature_names, feature_groups = build_all_features(
            data,
            include_sector_interactions=include_sector_interactions,
            config=config,
        )
        progress.log("2", "DONE", f"피처 {len(feature_names)}개")

    # Phase 3: 타겟 생성
    if precomputed_targets is not None:
        targets = precomputed_targets
        print("[Backtest] Phase 3: 사전 계산된 타겟 사용")
    else:
        print("[Backtest] Phase 3: 타겟 생성 중...")
        targets = build_targets(data, n_remove=pca_n_remove, config=config)
        progress.log("3", "DONE", "타겟 생성 완료")

    # Pre-listing backfill masking (OFF by default). Mask targets BEFORE
    # walk_forward_train so phantom pre-listing rows never poison training.
    if getattr(config, "listing_mask_enabled", False):
        targets = mask_pre_listing(targets, config.listing_dates, inclusive=True)
        print(f"[Backtest] listing mask applied: targets "
              f"({', '.join(config.listing_dates)})")

    # Phase 4: 모델 학습 및 예측
    all_dates = data.dates
    ewma_tracker = None
    if precomputed_models is not None and precomputed_predictions is not None:
        models = precomputed_models
        predictions = precomputed_predictions
        raw_predictions = precomputed_raw_predictions if precomputed_raw_predictions is not None else precomputed_predictions
        print("[Backtest] Phase 4: 사전 훈련된 모델/예측 사용")
    else:
        print("[Backtest] Phase 4: 모델 학습 및 예측 중...")
        models, predictions, raw_predictions, ewma_tracker = walk_forward_train(
            panel, targets, feature_names, all_dates, config=config,
        )
        progress.log("4", "DONE", f"모델 {len(models)}개")

    result.models = models
    result.raw_predictions = raw_predictions
    result.targets = targets
    result.panel = panel
    result.feature_names = feature_names
    result.feature_groups = feature_groups
    result.model_quality = getattr(ewma_tracker, "model_quality", None)
    if result.model_quality is None:
        result.model_quality = summarize_cached_model_quality(models, config)
        mq = result.model_quality
        if mq["degenerate_rate"] > mq["max_degenerate_model_rate"]:
            msg = (
                f"[Backtest] Cached model degenerate rate {mq['degenerate_rate']:.1%} "
                f"exceeds max_degenerate_model_rate={mq['max_degenerate_model_rate']:.1%} "
                f"({mq['degenerate_retrains']}/{mq['total_retrains']})."
            )
            print("WARNING: " + msg)
            if mq["fail_on_degenerate_model_rate"]:
                raise RuntimeError(msg)
    result.data_quality = getattr(data, "data_quality", None)

    # Pre-listing backfill masking (OFF by default). Mask predictions BEFORE
    # any overlay (PEAD/tilt/etc.) so phantom pre-listing alpha never enters
    # the overlays or the optimizer. NaN alpha lets the MVO pin w==bm, and the
    # masked cap-weighted BM already gives these names bm==0.
    if getattr(config, "listing_mask_enabled", False):
        predictions = mask_pre_listing(predictions, config.listing_dates, inclusive=True)
        if raw_predictions is not None:
            raw_predictions = mask_pre_listing(
                raw_predictions, config.listing_dates, inclusive=True
            )
            result.raw_predictions = raw_predictions
        print(f"[Backtest] listing mask applied: predictions "
              f"({', '.join(config.listing_dates)})")

    # REDESIGN U (2026-04-14): PEAD post-process boost.
    # Stocks with positive pre-earnings revisions get +boost_w × decay × quality
    # in the days following an earnings announcement. Panel/model untouched.
    if getattr(config, "pead_boost_enabled", False):
        if getattr(data, "earnings_timeline", None) is not None:
            predictions = apply_pead_boost(predictions, data, config)
            print(f"[Backtest] REDESIGN U: PEAD boost applied "
                  f"(weight={config.pead_boost_weight}, decay_days={config.pead_decay_days}, "
                  f"max_days={config.pead_max_days})")
        else:
            print("[Backtest] REDESIGN U: PEAD boost SKIPPED (earnings_timeline not loaded)")

    # REDESIGN iter19 (2026-04-17): Growth/Revision tilt.
    # Tilts OW toward growing + revised-up names, away from pure quality/margin plays.
    if getattr(config, "growth_tilt_enabled", False):
        predictions = apply_growth_tilt(predictions, data, config)

    # Phase 3 (2026-04-24): Value-trap gate.
    # Zeroes scores for cheap+bad_mom+margin_accel CRM-like profiles that are
    # empirically destructive (-1.99%/20d in P3). Panel/model untouched.
    if getattr(config, "value_trap_gate_enabled", False):
        predictions = apply_value_trap_gate(predictions, panel, config)

    # Signal stability shrinkage (OFF by default, infra added 2026-04-20).
    # Damps retrain-boundary score jumps to reduce turnover. Pure post-process.
    if getattr(config, "signal_stability_lambda", 0.0) > 0.0:
        predictions = apply_signal_stability_shrinkage(
            predictions, raw_predictions, config, retrain_freq=config.retrain_freq,
        )
        print(f"[Backtest] signal-stability shrinkage applied "
              f"(lambda={config.signal_stability_lambda}, "
              f"retrain_freq={config.retrain_freq})")

    result.predictions = predictions  # save POST-gate predictions for downstream

    print("[Backtest] Phase 5-6: 포트폴리오 구축 중...")
    returns = data.returns
    # Use the actually-loaded universe (data.tickers). Previously this
    # was [t for t in TICKERS if t in returns.columns], which could drift
    # from what assembly/conditioning/factor used for feature construction
    # and from what summary/phase1 validation reported.
    tickers = list(data.tickers)
    n_tickers = len(tickers)
    risk_returns = getattr(data, "raw_returns", None)
    if risk_returns is not None:
        risk_returns = risk_returns.reindex(index=returns.index, columns=tickers)
    print_optimizer_config(n_tickers=n_tickers, config=config)
    sector_map = get_sector_map(data)

    # REDESIGN A (2026-04): cap-weighted benchmark by default.
    # Previous EW(1/n) fought real mega-cap concentration and neutralised
    # the model's edge on large names. See get_benchmark_fn docstring.
    bm_weights_fn = get_benchmark_fn(data, tickers, config=config)
    print(f"[Backtest] Benchmark type: {getattr(config, 'benchmark_type', 'equal_weight')}")

    # S&P 500 수익률 (Factor_Returns에서 추출)
    has_spx = data.has_factor_data() and "SPX" in data.factor_returns.columns
    spx_factor = data.factor_returns["SPX"] if has_spx else None

    # Optimizer with config forwarding
    # Factor-neutral live coverage accumulator (review M1). Stays all-zero and
    # unattached when factor_neutral is OFF, so OFF-default metrics are unchanged.
    _fn_telemetry = {"dates": 0, "cells": 0, "imputed": 0, "inert_dates": 0}

    def _optimizer_fn(pred_row, hist_returns, prev_w, s_map, bm_w, diagnostics=None):
        cov_matrix = estimate_covariance(hist_returns, bm_weights=bm_w, config=config)
        if diagnostics is not None:
            diagnostics["cov_matrix"] = cov_matrix
            diagnostics["max_te_annual"] = config.max_te_annual
            diagnostics["sector_deviation"] = config.sector_deviation
        # Factor-neutral per-date style loadings (P3). Built here from the
        # captured panel + this rebalance date (pred_row.name); stays None when
        # disabled so the optimizer path is bit-identical (factor_pen -> 0).
        factor_loadings = None
        if getattr(config, "factor_neutral_enabled", False) and panel is not None:
            cols = [config.factor_neutral_loadings[a]
                    for a in config.factor_neutral_axes
                    if a in config.factor_neutral_loadings]
            try:
                sub = panel.xs(pred_row.name, level="date").reindex(pred_row.index)[cols]
                finite = np.isfinite(sub.values)
                factor_loadings = np.where(finite, sub.values, 0.0)
                _fn_telemetry["dates"] += 1
                _fn_telemetry["cells"] += int(finite.size)
                _fn_telemetry["imputed"] += int((~finite).sum())
                if not finite.any():
                    _fn_telemetry["inert_dates"] += 1  # whole-date penalty -> 0
            except (KeyError, ValueError):
                factor_loadings = None
                _fn_telemetry["dates"] += 1
                _fn_telemetry["inert_dates"] += 1  # loadings unavailable -> penalty off
        return optimize_portfolio(
            expected_returns=pred_row,
            cov_matrix=cov_matrix,
            prev_weights=prev_w,
            sector_map=s_map if s_map else None,
            bm_weights=bm_w,
            config=config,
            diagnostics=diagnostics,
            factor_loadings=factor_loadings,
        )

    # Delegate to simulate_portfolio (the shared inner loop)
    sim_result = simulate_portfolio(
        predictions=predictions,
        returns=returns,
        tickers=tickers,
        all_dates=all_dates,
        sector_map=sector_map,
        rebalance_freq=rebalance_freq,
        one_way_tc=config.one_way_tc,
        optimizer_fn=_optimizer_fn,
        targets=targets,
        bm_weights_fn=bm_weights_fn,   # REDESIGN A: cap-weighted by default
        rebal_check_fn=None,  # default fixed-period
        weight_drift=True,
        bm_drift=True,
        track_ic=True,
        track_spx=has_spx,
        raw_predictions=raw_predictions,
        spx_series=spx_factor,
        track_daily_weights=True,
        config=config,  # REDESIGN J: pass config for turnover controls
        risk_returns=risk_returns,
    )

    # Transfer simulation results into the already-populated result object
    result.portfolio_returns = sim_result.portfolio_returns
    result.benchmark_returns = sim_result.benchmark_returns
    result.turnover = sim_result.turnover
    result.portfolio_weights = sim_result.portfolio_weights
    result.daily_weights = sim_result.daily_weights
    result.ic_series = sim_result.ic_series
    result.spx_returns = sim_result.spx_returns
    result.optimizer_failures = sim_result.optimizer_failures
    result.optimizer_rebalances = sim_result.optimizer_rebalances
    result.optimizer_failure_rate = sim_result.optimizer_failure_rate
    result.optimizer_solver_counts = sim_result.optimizer_solver_counts
    result.optimizer_solver_solves = sim_result.optimizer_solver_solves
    result.optimizer_solver_fallbacks = sim_result.optimizer_solver_fallbacks
    result.optimizer_solver_fallback_rate = sim_result.optimizer_solver_fallback_rate
    result.optimizer_fallback_reason_counts = sim_result.optimizer_fallback_reason_counts

    # Surface factor-neutral live coverage once (review M1). Only when the
    # penalty actually ran (dates>0), so OFF-default runs add no attr/log.
    if _fn_telemetry["dates"] > 0:
        impute_frac = (_fn_telemetry["imputed"] / _fn_telemetry["cells"]
                       if _fn_telemetry["cells"] else None)
        result.factor_neutral_telemetry = {**_fn_telemetry, "impute_frac": impute_frac}
        print(f"[Backtest] factor-neutral live coverage: dates={_fn_telemetry['dates']} "
              f"impute_frac={impute_frac} inert_dates={_fn_telemetry['inert_dates']}")

    # C-5: 진행 로그 기록
    progress.log("5-6", "DONE", f"리밸런싱 {len(result.portfolio_weights)}회")

    print(result.summary())
    return result


# ---------------------------------------------------------------------------
# C-2: Validation Gate
# ---------------------------------------------------------------------------
def validate_backtest(result: BacktestResult, thresholds: dict = None) -> dict:
    """백테스트 결과 검증 게이트.

    Returns:
        dict with keys: passed (bool), checks (list of dicts)
    """
    import json
    from pathlib import Path

    if thresholds is None:
        thresholds = {
            "min_ic": 0.015,
            "max_annual_turnover": 4.0,  # 400%
            "max_optimizer_fail_rate": 0.10,
        }

    metrics = result.compute_metrics()
    checks = []

    # IC check
    avg_ic = metrics.get("avg_ic", 0)
    ic_pass = avg_ic >= thresholds["min_ic"]
    checks.append({
        "name": "Average IC",
        "value": round(avg_ic, 4),
        "threshold": thresholds["min_ic"],
        "passed": ic_pass,
    })

    # Turnover check
    annual_to = metrics.get("avg_annual_turnover", 0)
    to_pass = annual_to <= thresholds["max_annual_turnover"]
    checks.append({
        "name": "Annual Turnover",
        "value": round(annual_to, 2),
        "threshold": thresholds["max_annual_turnover"],
        "passed": to_pass,
    })

    # Optimizer failure rate
    total_rebals = getattr(result, "optimizer_rebalances", len(result.portfolio_weights))
    fail_count = getattr(result, "optimizer_failures", None)
    if fail_count is None:
        if total_rebals > 0:
            fail_count = sum(
                1 for w in result.portfolio_weights.values()
                if abs(w.std() - 0) < 1e-6
            )
            fail_rate = fail_count / total_rebals
        else:
            fail_rate = 0
    else:
        fail_rate = fail_count / total_rebals if total_rebals > 0 else 0
    opt_pass = fail_rate <= thresholds["max_optimizer_fail_rate"]
    checks.append({
        "name": "Optimizer Fail Rate",
        "value": round(fail_rate, 3),
        "threshold": thresholds["max_optimizer_fail_rate"],
        "passed": opt_pass,
    })

    all_passed = all(c["passed"] for c in checks)

    validation = {
        "passed": all_passed,
        "checks": checks,
        "metrics_snapshot": {
            "annual_return": round(metrics.get("annual_return", 0), 4),
            "sharpe_ratio": round(metrics.get("sharpe_ratio", 0), 2),
            "information_ratio": round(metrics.get("information_ratio", 0), 2),
            "max_drawdown": round(metrics.get("max_drawdown", 0), 4),
        }
    }

    # 결과 저장
    out_path = Path("./outputs/backtest_validation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # numpy 타입을 Python 네이티브 타입으로 변환
    def _to_native(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_native(v) for v in obj]
        return obj

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_to_native(validation), f, indent=2, ensure_ascii=False)

    # 콘솔 출력
    print("\n" + "=" * 50)
    print("  VALIDATION GATE")
    print("=" * 50)
    for c in checks:
        status = "PASS [O]" if c["passed"] else "FAIL [X]"
        print(f"  {c['name']:25s}: {c['value']:>8} (threshold: {c['threshold']}) [{status}]")
    print(f"\n  Overall: {'PASSED' if all_passed else 'FAILED'}")
    print("=" * 50)

    return validation
