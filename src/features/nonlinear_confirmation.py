"""Nonlinear confirmation features for winner formation.

The production ranker is deliberately shallow and often stops after only a
small number of trees.  These features therefore encode economically useful
``AND`` conditions explicitly instead of asking axis-aligned splits to learn
them from scratch.

Unlike a product, ``min(z(a), z(b))`` cannot turn two weak/negative parents
into a large positive value.  It is high only when *both* cross-sectional
signals are high, which matches the confirmation hypotheses below.
"""

from typing import Dict

import numpy as np
import pandas as pd

from src.features.utils import cross_sectional_zscore


# feature name -> (parent_a, parent_b).  Every parent is already admitted by
# CORE_FEATURE_WHITELIST; the block adds expressiveness, not hidden raw inputs.
CONFIRMATION_PARENTS: Dict[str, tuple[str, str]] = {
    # Strong long-term price trend that is also close to its 52-week high.
    "nl_mom_breakout_confirm": ("momentum_252d", "dist_52w_high"),
    # Price trend confirmed by persistent analyst EPS revisions.
    "nl_mom_revision_confirm": ("momentum_252d", "eps_rev_ma_63d"),
    # A long-term winner whose more recent momentum is accelerating.
    "nl_mom_accel_confirm": ("momentum_252d", "mom_accel_63_252"),
    # Sales growth that is supported by a high level of profitability.
    "nl_growth_quality_confirm": ("best_sales_chg_252d", "best_roe_level_z"),
    # Breadth: EPS and sales revisions must agree rather than offset.
    "nl_revision_breadth_confirm": ("eps_rev_ma_63d", "sales_rev_ma_63d"),
}


def _soft_and(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Elementwise minimum with strict NaN propagation.

    Both inputs are cross-sectional z-scores.  ``min`` is a conservative
    fuzzy-AND: one weak leg is enough to keep the confirmation score weak.
    """
    out = a.where(a.le(b), b)
    return out.where(a.notna() & b.notna())


def _trend_efficiency(returns: pd.DataFrame) -> pd.DataFrame:
    """Signed 252-day path efficiency, bounded to [-1, 1].

    The numerator is the same arithmetic-return path used by the production
    momentum features.  Dividing by total absolute movement distinguishes a
    smooth compounder from a high-churn path that happens to end at the same
    cumulative return.  A 126-observation warm-up matches the existing
    52-week range convention and leaves pre-listing rows as NaN.
    """
    min_periods = 126
    direction = returns.rolling(252, min_periods=min_periods).sum()
    distance = returns.abs().rolling(252, min_periods=min_periods).sum()
    efficiency = direction / distance.replace(0.0, np.nan)
    return efficiency.clip(lower=-1.0, upper=1.0)


def build_nonlinear_confirmation_features(
    all_features: Dict[str, pd.DataFrame],
    data=None,
) -> Dict[str, pd.DataFrame]:
    """Build confirmation and path-shape features without future data.

    Missing parents are skipped rather than imputed here.  Assembly performs
    the same per-date missing-value treatment as every other feature after the
    block is admitted through its default-off configuration gate.
    """
    out: Dict[str, pd.DataFrame] = {}
    zcache: Dict[str, pd.DataFrame] = {}

    for name, (parent_a, parent_b) in CONFIRMATION_PARENTS.items():
        if parent_a not in all_features or parent_b not in all_features:
            continue
        for parent in (parent_a, parent_b):
            if parent not in zcache:
                zcache[parent] = cross_sectional_zscore(all_features[parent])
        out[name] = _soft_and(zcache[parent_a], zcache[parent_b])

    returns = getattr(data, "returns_masked", None) if data is not None else None
    if returns is None and data is not None:
        # Small test doubles and older cached UniverseData objects may expose
        # only returns.  Production UniverseData always prefers returns_masked.
        returns = getattr(data, "returns", None)
    if returns is not None:
        out["nl_trend_efficiency_252"] = _trend_efficiency(returns)

    return out
