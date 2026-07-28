# -*- coding: utf-8 -*-
"""S13.13 nonlinear interaction block.

Products of CS z-scores of parents that are ALL already in
CORE_FEATURE_WHITELIST — zero new information axes, pure expressiveness for
a capacity-starved ranker (mean ~42 trees, §S13.8) that cannot form
multiplicative structure from axis-aligned splits on its own.

Pre-check (§S13.13, 2026-07-27): 21d persistence 0.66–0.84, between-ticker
variance share 0.06–0.12 — all pass the S13.10 identity-leak rule and the
S13.9 low-pass-survival lesson. Built unconditionally (S8 idiom); admission
is gated at the core-whitelist filter by config.interaction_features_enabled.
"""

from typing import Dict

import pandas as pd

from src.features.utils import cross_sectional_zscore

# feature name -> (parent_a, parent_b); every parent must be a core feature.
INTERACTION_PARENTS: Dict[str, tuple] = {
    "ix_vol_mom": ("idio_vol_63d", "momentum_252d"),
    "ix_val_mom": ("best_px_bps_ratio_level_z", "momentum_252d"),
    "ix_rev_vol": ("eps_rev_ma_63d", "idio_vol_63d"),
    "ix_qual_val": ("best_roe_level_z", "best_px_bps_ratio_level_z"),
}


def build_interaction_features(
    all_features: Dict[str, pd.DataFrame],
    data=None,
) -> Dict[str, pd.DataFrame]:
    """Return {name: z(parent_a) * z(parent_b)} for every pair whose parents
    are present in *all_features*; absent parents skip the feature silently
    (assembly calls this unconditionally, before any filtering).

    §S13.13 amendment: when *data* is given, also builds
    ``mom_consistency_252`` — the 252d share of up days (min_periods 126,
    "continuous information" path shape). NaN returns propagate, so
    pre-listing rows stay NaN.
    """
    out: Dict[str, pd.DataFrame] = {}
    zcache: Dict[str, pd.DataFrame] = {}
    for name, (pa, pb) in INTERACTION_PARENTS.items():
        if pa not in all_features or pb not in all_features:
            continue
        for p in (pa, pb):
            if p not in zcache:
                zcache[p] = cross_sectional_zscore(all_features[p])
        out[name] = zcache[pa] * zcache[pb]

    rets = getattr(data, "returns", None) if data is not None else None
    if rets is not None:
        up = (rets > 0).astype(float).where(rets.notna())
        out["mom_consistency_252"] = up.rolling(252, min_periods=126).mean()
    return out
