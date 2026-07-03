"""Backward-compatible re-export. Use src.features directly for new code."""

from src.features.assembly import build_all_features
from src.features.utils import cross_sectional_zscore, safe_pct_change, clip_outliers, cs_rank
from src.features.sellside import (
    clean_revision_spikes,
    build_bounded_revision_features,
    build_sellside_features,
)
from src.features.accounting import (
    build_accounting_features,
    ACCOUNTING_BASE,
    VALUATION_SHEETS,
    LEVEL_SKIP_SHEETS,
    _add_cross_ratios,
)
from src.features.price import build_price_features
from src.features.conditioning import build_conditioning_features
from src.features.factor import build_factor_features
from src.features.interaction import (
    build_sector_interaction_features,
    INTERACTION_KEY_FEATURES,
)
