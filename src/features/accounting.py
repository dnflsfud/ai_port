"""Category 1: Accounting / Fundamental (~100 features)."""

import logging
import pandas as pd
import numpy as np
from typing import Dict

from src.data_loader import UniverseData
from src.features.utils import cross_sectional_zscore, safe_pct_change, cs_rank, rolling_tsz

logger = logging.getLogger(__name__)


ACCOUNTING_BASE = [
    "BEST_EPS", "BEST_SALES", "BEST_CALCULATED_FCF",
    "BEST_GROSS_MARGIN", "OPER_MARGIN", "BEST_CAPEX", "BEST_ROE",
]

VALUATION_SHEETS = [
    "BEST_PE_RATIO", "BEST_PEG_RATIO", "BEST_PX_BPS_RATIO",
    "BEST_EV_TO_BEST_EBITDA",
]

# EPS/Sales/Margin: 변화율·가속도만 중요, level 자체는 제거
LEVEL_SKIP_SHEETS = {"BEST_EPS", "BEST_SALES", "BEST_GROSS_MARGIN", "OPER_MARGIN"}


def build_accounting_features(data: UniverseData) -> Dict[str, pd.DataFrame]:
    features: Dict[str, pd.DataFrame] = {}

    # -- Base fundamentals --
    for sheet in ACCOUNTING_BASE:
        try:
            raw = data.get_sheet(sheet)
        except KeyError:
            logger.debug("accounting: sheet %s missing — skipping base fundamental block", sheet)
            continue
        p = sheet.lower()
        # 변화율 (4 windows)
        for w in [21, 63, 126, 252]:
            features[f"{p}_chg_{w}d"] = safe_pct_change(raw, w)
        # 가속도 (변화율의 변화율)
        features[f"{p}_accel"] = safe_pct_change(raw, 21) - safe_pct_change(raw, 63)
        # 변화의 변동성 (fundamental stability)
        chg21 = safe_pct_change(raw, 21)
        features[f"{p}_chg_vol"] = chg21.rolling(63, min_periods=21).std()

        # Level features: EPS/Sales/Margin은 제외 (변화율이 더 중요)
        if sheet not in LEVEL_SKIP_SHEETS:
            features[f"{p}_level_z"] = cross_sectional_zscore(raw)
            features[f"{p}_rank"] = cs_rank(raw)
            med = raw.rolling(252, min_periods=126).median()
            features[f"{p}_vs_median"] = (raw / med.replace(0, np.nan)) - 1

    # -- Valuation: 4 sheets × 7 features = 28 --
    # REDESIGN M (2026-04-13): level_z now uses per-ticker rolling 3Y
    # self-normalization BEFORE cross-sectional comparison. Previously we
    # compared raw PE/PEG/PBR levels across the universe, which implicitly
    # assumed all 60 names belong in the same distribution. That penalized
    # structurally high-multiple names (NVDA, LLY, COST) even when they
    # were cheap relative to their own 3Y history. The new score answers
    # "how far is this ticker from its own recent norm, compared to other
    # tickers' distance from theirs". The cs_rank helper stays on raw
    # levels to preserve a pure cross-sectional rank alongside.
    for sheet in VALUATION_SHEETS:
        try:
            raw = data.get_sheet(sheet)
        except KeyError:
            logger.debug("accounting: valuation sheet %s missing — skipping", sheet)
            continue
        p = sheet.lower()
        tsz = rolling_tsz(raw, window=756, min_periods=252)
        features[f"{p}_level_z"] = cross_sectional_zscore(tsz)
        features[f"{p}_chg_21d"] = safe_pct_change(raw, 21)
        features[f"{p}_chg_63d"] = safe_pct_change(raw, 63)
        features[f"{p}_accel"] = safe_pct_change(raw, 21) - safe_pct_change(raw, 63)
        med = raw.rolling(252, min_periods=126).median()
        features[f"{p}_vs_median"] = (raw / med.replace(0, np.nan)) - 1
        features[f"{p}_vol"] = safe_pct_change(raw, 21).rolling(63, min_periods=21).std()
        features[f"{p}_rank"] = cs_rank(raw)

    # -- Cross-ratios (~10) --
    _add_cross_ratios(data, features)

    return features


def _add_cross_ratios(data: UniverseData, features: Dict[str, pd.DataFrame]):
    """Accounting 교차비율 피처."""
    def _safe_get(name):
        try:
            return data.get_sheet(name)
        except KeyError:
            logger.debug("accounting: cross-ratio input %s missing — derived features will skip", name)
            return None

    eps = _safe_get("BEST_EPS")
    sales = _safe_get("BEST_SALES")
    gm = _safe_get("BEST_GROSS_MARGIN")
    om = _safe_get("OPER_MARGIN")
    fcf = _safe_get("BEST_CALCULATED_FCF")
    capex = _safe_get("BEST_CAPEX")
    roe = _safe_get("BEST_ROE")
    pe = _safe_get("BEST_PE_RATIO")
    mc = data.market_cap

    if eps is not None and sales is not None:
        # Earnings quality: EPS growth > Sales growth = quality
        features["earnings_quality_63d"] = safe_pct_change(eps, 63) - safe_pct_change(sales, 63)
        features["earnings_quality_252d"] = safe_pct_change(eps, 252) - safe_pct_change(sales, 252)

    if gm is not None and om is not None:
        features["op_leverage_63d"] = safe_pct_change(om, 63) - safe_pct_change(gm, 63)

    if fcf is not None and eps is not None:
        features["cash_conversion_z"] = cross_sectional_zscore(fcf / eps.replace(0, np.nan).abs())

    if capex is not None and sales is not None:
        ratio = capex / sales.replace(0, np.nan).abs()
        features["capex_intensity_z"] = cross_sectional_zscore(ratio)
        features["capex_intensity_chg"] = safe_pct_change(ratio, 63)

    if roe is not None and pe is not None:
        features["roe_pe_z"] = cross_sectional_zscore(roe / pe.replace(0, np.nan).abs())

    if eps is not None:
        features["mkcap_eps_divg"] = safe_pct_change(mc, 63) - safe_pct_change(eps, 63)

    # REDESIGN S (2026-04-14): Quality-gated value composite — value-trap filter.
    # Multiplicative AND-gate: only positive when cheap (low PE) AND profitable
    # (high ROE) AND analyst revisions are improving (high eps_rev). Cube-root
    # softens skew. Cross-sectional z applied for whitelist consistency.
    # Designed to give the model an explicit signal that "cheap alone isn't enough" —
    # the LightGBM tree splits would need 3 layers to learn this AND-gate from
    # raw PE/ROE/rev features, so providing it explicitly improves sample efficiency.
    rev_for_gate = None
    try:
        rev_for_gate = data.get_sheet("Factset_EPS_Revision")
    except KeyError:
        logger.debug("accounting: Factset_EPS_Revision missing — skipping quality_gated_value_z")
    if pe is not None and roe is not None and rev_for_gate is not None:
        rev_smoothed = rev_for_gate.rolling(63, min_periods=21).mean()
        cheap_rank = cs_rank(-pe.replace(0, np.nan))     # high = cheap
        quality_rank = cs_rank(roe)                       # high = profitable
        revision_rank = cs_rank(rev_smoothed)             # high = improving
        composite = (cheap_rank * quality_rank * revision_rank).clip(lower=0) ** (1.0 / 3.0)
        features["quality_gated_value_z"] = cross_sectional_zscore(composite)
