"""Unit tests — src.features.assembly.apply_core_filter (S8 extra_whitelist).

Purpose: (a) pin the CURRENT apply_core_filter contract so the S8 change is
surgical, and (b) drive the target interface (spec-s8-news-trend.md §2.2):

    apply_core_filter(features, feature_groups, extra_whitelist=None) -> None
    survivors = set(features) & (CORE_FEATURE_WHITELIST | (extra_whitelist or set()))

The function mutates ``features`` / ``feature_groups`` IN PLACE and returns None;
CORE_FEATURE_WHITELIST itself is never edited.

Idioms (CLAUDE.md): plain pytest functions, no fixtures, synthetic in-memory
dicts only. Symbols under construction imported INSIDE test bodies so
``pytest --collect-only`` reports zero collection errors.

Expected state before the S8 implementation lands:
  * test_current_contract_intersection_only ...... GREEN (legacy contract)
  * test_current_contract_returns_none_in_place .. GREEN (legacy contract)
  * test_extra_whitelist_none_is_current_behavior  RED   (kwarg absent -> TypeError)
  * test_extra_whitelist_adds_only_news_trend ..... RED   (kwarg absent -> TypeError)
"""

import pandas as pd


NEWS_TREND = "news_trend"
_WL_MEMBERS = ["beta_63d", "momentum_252d", "eps_rev"]  # real whitelist members
_DUMMIES = ["dummy_junk_a", "dummy_junk_b"]             # non-whitelist -> pruned


def _tiny_df():
    return pd.DataFrame({"x": [0.0, 1.0]})


def _synthetic_panel():
    """Fresh (features, feature_groups) each call — the filter mutates them."""
    names = _WL_MEMBERS + [NEWS_TREND] + _DUMMIES
    features = {n: _tiny_df() for n in names}
    feature_groups = {
        "Price": ["beta_63d", "momentum_252d"],
        "Sellside": ["eps_rev", NEWS_TREND],
        "Junk": list(_DUMMIES),
    }
    return features, feature_groups


def _check_premise(whitelist):
    assert NEWS_TREND not in whitelist, (
        f"premise broken: {NEWS_TREND!r} must NOT be a base CORE_FEATURE_WHITELIST member"
    )
    for m in _WL_MEMBERS:
        assert m in whitelist, f"stale fixture: {m!r} not in whitelist"


# ---------------------------------------------------------------------------
# Current contract (must stay GREEN through the S8 change)
# ---------------------------------------------------------------------------
def test_current_contract_intersection_only():
    """Legacy 2-arg call keeps EXACTLY the whitelist intersection: non-whitelist
    keys (news_trend + dummies) pruned, empty groups removed, survivors reflected
    in feature_groups."""
    from src.features.assembly import apply_core_filter, CORE_FEATURE_WHITELIST

    _check_premise(CORE_FEATURE_WHITELIST)
    features, feature_groups = _synthetic_panel()
    original = set(features.keys())

    apply_core_filter(features, feature_groups)

    assert set(features.keys()) == (original & set(CORE_FEATURE_WHITELIST))
    assert set(features.keys()) == set(_WL_MEMBERS)
    assert NEWS_TREND not in features
    for d in _DUMMIES:
        assert d not in features
    # feature_groups pruned to survivors; empty 'Junk' dropped; news_trend gone.
    assert feature_groups.get("Price") == ["beta_63d", "momentum_252d"]
    assert feature_groups.get("Sellside") == ["eps_rev"]
    assert "Junk" not in feature_groups


def test_current_contract_returns_none_in_place():
    """apply_core_filter returns None and mutates its arguments in place."""
    from src.features.assembly import apply_core_filter

    features, feature_groups = _synthetic_panel()
    ret = apply_core_filter(features, feature_groups)

    assert ret is None
    assert NEWS_TREND not in features  # mutation happened on the passed-in dict


# ---------------------------------------------------------------------------
# Target interface (RED until the S8 extra_whitelist kwarg is added)
# ---------------------------------------------------------------------------
def test_extra_whitelist_none_is_current_behavior():
    """extra_whitelist=None must be identical to the legacy 2-arg behaviour."""
    from src.features.assembly import apply_core_filter, CORE_FEATURE_WHITELIST

    _check_premise(CORE_FEATURE_WHITELIST)
    features, feature_groups = _synthetic_panel()
    original = set(features.keys())

    apply_core_filter(features, feature_groups, extra_whitelist=None)

    assert set(features.keys()) == (original & set(CORE_FEATURE_WHITELIST))
    assert NEWS_TREND not in features


def test_extra_whitelist_adds_only_news_trend():
    """extra_whitelist={"news_trend"} extends the survivor set by EXACTLY that
    one key: survivor == intersection ∪ {news_trend}; dummies still pruned;
    news_trend retained in its feature_group."""
    from src.features.assembly import apply_core_filter, CORE_FEATURE_WHITELIST

    _check_premise(CORE_FEATURE_WHITELIST)
    features, feature_groups = _synthetic_panel()
    original = set(features.keys())
    baseline = original & set(CORE_FEATURE_WHITELIST)  # legacy survivors

    apply_core_filter(features, feature_groups, extra_whitelist={NEWS_TREND})

    survivors = set(features.keys())
    assert survivors == baseline | {NEWS_TREND}
    assert survivors - baseline == {NEWS_TREND}  # exactly one added
    assert len(survivors) == len(baseline) + 1
    for d in _DUMMIES:
        assert d not in features  # non-whitelist dummies still pruned
    assert feature_groups.get("Sellside") == ["eps_rev", NEWS_TREND]
    assert "Junk" not in feature_groups


# ---------------------------------------------------------------------------
# §S11.7: lean momentum 컴포짓은 dense returns가 아니라 상장 전 NaN 뷰
# (returns_masked)를 소비해야 한다.
# ---------------------------------------------------------------------------
def test_lean_momentum_uses_masked_returns():
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd

    from src.features.assembly import build_lean_momentum_composites

    n_dates, listing_pos = 80, 70
    dates = pd.bdate_range("2022-01-03", periods=n_dates)
    rng = np.random.default_rng(3)
    tickers = ["AAA", "BBB", "NEW"]
    dense = pd.DataFrame(
        rng.normal(0.0, 0.02, size=(n_dates, 3)), index=dates, columns=tickers
    )
    masked = dense.copy()
    masked.iloc[:listing_pos, 2] = np.nan
    prices = (1 + dense).cumprod() * 100.0
    data = SimpleNamespace(returns=dense, returns_masked=masked, prices=prices)

    feats = build_lean_momentum_composites(data, tickers)
    # NEW는 실데이터 10일뿐 -> mom21/mom63 미형성 -> accel 전 구간 NaN
    assert feats["mom_accel_21_63"]["NEW"].isna().all()
    assert np.isfinite(feats["mom_accel_21_63"]["AAA"].iloc[-1])
