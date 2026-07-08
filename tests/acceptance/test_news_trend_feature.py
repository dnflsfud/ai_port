"""Acceptance tests — S8 news_trend sentiment feature arm (default-OFF).

Written from the spec (spec-s8-news-trend.md) BEFORE any implementation
exists. The target contract is:

  * config.py adds ``PipelineConfig.news_trend_feature_enabled: bool = False``.
  * assembly.py grows an OPTIONAL kwarg on ``apply_core_filter`` (spec §2.2):

        apply_core_filter(features, feature_groups, extra_whitelist=None)

    survivors = set(features) & (CORE_FEATURE_WHITELIST | (extra_whitelist or set()))
    The function mutates ``features``/``feature_groups`` in place, returns None.
    CORE_FEATURE_WHITELIST itself is NOT edited.
  * variants/exp_news_trend_feature.yaml exists, cloned from
    exp_mu_vol_scaling.yaml, with ``news_trend_feature_enabled: true`` in place
    of the ``mu_vol_scaling_enabled: true`` line, all iter15 pins unchanged.
  * run_variant.py is NOT modified — the new flag must NOT be added to the
    SAFE_FOR_CACHE_REUSE allowlist (feature-panel change forces a full re-run).

Independent references (spec §1): the feature key is exactly ``news_trend``
(NOT "news_sent_trend"), and it is NOT a member of CORE_FEATURE_WHITELIST, so
under the OFF path it is pruned. Expected survivor sets are computed here with
plain set algebra against the imported CORE_FEATURE_WHITELIST — the filter's
own pruning logic is not reused.

Idioms (project convention): plain pytest functions, no fixtures, synthetic
in-memory dicts only. Target symbols under construction are imported INSIDE
test bodies so ``pytest --collect-only`` reports zero collection errors.

RED-before-implementation expectation (per test):
  * test_config_flag_default_off .......... RED now: AttributeError (no field)
  * test_off_parity_default_call .......... GREEN now: pure parity guard, the
        current 2-arg signature already yields OFF behaviour; must STAY green.
  * test_off_parity_explicit_none ......... RED now: TypeError (kwarg absent)
  * test_on_adds_only_news_trend .......... RED now: TypeError (kwarg absent)
  * test_flag_not_cache_safe .............. GREEN now: flag not yet anywhere,
        stays green iff implementer never registers it as cache-safe.
  * test_variant_yaml_loads_and_valid ..... RED now: file missing +
        unknown-field (config lacks the flag) -> load_manifest raises.

======================================================================
합격기준(spec §3) ↔ 테스트 매핑표
----------------------------------------------------------------------
3.1  default-OFF: PipelineConfig().news_trend_feature_enabled is False
        -> test_config_flag_default_off
3.2  OFF parity: OFF-path apply_core_filter == CORE_FEATURE_WHITELIST 교집합,
     news_trend 제거, feature_groups 동일
        -> test_off_parity_default_call      (2-arg / no extra)
        -> test_off_parity_explicit_none     (extra_whitelist=None)
3.3  ON 동작: extra_whitelist={"news_trend"} -> survivor = OFF survivor ∪
     {news_trend} 정확히 1개 추가, 타 비-whitelist 여전히 제거, groups 잔존
        -> test_on_adds_only_news_trend
3.4  캐시 안전: SAFE_FOR_CACHE_REUSE 에 "news_trend_feature_enabled" 미포함
        -> test_flag_not_cache_safe
3.5  variant 로드 정합: exp_news_trend_feature.yaml 이 _valid_config_fields
     검증 통과(load raise 없음), 플래그 true 포함, mu_vol_scaling 미포함
        -> test_variant_yaml_loads_and_valid
======================================================================
"""

import inspect
from pathlib import Path

import pandas as pd

from src.config import PipelineConfig


# The single pre-registered feature key under test (spec §1). Spelled exactly
# "news_trend" everywhere — NOT "news_sent_trend".
NEWS_TREND = "news_trend"

# ai_port repo root: tests/acceptance/<file> -> parents[2] == ai_port/.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_VARIANT_YAML = _REPO_ROOT / "variants" / "exp_news_trend_feature.yaml"

# Real CORE_FEATURE_WHITELIST members used to build the synthetic panel. These
# are asserted to be genuine whitelist members at runtime (see _assert_premise).
_WL_MEMBERS = ["beta_63d", "momentum_252d", "eps_rev"]
# Non-whitelist dummies that MUST be pruned under both OFF and ON paths.
_DUMMIES = ["dummy_junk_a", "dummy_junk_b"]


# ---------------------------------------------------------------------------
# Synthetic-data helpers (NOT the implementation). Fresh dicts each call
# because apply_core_filter mutates its arguments in place.
# ---------------------------------------------------------------------------
def _tiny_df():
    return pd.DataFrame({"x": [0.0, 1.0]})


def _synthetic_panel():
    """Return (features, feature_groups) covering whitelist + news_trend +
    non-whitelist dummies, with news_trend co-located next to a real member so
    its group survival can be checked independently of the dummies' group."""
    names = _WL_MEMBERS + [NEWS_TREND] + _DUMMIES
    features = {n: _tiny_df() for n in names}
    feature_groups = {
        "Price": ["beta_63d", "momentum_252d"],
        "Sellside": ["eps_rev", NEWS_TREND],
        "Junk": list(_DUMMIES),
    }
    return features, feature_groups


def _assert_premise(whitelist):
    """The spec's pre-registration facts the tests rely on."""
    assert NEWS_TREND not in whitelist, (
        f"premise broken: {NEWS_TREND!r} must NOT be in CORE_FEATURE_WHITELIST "
        "(spec §1 pre-registers it as a conditional extra, not a base member)"
    )
    for m in _WL_MEMBERS:
        assert m in whitelist, f"test fixture stale: {m!r} not in whitelist"
    for d in _DUMMIES:
        assert d not in whitelist, f"dummy {d!r} unexpectedly in whitelist"


def _all_group_names(feature_groups):
    out = set()
    for names in feature_groups.values():
        out.update(names)
    return out


# ---------------------------------------------------------------------------
# 3.1  default-OFF flag
# ---------------------------------------------------------------------------
def test_config_flag_default_off():
    """A fresh PipelineConfig exposes the flag and it defaults to False."""
    cfg = PipelineConfig()
    assert hasattr(cfg, "news_trend_feature_enabled"), (
        "PipelineConfig is missing the S8 flag 'news_trend_feature_enabled'"
    )
    assert cfg.news_trend_feature_enabled is False


# ---------------------------------------------------------------------------
# 3.2  OFF parity (news_trend pruned; result == CORE_FEATURE_WHITELIST 교집합)
# ---------------------------------------------------------------------------
def test_off_parity_default_call():
    """The plain 2-arg call (no extra_whitelist) reproduces the legacy core
    filter EXACTLY: survivors == set(names) & CORE_FEATURE_WHITELIST, news_trend
    and every dummy pruned, empty group dropped. This is the byte-identical
    OFF-parity guard and must stay green across the implementation."""
    from src.features.assembly import apply_core_filter, CORE_FEATURE_WHITELIST

    _assert_premise(CORE_FEATURE_WHITELIST)
    features, feature_groups = _synthetic_panel()
    original_names = set(features.keys())

    apply_core_filter(features, feature_groups)  # OFF path: no extra whitelist

    expected_survivors = original_names & set(CORE_FEATURE_WHITELIST)
    assert set(features.keys()) == expected_survivors == set(_WL_MEMBERS)
    assert NEWS_TREND not in features, "OFF path must prune the news_trend key"
    for d in _DUMMIES:
        assert d not in features

    # feature_groups reflect survivors; news_trend gone; empty 'Junk' removed.
    assert NEWS_TREND not in _all_group_names(feature_groups)
    assert feature_groups.get("Sellside") == ["eps_rev"]
    assert "Junk" not in feature_groups


def test_off_parity_explicit_none():
    """extra_whitelist=None is identical to the OFF/default path (the new kwarg
    defaults to inert). RED now (kwarg absent -> TypeError)."""
    from src.features.assembly import apply_core_filter, CORE_FEATURE_WHITELIST

    _assert_premise(CORE_FEATURE_WHITELIST)
    features, feature_groups = _synthetic_panel()
    original_names = set(features.keys())

    apply_core_filter(features, feature_groups, extra_whitelist=None)

    assert set(features.keys()) == (original_names & set(CORE_FEATURE_WHITELIST))
    assert NEWS_TREND not in features
    assert NEWS_TREND not in _all_group_names(feature_groups)


# ---------------------------------------------------------------------------
# 3.3  ON behaviour (exactly one extra survivor: news_trend)
# ---------------------------------------------------------------------------
def test_on_adds_only_news_trend():
    """extra_whitelist={"news_trend"} keeps news_trend and NOTHING else beyond
    the OFF survivor set: survivor == OFF ∪ {news_trend}, len(+1), dummies still
    pruned, and news_trend retained in its feature_group. RED now (TypeError)."""
    from src.features.assembly import apply_core_filter, CORE_FEATURE_WHITELIST

    _assert_premise(CORE_FEATURE_WHITELIST)

    # OFF reference survivor set, computed independently (not via the filter).
    off_survivors = (set(_WL_MEMBERS) | {NEWS_TREND} | set(_DUMMIES)) & set(
        CORE_FEATURE_WHITELIST
    )
    assert NEWS_TREND not in off_survivors  # sanity: baseline prunes it

    features, feature_groups = _synthetic_panel()
    apply_core_filter(features, feature_groups, extra_whitelist={NEWS_TREND})

    on_survivors = set(features.keys())
    assert on_survivors == off_survivors | {NEWS_TREND}
    assert on_survivors - off_survivors == {NEWS_TREND}  # EXACTLY one added
    assert len(on_survivors) == len(off_survivors) + 1
    assert NEWS_TREND in features
    for d in _DUMMIES:
        assert d not in features, "ON path must still prune non-whitelist dummies"

    # news_trend survives inside its feature_group; 'Junk' still empty->removed.
    assert feature_groups.get("Sellside") == ["eps_rev", NEWS_TREND]
    assert "Junk" not in feature_groups


# ---------------------------------------------------------------------------
# 3.4  Cache safety — flag NOT registered as cache-reuse-safe
# ---------------------------------------------------------------------------
def test_flag_not_cache_safe():
    """The feature-panel flag must force a full pipeline re-run, i.e. it must
    NOT appear in run_variant.run()'s SAFE_FOR_CACHE_REUSE allowlist.

    NOTE: SAFE_FOR_CACHE_REUSE is a function-local frozenset inside
    run_variant.run() (not a module attribute), so we inspect the source of the
    allowlist literal directly. A positive control ('cov_lookback', a known
    cache-safe key) proves the extracted region really is the allowlist."""
    import run_variant

    src_text = inspect.getsource(run_variant.run)
    marker = "SAFE_FOR_CACHE_REUSE = frozenset({"
    assert marker in src_text, "allowlist literal not found in run_variant.run"

    start = src_text.index(marker)
    end = src_text.index("})", start)
    region = src_text[start:end]

    # Positive control: the region really is the cache-safe allowlist.
    assert "cov_lookback" in region, "region-extraction failed to capture the allowlist"
    # The S8 flag must be absent -> panel change invalidates the cache.
    assert "news_trend_feature_enabled" not in region, (
        "news_trend_feature_enabled must NOT be in SAFE_FOR_CACHE_REUSE "
        "(feature-panel change requires a full re-run; run_variant.py unchanged)"
    )


# ---------------------------------------------------------------------------
# 3.5  Variant YAML loads and validates
# ---------------------------------------------------------------------------
def test_variant_yaml_loads_and_valid():
    """variants/exp_news_trend_feature.yaml exists and load_manifest accepts it
    (all override keys are valid PipelineConfig fields), the S8 flag is set true,
    and the template's mu_vol_scaling_enabled line is gone. RED now: the file is
    missing AND the config field does not yet exist (load_manifest would raise on
    the unknown field), so this is a full end-to-end acceptance gate."""
    import run_variant

    assert _VARIANT_YAML.exists(), f"variant manifest missing: {_VARIANT_YAML}"

    manifest = run_variant.load_manifest(_VARIANT_YAML)  # raises on any problem
    overrides = manifest.get("overrides") or {}

    # Every override key is a known PipelineConfig field (spec §3.5 "통과").
    valid_fields = run_variant._valid_config_fields()
    assert set(overrides.keys()) <= valid_fields, (
        f"unknown override fields: {sorted(set(overrides) - valid_fields)}"
    )

    assert overrides.get("news_trend_feature_enabled") is True, (
        "variant must set news_trend_feature_enabled: true"
    )
    assert "mu_vol_scaling_enabled" not in overrides, (
        "the template's mu_vol_scaling_enabled line must be removed (spec §2.4)"
    )
    assert manifest.get("label") == "exp_news_trend_feature"
