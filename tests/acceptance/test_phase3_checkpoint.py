"""Acceptance tests for SPEC E1 — Phase 3 (targets) HMAC checkpoint.

Written BEFORE implementation (test-first). Run from ai_port with PYTHONPATH=. :

    C:/Users/westl/PycharmProjects/pythonProject/venv_vf_new/Scripts/python.exe \
        -m pytest tests/acceptance/test_phase3_checkpoint.py -v

House-style idioms: plain pytest functions (no fixtures), synthetic data only,
fast (no full backtest — 2-run byte-parity is the verifier protocol, not a unit
test here). Expected values computed INDEPENDENTLY. Every filesystem artifact is
written under a private tempfile dir so the real outputs/ tree is never touched.

------------------------------------------------------------------------------
PINNED IMPORT SURFACE (this is the implementation contract for E1)
------------------------------------------------------------------------------
Module: run_variant  (spec E1 modification scope = run_variant.py; internally it
may delegate to src.backtest's existing HMAC machinery — save_checkpoint /
_sign_file — but the three NEW names below MUST be importable from run_variant).

    run_variant.phase3_cache_token(config, upstream_token: str = "") -> str
        Deterministic token derived from the target/PCA config fields that
        build_targets() actually reads (src/target_engine.py:279-346):
            pca_n_remove, pca_components, pca_lookback, forward_horizon,
            multi_horizon_targets_enabled, multi_horizon_weights,
            regime_pca_weighted_enabled
        plus the upstream_token (Phase 1/2 checkpoint hash) for chaining.
        Same (config, upstream) -> identical string. A change to ANY target/PCA
        field OR to upstream_token -> different string. Optimizer / Phase-5/6
        fields (risk_aversion, max_weight, turnover_penalty, ...) do NOT change
        it — PCA specific-return targets are independent of the MVO stage.

    run_variant.save_phase3_checkpoint(targets, token, output_dir="outputs") -> None
        HMAC-sign + persist the targets DataFrame tagged with `token`.

    run_variant.load_phase3_checkpoint(token, output_dir="outputs")
                                                    -> Optional[pd.DataFrame]
        Return a bit-identical targets DataFrame iff a checkpoint exists whose
        stored token == `token` AND its signature verifies. Otherwise return
        None — GRACEFUL fallback to the recompute path, NEVER raising — for:
        absent file, token mismatch, signature mismatch, missing .sig, or a
        corrupted pickle. (This differs from src.backtest.load_checkpoint, which
        RAISES on a missing/mismatched signature; Phase 3 recompute is always
        correct + cheap, so corruption must degrade to recompute, not abort.)

------------------------------------------------------------------------------
ACCEPTANCE-CRITERION -> TEST MAPPING
------------------------------------------------------------------------------
(E1-a) Invalidation correctness — same config => same token; each target/PCA
       field change => token changes; optimizer-only field change => token
       unchanged; upstream (Phase 1/2) change => token changes.
        -> test_token_is_deterministic
        -> test_token_changes_on_each_target_pca_field
        -> test_token_stable_across_optimizer_only_fields
        -> test_token_chains_to_upstream
(E1-b) Save->load round-trip is bit-identical (values, dtype, index, columns).
        -> test_roundtrip_is_bit_identical
(E1-c) HMAC mismatch / corrupted pickle / missing signature / wrong token =>
       recompute fallback (load returns None, never raises).
        -> test_corrupted_pickle_returns_none
        -> test_missing_signature_returns_none
        -> test_wrong_token_invalidates
(E1-d) Checkpoint absent => load returns None so the caller takes the compute
       path.
        -> test_absent_checkpoint_returns_none

Pre-implementation, EVERY test fails at the `_api()` / `_build_report`-style
import step (the three names do not yet exist on run_variant) — the correct
TDD red state. `pytest --collect-only` stays clean because the not-yet-existing
symbols are imported inside the test bodies, never at module import time.
------------------------------------------------------------------------------
"""

import dataclasses
import importlib
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PipelineConfig


# ---------------------------------------------------------------------------
# Deferred import of the pinned surface. Kept out of module top-level so that
# `pytest --collect-only` succeeds before the symbols exist; each test that
# calls this fails (red) with a clear AttributeError/ImportError until E1 lands.
# ---------------------------------------------------------------------------
def _api():
    rv = importlib.import_module("run_variant")
    return (
        rv.phase3_cache_token,
        rv.save_phase3_checkpoint,
        rv.load_phase3_checkpoint,
    )


# Target/PCA fields build_targets() actually reads. Each must move the token.
_TARGET_PCA_PERTURBATIONS = {
    "pca_n_remove": 3,                       # default 2
    "pca_components": 6,                      # default 5
    "pca_lookback": 504,                     # default 252
    "forward_horizon": 10,                   # default 20
    "regime_pca_weighted_enabled": True,     # default False
    "multi_horizon_targets_enabled": True,   # default False
}

# Optimizer / Phase-5/6 fields — inside SAFE_FOR_CACHE_REUSE; targets do not
# depend on them, so the Phase 3 token MUST be invariant to these.
_OPTIMIZER_ONLY_PERTURBATIONS = {
    "risk_aversion": 99.0,
    "max_weight": 0.123,
    "turnover_penalty": 0.456,
}


def _small_targets():
    """A tiny, realistic targets panel: float64, DatetimeIndex, leading-NaN row
    (specific-return targets are undefined until the PCA lookback fills)."""
    dates = pd.bdate_range("2015-01-02", periods=8)
    tickers = ["AAPL", "MSFT", "NVDA", "AMZN"]
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        rng.normal(0.0, 0.02, (8, 4)), index=dates, columns=tickers
    )
    df.iloc[0, :] = np.nan
    return df


# ===========================================================================
# (E1-a) Invalidation correctness
# ===========================================================================
def test_token_is_deterministic():
    """Same config + same upstream => byte-identical token (enables reuse)."""
    phase3_cache_token, _, _ = _api()
    cfg = PipelineConfig()
    t1 = phase3_cache_token(cfg, upstream_token="UP")
    t2 = phase3_cache_token(cfg, upstream_token="UP")
    assert isinstance(t1, str) and len(t1) > 0
    assert t1 == t2
    # A freshly-constructed equal config gives the same token (no identity dep).
    assert phase3_cache_token(PipelineConfig(), upstream_token="UP") == t1


def test_token_changes_on_each_target_pca_field():
    """Perturbing ANY target/PCA field the target engine reads must change the
    token (constraint #3: the token includes the real dependent fields)."""
    phase3_cache_token, _, _ = _api()
    base = PipelineConfig()
    base_tok = phase3_cache_token(base, upstream_token="UP")
    for field, value in _TARGET_PCA_PERTURBATIONS.items():
        perturbed = dataclasses.replace(base, **{field: value})
        assert phase3_cache_token(perturbed, upstream_token="UP") != base_tok, (
            f"token must change when target/PCA field '{field}' changes"
        )

    # multi_horizon_weights only matters when the blend is enabled; changing the
    # weight map (with the blend ON) must also move the token.
    on_a = dataclasses.replace(
        base, multi_horizon_targets_enabled=True, multi_horizon_weights={5: 1.0}
    )
    on_b = dataclasses.replace(
        base,
        multi_horizon_targets_enabled=True,
        multi_horizon_weights={5: 1.0, 10: 1.0},
    )
    ta = phase3_cache_token(on_a, upstream_token="UP")
    tb = phase3_cache_token(on_b, upstream_token="UP")
    assert ta != tb
    assert ta != base_tok and tb != base_tok


def test_token_stable_across_optimizer_only_fields():
    """Optimizer / Phase-5/6 overrides do NOT invalidate the Phase 3 cache — PCA
    specific-return targets are independent of the MVO stage. This is what makes
    the checkpoint reusable across production optimizer sweeps."""
    phase3_cache_token, _, _ = _api()
    base = PipelineConfig()
    base_tok = phase3_cache_token(base, upstream_token="UP")
    for field, value in _OPTIMIZER_ONLY_PERTURBATIONS.items():
        perturbed = dataclasses.replace(base, **{field: value})
        assert phase3_cache_token(perturbed, upstream_token="UP") == base_tok, (
            f"optimizer-only field '{field}' must NOT change the Phase 3 token"
        )


def test_token_chains_to_upstream():
    """Same config but a different upstream (Phase 1/2) hash => different token,
    so a rebuilt input panel forces Phase 3 recompute (constraint #3 chaining)."""
    phase3_cache_token, _, _ = _api()
    cfg = PipelineConfig()
    t_u1 = phase3_cache_token(cfg, upstream_token="UPSTREAM_1")
    t_u2 = phase3_cache_token(cfg, upstream_token="UPSTREAM_2")
    assert t_u1 != t_u2


# ===========================================================================
# (E1-b) Bit-identical save -> load round-trip
# ===========================================================================
def test_roundtrip_is_bit_identical():
    """Persist then reload under the same token => the DataFrame comes back
    exactly (values incl. NaN, dtype, index, column names)."""
    _, save_phase3_checkpoint, load_phase3_checkpoint = _api()
    targets = _small_targets()
    with tempfile.TemporaryDirectory() as tmp:
        save_phase3_checkpoint(targets, token="TOK_RT", output_dir=tmp)
        loaded = load_phase3_checkpoint("TOK_RT", output_dir=tmp)

    assert loaded is not None
    pd.testing.assert_frame_equal(
        loaded, targets, check_exact=True, check_dtype=True, check_names=True
    )
    assert loaded.index.equals(targets.index)
    assert loaded.columns.equals(targets.columns)
    assert str(loaded.index.dtype) == str(targets.index.dtype)


# ===========================================================================
# (E1-c) Corruption / tampering => graceful recompute fallback (None, no raise)
# ===========================================================================
def _pkl_files(root):
    return list(Path(root).rglob("*.pkl"))


def _sig_files(root):
    return list(Path(root).rglob("*.sig"))


def test_corrupted_pickle_returns_none():
    """A tampered checkpoint pickle => HMAC mismatch => load returns None
    (recompute path), it must NOT raise."""
    _, save_phase3_checkpoint, load_phase3_checkpoint = _api()
    targets = _small_targets()
    with tempfile.TemporaryDirectory() as tmp:
        save_phase3_checkpoint(targets, token="TOK_C", output_dir=tmp)
        pkls = _pkl_files(tmp)
        assert pkls, "implementation must write a .pkl checkpoint under output_dir"
        for p in pkls:
            p.write_bytes(b"\x00not-a-valid-pickle\xff" * 8)
        result = load_phase3_checkpoint("TOK_C", output_dir=tmp)  # must not raise
    assert result is None


def test_missing_signature_returns_none():
    """Deleting the .sig => load returns None (graceful), unlike the strict
    src.backtest.load_checkpoint which raises. Phase 3 must degrade quietly."""
    _, save_phase3_checkpoint, load_phase3_checkpoint = _api()
    targets = _small_targets()
    with tempfile.TemporaryDirectory() as tmp:
        save_phase3_checkpoint(targets, token="TOK_S", output_dir=tmp)
        sigs = _sig_files(tmp)
        assert sigs, "implementation must HMAC-sign the checkpoint (.sig file)"
        for s in sigs:
            s.unlink()
        result = load_phase3_checkpoint("TOK_S", output_dir=tmp)  # must not raise
    assert result is None


def test_wrong_token_invalidates():
    """A checkpoint written under one token is NOT served for a different token
    (config/upstream changed => recompute)."""
    _, save_phase3_checkpoint, load_phase3_checkpoint = _api()
    targets = _small_targets()
    with tempfile.TemporaryDirectory() as tmp:
        save_phase3_checkpoint(targets, token="TOK_OLD", output_dir=tmp)
        assert load_phase3_checkpoint("TOK_NEW", output_dir=tmp) is None
        # Sanity: the matching token still serves it (guards against a loader
        # that returns None unconditionally, which would pass the line above).
        served = load_phase3_checkpoint("TOK_OLD", output_dir=tmp)
    assert served is not None
    pd.testing.assert_frame_equal(served, targets, check_exact=True)


# ===========================================================================
# (E1-d) Absent checkpoint => None (caller takes the compute path)
# ===========================================================================
def test_absent_checkpoint_returns_none():
    """No checkpoint on disk => load returns None (never raises), so run_variant
    falls through to build_targets()."""
    _, _, load_phase3_checkpoint = _api()
    with tempfile.TemporaryDirectory() as tmp:
        assert load_phase3_checkpoint("ANY_TOKEN", output_dir=tmp) is None
