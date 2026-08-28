"""Unit smoke tests for run_variant's Phase 3 (targets) HMAC checkpoint helpers.

Complements the pinned acceptance suite (tests/acceptance/test_phase3_checkpoint.py).
House style: plain functions, synthetic data, private tempdir so the real
outputs/ tree is untouched. Written test-first (red until E1 lands).
"""

import dataclasses
import tempfile

import numpy as np
import pandas as pd

import run_variant
from src.config import PipelineConfig


def _targets():
    dates = pd.bdate_range("2015-01-02", periods=6)
    df = pd.DataFrame(
        np.random.default_rng(1).normal(0, 0.02, (6, 3)),
        index=dates,
        columns=["AAPL", "MSFT", "NVDA"],
    )
    df.iloc[0, :] = np.nan
    return df


def test_token_deterministic_and_field_sensitive():
    cfg = PipelineConfig()
    tok = run_variant.phase3_cache_token(cfg, upstream_token="UP")
    assert isinstance(tok, str) and tok
    assert run_variant.phase3_cache_token(PipelineConfig(), upstream_token="UP") == tok
    # a target/PCA field moves the token
    changed = dataclasses.replace(cfg, pca_n_remove=3)
    assert run_variant.phase3_cache_token(changed, upstream_token="UP") != tok
    # upstream chaining moves the token
    assert run_variant.phase3_cache_token(cfg, upstream_token="OTHER") != tok


def test_token_ignores_optimizer_fields():
    cfg = PipelineConfig()
    tok = run_variant.phase3_cache_token(cfg, upstream_token="UP")
    opt = dataclasses.replace(cfg, risk_aversion=99.0)
    assert run_variant.phase3_cache_token(opt, upstream_token="UP") == tok


def test_roundtrip_and_absent():
    targets = _targets()
    with tempfile.TemporaryDirectory() as tmp:
        assert run_variant.load_phase3_checkpoint("T", output_dir=tmp) is None
        run_variant.save_phase3_checkpoint(targets, token="T", output_dir=tmp)
        loaded = run_variant.load_phase3_checkpoint("T", output_dir=tmp)
        assert loaded is not None
        pd.testing.assert_frame_equal(loaded, targets, check_exact=True)
        assert run_variant.load_phase3_checkpoint("WRONG", output_dir=tmp) is None


# ---------------------------------------------------------------------------
# Point-in-time universe guard on the cache-reuse branch (§S11.4). The cached
# branch never constructs UniverseData, so its __init__ guard cannot fire —
# check_cached_universe is the only checkpoint for stale-universe checkpoints.
# ---------------------------------------------------------------------------
def _ns_data(n):
    import types
    return types.SimpleNamespace(tickers=[f"T{i}" for i in range(n)])


def test_check_cached_universe_none_is_inert():
    cfg = PipelineConfig()  # expected_universe_size defaults to None
    run_variant.check_cached_universe(_ns_data(3), cfg)  # no raise


def test_check_cached_universe_match_passes():
    import types

    from src.data_loader import TICKERS

    cfg = PipelineConfig(expected_universe_size=250)
    # Canonical composition in reversed order: the guard is order-insensitive
    # (a cached panel's column order is internally consistent).
    data = types.SimpleNamespace(tickers=list(reversed(TICKERS)))
    run_variant.check_cached_universe(data, cfg)  # no raise


def test_check_cached_universe_mismatch_raises():
    import pytest

    cfg = PipelineConfig(expected_universe_size=150)
    with pytest.raises(ValueError):
        run_variant.check_cached_universe(_ns_data(149), cfg)


def test_check_cached_universe_composition_mismatch_raises():
    """개수는 250으로 같아도 구성이 현행 TICKERS와 다르면 (예: 교체 이전
    슬레이트의 스테일 캐시) 캐시 재사용을 거부해야 한다 (2026-07-21)."""
    import types

    import pytest

    from src.data_loader import TICKERS

    cfg = PipelineConfig(expected_universe_size=250)
    swapped = list(TICKERS)
    swapped[0] = "ZZZZ_STALE"
    data = types.SimpleNamespace(tickers=swapped)
    with pytest.raises(ValueError, match="composition"):
        run_variant.check_cached_universe(data, cfg)


# ---------------------------------------------------------------------------
# Checkpoint-reuse reachability invariant (structural review 2026-08-27).
#
# The Phase 1/2/4 reuse branch in run_variant.run() has never executed: nothing
# in the repo writes those checkpoints, so load_checkpoint() always returns None
# and the full-pipeline branch always runs. That made the module docstring's
# "Reuse Phase 1/2/4 checkpoints if available" false, and made §S15 review
# finding #12 ("reuse validates nothing about the config that produced the
# checkpoints") a latent trap rather than a live defect.
#
# These tests pin the true state. If someone adds a writer they fail ON PURPOSE:
# reuse would then serve a panel/models built by whatever variant ran last, so
# finding #12 must be addressed in the same change.
# ---------------------------------------------------------------------------
_PHASE124 = ("phase1", "phase2", "phase4")


def test_no_phase124_checkpoint_writer_exists():
    """No module writes a Phase 1/2/4 checkpoint => the reuse branch is inert."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    writers = []
    for path in list(root.glob("*.py")) + list(root.rglob("src/**/*.py")) + list(
        root.rglob("scripts/**/*.py")
    ):
        text = path.read_text(encoding="utf-8", errors="replace")
        for phase in _PHASE124:
            # save_checkpoint("phase1", ...) / save_checkpoint('phase2', ...)
            if re.search(rf"save_checkpoint\(\s*['\"]{phase}['\"]", text):
                writers.append(f"{path.relative_to(root).as_posix()}:{phase}")
    assert writers == [], (
        "A Phase 1/2/4 checkpoint writer was added: "
        f"{writers}. run_variant.run()'s reuse branch is now REACHABLE and will "
        "serve the panel/models produced by whatever variant ran last, with no "
        "config validation (decision log §S15 review finding #12). Add a config "
        "fingerprint check to the checkpoint payload before enabling reuse, then "
        "update this test and the run_variant module docstring."
    )


def test_load_checkpoint_absent_returns_none_so_full_pipeline_runs():
    """The reuse branch is guarded by `if cp1 and cp2 and cp4:` — with no
    writer, every load returns None, so run() falls through to the full
    pipeline. Pins the guard shape as well as the absent-file behaviour."""
    import inspect
    import tempfile

    from src.backtest import load_checkpoint

    with tempfile.TemporaryDirectory() as tmp:
        for phase in _PHASE124:
            assert load_checkpoint(phase, output_dir=tmp) is None

    src_text = inspect.getsource(run_variant.run)
    assert "if cp1 and cp2 and cp4:" in src_text, (
        "reuse-branch guard shape changed; re-verify the inertness invariant"
    )
