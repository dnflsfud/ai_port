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
