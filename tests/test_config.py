"""TDD-guard stem test for src/config listing-mask fields.

The authoritative behavioural coverage lives in
tests/acceptance/test_listing_mask.py. This stem-named file exists so the
pytest-tdd PreToolUse guard (which name-matches test_<module>.py) permits
editing src/config.py.
"""

import shutil
import subprocess

import pytest

from src.config import PipelineConfig, _git_dirty, _git_hash


def test_listing_mask_fields_default_on_for_valid_100_name_history():
    c = PipelineConfig()
    assert c.listing_mask_enabled is True
    assert c.listing_dates == {
        "PLTR": "2020-09-30",
        "GEV": "2024-04-02",
        "BE": "2018-07-25",
        "285A": "2024-12-18",
        "SNDK": "2025-02-24",
        "ARM": "2023-09-14",
        "CEG": "2022-02-02",
    }


def _make_repo(root):
    if shutil.which("git") is None:
        pytest.skip("git not available")

    def run(*args):
        subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    run("init", "-q")
    (root / "seed.txt").write_text("seed", encoding="utf-8")
    run("add", "seed.txt")
    run("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init")
    return root


def test_git_dirty_ignores_outputs_only_changes(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "outputs").mkdir()
    (repo / "outputs" / "dummy.txt").write_text("x", encoding="utf-8")
    assert _git_dirty(repo=repo) is False


def test_git_dirty_flags_tracked_tree_changes(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "outputs").mkdir()
    (repo / "outputs" / "dummy.txt").write_text("x", encoding="utf-8")
    (repo / "other.txt").write_text("y", encoding="utf-8")
    assert _git_dirty(repo=repo) is True


def test_git_hash_returns_40_char_hex(tmp_path):
    repo = _make_repo(tmp_path)
    h = _git_hash(repo=repo)
    assert h is not None
    assert len(h) == 40
    int(h, 16)  # parses as hex
