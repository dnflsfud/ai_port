import inspect
import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CLEAN_TREE_CHECK = 'git status --porcelain -- . ":(exclude)outputs"'


def test_primary_bat_preserves_legacy_order_and_validates_before_git():
    text = (ROOT / "run_and_upload.bat").read_text(encoding="utf-8")
    commands = [
        "-m pytest tests/ -q",
        "run_variant.py --variant variants\\iter15_65tkr_reb21_vtg.yaml --no-cache",
        "scripts\\export_operating_data.py",
        "run_variant.py --variant variants\\codex_causal_rank_65.yaml --no-cache",
        "scripts\\validate_portfolio_bundles.py",
        "git add -A",
        "git push -u origin main",
    ]
    positions = [text.index(command) for command in commands]
    assert positions == sorted(positions)


def test_scheduled_bat_remains_a_thin_wrapper():
    text = (ROOT / "run_and_upload_scheduled.bat").read_text(encoding="utf-8")
    assert 'set "AI_PORT_NO_DASHBOARD=1"' in text
    assert 'call "%~dp0run_and_upload.bat"' in text
    assert "codex_causal_rank_65" not in text


def test_primary_bat_refuses_dirty_tree_before_tests_and_before_git_add():
    # §S22 D-01: the run computes production from the working tree and then
    # `git add -A` publishes it, so code outside outputs/ must be committed.
    from src.config import _git_dirty

    text = (ROOT / "run_and_upload.bat").read_text(encoding="utf-8")
    first = text.index(CLEAN_TREE_CHECK)
    second = text.index(CLEAN_TREE_CHECK, first + 1)
    assert first < text.index("-m pytest tests/ -q")
    assert text.index("run_variant.py") > first
    assert second < text.index("git add -A")
    # same pathspec as the manifest's git_dirty flag
    assert '":(exclude)outputs"' in inspect.getsource(_git_dirty)


def test_primary_bat_python_steps_fail_on_any_nonzero_exit():
    # §S22 D-05: `if errorlevel 1` means >= 1, so a native crash (negative
    # NTSTATUS exit code) was treated as success.
    lines = (ROOT / "run_and_upload.bat").read_text(encoding="utf-8").splitlines()
    steps = [i for i, line in enumerate(lines) if line.startswith('"%PY%"')]
    assert len(steps) == 7
    for i in steps:
        assert lines[i + 1].startswith("if !errorlevel! neq 0 ("), lines[i + 1]


@pytest.mark.skipif(os.name != "nt", reason="cmd.exe batch file")
def test_primary_bat_guard_blocks_dirty_repo_and_ignores_outputs(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    shutil.copy(ROOT / "run_and_upload.bat", repo / "run_and_upload.bat")

    def git(*args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                       cwd=repo, check=True, capture_output=True)

    def run_bat():
        env = dict(os.environ, AI_PORT_NO_DASHBOARD="1")
        return subprocess.run(["cmd", "/c", str(repo / "run_and_upload.bat")], cwd=repo,
                              capture_output=True, text=True, env=env, timeout=300)

    git("init", "-q", "-b", "main")
    git("add", "-A")
    git("commit", "-q", "-m", "init")
    (repo / "outputs").mkdir()
    (repo / "outputs" / "metrics.json").write_text("{}")
    (repo / "stray.py").write_text("x = 1\n")

    dirty = run_bat()
    assert dirty.returncode == 1
    assert "uncommitted" in dirty.stdout
    assert "[2/10]" not in dirty.stdout

    (repo / "stray.py").unlink()
    clean = run_bat()  # only outputs/ changed: passes the guard, then stops at
    assert "[2/10]" in clean.stdout  # the test step (this scratch repo has no tests)
    assert "uncommitted" not in clean.stdout
    assert clean.returncode == 1


def test_dashboard_bat_keeps_existing_stages_before_challenger():
    text = (ROOT / "run_dashboard.bat").read_text(encoding="utf-8")
    positions = [text.index(x) for x in (
        '"%PY%" run_pictet_adoption.py',
        '"%PY%" scripts\\data_quality_report.py',
        '"%PY%" scripts\\export_operating_data.py',
        '"%PY%" run_variant.py --variant variants\\codex_causal_rank_65.yaml --no-cache',
        '"%PY%" scripts\\validate_portfolio_bundles.py',
        '"%PY%" -m streamlit run streamlit_app.py',
    )]
    assert positions == sorted(positions)
