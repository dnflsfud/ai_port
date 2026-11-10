# -*- coding: utf-8 -*-
"""§S28 operational fixes (decision log §S28 M7 / M8).

M8: the production gate must HOLD once the next grid rebalance date has passed in real time
(rebalance_overdue is row-count based and data_as_of_fresh_ok tolerates 3 sessions).
M7: the scheduled wrapper leaves a history line even when the run dies; the primary bat refuses
a nearly full disk and discards paper-ledger changes that a previous run never published.
"""
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.validate_portfolio_bundles import evaluate_production

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ M8
def _gate(**overrides):
    """Every binding check clear; registry built 2026-11-05 02:30 UTC = 11-04 21:30 ET, so the
    11-04 XNYS session has closed and data_as_of 2026-11-03 is 1 session old."""
    record = {
        "data_as_of": "2026-11-03", "rebalance_calendar": "XNYS", "rows_until_next_rebalance": 2,
        "git_dirty": False,
        "_risk_guardrails": {"estimated_te_breached": False, "top_name_active_risk_breached": False,
                             "top_sector_active_risk_breached": False},
        "_model_quality": {"split_audit": [{"prediction_date": "2026-10-01"}], "events": []},
        "performance": {"tracking_error": 0.03, "data_quality": {
            "tail_ffill_days": 0, "max_tail_ffill_days": 10,
            "currency": {"tg_px_ratio_suspect": {}, "tg_basis_events_consistent_ok": True}}},
        "_option_vol_cov": {"enabled": False, "applied": False},
        "_operations": {"latest_rebalance_used_fallback": False, "turnover_two_way_latest": 0.0,
                        "max_two_way_turnover": 0.15},
    }
    record.update(overrides)
    return evaluate_production(record, as_of_utc=datetime(2026, 11, 5, 2, 30, tzinfo=timezone.utc))


def test_production_gate_holds_when_the_next_rebalance_is_already_due():
    # next grid rebalance = 1 session after data_as_of = 11-04, which has closed: due but unpublished
    due = _gate(rows_until_next_rebalance=1)
    assert due["status"] == "HOLD"
    assert due["values"]["data_age_sessions"] == 1
    assert due["checks"]["data_as_of_fresh_ok"] is True            # the data-age check alone passes it
    assert due["checks"]["rebalance_schedule_current_ok"] is False
    assert due["values"]["rows_until_next_rebalance"] == 1

    not_due = _gate(rows_until_next_rebalance=2)                   # next rebalance = 11-05, still open
    assert not_due["checks"]["rebalance_schedule_current_ok"] is True
    assert not_due["status"] == "PRODUCTION"


def test_rebalance_schedule_check_fails_closed_on_missing_or_invalid_input():
    for bad in (None, True, 0, "2"):
        gate = _gate(rows_until_next_rebalance=bad)
        assert gate["checks"]["rebalance_schedule_current_ok"] is None, bad
        assert gate["status"] == "HOLD", bad
    assert _gate(data_as_of=None)["checks"]["rebalance_schedule_current_ok"] is None
    assert evaluate_production({})["checks"]["rebalance_schedule_current_ok"] is None


# ------------------------------------------------------------------ M7
@pytest.mark.skipif(os.name != "nt", reason="cmd.exe batch file")
def test_scheduled_wrapper_logs_a_start_line_before_the_run_and_an_exit_line_after(tmp_path):
    # On 2026-11-09 the run died with a full disk and the only history line (written after the
    # run) was lost with it; a `start` without a matching `exit` now shows a run that died.
    shutil.copy(ROOT / "run_and_upload_scheduled.bat", tmp_path / "run_and_upload_scheduled.bat")
    (tmp_path / "run_and_upload.bat").write_bytes(
        b'@echo off\r\n'
        b'copy /y "%~dp0logs\\scheduled_run_history.log" "%~dp0history_during_run.txt" >nul\r\n'
        b'exit /b 3\r\n')
    done = subprocess.run(["cmd", "/c", str(tmp_path / "run_and_upload_scheduled.bat")], cwd=tmp_path,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 3
    lines = (tmp_path / "logs" / "scheduled_run_history.log").read_text().splitlines()
    ts = lines[0].split(" ")[0]
    assert lines == [f"{ts} start log=scheduled_run_{ts}.log", f"{ts} exit=3 log=scheduled_run_{ts}.log"]
    assert (tmp_path / "history_during_run.txt").read_text().splitlines() == lines[:1]
    assert (tmp_path / "logs" / "scheduled_run_last.log").exists()


def _scratch_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    shutil.copy(ROOT / "run_and_upload.bat", repo / "run_and_upload.bat")

    def git(*args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                       cwd=repo, check=True, capture_output=True)

    def run_bat(**extra):
        env = {k: v for k, v in os.environ.items() if k != "AI_PORT_MIN_FREE_GB"}
        env.update(AI_PORT_NO_DASHBOARD="1", **extra)
        return subprocess.run(["cmd", "/c", str(repo / "run_and_upload.bat")], cwd=repo,
                              capture_output=True, text=True, env=env, timeout=300)

    git("init", "-q", "-b", "main")
    return repo, git, run_bat


@pytest.mark.skipif(os.name != "nt", reason="cmd.exe batch file")
def test_primary_bat_refuses_to_start_on_a_nearly_full_disk(tmp_path):
    repo, git, run_bat = _scratch_repo(tmp_path)
    git("add", "-A")
    git("commit", "-q", "-m", "init")

    full = run_bat(AI_PORT_MIN_FREE_GB="1000000000")
    assert "ERROR: less than 1000000000 GB free" in full.stdout
    assert "[2/10]" not in full.stdout
    assert full.returncode == 1

    default = run_bat()       # 10 GB default: passes the guard, then stops at the test step
    assert "[2/10]" in default.stdout  # (this scratch repo has no tests)
    assert "GB free" not in default.stdout


@pytest.mark.skipif(os.name != "nt", reason="cmd.exe batch file")
def test_primary_bat_discards_unpublished_paper_ledger_changes(tmp_path):
    # M7(b): a run that fails at validation/commit leaves outputs\paper_ledger modified, and a
    # rebalance is frozen by the first run that PUBLISHES it - the next run starts from HEAD.
    repo, git, run_bat = _scratch_repo(tmp_path)
    state = repo / "outputs" / "paper_ledger" / "state.json"
    state.parent.mkdir(parents=True)
    state.write_text("A")
    git("add", "-A")
    git("commit", "-q", "-m", "init")
    state.write_text("B")     # unpublished; outputs/ is excluded from the dirty-tree check

    done = run_bat(AI_PORT_MIN_FREE_GB="0")
    assert "[2/10]" in done.stdout
    assert state.read_text() == "A"

    # a run that dies between `git add -A` and `git commit` leaves the change STAGED
    state.write_text("C")
    git("add", "-A")
    staged = run_bat(AI_PORT_MIN_FREE_GB="0")
    assert "[2/10]" in staged.stdout
    assert state.read_text() == "A"
