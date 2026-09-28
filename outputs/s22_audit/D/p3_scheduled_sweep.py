"""D-probe 3 (read-only git): the scheduled job publishes uncommitted source code
and the production bundle it builds from that dirty tree passes validation.

Run from the ai_port repo root. Only read-only git commands are used.
"""
import json
import re
import subprocess
from pathlib import Path


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout


code_re = re.compile(r"^(src/|scripts/|tests/|variants/|run_variant\.py|requirements)")
for sha in ("c28ecf9", "bf6e055"):
    subject = git("log", "-1", "--format=%s", sha).strip()
    files = [f for f in git("show", "--name-only", "--format=", sha).splitlines() if f]
    code = [f for f in files if code_re.match(f)]
    manifest = json.loads(git("show", f"{sha}:outputs/codex_causal_rank_65/experiment_manifest.json"))
    parent = git("rev-parse", f"{sha}^").strip()
    print(f"{sha} subject={subject!r} code_files_committed={len(code)} "
          f"run_git_dirty={manifest.get('git_dirty')} "
          f"run_git_hash_is_parent={manifest.get('git_hash') == parent}")
    print("   e.g.", code[:4])

bat = Path("run_and_upload.bat").read_text(encoding="utf-8", errors="replace")
print("bat stages whole tree (git add -A):", "git add -A" in bat)
validator = Path("scripts/validate_portfolio_bundles.py").read_text(encoding="utf-8")
print("validator checks git_dirty:", "git_dirty" in validator)
