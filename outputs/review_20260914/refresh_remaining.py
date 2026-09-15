"""Run the remaining validation stages sequentially with one verified Excel parse.

The cache is process-local, keyed by the FULL source-file identity including
SHA-256, and each consumer receives independent deep copies. No checkpoint or
model reuse is enabled. run_variant and export still verify their own inputs.
"""
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import traceback

from src import data_loader
from src.run_integrity import _file_identity
from run_variant import run
from scripts.export_operating_data import main as export

OUT = Path(__file__).resolve().parent
original_loader = data_loader.load_all_sheets
cached_identity = None
cached_raw = None


def verified_loader(path):
    global cached_identity, cached_raw
    identity = _file_identity(path, required=True)
    if identity != cached_identity:
        raw = original_loader(path)
        if _file_identity(path, required=True) != identity:
            raise ValueError("Workbook changed during Excel parsing")
        cached_raw, cached_identity = raw, identity
    return {name: frame.copy(deep=True) for name, frame in cached_raw.items()}


data_loader.load_all_sheets = verified_loader
stages = [
    ("challenger_run", lambda: run(Path("variants/iter15_65tkr_reb21_vtg.yaml"), no_cache=True)),
    ("production_export", lambda: export([
        "--variant", "variants/codex_causal_rank_65.yaml",
        "--operating-dir", "outputs/operating_codex_causal_rank_65"])),
    ("challenger_export", lambda: export([
        "--variant", "variants/iter15_65tkr_reb21_vtg.yaml",
        "--operating-dir", "outputs/operating"])),
]
for name, action in stages:
    print(f"START {name}", flush=True)
    with (OUT / f"{name}.log").open("w", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                result = action()
                if result not in (None, 0):
                    raise RuntimeError(f"{name} returned {result}")
            except BaseException:
                traceback.print_exc()
                raise
    print(f"DONE {name}", flush=True)
