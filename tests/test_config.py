"""TDD-guard stem test for src/config listing-mask fields.

The authoritative behavioural coverage lives in
tests/acceptance/test_listing_mask.py. This stem-named file exists so the
pytest-tdd PreToolUse guard (which name-matches test_<module>.py) permits
editing src/config.py.
"""

import json
import shutil
import subprocess
from datetime import datetime, timezone

import pytest

from src.config import (
    PipelineConfig,
    _git_dirty,
    _git_hash,
    data_vintage_fingerprint,
    dump_experiment_manifest,
)


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
        # S11 expansion — IPO listings
        "DELL": "2018-12-28",
        "ABNB": "2020-12-10",
        "UMG": "2021-09-21",
        # S11 expansion — corporate-action continuity masks (post-event only)
        "GE": "2024-04-02",
        "TT": "2020-03-02",
        "BN": "2022-12-12",
        # S11.4 coverage audit — unregistered leading-constant backfills
        "ANET": "2014-06-06",
        "RACE": "2015-10-21",
        "LITE": "2015-07-27",
        "VST": "2016-10-05",
        "SPOT": "2018-04-03",
        "VRT": "2018-08-01",
        # S13.3 — PIT-contract §6 pre-registered IPOs (audit-confirmed)
        "TTD": "2016-09-21",
        "UBER": "2019-05-10",
        "CRWD": "2019-06-12",
        "DDOG": "2019-09-19",
        "DASH": "2020-12-09",
        "RBLX": "2021-03-10",
        # S13.3 audit-surfaced spin backfills
        "PYPL": "2015-07-07",
        "KEYS": "2014-10-21",
        # S13.3 corporate-action continuity masks
        "WDC": "2025-02-24",
        "COF": "2025-05-19",
        # S14 250-name expansion — IPO constant ghosts (§S14.2)
        "TEAM": "2015-12-10",
        "ZS": "2018-03-16",
        "NET": "2019-09-13",
        "SNOW": "2020-09-16",
        "APP": "2021-04-15",
        "RDDT": "2024-03-21",
        # S14 moving-predecessor continuity masks (TKO=WWE, HWM=Arconic)
        "TKO": "2023-09-12",
        "HWM": "2020-04-01",
    }


def test_standard_idio_vol_feature_is_explicitly_opt_in():
    c = PipelineConfig()
    assert c.standard_idio_vol_feature_enabled is False
    assert c.vol_quality_tilt_vol_feature == "idio_vol_63d"

    enabled = PipelineConfig(
        standard_idio_vol_feature_enabled=True,
        vol_quality_tilt_vol_feature="idio_vol_capm_63d",
    )
    assert enabled.standard_idio_vol_feature_enabled is True
    assert enabled.vol_quality_tilt_vol_feature == "idio_vol_capm_63d"


def test_spain_market_suffix_maps_to_eur():
    from src.data_loader import MARKET_TO_CURRENCY, FX_QUOTE_SPECS
    assert MARKET_TO_CURRENCY["SM"] == "EUR"
    # EUR conversion spec must already exist — S11 adds no new FX pair.
    assert "EUR" in FX_QUOTE_SPECS


def test_milan_market_suffix_maps_to_eur():
    from src.data_loader import MARKET_TO_CURRENCY, FX_QUOTE_SPECS
    # §S14: new exchange code IM (Milan — UCG, ENEL), no new FX pair.
    assert MARKET_TO_CURRENCY["IM"] == "EUR"
    assert "EUR" in FX_QUOTE_SPECS


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


# ---------------------------------------------------------------------------
# 데이터 빈티지 지문 (구조 리뷰 2026-08-31, O7). 유효 빈티지 =
# (ai_signal_data.xlsx, Index.xlsx) mtime 쌍(§S13.47) — 산출물만 보고 두 런이
# 같은 빈티지인지 검증할 수 있어야 한다.
# ---------------------------------------------------------------------------
def _stamp(path):
    return datetime.fromtimestamp(
        path.stat().st_mtime, timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_data_vintage_fingerprint_records_mtime_and_tolerates_absent_file(tmp_path):
    workbook = tmp_path / "ai_signal_data.xlsx"
    workbook.write_bytes(b"x" * 17)
    absent = tmp_path / "Index.xlsx"  # never created
    fp = data_vintage_fingerprint(
        PipelineConfig(data_path=str(workbook), fx_source_path=str(absent))
    )

    assert fp["data_path"] == str(workbook)
    assert fp["data_size_bytes"] == 17
    assert fp["data_mtime_utc"] == _stamp(workbook)
    # 없는 파일은 예외가 아니라 None (지문이 런을 깨뜨리면 안 된다)
    assert fp["fx_source_path"] == str(absent)
    assert fp["fx_mtime_utc"] is None
    assert fp["fx_size_bytes"] is None


def test_manifest_adds_data_vintage_and_keeps_existing_keys(tmp_path):
    workbook = tmp_path / "wb.xlsx"
    workbook.write_bytes(b"y" * 5)
    cfg = PipelineConfig(
        data_path=str(workbook), fx_source_path=str(tmp_path / "none.xlsx")
    )
    path = dump_experiment_manifest(config=cfg, output_dir=str(tmp_path / "out"))
    manifest = json.loads(path.read_text(encoding="utf-8"))

    for key in ("timestamp_utc", "git_hash", "git_dirty", "config"):
        assert key in manifest
    assert manifest["config"]["data_path"] == str(workbook)
    assert manifest["data_vintage"] == data_vintage_fingerprint(cfg)
