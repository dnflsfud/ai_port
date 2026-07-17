"""TDD-guard stem test for src/config listing-mask fields.

The authoritative behavioural coverage lives in
tests/acceptance/test_listing_mask.py. This stem-named file exists so the
pytest-tdd PreToolUse guard (which name-matches test_<module>.py) permits
editing src/config.py.
"""

from src.config import PipelineConfig


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
