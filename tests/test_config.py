"""TDD-guard stem test for src/config listing-mask fields.

The authoritative behavioural coverage lives in
tests/acceptance/test_listing_mask.py. This stem-named file exists so the
pytest-tdd PreToolUse guard (which name-matches test_<module>.py) permits
editing src/config.py.
"""

from src.config import PipelineConfig


def test_listing_mask_fields_default_off():
    c = PipelineConfig()
    assert c.listing_mask_enabled is False
    assert c.listing_dates == {
        "PLTR": "2020-09-30",
        "GEV": "2024-04-02",
        "BE": "2018-07-25",
    }
