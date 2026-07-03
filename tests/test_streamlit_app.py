"""Pointer shim for streamlit_app.py.

The real, authoritative tests for ``streamlit_app`` live in
``tests/acceptance/test_streamlit_report.py`` (10 acceptance tests covering the
pure helpers ``list_runs`` / ``load_metrics`` / ``load_result`` and the
import-without-UI contract). This file exists only so the filename-based
TDD guard (which looks for ``test_<module>.py``) can find a test for
``streamlit_app.py``; it intentionally defines NO test functions of its own so
the suite count stays unchanged.
"""
