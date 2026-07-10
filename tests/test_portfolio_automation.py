from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
