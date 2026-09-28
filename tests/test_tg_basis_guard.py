"""§S22 A-04 (decision log §S23): orphaned pending TG-basis events."""
import json

from src.tg_basis_guard import annotate_basis_guard


def test_orphan_pending_is_released_but_reflagged_on_return(tmp_path):
    (tmp_path / "tg_basis_state.json").write_text(json.dumps({
        "schema_version": 1,
        "baseline": {"AAA": 1.10, "OLD": 1.10},
        "pending": {"OLD": {"previous": 1.10, "now": 1.60}},
    }))
    dq = {"currency": {"tg_px_ratio_median": {"AAA": 1.10, "NEW": 1.20},
                       "tg_px_ratio_suspect": {}}}
    assert annotate_basis_guard(tmp_path, dq) == {}
    assert dq["currency"]["tg_basis_guard_ok"] is True
    state = json.loads((tmp_path / "tg_basis_state.json").read_text(encoding="utf-8"))
    assert state["baseline"]["OLD"] == 1.10  # original normal basis kept
    dq = {"currency": {"tg_px_ratio_median": {"AAA": 1.10, "OLD": 1.60},
                       "tg_px_ratio_suspect": {}}}
    assert set(annotate_basis_guard(tmp_path, dq)) == {"OLD"}
    assert dq["currency"]["tg_basis_guard_ok"] is False
