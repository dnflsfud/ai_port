from run_selection_bias import _PKL_FALLBACK_ORDER


def test_fallback_order_puts_current_production_first():
    names = [p.parent.name for p in _PKL_FALLBACK_ORDER]
    # codex_causal_rank_65 is production (2026-07-11), iter15 is legacy challenger.
    assert names[0] == "codex_causal_rank_65"
    assert names[1] == "iter15_65tkr_reb21_vtg"
