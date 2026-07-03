"""alpha_attribution config fields: OFF by default, no perturbation of existing defaults."""
from src.config import PipelineConfig, DEFAULT_CONFIG


def test_attribution_off_by_default():
    c = PipelineConfig()
    assert c.alpha_attribution_enabled is False
    assert c.alpha_attribution_n_dates == 8


def test_attribution_fields_do_not_perturb_existing_defaults():
    c = PipelineConfig()
    # existing OFF-default precedent unchanged
    assert c.bm_proportional_cap_enabled == DEFAULT_CONFIG.bm_proportional_cap_enabled
    assert c.max_active_share == DEFAULT_CONFIG.max_active_share
    assert c.value_trap_gate_enabled == DEFAULT_CONFIG.value_trap_gate_enabled
