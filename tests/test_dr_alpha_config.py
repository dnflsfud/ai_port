"""Task 1 — CS-DR-Alpha config fields: present, off/no-op, off-path parity."""
from src.config import PipelineConfig, DEFAULT_CONFIG


def test_dr_alpha_defaults_present_and_off():
    c = PipelineConfig()
    assert c.dr_alpha_enabled is False
    assert c.dr_alpha_arch == "linear"
    assert c.dr_alpha_residual is True
    assert c.dr_alpha_gamma == 1.0
    assert c.dr_alpha_embargo == 20
    assert c.dr_alpha_turnover_lambda >= 0.0
    assert c.dr_alpha_seed == 42
    assert c.dr_alpha_hidden == 16
    assert c.dr_alpha_lr > 0.0
    assert c.dr_alpha_epochs > 0
    assert c.dr_alpha_l2 >= 0.0
    assert c.dr_alpha_warm_start is True
    assert c.dr_alpha_val_months == 6


def test_dr_alpha_offpath_parity():
    # New fields must not perturb any existing default value.
    c = PipelineConfig()
    assert c.rebalance_freq == DEFAULT_CONFIG.rebalance_freq
    assert c.max_te_annual == DEFAULT_CONFIG.max_te_annual
    assert c.retrain_freq == DEFAULT_CONFIG.retrain_freq
