"""Task 2 — CS-DR-Alpha learning primitive (src/rl/dr_alpha.py)."""
import numpy as np
import torch

from src.rl.dr_alpha import (
    DRAlphaPolicy, xs_zscore, active_weights,
    differentiable_sharpe, fold_portfolio_returns, train_fold,
)
from src.config import PipelineConfig


def test_active_weights_normalized():
    z = torch.tensor([2.0, -1.0, -1.0, 0.5, -0.5])
    a = active_weights(z)
    assert abs(float(a.abs().sum()) - 1.0) < 1e-6
    assert abs(float(a.sum())) < 1e-6


def test_sharpe_finite_difference_gradient():
    torch.manual_seed(0)
    R = torch.randn(40, requires_grad=True)
    s = differentiable_sharpe(R)
    s.backward()
    g_auto = R.grad.clone()
    eps = 1e-4
    g_fd = torch.zeros_like(R)
    with torch.no_grad():
        for i in range(R.numel()):
            Rp = R.clone(); Rp[i] += eps
            Rm = R.clone(); Rm[i] -= eps
            g_fd[i] = (differentiable_sharpe(Rp) - differentiable_sharpe(Rm)) / (2 * eps)
    assert torch.allclose(g_auto, g_fd, atol=1e-3)


def test_residual_gamma_zero_equals_prior():
    torch.manual_seed(0)
    dates = 5
    scores = [torch.randn(8) for _ in range(dates)]
    labels = [torch.randn(8) for _ in range(dates)]
    prior = [torch.randn(8) for _ in range(dates)]
    R_g0 = fold_portfolio_returns(scores, labels, prior, tc=0.001, lam=0.1, gamma=0.0, residual=True)
    R_ref = []
    prev = None
    for d in range(dates):
        a = active_weights(xs_zscore(prior[d]))
        r = (a * labels[d]).sum()
        tc_ = torch.zeros(()) if prev is None else 0.001 * (a - prev).abs().sum()
        R_ref.append(r - 0.1 * tc_)
        prev = a
    assert torch.allclose(R_g0, torch.stack(R_ref), atol=1e-6)


def test_recovers_known_signal():
    rng = np.random.default_rng(0)
    F, N, D = 4, 20, 80
    X = [torch.tensor(rng.normal(size=(N, F)), dtype=torch.float32) for _ in range(D)]
    lab = [(x[:, 0] + 0.2 * torch.randn(N)) for x in X]
    prior = [torch.zeros(N) for _ in range(D)]
    cfg = PipelineConfig()
    cfg.dr_alpha_epochs = 400
    cfg.dr_alpha_lr = 5e-2
    pol, info = train_fold(X, lab, prior, X[-10:], lab[-10:], prior[-10:], F, cfg)
    w = pol.forward(torch.eye(F))
    assert int(torch.argmax(w.abs())) == 0
    assert info["val_sharpe"] > 0.5


def test_determinism():
    rng = np.random.default_rng(1)
    F, N, D = 4, 12, 40
    X = [torch.tensor(rng.normal(size=(N, F)), dtype=torch.float32) for _ in range(D)]
    lab = [x[:, 0] for x in X]
    prior = [torch.zeros(N) for _ in range(D)]
    cfg = PipelineConfig()
    cfg.dr_alpha_epochs = 50
    p1, _ = train_fold(X, lab, prior, [], [], [], F, cfg)
    p2, _ = train_fold(X, lab, prior, [], [], [], F, cfg)
    assert torch.allclose(p1.forward(torch.eye(F)), p2.forward(torch.eye(F)), atol=1e-6)
