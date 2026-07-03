"""CS-DR-Alpha learning primitive.

A torch policy pi_theta maps per-stock features -> a scalar score. It is trained
by gradient ascent on a *differentiable, turnover-penalised cross-sectional
Sharpe* (Moody & Saffell "Direct Reinforcement", 1998/2001 — policy-gradient on
a financial utility; no value function, no PPO). This is the genuine RL
value-add over LightGBM's MSE objective: the reward is the IR-net-of-cost the
system actually cares about, which MSE cannot express.

This module is pure torch/numpy — no I/O, no pipeline imports. The temporal
walk-forward protocol and ticker alignment live in src/rl/dr_walkforward.py;
the functions here assume each per-date tensor is already aligned to a fixed
fold universe (same length & order across consecutive dates).
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch

EPS = 1e-8


def xs_zscore(s: torch.Tensor) -> torch.Tensor:
    """Cross-sectional z-score (zero mean, unit std), epsilon-safe."""
    mu = s.mean()
    sd = s.std(unbiased=False)
    return (s - mu) / (sd + EPS)


def active_weights(z: torch.Tensor) -> torch.Tensor:
    """Active tilt a = z / (sum|z| + eps).

    With a zero-mean input z (e.g. from xs_zscore) this is dollar-neutral
    (sum a ~ 0) with gross L1 = 1. Scale is irrelevant to the downstream MVO
    (which re-sizes); fixing L1 = 1 makes the return/turnover trade-off
    well-posed for the Sharpe objective.
    """
    return z / (z.abs().sum() + EPS)


def differentiable_sharpe(R: torch.Tensor, ann: float = 1.0) -> torch.Tensor:
    """mean(R) / std(R) * sqrt(ann). std is population (unbiased=False)."""
    mu = R.mean()
    sd = R.std(unbiased=False)
    return mu / (sd + EPS) * math.sqrt(ann)


class DRAlphaPolicy(torch.nn.Module):
    """Per-stock score map. arch in {"linear", "tiny"}."""

    def __init__(self, n_features: int, arch: str = "linear",
                 hidden: int = 16, seed: int = 42):
        super().__init__()
        torch.manual_seed(int(seed))
        self.arch = arch
        if arch == "linear":
            self.net: torch.nn.Module = torch.nn.Linear(n_features, 1)
        elif arch == "tiny":
            self.net = torch.nn.Sequential(
                torch.nn.Linear(n_features, hidden),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden, 1),
            )
        else:
            raise ValueError(f"unknown dr_alpha_arch {arch!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, F) -> (N,)
        return self.net(x).squeeze(-1)


def fold_portfolio_returns(
    scores_by_date: List[torch.Tensor],   # per date: policy output (N_d,)
    labels_by_date: List[torch.Tensor],   # per date: 20d fwd specific return (N_d,)
    prior_by_date: List[torch.Tensor],    # per date: LGBM z (N_d,)
    tc: float,
    lam: float,
    gamma: float,
    residual: bool,
) -> torch.Tensor:
    """Differentiable net active-return series R_d = r_d - lam * tc * sum|da|.

    residual: combined = prior + gamma * xs_zscore(score)  (gamma=0 -> baseline)
    else:     combined = xs_zscore(score)
    a_d = active_weights(xs_zscore(combined)); r_d = sum(a_d * label_d).
    Consecutive dates are assumed aligned (same length & ticker order).
    """
    R: List[torch.Tensor] = []
    prev_a: Optional[torch.Tensor] = None
    for s, lab, pri in zip(scores_by_date, labels_by_date, prior_by_date):
        if residual:
            combined = pri + gamma * xs_zscore(s)
        else:
            combined = xs_zscore(s)
        a = active_weights(xs_zscore(combined))
        r = (a * lab).sum()
        if prev_a is None or prev_a.shape != a.shape:
            turn = torch.zeros((), dtype=a.dtype)
        else:
            turn = tc * (a - prev_a).abs().sum()
        R.append(r - lam * turn)
        prev_a = a
    return torch.stack(R)


def train_fold(
    X_by_date: List[torch.Tensor],
    labels_by_date: List[torch.Tensor],
    prior_by_date: List[torch.Tensor],
    Xval_by_date: List[torch.Tensor],
    labelval_by_date: List[torch.Tensor],
    priorval_by_date: List[torch.Tensor],
    n_features: int,
    config,
    warm_start_state: Optional[dict] = None,
) -> Tuple[DRAlphaPolicy, dict]:
    """Fit a policy on one fold; early-stop on validation Sharpe if val given.

    Returns (best policy, info) where info has val_sharpe / train_sharpe / epochs.
    """
    seed = int(getattr(config, "dr_alpha_seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    arch = getattr(config, "dr_alpha_arch", "linear")
    hidden = int(getattr(config, "dr_alpha_hidden", 16))
    lr = float(getattr(config, "dr_alpha_lr", 1e-2))
    epochs = int(getattr(config, "dr_alpha_epochs", 300))
    l2 = float(getattr(config, "dr_alpha_l2", 1e-3))
    lam = float(getattr(config, "dr_alpha_turnover_lambda", 0.10))
    gamma = float(getattr(config, "dr_alpha_gamma", 1.0))
    residual = bool(getattr(config, "dr_alpha_residual", True))
    tc = float(getattr(config, "one_way_tc", 0.001))
    ann = 252.0 / max(int(getattr(config, "rebalance_freq", 21)), 1)

    policy = DRAlphaPolicy(n_features, arch=arch, hidden=hidden, seed=seed)
    if warm_start_state is not None:
        try:
            policy.load_state_dict(warm_start_state)
        except Exception:
            pass  # shape/arch mismatch -> fresh init (logged by caller)

    opt = torch.optim.Adam(policy.parameters(), lr=lr, weight_decay=l2)

    have_val = len(Xval_by_date) > 0
    best_val = -float("inf")
    best_state = {k: v.clone() for k, v in policy.state_dict().items()}
    best_epoch = 0
    patience, since_improve = 40, 0

    def _series(Xs, labs, pris):
        scores = [policy.forward(x) for x in Xs]
        return fold_portfolio_returns(scores, labs, pris, tc, lam, gamma, residual)

    last_train_sharpe = 0.0
    for ep in range(epochs):
        policy.train()
        opt.zero_grad()
        R = _series(X_by_date, labels_by_date, prior_by_date)
        loss = -differentiable_sharpe(R, ann=ann)
        loss.backward()
        opt.step()
        last_train_sharpe = float(differentiable_sharpe(R.detach(), ann=ann))

        if have_val:
            policy.eval()
            with torch.no_grad():
                Rv = _series(Xval_by_date, labelval_by_date, priorval_by_date)
                vs = float(differentiable_sharpe(Rv, ann=ann))
            if vs > best_val + 1e-6:
                best_val = vs
                best_state = {k: v.clone() for k, v in policy.state_dict().items()}
                best_epoch = ep
                since_improve = 0
            else:
                since_improve += 1
                if since_improve >= patience:
                    break

    if have_val:
        policy.load_state_dict(best_state)
        val_sharpe = best_val
    else:
        val_sharpe = last_train_sharpe

    info = {
        "val_sharpe": float(val_sharpe),
        "train_sharpe": float(last_train_sharpe),
        "epochs": int(best_epoch if have_val else epochs),
        "n_train_dates": len(X_by_date),
        "n_val_dates": len(Xval_by_date),
    }
    return policy, info
