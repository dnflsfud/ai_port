"""D-probe 2: production HOLD gate is blind to a benchmark fallback.

1. Build an infeasible rebalance with the PRODUCTION config (variant overrides):
   the drifted book carries a +30% active position in one name, so the
   0.04 per-name active cap is unreachable inside max_single_turnover=0.15.
   optimize_portfolio returns bm_weights (used_fallback=True) and the
   post-execution projection also fails and returns its fallback (= that bm
   target), so the executed book jumps to the benchmark with L1 turnover 0.60
   (4x the 0.15 hard cap).
2. Compute risk.json guardrails for that executed book with the exact formulas
   of scripts/export_operating_data.py:1684-1749 (copied, not imported: they
   are inline in main()).
3. Feed them to scripts/validate_portfolio_bundles.evaluate_production with an
   otherwise healthy record and optimizer_failure_rate=1.0.
"""
import numpy as np
import pandas as pd
import yaml

from src.harness import build_override_config
from src.portfolio_optimizer import optimize_portfolio, project_portfolio_weights
from scripts.validate_portfolio_bundles import evaluate_production

manifest = yaml.safe_load(open("variants/codex_causal_rank_65.yaml", encoding="utf-8"))
ov = dict(manifest["overrides"]); ov["enforce_oos_holdout"] = False
cfg = build_override_config(ov)

rng = np.random.default_rng(1)
n = 40
tickers = [f"T{i}" for i in range(n)]
cap = rng.uniform(1.0, 2.0, n)
bm = cap / cap.sum()                      # all names < 0.04 -> no mega-cap pins
sector_map = {t: f"S{i % 4}" for i, t in enumerate(tickers)}
F = rng.standard_normal((n, 1)) * 0.01
cov = F @ F.T + np.diag(rng.uniform(1e-4, 4e-4, n))
mu = pd.Series(rng.standard_normal(n), index=tickers)

prev = bm.copy()
prev[0] += 0.30
prev[1:] -= 0.30 * bm[1:] / bm[1:].sum()   # still long-only, sums to one

diag_opt, diag_proj = {}, {}
target = optimize_portfolio(mu, cov, prev_weights=prev, sector_map=sector_map,
                            bm_weights=bm, config=cfg, diagnostics=diag_opt)
new_w = project_portfolio_weights(prev + 0.42 * (target - prev), mu, cov,
                                  prev_weights=prev, sector_map=sector_map,
                                  bm_weights=bm, config=cfg,
                                  fallback_weights=target, diagnostics=diag_proj)
print("MVO used_fallback:", diag_opt.get("used_fallback"), "reason:", diag_opt.get("fallback_reason"))
print("projection used_fallback:", diag_proj.get("used_fallback"), "reason:", diag_proj.get("fallback_reason"))
print("executed book == benchmark:", bool(np.allclose(new_w, bm, atol=1e-12)))
print(f"executed two-way turnover: {np.abs(new_w - prev).sum():.3f} (hard cap {cfg.max_single_turnover})")

# --- export risk.json guardrail formulas (export_operating_data.py:1684-1749)
te_limit = float(cfg.max_te_annual)
wv, av = new_w, new_w - bm
active_var = float(av @ cov @ av)
active_te = float(np.sqrt(max(active_var, 0.0)) * np.sqrt(252.0))
contrib = [(av[i] * (cov @ av)[i] / np.sqrt(active_var) * np.sqrt(252.0)) if active_var > 0 else None
           for i in range(n)]
top = max(range(n), key=lambda i: abs(contrib[i] or 0))
top_name_share = abs(contrib[top]) / active_te if (active_te > 0 and contrib[top] is not None) else None
top_sector_share = None  # sector_risk sums of None -> 0.0, active_te == 0 -> share stays None
guardrails = {
    "estimated_te_breached": bool(active_te > te_limit + 1e-6),
    "top_name_active_risk_breached": (top_name_share is not None and top_name_share > cfg.max_name_active_risk_share),
    "top_sector_active_risk_breached": (top_sector_share is not None and top_sector_share > cfg.max_sector_active_risk_share),
}
record = {
    "_risk_guardrails": guardrails,
    "_model_quality": {"degenerate_rate": 0.3, "live_model_age_retrains": 0,
                        "split_audit": [{"prediction_date": "2026-08-01"}], "events": []},
    "performance": {"tracking_error": 0.037, "optimizer_failure_rate": 1.0,
                    "data_quality": {"tail_ffill_days": 0, "max_tail_ffill_days": 10,
                                     "currency": {"tg_px_ratio_suspect": {},
                                                  "tg_px_ratio_jump_vs_prev": {},
                                                  "tg_basis_guard_ok": True}}},
}
gate = evaluate_production(record)
print("guardrails:", guardrails)
print("production_gate status with optimizer_failure_rate=1.0 and a benchmark book:", gate["status"])
