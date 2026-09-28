"""Probe C-02 (+ lag-once check): drive the REAL run_backtest (precomputed
Phase 2-4 inputs; real overlays / lag / simulate_portfolio / ECOS projection)
on a stub UniverseData whose Earnings_Timeline and iv30_z sheets are absent,
with pead_boost_enabled=True and option_vol_covariance_enabled=True (both
production-ON). The run completes with both components inert and nothing in
data_quality records it; the execution lag is applied exactly once."""
from src.config import PipelineConfig
from stub_common import BASE, preds, run

cfg = PipelineConfig(**BASE, pead_boost_enabled=True, option_vol_covariance_enabled=True)
res, out = run(cfg)
print("PEAD line:", [l for l in out.splitlines() if "PEAD" in l])
print("S13.41 ON line present:", any("S13.41 option-vol cov scaling ON" in l for l in out.splitlines()))
print("run completed, rebalances:", len(res.portfolio_weights))
print("pre_execution == input (PEAD added nothing):", bool(res.pre_execution_predictions.equals(preds)))
print("predictions == input.shift(1) (lag applied once):", bool(res.predictions.equals(preds.shift(1))))
print("data_quality keys flagging the skip:", sorted(res.data_quality or {}))
