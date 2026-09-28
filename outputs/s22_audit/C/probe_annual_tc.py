"""Probe C-03: the S15 annual_tc fix attaches one_way_tc to simulate_portfolio's
result, but run_backtest copies fields into a NEW BacktestResult and drops it,
so compute_metrics()['annual_tc'] falls back to the import-time ONE_WAY_TC
(0.0010) whenever config.one_way_tc differs. (P&L TC itself uses config.)"""
from src.config import PipelineConfig
from stub_common import BASE, run

cfg = PipelineConfig(**BASE, pead_boost_enabled=False, option_vol_covariance_enabled=False,
                     one_way_tc=0.0050)
res, _ = run(cfg)
m = res.compute_metrics()
print("hasattr(result, 'one_way_tc'):", hasattr(res, "one_way_tc"))
print(f"annual_tc reported = {m['annual_tc']:.6f}; turnover*config.one_way_tc = "
      f"{m['avg_annual_turnover'] * cfg.one_way_tc:.6f}; "
      f"ratio = {m['annual_tc'] / (m['avg_annual_turnover'] * cfg.one_way_tc):.3f}")
