"""Probe C-01: tuning_mode='tuning' (enforce_oos_holdout + train_cutoff_date)
stops PREDICTIONS at the cutoff, but simulate_portfolio keeps booking P&L of the
last (frozen, drifting) book through the reserved OOS window, so the headline
metrics a tuner compares (IR/TE/sub-periods) contain reserved-window returns.
Uses the real walk_forward_train + simulate_portfolio + BacktestResult; toy data."""
import numpy as np, pandas as pd
from src.config import PipelineConfig
from src.model_trainer import walk_forward_train
from src.backtest import simulate_portfolio

rng = np.random.default_rng(1)
dates = pd.bdate_range("2020-01-01", periods=420)
tickers = [f"T{i:02d}" for i in range(20)]
rets = pd.DataFrame(rng.normal(0, 0.01, (len(dates), len(tickers))), index=dates, columns=tickers)
idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
panel = pd.DataFrame(rng.normal(size=(len(idx), 3)), index=idx, columns=["f0", "f1", "f2"])
targets = rets.rolling(5).sum().shift(-5)          # toy 5d forward label
cutoff = dates[300]
cfg = PipelineConfig(train_window=150, val_window=20, retrain_freq=21, forward_horizon=5,
                     causal_validation_enabled=True, ewma_enabled=False, listing_mask_enabled=False,
                     enforce_oos_holdout=True, train_cutoff_date=str(cutoff.date()),
                     lgbm_params={"objective": "regression", "n_estimators": 20, "verbose": -1,
                                  "random_state": 0, "min_child_samples": 20})
_, preds, raw, _ = walk_forward_train(panel, targets, ["f0", "f1", "f2"], dates, config=cfg)
print("last date with predictions:", preds.dropna(how="all").index.max().date(), "cutoff:", cutoff.date())

def opt(pred_row, hist, prev_w, s_map, bm_w):          # simple active tilt (no cvxpy)
    w = bm_w + 0.02 * np.sign(pred_row.fillna(0).values)
    w = np.clip(w, 0, None); return w / w.sum()

res = simulate_portfolio(preds, rets, tickers, dates, rebalance_freq=21, one_way_tc=0.001,
                         optimizer_fn=opt, targets=targets, config=cfg)
pr, br = res.portfolio_returns, res.benchmark_returns
post = pr.index > cutoff
act = (pr - br)
print("P&L rows after cutoff:", int(post.sum()), " rebalances after cutoff:",
      int((res.turnover.index > cutoff).sum()))
print("nonzero active-return rows after cutoff:", int((act[post].abs() > 1e-12).sum()))
from src.utils import compute_performance_metrics
full = res.compute_metrics()["information_ratio"]
pre_only = compute_performance_metrics(pr[~post], br[~post])["information_ratio"]
print(f"reported IR (includes reserved window) = {full:.4f}; same metric on pre-cutoff rows only = {pre_only:.4f}")
