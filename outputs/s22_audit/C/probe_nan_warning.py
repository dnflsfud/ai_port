"""Probe C-04: simulate_portfolio's inf guard uses ~np.isfinite, so an ordinary
NaN prediction (e.g. a pre-listing / listing-masked name, which production has
on most 2019-2025 rebalances) fires the 'non-finite ... Check upstream model
output for inf/-inf' WARNING on every rebalance. No inf is present here."""
import logging
import numpy as np, pandas as pd
from src.backtest import simulate_portfolio

rng = np.random.default_rng(0)
dates = pd.bdate_range("2022-01-03", periods=90)
tickers = [f"T{i:02d}" for i in range(15)]
rets = pd.DataFrame(rng.normal(0, 0.01, (90, 15)), index=dates, columns=tickers)
preds = pd.DataFrame(rng.normal(size=(90, 15)), index=dates, columns=tickers)
preds["T14"] = np.nan                                    # one not-yet-listed name
assert np.isinf(preds.values).sum() == 0

records = []
class H(logging.Handler):
    def emit(self, r): records.append(r.getMessage())
logging.getLogger("src.backtest").addHandler(H())
logging.getLogger("src.backtest").setLevel(logging.WARNING)

opt = lambda pred_row, hist, prev_w, s_map, bm_w: bm_w
res = simulate_portfolio(preds, rets, tickers, dates, rebalance_freq=21, optimizer_fn=opt)
hits = [m for m in records if "non-finite prediction" in m]
print("rebalances:", len(res.portfolio_weights), " inf cells:", 0, " 'non-finite' warnings:", len(hits))
print("first:", hits[0] if hits else None)
