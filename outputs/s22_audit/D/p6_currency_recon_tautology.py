"""D-probe 6: the currency "exact reconciliation" gate cannot fail.

build_currency_attribution defines fx_effect := r_usd - r_local, then checks
local + fx_effect - usd == 0.  Feed it LOCAL returns that are pure noise
(unrelated to the USD returns) and a JPY name with an FX series that moves the
opposite way: reconciliation.passed stays True and the validator's identical
checks would pass too.
"""
import numpy as np
import pandas as pd

from scripts.export_operating_data import build_currency_attribution

rng = np.random.default_rng(3)
dates = pd.bdate_range("2026-01-01", periods=60)
tickers = ["AAA", "JPX"]
usd = pd.DataFrame(rng.normal(0, 0.01, (60, 2)), index=dates, columns=tickers)
garbage_local = pd.DataFrame(rng.normal(0, 0.05, (60, 2)), index=dates, columns=tickers)
fx_ret = pd.DataFrame({"JPX": rng.normal(0, 0.005, 60)}, index=dates)
fx_rate = (1 + fx_ret).cumprod() * 0.0067
w = pd.DataFrame(0.5, index=dates, columns=tickers)
out = build_currency_attribution(
    tickers=tickers, dates=dates, usd_returns=usd, local_returns=garbage_local,
    fx_returns=fx_ret, fx_rates_usd_per_local=fx_rate,
    currency_map={"AAA": "USD", "JPX": "JPY"},
    portfolio_entering_weights=w, benchmark_entering_weights=w,
    latest_portfolio_weights=w.iloc[-1], latest_benchmark_weights=w.iloc[-1],
    data_quality={"fx_data_as_of": str(dates[-1].date())},
)
fx_usd_name = out["by_ticker"][0]
print("USD-listed name gets a nonzero 'FX' contribution:",
      abs(fx_usd_name["portfolio_fx_contribution"]) > 1e-6, fx_usd_name["portfolio_fx_contribution"])
print("reconciliation.passed with garbage local returns:", out["reconciliation"]["passed"])
