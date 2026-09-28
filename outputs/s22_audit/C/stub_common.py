"""Shared 20-name toy UniverseData stub + precomputed Phase 2-4 inputs for the
run_backtest probes (no workbook, no pkl). Run probes with CWD = this folder so
run_backtest's ProgressLogger writes ./outputs/progress.md HERE, not in the repo."""
import numpy as np, pandas as pd

rng = np.random.default_rng(3)
dates = pd.bdate_range("2021-01-01", periods=200)
tickers = [f"T{i:02d}" for i in range(20)]
rets = pd.DataFrame(rng.normal(0.0003, 0.01, (200, 20)), index=dates, columns=tickers)
caps = pd.DataFrame(np.tile(np.linspace(50, 150, 20), (200, 1)), index=dates, columns=tickers)
idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
panel = pd.DataFrame({"f0": rng.normal(size=len(idx))}, index=idx)
targets = rets.rolling(20).sum().shift(-20)
preds = pd.DataFrame(rng.normal(size=(200, 20)), index=dates, columns=tickers)
preds.iloc[:130] = np.nan                       # walk-forward burn-in


class StubData:
    def __init__(self):
        self.tickers, self.dates = tickers, dates
        self.returns, self.raw_returns, self.market_cap = rets, rets.copy(), caps
        self.meta = pd.DataFrame({"sector": ["S%d" % (i % 4) for i in range(20)]}, index=tickers)
        self.listing_dates, self.data_quality = {}, {}
        self.earnings_timeline = None              # Earnings_Timeline sheet absent
        self.factor_data, self.factor_prices = {}, None

    def has_factor_data(self):
        return False

    def get_sheet(self, name):                     # iv30_z (and every sheet) absent
        raise KeyError(name)


BASE = dict(listing_mask_enabled=False, vol_quality_tilt_enabled=False,
            growth_tilt_enabled=False, value_trap_gate_enabled=False,
            execution_signal_lag_days=1, static_execution_enabled=True,
            sp500_benchmark_enabled=False, max_te_annual=0.035,
            sector_deviation=0.5, mega_cap_protection_enabled=False)


def run(cfg):
    import io, contextlib, logging
    from src.backtest import run_backtest
    logging.disable(logging.CRITICAL)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = run_backtest(StubData(), precomputed_panel=panel, precomputed_feature_names=["f0"],
                           precomputed_feature_groups={}, precomputed_targets=targets,
                           precomputed_models={}, precomputed_predictions=preds, config=cfg)
    return res, buf.getvalue()
