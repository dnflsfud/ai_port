"""B-01 probe: PCA target subtracts a ONE-DAY trailing mean from a 20-DAY return.

compute_specific_returns fits sklearn PCA on daily returns (mean_ = trailing
252d DAILY mean) and then calls pca.transform / inverse_transform on a 20-day
cumulative forward return.  transform subtracts mean_, inverse_transform adds
it back, so spec = (I-P)(fwd - mean_daily): the label carries -(I-P)*mean_daily,
a deterministic function of PAST returns.

Null world: i.i.d. returns (factor + idio), NO momentum / no predictability.
A correct residual target must have ~0 rank IC with trailing 252d momentum.
We measure the rank IC of momentum_252d vs (a) the production target and
(b) the same construction without the mean offset, on identical data.
"""
import copy
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, r"C:/Users/westl/PycharmProjects/pythonProject/venv_vf_new/machine/re_study/c2/ai_port")
from src.config import DEFAULT_CONFIG  # noqa: E402
from src.target_engine import compute_specific_returns, compute_forward_returns  # noqa: E402

rng = np.random.default_rng(7)
N, T = 150, 760
dates = pd.bdate_range("2020-01-01", periods=T)
tickers = [f"T{i:03d}" for i in range(N)]
beta = rng.uniform(0.6, 1.4, N)
gamma = rng.normal(0.0, 1.0, N)
mkt = rng.normal(0.0004, 0.010, T)
f2 = rng.normal(0.0, 0.006, T)
eps = rng.normal(0.0, 0.018, (T, N))          # ~29% annual idio vol, i.i.d.
ret = pd.DataFrame(np.outer(mkt, beta) + np.outer(f2, gamma) + eps,
                   index=dates, columns=tickers)

cfg = copy.copy(DEFAULT_CONFIG)
cfg.pca_components, cfg.pca_n_remove, cfg.pca_lookback, cfg.forward_horizon = 5, 2, 252, 20
cfg.pca_vol_standardize = False

prod = compute_specific_returns(ret, n_remove=2, config=cfg)   # production code path

# Same loop, identical PCA, but no daily-mean offset in the 20d projection.
fwd = compute_forward_returns(ret, 20)
alt = pd.DataFrame(np.nan, index=dates, columns=tickers)
for t in range(252, T - 20):
    hist = ret.iloc[t - 252:t].values
    pca = PCA(n_components=5).fit(hist)
    f = fwd.iloc[t].values.reshape(1, -1)
    comps = pca.components_[:2]
    alt.iloc[t] = (f - (f @ comps.T) @ comps).ravel()

mom = ret.rolling(252, min_periods=252).sum()   # == price.py momentum_252d

def ic_series(label):
    out = []
    for dt in label.index:
        a, b = mom.loc[dt], label.loc[dt]
        m = a.notna() & b.notna()
        if m.sum() > 30:
            out.append(a[m].rank().corr(b[m].rank()))
    return pd.Series(out)

ic_prod, ic_alt = ic_series(prod), ic_series(alt)
d = ic_prod - ic_alt
offset = (prod - alt).stack()
print(f"dates={len(ic_prod)}")
print(f"mean rank IC(momentum_252d, production target) = {ic_prod.mean():+.4f}")
print(f"mean rank IC(momentum_252d, no-offset target)  = {ic_alt.mean():+.4f}")
print(f"per-date IC difference: mean {d.mean():+.4f}, sd {d.std():.4f}, "
      f"share of dates < 0 = {(d < 0).mean():.3f}")
print(f"sd(offset)/sd(target) = {offset.std() / alt.stack().std():.4f}")
print(f"corr(offset, -momentum_252d) = "
      f"{pd.concat([offset, -mom.stack()], axis=1).dropna().corr().iloc[0, 1]:+.3f}")
