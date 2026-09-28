"""B-01 deterministic check: a name whose 20-day forward return is exactly 0
must get a 0 residual label if the target only removes common factors.
compute_specific_returns returns -(I-P2) * mean_daily instead, i.e. a label
that is a pure function of the PAST 252 days (sign opposite to past drift)."""
import copy
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, r"C:/Users/westl/PycharmProjects/pythonProject/venv_vf_new/machine/re_study/c2/ai_port")
from src.config import DEFAULT_CONFIG  # noqa: E402
from src.target_engine import compute_specific_returns  # noqa: E402

rng = np.random.default_rng(1)
N, L, H = 40, 252, 20
dates = pd.bdate_range("2024-01-01", periods=L + H + 1)
drift = np.linspace(-0.004, 0.004, N)                   # past daily drift per name
past = rng.normal(0, 0.01, (L, N)) + rng.normal(0, 0.01, (L, 1)) + drift
ret = np.vstack([past, np.zeros((H + 1, N))])            # forward window: all exactly 0
ret = pd.DataFrame(ret, index=dates, columns=[f"T{i}" for i in range(N)])

cfg = copy.copy(DEFAULT_CONFIG)
cfg.pca_components, cfg.pca_n_remove, cfg.pca_lookback, cfg.forward_horizon = 5, 2, L, H
cfg.pca_vol_standardize = False
spec = compute_specific_returns(ret, n_remove=2, config=cfg).iloc[L]   # t = L, fwd window = zeros

pca = PCA(n_components=5).fit(ret.iloc[:L].values)
mu = pca.mean_
P2 = pca.components_[:2].T @ pca.components_[:2]
expected = -(mu - P2 @ mu)
print("forward 20d return of every name: 0.0")
print(f"label range: [{spec.min():+.5f}, {spec.max():+.5f}]  (should be all 0)")
print(f"max |label - (-(I-P2) mean_daily)| = {np.abs(spec.values - expected).max():.2e}")
print(f"corr(label, past 252d mean return) = {np.corrcoef(spec.values, mu)[0, 1]:+.3f}")
