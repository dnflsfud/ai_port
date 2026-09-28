"""D-probe 1: S13.41 option-vol scale row t consumes close-t information.

The production optimizer (src/backtest.py:_optimizer_fn) multiplies the
covariance by ``_optvol_scale.loc[pred_row.name]`` where pred_row.name is the
rebalance date t, while the covariance window itself is
``risk_source.iloc[hist_start:t_idx]`` (strictly before t) and the alpha is
lagged one row (execution_signal_lag_days=1).  This probe shows that the scale
value at row t changes when ONLY iv30_z[t] or ONLY the return r[t] changes,
i.e. the risk model uses one more day of information than every other input.
"""
import numpy as np
import pandas as pd

from src.option_vol_cov import build_option_vol_scale

rng = np.random.default_rng(0)
n, k = 700, 40
idx = pd.bdate_range("2015-01-01", periods=n)
cols = [f"T{i}" for i in range(k)]
vol = rng.uniform(0.01, 0.03, size=k)
# persistent vol regime z (AR(1)); iv30_z is an informative noisy read of it,
# so the preregistered model B learns a material iv30_z coefficient.
z = np.zeros((n, k))
for t in range(1, n):
    z[t] = 0.97 * z[t - 1] + 0.25 * rng.standard_normal(k)
rets = pd.DataFrame(rng.standard_normal((n, k)) * vol * np.exp(0.4 * z),
                    index=idx, columns=cols)
ivz = pd.DataFrame(z + 0.3 * rng.standard_normal((n, k)), index=idx, columns=cols)

R = 400  # inside the estimation block that starts at row 378 (252 + 2*63)
base = build_option_vol_scale(rets, ivz)

ivz2 = ivz.copy()
ivz2.iloc[R] = ivz2.iloc[R] + 3.0            # perturb ONLY iv30_z on day t
s_iv = build_option_vol_scale(rets, ivz2)

rets2 = rets.copy()
rets2.iloc[R] = rets2.iloc[R] * 8.0            # perturb ONLY the return of day t
s_r = build_option_vol_scale(rets2, ivz)

d_iv_t = float((s_iv.iloc[R] - base.iloc[R]).abs().max())
d_iv_prev = float((s_iv.iloc[R - 1] - base.iloc[R - 1]).abs().max())
d_r_t = float((s_r.iloc[R] - base.iloc[R]).abs().max())
d_r_prev = float((s_r.iloc[R - 1] - base.iloc[R - 1]).abs().max())
print(f"scale[t] moves when only iv30_z[t] changes : {d_iv_t > 1e-6}  (max|d|={d_iv_t:.3e})")
print(f"scale[t-1] unaffected by iv30_z[t]           : {d_iv_prev == 0.0}")
print(f"scale[t] moves when only r[t] changes        : {d_r_t > 1e-6}  (max|d|={d_r_t:.3e})")
print(f"scale[t-1] unaffected by r[t]                : {d_r_prev == 0.0}")
print("RESULT: optvol scale row t is a function of close-t data ->",
      "SAME-BAR" if (d_iv_t > 1e-6 and d_r_t > 1e-6 and d_iv_prev == 0.0) else "lagged")
