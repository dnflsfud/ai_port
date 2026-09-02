"""Single-process empirical audit pack from backtest_result.pkl (S0' certified run).
Writes compact CSV/JSON summaries to OUT so review agents never need to load the 274MB pkl.
"""
import sys, os, json, pickle, gc, time
import numpy as np
import pandas as pd

PKL = sys.argv[1]
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
res = pickle.load(open(PKL, 'rb'))
print('loaded', round(time.time() - t0, 1), 's')

meta = {}
panel = res.panel
meta['panel_shape'] = list(panel.shape)
meta['panel_index_names'] = list(panel.index.names) if hasattr(panel.index, 'names') else str(type(panel.index))
meta['panel_index_sample'] = [str(x) for x in panel.index[:3]]
meta['feature_names'] = list(res.feature_names)
meta['feature_groups'] = {k: list(v) for k, v in res.feature_groups.items()} if isinstance(res.feature_groups, dict) else str(res.feature_groups)
meta['data_quality'] = {k: (v if isinstance(v, (int, float, str, bool, list, dict)) else str(v)) for k, v in res.data_quality.items()}
meta['model_quality'] = {k: (v if isinstance(v, (int, float, str, bool, list, dict)) else str(v)) for k, v in res.model_quality.items()}
for k in ['optimizer_failure_rate', 'optimizer_failures', 'optimizer_rebalances', 'optimizer_solver_counts',
          'optimizer_solver_fallback_rate', 'optimizer_solver_solves', 'optimizer_solver_fallbacks',
          'optimizer_fallback_reason_counts', 'execution_signal_lag_days', 'benchmark_type']:
    v = getattr(res, k, None)
    meta[k] = v if isinstance(v, (int, float, str, bool, list, dict)) else str(v)

# ---- normalise panel to MultiIndex (date, ticker) ----
if isinstance(panel.index, pd.MultiIndex):
    names = list(panel.index.names)
    # detect which level is date
    lvl0 = panel.index.get_level_values(0)
    if np.issubdtype(lvl0.dtype, np.datetime64):
        date_lvl, tk_lvl = 0, 1
    else:
        date_lvl, tk_lvl = 1, 0
    panel = panel.copy()
    panel.index = panel.index.set_names(['date', 'ticker']) if date_lvl == 0 else panel.index.swaplevel().set_names(['date', 'ticker'])
    if date_lvl != 0:
        panel = panel.sort_index()
else:
    raise SystemExit('panel index is not MultiIndex: %s' % type(panel.index))
meta['dates_n'] = int(panel.index.get_level_values('date').nunique())
meta['tickers_n'] = int(panel.index.get_level_values('ticker').nunique())
tickers = sorted(panel.index.get_level_values('ticker').unique().tolist())
meta['tickers'] = tickers
feats = [c for c in panel.columns]
X = panel.astype('float32')
del panel; gc.collect()

# ---- A. per-feature global stats ----
rows = []
absX = X.abs()
for c in feats:
    s = X[c]
    a = absX[c]
    valid = s.notna()
    n = int(valid.sum())
    rows.append({
        'feature': c,
        'nan_pct': float(100 * (1 - n / len(s))),
        'zero_exact_pct': float(100 * (s == 0).sum() / max(n, 1)),
        'sat_abs_ge_4_99_pct': float(100 * (a >= 4.99).sum() / max(n, 1)),
        'abs_ge_3_pct': float(100 * (a >= 3).sum() / max(n, 1)),
        'mean': float(s.mean()), 'std': float(s.std()),
        'p01': float(s.quantile(0.01)), 'p50': float(s.median()), 'p99': float(s.quantile(0.99)),
        'min': float(s.min()), 'max': float(s.max()),
    })
A = pd.DataFrame(rows).set_index('feature')

# ---- per-date cross-sectional std / constant fraction ----
g_date = X.groupby(level='date')
cs_std = g_date.std()
cs_cnt = g_date.count()
cs_nuniq_low = (cs_std < 0.05)
A['dates_cs_std_lt_0_05_pct'] = 100 * cs_nuniq_low.sum() / len(cs_std)
A['dates_cs_std_eq_0_pct'] = 100 * (cs_std == 0).sum() / len(cs_std)
A['median_cs_std'] = cs_std.median()
A['median_cs_count'] = cs_cnt.median()
# per-date fraction of exact zeros
cs_zero = g_date.apply(lambda d: (d == 0).mean()) if False else None  # too slow; use vectorised below
Z = (X == 0).astype('float32')
Z[X.isna()] = np.nan
cs_zero_frac = Z.groupby(level='date').mean()
A['dates_zero_frac_gt_50_pct'] = 100 * (cs_zero_frac > 0.5).sum() / len(cs_zero_frac)
del Z; gc.collect()
cs_std.to_csv(os.path.join(OUT, 'cs_std_by_date.csv'))
cs_zero_frac.to_csv(os.path.join(OUT, 'cs_zero_frac_by_date.csv'))

# ---- per-ticker medians & pinned detection ----
g_tk = X.groupby(level='ticker')
tk_med = g_tk.median()
tk_std = g_tk.std()
tk_cnt = g_tk.count()
tk_med.to_csv(os.path.join(OUT, 'ticker_median.csv'))
tk_std.to_csv(os.path.join(OUT, 'ticker_std.csv'))
pinned = {}
for c in feats:
    m = tk_med[c]
    s = tk_std[c]
    pin = m[(m.abs() > 2.0) & (tk_cnt[c] > 252)].sort_values()
    lowvar = s[(s < 0.15) & (tk_cnt[c] > 252)]
    pinned[c] = {
        'n_pinned_abs_median_gt_2': int(len(pin)),
        'pinned': {k: round(float(v), 3) for k, v in pin.head(12).items()},
        'n_ticker_std_lt_0_15': int(len(lowvar)),
        'lowvar': {k: round(float(v), 3) for k, v in lowvar.head(12).items()},
    }
A['n_tickers_pinned_abs_med_gt_2'] = pd.Series({c: pinned[c]['n_pinned_abs_median_gt_2'] for c in feats})
A['n_tickers_std_lt_0_15'] = pd.Series({c: pinned[c]['n_ticker_std_lt_0_15'] for c in feats})
json.dump(pinned, open(os.path.join(OUT, 'pinned_tickers.json'), 'w'), indent=1)

# ---- suffix/exchange grouping (infer from ticker string) ----
def suffix(t):
    parts = str(t).split()
    if len(parts) >= 2:
        return parts[-1]
    # numeric-only tickers (KR/JP) -> mark
    if str(t).isdigit():
        return 'NUMERIC'
    return 'NOSFX'
sfx = pd.Series({t: suffix(t) for t in tickers})
meta['ticker_suffix_counts'] = sfx.value_counts().to_dict()
grp_std = {}
for gname, gt in sfx.groupby(sfx):
    tks = list(gt.index)
    if len(tks) < 3:
        continue
    sub = tk_med.loc[tk_med.index.intersection(tks)]
    grp_std[gname] = {'n': len(tks), 'median_of_ticker_medians': sub.median().round(3).to_dict()}
json.dump(grp_std, open(os.path.join(OUT, 'group_median_by_suffix.json'), 'w'), indent=1)

# ---- lag-1 autocorrelation per ticker (median over tickers), sampled features all ----
def lag1_ac(df):
    out = {}
    for c in feats:
        s = df[c]
        s1 = s.shift(1)
        v = s.notna() & s1.notna()
        if v.sum() < 100:
            out[c] = np.nan
            continue
        out[c] = float(np.corrcoef(s[v], s1[v])[0, 1])
    return pd.Series(out)
ac_rows = []
for t in tickers[::5]:  # every 5th ticker -> 50 tickers
    ac_rows.append(lag1_ac(X.xs(t, level='ticker')))
AC = pd.DataFrame(ac_rows)
A['median_lag1_autocorr_50tk'] = AC.median()

# ---- listing-entry ramp: mean |x| first 21 valid rows vs overall per ticker ----
ramp = {}
first_valid = {}
for t in tickers:
    d = X.xs(t, level='ticker')
    anyvalid = d.notna().any(axis=1)
    if anyvalid.sum() == 0:
        continue
    fv = anyvalid.idxmax()
    first_valid[t] = str(fv)[:10]
    head = d.loc[fv:].iloc[:21].abs().mean()
    ramp[t] = head
RAMP = pd.DataFrame(ramp).T  # ticker x feature
overall_abs_mean = absX.groupby(level='ticker').mean()
ratio = (RAMP / overall_abs_mean.reindex(RAMP.index)).replace([np.inf, -np.inf], np.nan)
# tickers whose first valid date is after 2014 (late entrants)
late = [t for t, d in first_valid.items() if d > '2014-06-30']
meta['late_entrants'] = {t: first_valid[t] for t in late}
A['late_entrant_ramp_ratio_median'] = ratio.loc[ratio.index.intersection(late)].median() if late else np.nan
A['late_entrant_ramp_absmean_median'] = RAMP.loc[RAMP.index.intersection(late)].median() if late else np.nan
ratio.loc[ratio.index.intersection(late)].round(2).to_csv(os.path.join(OUT, 'late_entrant_ramp_ratio.csv'))
del absX; gc.collect()

# ---- B. feature correlation on 5% row sample ----
samp = X.sample(frac=0.05, random_state=0)
C = samp.corr()
C.round(3).to_csv(os.path.join(OUT, 'feature_corr_sample.csv'))
pairs = []
cols = list(C.columns)
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        v = C.iloc[i, j]
        if abs(v) >= 0.85:
            pairs.append({'a': cols[i], 'b': cols[j], 'corr': round(float(v), 3)})
pairs = sorted(pairs, key=lambda p: -abs(p['corr']))
json.dump(pairs, open(os.path.join(OUT, 'high_corr_pairs.json'), 'w'), indent=1)
del samp, C; gc.collect()

A.round(4).to_csv(os.path.join(OUT, 'feature_audit.csv'))
del X; gc.collect()
print('features done', round(time.time() - t0, 1), 's')

# ---- C. predictions ----
P = {}
for nm in ['raw_predictions', 'pre_overlay_predictions', 'predictions', 'pre_execution_predictions']:
    df = getattr(res, nm, None)
    if df is not None:
        P[nm] = df
pred_summary = {}
if P:
    base = P.get('predictions')
    for nm, df in P.items():
        nanpct = float(100 * df.isna().mean().mean())
        cs_disp = df.std(axis=1)
        pred_summary[nm] = {
            'shape': list(df.shape), 'nan_pct': nanpct,
            'cs_std_median': float(cs_disp.median()), 'cs_std_min': float(cs_disp.min()), 'cs_std_max': float(cs_disp.max()),
            'dates_all_nan': int(df.isna().all(axis=1).sum()),
            'dates_cs_std_lt_1e-6': int((cs_disp < 1e-6).sum()),
            'first_date': str(df.index[0])[:10], 'last_date': str(df.index[-1])[:10],
        }
        if base is not None and nm != 'predictions':
            rc = []
            for d in df.index[::10]:
                a = df.loc[d]; b = base.loc[d]
                v = a.notna() & b.notna()
                if v.sum() > 10:
                    rc.append(a[v].rank().corr(b[v].rank()))
            pred_summary[nm]['rank_corr_vs_predictions_median'] = float(np.nanmedian(rc)) if rc else None
            pred_summary[nm]['rank_corr_vs_predictions_min'] = float(np.nanmin(rc)) if rc else None
    # lag-1 rank autocorr of predictions
    df = P.get('predictions')
    rc = []
    for i in range(1, len(df.index), 5):
        a = df.iloc[i]; b = df.iloc[i - 1]
        v = a.notna() & b.notna()
        if v.sum() > 10:
            rc.append(a[v].rank().corr(b[v].rank()))
    pred_summary['predictions_lag1_rank_autocorr_median'] = float(np.nanmedian(rc))
    # per-ticker mean prediction (persistent tilt)
    tm = df.mean().sort_values()
    pred_summary['predictions_ticker_mean_bottom10'] = tm.head(10).round(4).to_dict()
    pred_summary['predictions_ticker_mean_top10'] = tm.tail(10).round(4).to_dict()
    pred_summary['predictions_ticker_mean_std_across_tickers'] = float(tm.std())
    pred_summary['predictions_global_std'] = float(df.stack().std())
json.dump(pred_summary, open(os.path.join(OUT, 'predictions_summary.json'), 'w'), indent=1)

# ---- D. targets ----
T = res.targets
tg = {
    'shape': list(T.shape), 'nan_pct': float(100 * T.isna().mean().mean()),
    'valid_per_date_median': float(T.notna().sum(axis=1).median()),
    'dates_zero_valid': int((T.notna().sum(axis=1) == 0).sum()),
    'dates_zero_valid_list_head': [str(d)[:10] for d in T.index[T.notna().sum(axis=1) == 0][:20]],
    'cs_std_median': float(T.std(axis=1).median()),
    'first_date': str(T.index[0])[:10], 'last_date': str(T.index[-1])[:10],
    'last_valid_date': str(T.dropna(how='all').index[-1])[:10],
}
json.dump(tg, open(os.path.join(OUT, 'targets_summary.json'), 'w'), indent=1)

# ---- E. weights / turnover / IC ----
pw = res.portfolio_weights
wrows = []
prev = None
for d in sorted(pw.keys()):
    w = pw[d]
    ws = pd.Series(w) if not isinstance(w, pd.Series) else w
    ws = ws.astype(float)
    row = {'date': str(d)[:10], 'sum': float(ws.sum()), 'min': float(ws.min()), 'max': float(ws.max()),
           'n_pos': int((ws > 1e-6).sum()), 'n_neg': int((ws < -1e-9).sum()), 'n_nan': int(ws.isna().sum()),
           'top5_sum': float(ws.sort_values(ascending=False).head(5).sum()),
           'n_ge_0_15': int((ws > 0.15 + 1e-6).sum())}
    if prev is not None:
        al = ws.reindex(ws.index.union(prev.index)).fillna(0)
        pl = prev.reindex(al.index).fillna(0)
        row['l1_vs_prev'] = float((al - pl).abs().sum())
    prev = ws
    wrows.append(row)
W = pd.DataFrame(wrows)
W.to_csv(os.path.join(OUT, 'portfolio_weights_summary.csv'), index=False)
dw = res.daily_weights
dwrows = []
for d in sorted(dw.keys())[::10]:
    w = dw[d]
    ws = pd.Series(w).astype(float) if not isinstance(w, pd.Series) else w.astype(float)
    dwrows.append({'date': str(d)[:10], 'sum': float(ws.sum()), 'min': float(ws.min()), 'max': float(ws.max()), 'n_nan': int(ws.isna().sum())})
pd.DataFrame(dwrows).to_csv(os.path.join(OUT, 'daily_weights_summary_every10.csv'), index=False)
ic = res.ic_series
to = res.turnover
ash = res.active_share_series
ser = {
    'ic': {'n': int(ic.notna().sum()), 'mean': float(ic.mean()), 'std': float(ic.std()), 'min': float(ic.min()), 'max': float(ic.max()),
           'first': str(ic.index[0])[:10], 'last': str(ic.index[-1])[:10], 'n_nan': int(ic.isna().sum())},
    'turnover': {'n': int(to.notna().sum()), 'mean': float(to.mean()), 'median': float(to.median()), 'max': float(to.max()), 'min': float(to.min()),
                 'first': str(to.index[0])[:10]},
    'active_share': {'mean': float(ash.mean()), 'min': float(ash.min()), 'max': float(ash.max())},
}
pr = res.portfolio_returns; br = res.benchmark_returns
act = (pr - br)
ser['returns'] = {
    'n_port': int(len(pr)), 'n_bm': int(len(br)), 'n_nan_port': int(pr.isna().sum()), 'n_nan_bm': int(br.isna().sum()),
    'first': str(pr.index[0])[:10], 'last': str(pr.index[-1])[:10],
    'n_zero_port_days': int((pr == 0).sum()), 'n_zero_bm_days': int((br == 0).sum()),
    'port_ann_ret_252': float(pr.mean() * 252), 'bm_ann_ret_252': float(br.mean() * 252),
    'active_ann_252': float(act.mean() * 252), 'te_ann_252': float(act.std() * np.sqrt(252)),
    'ir_252': float(act.mean() * 252 / (act.std() * np.sqrt(252))),
    'rows_per_year': float(len(pr) / ((pr.index[-1] - pr.index[0]).days / 365.25)),
    'max_abs_daily_port': float(pr.abs().max()), 'max_abs_daily_active': float(act.abs().max()),
    'worst5_active_days': {str(k)[:10]: round(float(v), 4) for k, v in act.nsmallest(5).items()},
    'best5_active_days': {str(k)[:10]: round(float(v), 4) for k, v in act.nlargest(5).items()},
}
json.dump(ser, open(os.path.join(OUT, 'series_summary.json'), 'w'), indent=1)
pd.DataFrame({'port': pr, 'bm': br}).to_csv(os.path.join(OUT, 'daily_returns.csv'))
ic.to_csv(os.path.join(OUT, 'ic_series.csv')); to.to_csv(os.path.join(OUT, 'turnover.csv')); ash.to_csv(os.path.join(OUT, 'active_share.csv'))

# ---- F. models ----
mrows = []
ids = {}
for k in sorted(res.models.keys(), key=lambda x: str(x)):
    m = res.models[k]
    mid = id(m)
    ids.setdefault(mid, []).append(str(k)[:10])
    row = {'key': str(k)[:10], 'obj_id': mid, 'type': type(m).__name__}
    try:
        booster = m.booster_ if hasattr(m, 'booster_') else m
        row['n_trees'] = int(booster.num_trees()) if hasattr(booster, 'num_trees') else None
        row['best_iteration'] = int(getattr(m, 'best_iteration_', None) or 0) if hasattr(m, 'best_iteration_') else None
        fn = list(booster.feature_name()) if hasattr(booster, 'feature_name') else []
        row['n_features'] = len(fn)
        imp = booster.feature_importance(importance_type='gain') if hasattr(booster, 'feature_importance') else None
        if imp is not None and len(fn) == len(imp):
            s = pd.Series(imp, index=fn)
            s = s / s.sum() if s.sum() > 0 else s
            row['top5_gain'] = s.sort_values(ascending=False).head(5).round(4).to_dict()
            row['n_zero_gain'] = int((s == 0).sum())
    except Exception as e:
        row['err'] = str(e)[:200]
    mrows.append(row)
json.dump({'models': mrows, 'shared_object_groups': [v for v in ids.values() if len(v) > 1]}, open(os.path.join(OUT, 'models_summary.json'), 'w'), indent=1)
json.dump(meta, open(os.path.join(OUT, 'meta.json'), 'w'), indent=1, default=str)
print('all done', round(time.time() - t0, 1), 's')
