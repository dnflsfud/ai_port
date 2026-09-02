"""Raw workbook probe (reads only 7 sheets). Outputs compact JSON/CSV to OUT.
Checks: (1) negative-equity ROE/PB, (2) asynchronous-trading beta/corr by exchange, (3) tg_upside raw ratio.
"""
import sys, os, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
from src.config import PipelineConfig
from src import data_loader as dl

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
path = PipelineConfig().data_path
SHEETS = ['Universe_Meta', 'BEST_ROE', 'BEST_PX_BPS_RATIO', 'BEST_EPS', 'Daily_Returns', 'PX_LAST', 'Factset_TG_Price']
t0 = time.time()
raw = pd.read_excel(path, sheet_name=SHEETS)
print('read', round(time.time() - t0, 1), 's', {k: v.shape for k, v in raw.items()})

def std(df):
    for fn in ('_rename_bloomberg_equity_columns', '_standardize_index', '_standardize_columns'):
        f = getattr(dl, fn, None)
        if f is not None:
            try:
                df = f(df)
            except Exception as e:
                print('std fail', fn, e)
    return df

meta = dl.load_universe_meta(raw)
meta.to_csv(os.path.join(OUT, 'universe_meta.csv'))
print('meta cols', list(meta.columns)[:12], 'n', len(meta))
print(meta.head(3).to_string())
sheets = {k: std(v) for k, v in raw.items() if k != 'Universe_Meta'}
for k, v in sheets.items():
    print(k, v.shape, 'index', str(v.index[:2].tolist()), 'cols', v.columns[:4].tolist(), 'dtype', v.index.dtype)
tickers = [t for t in meta.index if t in sheets['PX_LAST'].columns]
print('tickers matched', len(tickers))

# ---- (1) negative equity: ROE / PB ----
roe = sheets['BEST_ROE'].reindex(columns=tickers)
pb = sheets['BEST_PX_BPS_RATIO'].reindex(columns=tickers)
eps = sheets['BEST_EPS'].reindex(columns=tickers)
rows = []
for t in tickers:
    r, p, e = roe[t], pb[t], eps[t]
    rv = r.dropna(); pv = p.dropna(); ev = e.dropna()
    rows.append({
        'ticker': t,
        'roe_n': len(rv), 'roe_neg_pct': float(100 * (rv < 0).mean()) if len(rv) else np.nan, 'roe_median': float(rv.median()) if len(rv) else np.nan,
        'roe_abs_gt_100_pct': float(100 * (rv.abs() > 100).mean()) if len(rv) else np.nan,
        'pb_n': len(pv), 'pb_nan_pct': float(100 * p.isna().mean()), 'pb_neg_pct': float(100 * (pv < 0).mean()) if len(pv) else np.nan, 'pb_median': float(pv.median()) if len(pv) else np.nan,
        'pb_gt_50_pct': float(100 * (pv > 50).mean()) if len(pv) else np.nan,
        'eps_pos_pct': float(100 * (ev > 0).mean()) if len(ev) else np.nan,
    })
NE = pd.DataFrame(rows).set_index('ticker')
NE['flag_roe_neg_eps_pos'] = (NE['roe_neg_pct'] > 50) & (NE['eps_pos_pct'] > 80)
NE['flag_pb_neg'] = NE['pb_neg_pct'] > 20
NE['flag_pb_nan'] = NE['pb_nan_pct'] > 50
NE['flag_roe_extreme'] = NE['roe_abs_gt_100_pct'] > 20
NE.round(3).to_csv(os.path.join(OUT, 'negative_equity_probe.csv'))
flag = NE[NE[['flag_roe_neg_eps_pos', 'flag_pb_neg', 'flag_pb_nan', 'flag_roe_extreme']].any(axis=1)]
print('=== negative-equity / extreme flags ===')
print(flag[['roe_neg_pct', 'roe_median', 'roe_abs_gt_100_pct', 'pb_nan_pct', 'pb_neg_pct', 'pb_median', 'pb_gt_50_pct', 'eps_pos_pct']].round(2).to_string())
# how many are positive-EPS names with negative or |ROE|>100
print('roe unit sample (AAPL, MSFT, JPM):', {t: float(roe[t].dropna().median()) for t in ['AAPL', 'MSFT', 'JPM'] if t in roe})
print('pb unit sample:', {t: float(pb[t].dropna().median()) for t in ['AAPL', 'MSFT', 'JPM'] if t in pb})

# ---- (2) asynchronous trading ----
ret = sheets['Daily_Returns'].reindex(columns=tickers).astype(float)
med_abs = float(ret.abs().stack().median())
scale = 0.01 if med_abs > 0.2 else 1.0  # percent vs decimal
ret = ret * scale
print('returns median abs (raw)', med_abs, 'scale applied', scale)
exch = meta['exchange_code'] if 'exchange_code' in meta else pd.Series(index=meta.index, dtype=object)
cur = meta['currency'] if 'currency' in meta else pd.Series(index=meta.index, dtype=object)
def region(t):
    c = str(cur.get(t, '')).upper(); x = str(exch.get(t, '')).upper()
    if c == 'USD' and x in ('', 'NONE', 'NAN', 'US', 'UN', 'UW', 'UA', 'UQ', 'UR'):
        return 'US'
    if c in ('JPY', 'KRW', 'TWD', 'HKD', 'CNY', 'INR', 'AUD', 'SGD'):
        return 'ASIA'
    if c in ('EUR', 'GBP', 'GBp', 'CHF', 'DKK', 'SEK', 'NOK'):
        return 'EUROPE'
    return 'OTHER:' + c + '/' + x
reg = pd.Series({t: region(t) for t in tickers})
print('region counts', reg.value_counts().to_dict())
us = [t for t in tickers if reg[t] == 'US']
mkt_us = ret[us].mean(axis=1)
mkt_all = ret.mean(axis=1)
arows = []
for t in tickers:
    s = ret[t]
    df = pd.DataFrame({'r': s, 'm0': mkt_us, 'm_lag1': mkt_us.shift(1), 'm_lead1': mkt_us.shift(-1), 'a0': mkt_all}).dropna()
    if len(df) < 500:
        continue
    def beta(x, y):
        return float(np.cov(x, y)[0, 1] / np.var(y, ddof=1))
    b0 = beta(df['r'], df['m0']); bl = beta(df['r'], df['m_lag1']); bf = beta(df['r'], df['m_lead1'])
    arows.append({'ticker': t, 'region': reg[t], 'currency': cur.get(t), 'exchange': exch.get(t),
                  'corr_lag0_us': float(df['r'].corr(df['m0'])), 'corr_stock_t_vs_us_tm1': float(df['r'].corr(df['m_lag1'])),
                  'corr_stock_t_vs_us_tp1': float(df['r'].corr(df['m_lead1'])), 'beta0_us': b0, 'beta_lag1_us': bl, 'beta_lead1_us': bf,
                  'dimson_beta': b0 + bl + bf, 'dimson_ratio': (b0 + bl + bf) / b0 if abs(b0) > 1e-9 else np.nan,
                  'corr_lag0_all': float(df['r'].corr(df['a0']))})
AS = pd.DataFrame(arows).set_index('ticker')
AS.round(4).to_csv(os.path.join(OUT, 'async_probe.csv'))
G = AS.groupby('region')[['corr_lag0_us', 'corr_stock_t_vs_us_tm1', 'corr_stock_t_vs_us_tp1', 'beta0_us', 'beta_lag1_us', 'dimson_beta', 'dimson_ratio', 'corr_lag0_all']].median()
print('=== async by region (median) ===')
print(G.round(3).to_string())
print('=== ASIA names ===')
print(AS[AS['region'] == 'ASIA'][['currency', 'corr_lag0_us', 'corr_stock_t_vs_us_tm1', 'beta0_us', 'beta_lag1_us', 'dimson_ratio']].round(3).to_string())
# region-level cross-correlation of region EW returns with US EW
regs = {}
for rname in ['ASIA', 'EUROPE']:
    tk = [t for t in tickers if reg[t] == rname]
    if not tk:
        continue
    m = ret[tk].mean(axis=1)
    d = pd.DataFrame({'m': m, 'us0': mkt_us, 'us_lag1': mkt_us.shift(1)}).dropna()
    regs[rname] = {'n': len(tk), 'corr_same_day': float(d['m'].corr(d['us0'])), 'corr_region_t_vs_us_tm1': float(d['m'].corr(d['us_lag1']))}
print('region EW cross-corr', regs)

# ---- (3) tg_upside raw ratio ----
tg = sheets['Factset_TG_Price'].reindex(columns=tickers).astype(float)
px = sheets['PX_LAST'].reindex(columns=tickers).astype(float)
scale_map = {t: (0.01 if str(exch.get(t, '')).upper() == 'LN' else 1.0) for t in tickers}
px_s = px.mul(pd.Series(scale_map), axis=1)
up = tg / px_s - 1
um = up.median().sort_values()
tgj = {'bottom12': um.head(12).round(3).to_dict(), 'top12': um.tail(12).round(3).to_dict(),
       'median_all': float(um.median()), 'tg_px_ratio_median_by_ticker_outside_0.5_2': {t: round(float(v), 3) for t, v in (tg / px_s).median().items() if not (0.5 <= v <= 2.0)}}
print('=== tg_upside raw median ===', json.dumps(tgj, ensure_ascii=False)[:1500])
json.dump({'negative_equity_flags': flag.index.tolist(), 'async_by_region': G.round(4).to_dict(), 'region_ew_xcorr': regs, 'tg': tgj,
           'returns_scale': scale, 'region_counts': reg.value_counts().to_dict()}, open(os.path.join(OUT, 'raw_probe.json'), 'w'), indent=1, default=str)
print('done', round(time.time() - t0, 1), 's')
