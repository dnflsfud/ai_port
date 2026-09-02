import sys, pickle, time
sys.path.insert(0, r"C:/Users/westl/PycharmProjects/pythonProject/venv_vf_new/machine/re_study/c2/ai_port")
import numpy as np, pandas as pd
t0=time.time()
with open(r"C:/Users/westl/PycharmProjects/pythonProject/venv_vf_new/machine/re_study/c2/ai_port/outputs/s16_7_name_risk_cap/backtest_result.pkl","rb") as fh:
    r = pickle.load(fh)
print("load sec", round(time.time()-t0,1))
print([a for a in dir(r) if not a.startswith("__")][:80])
pw = r.portfolio_weights
k = sorted(pw.keys())
print("pw n", len(k), type(k[0]), k[0], k[-1], type(pw[k[0]]))
v = pw[k[-1]]
print("pw value:", getattr(v,'shape',None), (v.index[:5].tolist() if hasattr(v,'index') else None), (v.sum() if hasattr(v,'sum') else None))
print("weights nonzero", int((np.asarray(v)>1e-6).sum()))
m = r.models
mk = sorted(m.keys())
print("models n", len(mk), mk[0], type(m[mk[0]]))
mm = m[mk[-1]]
af = getattr(mm, "_active_features", None)
print("active feats", None if af is None else (len(af), af[:5]))
print("booster feats", mm.booster_.num_feature(), mm.booster_.feature_name()[:5])
print("panel", r.panel.shape, r.panel.index.names, r.panel.columns[:5].tolist())
tg = r.targets
print("targets", tg.shape, tg.index[:2].tolist(), "nan pct", float(tg.isna().mean().mean()))
print("targets first valid date", tg.dropna(how='all').index.min(), "PLTR first valid", tg['PLTR'].first_valid_index(), "GEV", tg['GEV'].first_valid_index(), "285A", tg['285A'].first_valid_index())
print("targets row valid count median", tg.notna().sum(axis=1).median(), "min>0", tg.notna().sum(axis=1)[tg.notna().sum(axis=1)>0].min())
print("preds first valid", r.predictions.dropna(how='all').index.min())
print("tickers panel==targets cols", list(r.panel.index.get_level_values('ticker').unique()) == list(tg.columns))
print("feature_names n", len(r.feature_names))
print("ic_series", r.ic_series.index[:2].tolist(), len(r.ic_series))
print("daily_weights", type(r.daily_weights), getattr(r.daily_weights,'shape',None))
print("benchmark weights attr?", [a for a in dir(r) if 'bench' in a.lower() or 'bm' in a.lower()])
