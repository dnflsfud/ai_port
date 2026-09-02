import pickle, sys, time, os
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
t0=time.time()
r = pickle.load(open("outputs/s16_7_name_risk_cap/backtest_result.pkl","rb"))
print("loaded %.1fs type %s" % (time.time()-t0, type(r)))
attrs = [a for a in dir(r) if not a.startswith("__")]
for a in attrs:
    try: v = getattr(r, a)
    except Exception as e: print(a, "ERR", e); continue
    if callable(v): continue
    if isinstance(v, pd.DataFrame): print(f"{a}: DataFrame {v.shape} idx[{v.index[0]}..{v.index[-1]}]")
    elif isinstance(v, pd.Series): print(f"{a}: Series {v.shape} idx[{v.index[0]}..{v.index[-1]}] head={v.head(2).to_dict()}")
    elif isinstance(v, dict): 
        ks=list(v.keys()); print(f"{a}: dict n={len(v)} keys[:3]={ks[:3]} valtype={type(v[ks[0]]) if ks else None}")
    elif isinstance(v, (list,tuple)): print(f"{a}: {type(v).__name__} n={len(v)} first={str(v[:2])[:200]}")
    elif isinstance(v, np.ndarray): print(f"{a}: ndarray {v.shape}")
    else: print(f"{a}: {type(v).__name__} = {str(v)[:300]}")
