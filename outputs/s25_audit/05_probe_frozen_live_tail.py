import sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, r"C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port")
from src import data_loader as dl
raw = pickle.load(open(sys.argv[1], "rb"))
raw2, _ = dl.restrict_to_business_days(raw)
meta = dl.load_universe_meta(raw2); tickers = list(meta.index)
def prep(name):
    df = dl._standardize_columns(raw2[name].copy())
    if name in dl.BLOOMBERG_EQUITY_SHEETS: df = dl._rename_bloomberg_equity_columns(df)
    df = dl._filter_tickers(dl._standardize_index(df), tickers=tickers)
    return df.apply(pd.to_numeric, errors="coerce")
for sh in ["BEST_PX_BPS_RATIO", "BEST_PE_RATIO", "BEST_PEG_RATIO", "BEST_CALCULATED_FCF", "BEST_CAPEX", "BEST_GROSS_MARGIN", "BEST_EV_TO_BEST_EBITDA"]:
    df = prep(sh); v = df.to_numpy(float)
    eq = np.isclose(v, v[-1], rtol=1e-10, atol=1e-12, equal_nan=True)
    tr = pd.Series([(np.argmax(~eq[::-1, j]) if (~eq[:, j]).any() else len(v)) for j in range(v.shape[1])], index=df.columns)
    st = tr[tr >= 252].sort_values(ascending=False)
    print(f"{sh}: frozen >=252 rows at the live tail: {len(st)} names ->",
          ", ".join(f"{t}({n}r, since {df.index[-n].date() if n < len(df) else df.index[0].date()}, val {df[t].iloc[-1]:.2f}, rank {df.iloc[-1].rank(pct=True)[t]:.2f})" for t, n in st.head(14).items()))
