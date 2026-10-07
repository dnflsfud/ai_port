"""Read-only probe: leading flat (source-bfilled) prefix per (sheet, ticker) vs listing date."""
import sys, pickle, json
import numpy as np, pandas as pd
sys.path.insert(0, r"C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port")
from src import data_loader as dl
from src.config import PipelineConfig

raw = pickle.load(open(sys.argv[1], "rb"))
cfg = PipelineConfig()
raw2, _ = dl.restrict_to_business_days(raw)
meta = dl.load_universe_meta(raw2)
tickers = list(meta.index)
m = json.load(open(r"C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port\outputs\codex_causal_rank_65\metrics.json", encoding="utf-8"))
listing = {t: pd.Timestamp(d) for t, d in m["data_quality"]["listing_mask"]["dates"].items()}

def prep(name):
    df = raw2[name].copy()
    df = dl._standardize_columns(df)
    if name in dl.SENT_TREND_SHEETS:
        return None
    if name in dl.BLOOMBERG_EQUITY_SHEETS:
        df = dl._rename_bloomberg_equity_columns(df)
    try:
        df = dl._standardize_index(df)
    except Exception:
        return None
    df = dl._filter_tickers(df, tickers=tickers)
    if df.shape[1] < 50:
        return None
    return df.apply(pd.to_numeric, errors="coerce")

rows = []
for name in raw2:
    if name in dl.SKIP_SHEETS or name in dl.FACTOR_SHEETS:
        continue
    df = prep(name)
    if df is None:
        continue
    idx = df.index
    start = idx[0]
    for t in df.columns:
        s = df[t]
        lst = listing.get(t, start)
        post = s[s.index >= lst]
        nn = post.dropna()
        if nn.empty:
            rows.append((name, t, "all_nan", 0, 0, 0, None, None)); continue
        first_obs = nn.index[0]
        lead_nan = int((post.index < first_obs).sum())
        vals = nn.to_numpy(float)
        same = np.isclose(vals, vals[0], rtol=1e-10, atol=1e-12)
        chg = np.flatnonzero(~same)
        lead_run = int(chg[0]) if len(chg) else len(vals)
        # max later constant run
        later = vals[lead_run:]
        if len(later) > 1:
            d = np.flatnonzero(~np.isclose(later[1:], later[:-1], rtol=1e-10, atol=1e-12))
            edges = np.concatenate(([-1], d, [len(later) - 1]))
            max_later = int(np.diff(edges).max())
        else:
            max_later = 0
        first_change = nn.index[lead_run] if lead_run < len(nn) else None
        rows.append((name, t, "ok", lead_nan, lead_run, max_later,
                     str(first_obs.date()), str(first_change.date()) if first_change is not None else None))
out = pd.DataFrame(rows, columns=["sheet", "ticker", "status", "lead_nan", "lead_run", "max_later_run", "first_obs", "first_change"])
out["listing"] = out["ticker"].map(lambda t: str(listing.get(t, pd.NaT))[:10])
out.to_csv(sys.argv[2], index=False)
out["suspect"] = (out["lead_run"] >= 40) & (out["lead_run"] > 3 * out["max_later_run"].clip(lower=1))
g = out.groupby("sheet").agg(n=("ticker", "size"), lead_nan_names=("lead_nan", lambda x: int((x > 0).sum())),
                             suspect=("suspect", "sum"), med_lead_run=("lead_run", "median"),
                             p90_lead_run=("lead_run", lambda x: float(np.percentile(x, 90))),
                             max_lead_run=("lead_run", "max"), med_later=("max_later_run", "median"))
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(g.sort_values("suspect", ascending=False).to_string())
