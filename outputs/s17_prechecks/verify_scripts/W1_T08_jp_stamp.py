"""W1 T-08: is the JP PX_LAST series T+1 stamped? Uses M2 sheet cache (m2_sheets.pkl)."""
import pickle, os, json
import numpy as np, pandas as pd
V = os.environ["V"]
d = pickle.load(open(os.path.join(V, "m2_sheets.pkl"), "rb"))
px = d["PX_LAST"]; rt = d["Daily_Returns"]; meta = d["Universe_Meta"]; idx = d["Index_PX_LAST_subset"]
jp = [t for t in meta.index[meta["exchange_code"].astype(str) == "JP"] if t in px.columns and t != "285A"]
us = [t for t in meta.index[meta["exchange_code"].astype(str) == "US"] if t in px.columns]
out = {"n_jp": len(jp)}

# (1) year-start rows: leading flat run for JP in 2014 and whether it recurs in 2015/2016
for y, first_td in [("2014", "2014-01-06"), ("2015", "2015-01-05"), ("2016", "2016-01-04"), ("2017", "2017-01-04")]:
    win = px.loc[f"{y}-01-01":f"{y}-01-09", jp]
    flat = (win.diff().abs() < 1e-9).all(axis=1)
    out[f"jp_flat_rows_{y}"] = [str(k.date()) for k, v in flat.items() if v]
    out[f"jp_first_change_{y}"] = str(win.index[(win.diff().abs() > 1e-9).any(axis=1)][0].date())
    out[f"jp_ret_on_first_td_{y}"] = float(rt.loc[first_td, jp].mean()) if first_td in rt.index else None
# NKY / TPX in Index.xlsx around 2014-01-06
nk = idx.loc["2014-01-01":"2014-01-09", ["NKY Index", "TPX Index", "SPX Index", "USDJPY Curncy"]]
out["index_2014_jan"] = {str(k.date()): [round(float(x), 2) for x in v] for k, v in nk.iterrows()}
out["NKY_ret_2014-01-07"] = float(idx["NKY Index"].loc["2014-01-07"] / idx["NKY Index"].loc["2014-01-06"] - 1)
out["7203_ret_2014-01-07_px"] = float(px.loc["2014-01-07", "7203"] / px.loc["2014-01-06", "7203"] - 1)
out["7203_ret_2014-01-07_sheet"] = float(rt.loc["2014-01-07", "7203"])
out["7203_ret_2014-01-06_sheet"] = float(rt.loc["2014-01-06", "7203"])
out["jp_ew_ret_2014-01-07_sheet"] = float(rt.loc["2014-01-07", jp].mean())
out["jp_ew_ret_2014-01-08_sheet"] = float(rt.loc["2014-01-08", jp].mean())
out["NKY_ret_2014-01-08"] = float(idx["NKY Index"].loc["2014-01-08"] / idx["NKY Index"].loc["2014-01-07"] - 1)

# (2) JP-only holidays (US open): flat on the holiday row (no shift) or on the next row (T+1)?
jp_hol = ["2014-01-13", "2014-02-11", "2014-03-21", "2014-04-29", "2014-05-05", "2014-05-06", "2014-07-21", "2014-09-15", "2014-09-23",
          "2014-10-13", "2014-11-03", "2014-11-24", "2014-12-23", "2015-01-12", "2015-02-11", "2015-04-29", "2015-05-04", "2015-05-05",
          "2015-05-06", "2015-07-20", "2015-09-21", "2015-09-22", "2015-09-23", "2015-10-12", "2015-11-03", "2015-11-23", "2015-12-23",
          "2016-01-11", "2016-02-11", "2016-03-21", "2016-04-29", "2016-05-03", "2016-05-04", "2016-05-05", "2016-07-18", "2016-09-19",
          "2016-09-22", "2016-10-10", "2016-11-03", "2016-11-23", "2016-12-23", "2024-01-08", "2024-02-12", "2024-02-23", "2024-03-20",
          "2024-04-29", "2024-05-03", "2024-05-06", "2024-07-15", "2024-08-12", "2024-09-16", "2024-09-23", "2024-10-14", "2024-11-04"]
rows = []
for h in jp_hol:
    h = pd.Timestamp(h)
    if h not in px.index: continue
    i = px.index.get_loc(h)
    nxt = px.index[i + 1]
    flat_h = float((rt.loc[h, jp].abs() < 1e-9).mean())
    flat_n = float((rt.loc[nxt, jp].abs() < 1e-9).mean())
    us_h = float((rt.loc[h, us].abs() < 1e-9).mean())
    rows.append({"holiday": str(h.date()), "jp_flat_frac_on_holiday": flat_h, "jp_flat_frac_next_row": flat_n, "us_flat_frac_on_holiday": us_h, "weekday": h.day_name()})
df = pd.DataFrame(rows)
out["jp_holiday_test_n"] = len(df)
out["jp_flat_on_holiday_mean"] = float(df["jp_flat_frac_on_holiday"].mean())
out["jp_flat_next_row_mean"] = float(df["jp_flat_frac_next_row"].mean())
out["us_flat_on_jp_holiday_mean"] = float(df["us_flat_frac_on_holiday"].mean())
out["n_holidays_jp_fully_flat_on_day"] = int((df["jp_flat_frac_on_holiday"] >= 0.99).sum())
out["n_holidays_jp_fully_flat_next"] = int((df["jp_flat_frac_next_row"] >= 0.99).sum())
df.to_csv(os.path.join(V, "W1_T08_jp_holidays.csv"), index=False)

# (3) US holidays when JP open: JP should move on the row, US flat
us_hol = ["2014-01-20", "2014-02-17", "2014-05-26", "2014-07-04", "2014-09-01", "2014-11-27", "2015-01-19", "2015-02-16", "2015-05-25",
          "2015-07-03", "2015-09-07", "2015-11-26", "2024-01-15", "2024-02-19", "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28"]
rows2 = []
for h in us_hol:
    h = pd.Timestamp(h)
    if h not in px.index: continue
    rows2.append({"us_holiday": str(h.date()), "us_flat": float((rt.loc[h, us].abs() < 1e-9).mean()), "jp_flat": float((rt.loc[h, jp].abs() < 1e-9).mean())})
df2 = pd.DataFrame(rows2)
out["us_holiday_test"] = {"n": len(df2), "us_flat_mean": float(df2["us_flat"].mean()), "jp_flat_mean": float(df2["jp_flat"].mean())}

# (4) lag cross-correlation 7203/JP-EW vs NKY (in-sample redo of M2 step2b)
nky = idx["NKY Index"].pct_change()
jpew = rt[jp].mean(axis=1)
common = jpew.index.intersection(nky.index)
x = jpew.reindex(common); yv = nky.reindex(common)
out["xcorr_JPEW_NKY"] = {f"lag{k}": float(x.corr(yv.shift(k))) for k in [-1, 0, 1, 2]}
out["xcorr_7203_NKY"] = {f"lag{k}": float(rt["7203"].reindex(common).corr(yv.shift(k))) for k in [-1, 0, 1, 2]}
# 2024-08-05
out["7203_ret_2024-08-05_sheet"] = float(rt.loc["2024-08-05", "7203"]); out["7203_ret_2024-08-06_sheet"] = float(rt.loc["2024-08-06", "7203"])
out["NKY_ret_2024-08-05"] = float(idx["NKY Index"].loc["2024-08-05"] / idx["NKY Index"].loc["2024-08-02"] - 1)
json.dump(out, open(os.path.join(V, "W1_T08.json"), "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
