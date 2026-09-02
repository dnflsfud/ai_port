import pickle, sys, os
import numpy as np, pandas as pd
V = r"C:/Users/westl/AppData/Local/Temp/claude/C--Users-westl-PycharmProjects-pythonProject-venv-vf-new-machine-re-study-c2-ai-port/0caf960f-9660-4a4b-a91e-809a2f4ab53d/scratchpad/w3/verify"
d = pickle.load(open(os.path.join(V, "m2_sheets.pkl"), "rb"))
print("m2 keys:", list(d.keys()))
for k, v in d.items():
    if isinstance(v, pd.DataFrame):
        print(k, v.shape, v.index.min(), v.index.max(), "n", len(v))
px = d["PX_LAST"]; rt = d["Daily_Returns"]; meta = d["Universe_Meta"]
print("meta cols", list(meta.columns)); print(meta.head(3))
jp = [t for t in ["4063","6146","6758","6857","6981","7011","7203","7974","8035","8306","8316","9432","9433"] if t in px.columns]
print("JP present", len(jp), jp)
print("--- PX_LAST JP 2013-12-27..2014-01-09 ---")
print(px.loc["2013-12-27":"2014-01-09", jp].round(2).to_string())
print("--- PX_LAST US/EU/KR sample same window ---")
smp = [c for c in ["AAPL","MSFT","ASML","SAP","005930","000660","HSBA","NOVO"] if c in px.columns]
print(px.loc["2013-12-27":"2014-01-09", smp].round(2).to_string())
print("--- 7203 & NKY around 2024-08-05 ---")
idx = d.get("Index_PX_LAST_subset")
print(px.loc["2024-08-01":"2024-08-08", ["7203","6857"]].round(1).to_string())
if idx is not None:
    print("Index subset cols", list(idx.columns))
    print(idx.loc["2024-08-01":"2024-08-08"].round(2).to_string())
print("--- T-09(a): US ticker returns on rebal rows at US holidays ---")
us = meta.index[meta["exchange_code"].astype(str).str.upper().isin(["US","UN","UW","UQ","UA"])].tolist() if "exchange_code" in meta.columns else []
print("n US by exchange_code", len(us), "exchange codes:", meta["exchange_code"].value_counts().to_dict())
for h in ["2018-12-25","2024-02-19","2025-12-25","2026-06-19"]:
    h = pd.Timestamp(h)
    if h in rt.index:
        r = rt.loc[h]
        rus = r.reindex(us)
        print(h.date(), "in Daily_Returns; US n", rus.notna().sum(), "US zero", int((rus.abs() < 1e-12).sum()), "US nonzero", int((rus.abs() >= 1e-12).sum()), "US NaN", int(rus.isna().sum()), "| nonUS nonzero", int((r.drop(us, errors='ignore').abs() >= 1e-12).sum()))
    else:
        print(h.date(), "NOT in Daily_Returns index")
