"""M2 verify — step 0: extract the workbook sheets needed (cached to scratchpad).

Run from ai_port with PYTHONPATH=. ; writes only into the scratchpad verify dir.
"""
import os, sys, time, pickle
import pandas as pd
import numpy as np

sys.path.insert(0, os.getcwd())
from src.config import PipelineConfig
from src.data_loader import _rename_bloomberg_equity_columns, load_universe_meta

OUT = r"C:\Users\westl\AppData\Local\Temp\claude\C--Users-westl-PycharmProjects-pythonProject-venv-vf-new-machine-re-study-c2-ai-port\0caf960f-9660-4a4b-a91e-809a2f4ab53d\scratchpad\w3\verify"
cfg = PipelineConfig()
print("data_path", cfg.data_path)
print("fx_source_path", cfg.fx_source_path)

t0 = time.time()
sheets = pd.read_excel(cfg.data_path, sheet_name=["Daily_Returns", "PX_LAST", "Universe_Meta", "CUR_MKT_CAP"])
print("read workbook sheets in %.0fs" % (time.time() - t0))

out = {}
for name in ["Daily_Returns", "PX_LAST", "CUR_MKT_CAP"]:
    df = sheets[name]
    print(name, df.shape, "date dtype:", df["date"].dtype, "first raw date:", df["date"].iloc[0])
    df = df.set_index(pd.to_datetime(df["date"])).drop(columns=["date"]).sort_index()
    df = _rename_bloomberg_equity_columns(df)
    df = df.apply(pd.to_numeric, errors="coerce")
    print("   ->", df.shape, df.index.min(), df.index.max(), "cols sample", list(df.columns[:5]))
    out[name] = df

meta = load_universe_meta({"Universe_Meta": sheets["Universe_Meta"]})
print("Universe_Meta", meta.shape, list(meta.columns))
print(meta["currency"].value_counts())
print(meta["exchange_code"].value_counts())
out["Universe_Meta"] = meta

# Index.xlsx: header, then NKY / SPX / FX columns
t0 = time.time()
hdr = pd.read_excel(cfg.fx_source_path, sheet_name="PX_LAST", nrows=0).columns.tolist()
print("Index.xlsx PX_LAST header (%d cols) in %.0fs" % (len(hdr), time.time() - t0))
print(hdr)
want = [c for c in hdr if str(c).strip().casefold() == "date"]
for c in hdr:
    cu = str(c).upper()
    if any(k in cu for k in ["NKY", "SPX", "USDJPY", "USDKRW", "EURUSD", "USDCHF", "GBPUSD", "USDDKK", "KOSPI", "TPX"]):
        want.append(c)
want = list(dict.fromkeys(want))
print("reading Index.xlsx cols:", want)
t0 = time.time()
idx = pd.read_excel(cfg.fx_source_path, sheet_name="PX_LAST", usecols=want)
dcol = want[0]
idx[dcol] = pd.to_datetime(idx[dcol], errors="coerce")
idx = idx.dropna(subset=[dcol]).drop_duplicates(subset=[dcol], keep="last").set_index(dcol).sort_index()
idx = idx.apply(pd.to_numeric, errors="coerce")
print("Index.xlsx subset", idx.shape, idx.index.min(), idx.index.max(), "in %.0fs" % (time.time() - t0))
print(idx.tail(3))
out["Index_PX_LAST_subset"] = idx

# Also check the workbook's own Factor sheets for NKY
try:
    fhdr = pd.read_excel(cfg.data_path, sheet_name="Factor_Returns", nrows=0).columns.tolist()
    print("Factor_Returns header:", fhdr)
    out["Factor_Returns_header"] = fhdr
except Exception as e:
    print("Factor_Returns header read failed:", e)

with open(os.path.join(OUT, "m2_sheets.pkl"), "wb") as f:
    pickle.dump(out, f, protocol=4)
print("saved m2_sheets.pkl")
