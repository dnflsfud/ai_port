"""M3 step 0: read the 5 workbook sheets once and cache as pickles (scratch only)."""
import sys, time, pickle
sys.path.insert(0, r"C:/Users/westl/PycharmProjects/pythonProject/venv_vf_new/machine/re_study/c2/ai_port")
import pandas as pd
from src.config import PipelineConfig
from src.data_loader import _rename_bloomberg_equity_columns

CACHE = r"C:/Users/westl/AppData/Local/Temp/claude/C--Users-westl-PycharmProjects-pythonProject-venv-vf-new-machine-re-study-c2-ai-port/0caf960f-9660-4a4b-a91e-809a2f4ab53d/scratchpad/w3/verify/cache"
SHEETS = ["BEST_ROE", "BEST_PX_BPS_RATIO", "BEST_PE_RATIO", "BEST_EPS", "CUR_MKT_CAP"]
path = PipelineConfig().data_path
t0 = time.time()
raw = pd.read_excel(path, sheet_name=SHEETS)
print("read_excel sec", round(time.time() - t0, 1))
for name, df in raw.items():
    if "date" in df.columns:
        df = df.set_index(pd.to_datetime(df["date"])).drop(columns=["date"]).sort_index()
    else:
        first = df.columns[0]
        df = df.set_index(pd.to_datetime(df[first])).drop(columns=[first]).sort_index()
    df = _rename_bloomberg_equity_columns(df)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc["2014-01-01":]
    df = df.apply(pd.to_numeric, errors="coerce")
    with open(f"{CACHE}/{name}.pkl", "wb") as fh:
        pickle.dump(df, fh)
    print(name, df.shape, df.index.min().date(), df.index.max().date(), "ncol_equity", len(df.columns))
