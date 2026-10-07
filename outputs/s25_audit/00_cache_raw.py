import sys, time, pickle
sys.path.insert(0, r"C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port")
from src.data_loader import load_all_sheets
t = time.time()
raw = load_all_sheets(r"C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\ai_signal_data.xlsx")
out = sys.argv[1]
with open(out, "wb") as f:
    pickle.dump(raw, f, protocol=4)
print("sheets", len(raw), "elapsed", round(time.time() - t, 1))
for k, v in raw.items():
    print(k, v.shape)
