"""§S25.3 generator-pair mechanism probe (read-only).

Compares the OLD-generator workbook (working tree minus the 2026-10-07 call sites) with the NEW-generator
workbook (S25 bfill-prefix masking + S25.1 cleaning), both built from the same 2026-10-02 Bloomberg pull.
Expected: every difference is a cell that is NaN in NEW and non-NaN in OLD (masking / cleaning); no other
value changes; identical sheet sets, date ranges and ticker columns.

usage: python pair_workbook_diff.py <old.xlsx> <new.xlsx> <out.csv>
"""
import sys, time
import numpy as np, pandas as pd

sys.path.insert(0, r"C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_port")
from src.data_loader import load_all_sheets, _rename_bloomberg_equity_columns, _standardize_columns, _standardize_index  # noqa: E402


def std(df):
    df = _rename_bloomberg_equity_columns(_standardize_columns(df.copy()))
    try:
        return _standardize_index(df)
    except Exception:
        return df


old_path, new_path, out_csv = sys.argv[1:4]
t = time.time()
import pickle, os
def cached(path, tag):
    cache = os.path.join(os.path.dirname(out_csv), f"wb_cache_{tag}.pkl")
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    d = load_all_sheets(path)
    pickle.dump(d, open(cache, "wb"), protocol=4)
    return d
old = cached(old_path, "old"); print("old loaded", len(old), round(time.time() - t))
new = cached(new_path, "new"); print("new loaded", len(new), round(time.time() - t))

print("sheet sets equal:", set(old) == set(new), "| only old:", sorted(set(old) - set(new)), "| only new:", sorted(set(new) - set(old)))
rows = []
other_changes = {}
for name in sorted(set(old) & set(new)):
    a, b = std(old[name]), std(new[name])
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        print(f"[SHAPE/COLUMNS DIFFER] {name}: old {a.shape} new {b.shape}")
        continue
    if not a.index.equals(b.index):
        print(f"[INDEX DIFFERS] {name}")
    an = a.apply(pd.to_numeric, errors="coerce"); bn = b.apply(pd.to_numeric, errors="coerce")
    masked = an.notna() & bn.isna()
    unmasked = an.isna() & bn.notna()
    both = an.notna() & bn.notna()
    changed = both.to_numpy(dtype=bool) & ~np.isclose(an.fillna(0).to_numpy(dtype=float), bn.fillna(0).to_numpy(dtype=float), rtol=0, atol=1e-9, equal_nan=True)
    n_masked, n_unmasked, n_changed = int(masked.to_numpy(dtype=bool).sum()), int(unmasked.to_numpy(dtype=bool).sum()), int(changed.sum())
    if n_unmasked or n_changed:
        other_changes[name] = (n_unmasked, n_changed)
    if n_masked:
        per_ticker = masked.sum()
        per_ticker = per_ticker[per_ticker > 0].sort_values(ascending=False)
        idx = pd.to_datetime(an.index, errors="coerce") if not isinstance(an.index, pd.DatetimeIndex) else an.index
        for tk, cnt in per_ticker.items():
            m = masked[tk].to_numpy(dtype=bool)
            first_masked = idx[m].min(); last_masked = idx[m].max()
            first_new_valid = idx[bn[tk].notna().to_numpy()].min() if bn[tk].notna().any() else pd.NaT
            rows.append({"sheet": name, "ticker": tk, "masked_cells": int(cnt),
                         "first_masked": first_masked.date() if pd.notna(first_masked) else None,
                         "last_masked": last_masked.date() if pd.notna(last_masked) else None,
                         "first_valid_new": first_new_valid.date() if pd.notna(first_new_valid) else None,
                         "old_value_at_first_masked": float(an[tk].to_numpy()[m][0]) if cnt else None,
                         "all_masked": bool(bn[tk].isna().all())})
df = pd.DataFrame(rows)
df.to_csv(out_csv, index=False)
print("\nOTHER changes (unmasked / value-changed cells) by sheet:", other_changes or "NONE")
# Derived 252D rolling z sheets legitimately change in the window after a masked raw prefix; beyond that window the only
# admissible difference is a NaN<->0.0 flip inside a constant run (pandas rolling-std floating state). Classify and save.
import json
Z_PAIRS = {"iv30_z": "iv30", "iv_term_structure_z": "iv_term_structure", "downside_skew_z": "downside_skew", "vol_risk_premium_z": "vol_risk_premium"}
zclass = {}
for z, raw in Z_PAIRS.items():
    if z not in new or raw not in new:
        continue
    a, b = std(old[z]).apply(pd.to_numeric, errors="coerce"), std(new[z]).apply(pd.to_numeric, errors="coerce")
    ra, rb = std(old[raw]).apply(pd.to_numeric, errors="coerce"), std(new[raw]).apply(pd.to_numeric, errors="coerce")
    masked_raw = (ra.notna() & rb.isna()).to_numpy(dtype=bool)
    within = np.zeros(a.shape, bool)
    for j in range(a.shape[1]):
        m = masked_raw[:, j]
        last = np.where(m)[0].max() if m.any() else -1
        within[: last + 253, j] = True
    status_diff = (a.notna() != b.notna()).to_numpy(dtype=bool)
    val_diff = (a.notna() & b.notna()).to_numpy(dtype=bool) & ~np.isclose(a.fillna(0).to_numpy(dtype=float), b.fillna(0).to_numpy(dtype=float), atol=1e-9)
    beyond = ~within & (status_diff | val_diff)
    av, bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    nan_to_zero = beyond & np.isnan(av) & (bv == 0.0)
    zero_to_nan = beyond & np.isnan(bv) & (av == 0.0)
    other = beyond & ~nan_to_zero & ~zero_to_nan
    zclass[z] = {"within_window_changes": int(((status_diff | val_diff) & within).sum()), "beyond_window_cells": int(beyond.sum()),
                 "beyond_nan_to_zero": int(nan_to_zero.sum()), "beyond_zero_to_nan": int(zero_to_nan.sum()), "beyond_other": int(other.sum()),
                 "beyond_tickers": sorted(set(a.columns[beyond.any(axis=0)]))}
json.dump(zclass, open(out_csv.replace(".csv", "_zclass.json"), "w", encoding="utf-8"), indent=1)
print("Z-SHEET CLASSIFICATION:", json.dumps(zclass))
if len(df):
    print("\nmasked cells by sheet:\n", df.groupby("sheet")["masked_cells"].agg(["sum", "count"]).sort_values("sum", ascending=False).to_string())
    print("\ntop 25 (sheet, ticker):\n", df.sort_values("masked_cells", ascending=False).head(25).to_string(index=False))
    print("\nall-masked columns:", df[df["all_masked"]][["sheet", "ticker"]].to_dict("records"))
    for s, tk in (("BEST_PE_RATIO", "UBER"), ("BEST_ROE", "FICO"), ("EQY_REC_CONS", "AON"), ("OPER_MARGIN", "CS")):
        hit = df[(df["sheet"] == s) & (df["ticker"] == tk)]
        print(f"check {s}/{tk}:", hit.to_dict("records") if len(hit) else "no masked cells")
for name in ("PX_LAST", "BEST_PE_RATIO", "NEWS_SENTIMENT_DAILY_AVG"):
    if name in new:
        idx = pd.to_datetime(std(new[name]).index, errors="coerce")
        print(f"{name}: new date range {idx.min().date()} .. {idx.max().date()} rows {len(idx)}")
print("elapsed", round(time.time() - t))
