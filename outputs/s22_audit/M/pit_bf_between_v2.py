"""M-01 프로브: PIT 롤링 1FY/2FY 라면 블렌디드 선행(BEST_SALES, 1BF)은 [min(1FY,2FY), max] 안에 있어야 한다.
슬로프 시트 유효 셀(생성기 마스크 후 non-NaN, 평일)에서 연도별 포함 비율과 셀 수. 허용오차 ±2%."""
import json, sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from scripts.precheck_s20_fy2_sales_revision import read_sheet
b, out = sys.argv[1], sys.argv[2]
sl = read_sheet(b, "Fwd_Sales_Slope_1FY2FY"); sl.columns = [str(x).split()[0] for x in sl.columns]
bf, s1, s2 = (read_sheet(b, n) for n in ("BEST_SALES", "BEST_SALES_1FY", "BEST_SALES_2FY"))
d = sl.index.intersection(bf.index).intersection(s1.index).intersection(s2.index)
d = d[d.dayofweek < 5]
c = [x for x in sl.columns if x in bf.columns and x in s1.columns and x in s2.columns]
S, B, F1, F2 = (x.loc[d, c].to_numpy(dtype=float) for x in (sl, bf, s1, s2))
valid = np.isfinite(S) & np.isfinite(B) & np.isfinite(F1) & np.isfinite(F2)
inside = (B >= np.minimum(F1, F2) * 0.98) & (B <= np.maximum(F1, F2) * 1.02)
years = np.asarray(d.year)
rows = []
for y in sorted(set(years)):
    m = valid & (years == y)[:, None]
    n = int(m.sum())
    rows.append({"year": int(y), "valid_cells": n,
                 "share_inside": round(float(inside[m].mean()), 4) if n else None,
                 "median_abs_log_bf_over_1fy": round(float(np.median(np.abs(np.log(B[m] / F1[m])))), 3) if n else None})
    print(rows[-1])
json.dump({"probe": "M-01 BF-bracketing PIT test", "tolerance": 0.02, "tickers": len(c), "by_year": rows},
          open(out, "w", encoding="utf-8"), indent=1)
