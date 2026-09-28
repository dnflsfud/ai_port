"""M-01 프로브 3: ZS(7월 결산, 9월 초 연간 발표)의 슬로프 이력이 09-03 → 09-11 번들 사이에 연도 전반에서 바뀌었는가.
대조군 AAPL(9월 결산, 10월 말 발표 — 이 구간에 롤 없음)."""
import gc, pickle, sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
out = {}
for k, p in {"0903": "outputs/s17_2_s0recert/backtest_result.pkl", "0911": "outputs/s18_7_s0recert/backtest_result.pkl"}.items():
    r = pickle.load(open(p, "rb"))
    out[k] = r.panel["fwd_sales_slope_level"].unstack("ticker")[["ZS", "AAPL", "MSFT"]]
    del r; gc.collect()
a, b = out["0903"], out["0911"]
d = a.index.intersection(b.index)
for t in ("ZS", "AAPL", "MSFT"):
    diff = (a.loc[d, t] - b.loc[d, t])
    by = diff.abs().groupby(diff.index.year).agg(["count", lambda s: float((s > 1e-6).mean()), "median"])
    by.columns = ["n", "share_changed", "median_abs_dz"]
    print(f"== {t}\n{by.round(4).to_string()}")
