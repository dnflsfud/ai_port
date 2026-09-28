"""M-01 프로브 2: 빈티지(번들)가 다른 production pkl 패널 사이에서 fwd_sales_slope_level 과거 이력이
종목 단위로 통째로 바뀌는가. 패널 값은 날짜별 CS z-score 이므로 날짜별 순위(pct)로 비교한다.
공통 구간 2019-01-01..2024-12-31, 종목별 median |Δrank_pct|."""
import gc, pickle, sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
paths = {"0903": "outputs/s17_2_s0recert/backtest_result.pkl",
         "0911": "outputs/s18_7_s0recert/backtest_result.pkl",
         "0915": sys.argv[1] + "/snap/backtest_result.pkl"}
feat = {}
for k, p in paths.items():
    r = pickle.load(open(p, "rb"))
    s = r.panel["fwd_sales_slope_level"].unstack("ticker")
    feat[k] = s.loc["2019-01-01":"2024-12-31"].rank(axis=1, pct=True)
    del r; gc.collect()
    print(k, feat[k].shape, flush=True)
for a, b in (("0903", "0911"), ("0911", "0915"), ("0903", "0915")):
    x, y = feat[a], feat[b]
    d = x.index.intersection(y.index); c = x.columns.intersection(y.columns)
    diff = (x.loc[d, c] - y.loc[d, c]).abs()
    med = diff.median().sort_values(ascending=False)
    print(f"{a} vs {b}: tickers with median|Δrank|>0.05: {int((med > 0.05).sum())} / {len(med)}; top: "
          + ", ".join(f"{t}={v:.3f}" for t, v in med.head(8).items()))
