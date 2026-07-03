"""유틸리티 함수."""

import pandas as pd
import numpy as np
from pathlib import Path


def ensure_dir(path: str) -> Path:
    """디렉토리 생성 (존재하지 않으면)."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def annualise_return(daily_returns: pd.Series, ann_factor: int = 252) -> float:
    """Geometric annualised return (CAGR)."""
    n = len(daily_returns)
    if n == 0:
        return 0.0
    total = (1 + daily_returns).prod()
    return total ** (ann_factor / n) - 1


def compute_performance_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series = None,
    ann_factor: int = 252,
) -> dict:
    """Canonical performance metrics computation.

    Uses geometric annualisation and consistent ddof=1 (sample std).
    """
    port = portfolio_returns.dropna()
    n = len(port)
    if n == 0:
        return {}

    ann_ret = annualise_return(port, ann_factor)
    ann_vol = port.std() * np.sqrt(ann_factor)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    # Drawdown
    cum = (1 + port).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum / rolling_max) - 1
    max_dd = drawdown.min()

    result = {
        "annual_return": ann_ret,
        "annual_vol": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
    }

    if benchmark_returns is not None:
        # Item 12: ffill before zero-fill. Pure .fillna(0) on a benchmark
        # series injects a spurious "benchmark is flat today" on missing
        # dates, which biases active_return / IR. ffill propagates the
        # last known BM return, then 0 is a last-resort boundary guard.
        bm = benchmark_returns.reindex(port.index).ffill().fillna(0)
        active = port - bm
        active_ret = annualise_return(active, ann_factor)
        active_vol = active.std() * np.sqrt(ann_factor)
        ir = active_ret / active_vol if active_vol > 0 else 0.0
        result.update({
            "active_return": active_ret,
            "tracking_error": active_vol,
            "information_ratio": ir,
        })

    return result


def compute_beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Realized regression beta cov(port,bm)/var(bm) on overlapping finite rows.

    Returns nan if <2 overlapping points or var(bm)==0. Sample (ddof=1) moments.
    """
    aligned = pd.concat(
        [portfolio_returns.rename("p"), benchmark_returns.rename("b")], axis=1
    ).dropna()
    if len(aligned) < 2:
        return float("nan")
    p = aligned["p"].values
    b = aligned["b"].values
    var_b = b.var(ddof=1)
    if not np.isfinite(var_b) or var_b <= 0.0:
        return float("nan")
    return float(np.cov(p, b, ddof=1)[0, 1] / var_b)


def format_metrics(metrics: dict) -> str:
    """Format metrics dict into a readable string."""
    lines = []
    for key, val in metrics.items():
        if isinstance(val, float):
            if 'ratio' in key or 'sharpe' in key:
                lines.append(f"  {key}: {val:.2f}")
            else:
                lines.append(f"  {key}: {val:.2%}")
    return "\n".join(lines)


def rolling_ic(predictions: pd.DataFrame, realized: pd.DataFrame, window: int = 252) -> pd.Series:
    """Rolling Information Coefficient (vectorized)."""
    # Align on common dates
    common_dates = predictions.index.intersection(realized.index)
    pred_aligned = predictions.loc[common_dates]
    real_aligned = realized.loc[common_dates]

    # Compute Spearman IC per date
    def _row_spearman(i):
        p = pred_aligned.iloc[i]
        r = real_aligned.iloc[i]
        valid = p.notna() & r.notna()
        if valid.sum() < 3:
            return np.nan
        return p[valid].corr(r[valid], method="spearman")

    ic_values = [_row_spearman(i) for i in range(len(common_dates))]
    ic_daily = pd.Series(ic_values, index=common_dates, dtype=float)
    # Reindex to full prediction dates
    ic_daily = ic_daily.reindex(predictions.index)

    return ic_daily.rolling(window, min_periods=window // 2).mean()
