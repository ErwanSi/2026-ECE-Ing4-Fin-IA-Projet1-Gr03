import numpy as np
import pandas as pd


def wealth_curve(returns: pd.Series) -> pd.Series:
    """
    returns: daily arithmetic returns series.
    """
    return (1.0 + returns).cumprod()


def max_drawdown(wealth: pd.Series) -> float:
    """
    wealth: cumulative wealth series.
    """
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return float(dd.min())


def perf_metrics(returns: pd.Series) -> dict:
    """
    Annualised approx: mean*252, vol*sqrt(252), sharpe = mean/vol.
    """
    mu_d = float(returns.mean())
    vol_d = float(returns.std(ddof=1))
    mu_a = mu_d * 252.0
    vol_a = vol_d * np.sqrt(252.0)
    sharpe = mu_a / vol_a if vol_a > 1e-12 else 0.0

    w = wealth_curve(returns)
    mdd = max_drawdown(w)

    return {
        "mean": float(mu_a),
        "vol": float(vol_a),
        "sharpe": float(sharpe),
        "maxDD": float(mdd),
    }
