import numpy as np
import pandas as pd


def max_drawdown(cum: pd.Series) -> float:
    peak = cum.cummax()
    dd = cum / peak - 1.0
    return float(dd.min())


def annualised_metrics(daily_returns: pd.Series) -> dict:
    mu = float(daily_returns.mean()) * 252.0
    vol = float(daily_returns.std(ddof=0)) * np.sqrt(252.0)
    sharpe = mu / (vol + 1e-12)
    cum = (1.0 + daily_returns).cumprod()
    mdd = max_drawdown(cum)
    return {"mean": mu, "vol": vol, "sharpe": sharpe, "maxDD": mdd}


def walk_forward_apply_weights(rets: pd.DataFrame, weights_by_date: dict, rebalance_step: int):
    """
    Given returns DataFrame and a dict {rebalance_index: weight_vector},
    build daily portfolio returns by holding weights until next rebalance.
    """
    idx = rets.index
    n = rets.shape[1]
    port = pd.Series(index=idx, dtype=float)

    rebal_points = sorted(weights_by_date.keys())
    for k, t0 in enumerate(rebal_points):
        w = np.asarray(weights_by_date[t0]).reshape(n)
        t1 = rebal_points[k + 1] if k + 1 < len(rebal_points) else len(idx)
        sl = slice(t0, t1)
        port.iloc[sl] = (rets.iloc[sl].values @ w)

    return port.dropna()
