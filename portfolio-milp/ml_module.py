import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


def _make_supervised(series: pd.Series, lags: int):
    """
    Create supervised dataset:
    X_t = [r_{t-1}, ..., r_{t-lags}], y_t = r_t
    """
    r = series.values
    X, y = [], []
    for t in range(lags, len(r)):
        X.append(r[t - lags : t])
        y.append(r[t])
    if len(X) == 0:
        return None, None
    return np.asarray(X), np.asarray(y)


def fit_quantile_models(returns: pd.DataFrame, lags: int = 5, lookback: int = 252):
    """
    Train 3 quantile models per asset on last `lookback` rows of returns.
    Returns dict[ticker] = {q: model or None, 'fallback': float, 'r2': float or None}
    """
    models = {}
    window = returns.tail(lookback).copy()

    for tkr in window.columns:
        s = window[tkr].dropna()
        fallback = float(s.mean()) if len(s) > 0 else 0.0

        X, y = _make_supervised(s, lags=lags)
        if X is None or len(y) < 50:
            models[tkr] = {
                0.05: None,
                0.50: None,
                0.95: None,
                "fallback": fallback,
                "trained": False,
                "rows": 0 if X is None else int(len(y)),
                "r2": None,
            }
            continue

        # simple holdout for diagnostics (not for tuning)
        split = int(0.8 * len(y))
        Xtr, Xte = X[:split], X[split:]
        ytr, yte = y[:split], y[split:]

        bucket = {}
        for q in (0.05, 0.50, 0.95):
            m = GradientBoostingRegressor(
                loss="quantile",
                alpha=q,
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
            )
            m.fit(Xtr, ytr)
            bucket[q] = m

        # quick R2 on median model (finance can be negative: normal)
        try:
            yhat = bucket[0.50].predict(Xte)
            ss_res = float(((yte - yhat) ** 2).sum())
            ss_tot = float(((yte - yte.mean()) ** 2).sum()) + 1e-12
            r2 = 1.0 - ss_res / ss_tot
        except Exception:
            r2 = None

        bucket["fallback"] = fallback
        bucket["trained"] = True
        bucket["rows"] = int(len(y))
        bucket["r2"] = r2
        models[tkr] = bucket

    return models


def predict_quantiles(models, recent_returns_row: pd.Series, lags: int = 5):
    """
    Produce q05/q50/q95 predictions per asset using the models and last `lags` returns.
    recent_returns_row is not enough by itself; caller should provide a vector of last lags.
    Here we accept a dict asset->lags vector via recent_returns_row[ticker] being array-like.
    """
    q05, q50, q95 = {}, {}, {}
    for tkr, pack in models.items():
        fb = pack.get("fallback", 0.0)
        x = recent_returns_row.get(tkr, None)

        if x is None or pack.get("trained") is not True:
            q05[tkr] = fb
            q50[tkr] = fb
            q95[tkr] = fb
            continue

        x = np.asarray(x).reshape(1, -1)
        if x.shape[1] != lags:
            q05[tkr] = fb
            q50[tkr] = fb
            q95[tkr] = fb
            continue

        try:
            q05[tkr] = float(pack[0.05].predict(x)[0])
            q50[tkr] = float(pack[0.50].predict(x)[0])
            q95[tkr] = float(pack[0.95].predict(x)[0])
        except Exception:
            q05[tkr] = fb
            q50[tkr] = fb
            q95[tkr] = fb

    return q05, q50, q95


def build_scenarios_from_quantiles(q05, q50, q95, S: int = 100, seed: int = 42):
    """
    Build S Monte-Carlo scenarios per asset using a piecewise-linear quantile mapping.
    We sample u~U(0,1) and interpolate between (0.05->0.50) and (0.50->0.95).
    """
    rng = np.random.default_rng(seed)
    tickers = list(q50.keys())
    n = len(tickers)
    U = rng.random((S, n))

    R = np.zeros((S, n), dtype=float)
    for j, tkr in enumerate(tickers):
        a = float(q05[tkr])
        b = float(q50[tkr])
        c = float(q95[tkr])

        # map u in [0,1] -> returns using 3-point quantile curve
        for s in range(S):
            u = U[s, j]
            if u <= 0.50:
                # interpolate 0.05..0.50
                # clamp u below 0.05
                uu = max(u, 0.05)
                R[s, j] = a + (b - a) * (uu - 0.05) / (0.50 - 0.05)
            else:
                # interpolate 0.50..0.95
                uu = min(u, 0.95)
                R[s, j] = b + (c - b) * (uu - 0.50) / (0.95 - 0.50)

    return tickers, R
