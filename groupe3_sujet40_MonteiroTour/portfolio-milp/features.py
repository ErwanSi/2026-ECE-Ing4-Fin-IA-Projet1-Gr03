import numpy as np


def make_features(returns: np.ndarray, lags: int = 5, vol_window: int = 20):
    """
    Build simple features for ML on returns:
      - lagged returns (t-1..t-lags)
      - rolling volatility
      - rolling mean
    returns: (T,) array
    """
    r = np.asarray(returns, dtype=float).reshape(-1)
    T = len(r)

    X = []
    y = []

    for t in range(max(lags, vol_window), T - 1):
        lags_vec = [r[t - k] for k in range(1, lags + 1)]
        window = r[t - vol_window : t]
        vol = float(np.std(window, ddof=1))
        mom = float(np.mean(window))
        X.append(lags_vec + [vol, mom])
        y.append(r[t + 1])  # next return

    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)
