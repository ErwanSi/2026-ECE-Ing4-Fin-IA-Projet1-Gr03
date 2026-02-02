import numpy as np
from sklearn.linear_model import Ridge
from features import make_features


def train_ridge_mu(returns_matrix: np.ndarray):
    """
    Simple per-asset Ridge model predicting next return.
    returns_matrix: (T,n)
    Returns: list of fitted models
    """
    R = np.asarray(returns_matrix, dtype=float)
    _, n = R.shape
    models = []
    for i in range(n):
        X, y = make_features(R[:, i])
        m = Ridge(alpha=1.0)
        m.fit(X, y)
        models.append(m)
    return models


def predict_mu(models, returns_matrix_recent: np.ndarray):
    """
    Predict expected return per asset from recent window.
    """
    R = np.asarray(returns_matrix_recent, dtype=float)
    _, n = R.shape
    mu = np.zeros(n, dtype=float)
    for i in range(n):
        X, _ = make_features(R[:, i])
        mu[i] = models[i].predict(X[-1].reshape(1, -1))[0]
    return mu
