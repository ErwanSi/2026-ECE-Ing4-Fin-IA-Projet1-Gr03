import numpy as np


def bootstrap_scenarios(train_returns: np.ndarray, n_scenarios: int, seed: int = 42) -> np.ndarray:
    """
    Bootstraps i.i.d. daily return scenarios from historical daily returns.

    train_returns: shape (T, N)
    returns: shape (S, N)
    """
    if train_returns.ndim != 2:
        raise ValueError("train_returns must be 2D (T, N).")
    T, N = train_returns.shape
    if T < 5:
        raise ValueError("Not enough training data for bootstrap.")

    rng = np.random.default_rng(seed)
    idx = rng.integers(low=0, high=T, size=n_scenarios)
    return train_returns[idx, :].astype(float)
