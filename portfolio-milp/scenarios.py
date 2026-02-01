import numpy as np
import pandas as pd


def make_scenarios_multivariate(
    train_returns: pd.DataFrame,
    q_pred: pd.DataFrame,
    S: int = 100,
    seed: int = 42,
    clip: float = 0.20,
):
    """
    Génère des scénarios multi-actifs pour la CVaR (MILP/CSP).

    Objectif "niveau pro" :
    - préserver la dépendance cross-asset => on bootstrap des dates communes
    - recentrer sur la médiane prédite ML (q50)
    - ajuster l'échelle par actif via IQR (q95-q5) pour refléter une variance conditionnelle

    Retour :
      ndarray shape (S, n) des rendements scénarios.
    """
    rng = np.random.default_rng(seed)
    X = train_returns.values
    T, n = X.shape

    if T < 80:
        raise ValueError("Fenêtre train trop courte pour des scénarios robustes.")

    # Bootstrap dates communes pour préserver corrélations
    X_centered = X - X.mean(axis=0, keepdims=True)
    idx = rng.integers(0, T, size=S)
    eps = X_centered[idx, :]  # (S, n)

    tickers = list(train_returns.columns)

    # Centre conditionnel = médiane prédite
    center = q_pred.loc[tickers, "q50"].values.reshape(1, -1)

    # Ajustement d'échelle conditionnel via IQR
    hist_q05 = np.quantile(X, 0.05, axis=0)
    hist_q95 = np.quantile(X, 0.95, axis=0)
    hist_iqr = np.maximum(hist_q95 - hist_q05, 1e-8)

    pred_q05 = q_pred.loc[tickers, "q5"].values
    pred_q95 = q_pred.loc[tickers, "q95"].values
    pred_iqr = np.maximum(pred_q95 - pred_q05, 1e-8)

    scale = (pred_iqr / hist_iqr).reshape(1, -1)
    scale = np.clip(scale, 0.5, 2.0)  # stabilité solveur

    scenarios = center + scale * eps
    scenarios = np.clip(scenarios, -clip, clip)  # clamp extrêmes

    return scenarios
