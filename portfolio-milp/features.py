import numpy as np
import pandas as pd


def compute_drawdown_from_returns(r: pd.Series) -> pd.Series:
    """
    Drawdown calculé à partir d'une série de rendements.

    V_t = cumprod(1 + r_t)
    DD_t = 1 - V_t / max_{u<=t} V_u
    """
    v = (1.0 + r.fillna(0.0)).cumprod()
    peak = v.cummax()
    dd = 1.0 - (v / peak)
    return dd


def make_features_for_asset(r: pd.Series, n_lags: int = 10) -> pd.DataFrame:
    """
    Construit un dataset supervisé pour un actif.

    Cible:
      y(t) = r(t+1)

    Features:
      - lags 1..n_lags
      - momentum 20/60
      - volatilité 20/60
      - drawdown courant

    IMPORTANT:
      Le dataset est construit uniquement sur la fenêtre train -> pas de fuite OOS.
    """
    df = pd.DataFrame({"r": r}).copy()

    for k in range(1, n_lags + 1):
        df[f"lag_{k}"] = df["r"].shift(k)

    df["mom_20"] = df["r"].rolling(20).mean()
    df["mom_60"] = df["r"].rolling(60).mean()

    df["vol_20"] = df["r"].rolling(20).std()
    df["vol_60"] = df["r"].rolling(60).std()

    df["dd"] = compute_drawdown_from_returns(df["r"])

    # cible: rendement suivant
    df["y"] = df["r"].shift(-1)

    # Supprime NaN induits par rolling/shift
    df = df.dropna()

    return df


def last_feature_row(train_returns: pd.Series, n_lags: int = 10) -> np.ndarray:
    """
    Dernière ligne de features X(t) (sans y), pour prédire r(t+1) au rebalancing.

    Retour :
      ndarray shape (1, d)
    """
    df = make_features_for_asset(train_returns, n_lags=n_lags)
    X = df.drop(columns=["r", "y"]).values
    return X[-1].reshape(1, -1)
