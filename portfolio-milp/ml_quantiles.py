import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from features import make_features_for_asset, last_feature_row


def fit_quantile_models(
    train_returns: pd.DataFrame,
    quantiles=(0.05, 0.5, 0.95),
    n_lags: int = 10,
):
    """
    ML "utile" : quantile regression pour estimer la distribution conditionnelle.

    Pourquoi c'est cohérent avec le sujet ?
    - On veut faire CVaR (risque de queue) => quantiles utiles (q05/q95)
    - On ne fait pas du ML gadget; le ML sert la formulation de risque.

    Retour :
      models[ticker][q] = modèle entraîné
      x_last[ticker] = features au dernier jour du train, pour prédire r(t+1)
    """
    models = {}
    x_last = {}

    for t in train_returns.columns:
        df = make_features_for_asset(train_returns[t], n_lags=n_lags)
        X = df.drop(columns=["r", "y"]).values
        y = df["y"].values

        x_last[t] = last_feature_row(train_returns[t], n_lags=n_lags)

        models[t] = {}
        for q in quantiles:
            m = GradientBoostingRegressor(
                loss="quantile",
                alpha=float(q),
                n_estimators=300,
                max_depth=3,
                learning_rate=0.05,
                random_state=42,
            )
            m.fit(X, y)
            models[t][q] = m

    return models, x_last


def predict_quantiles(models, x_last, quantiles=(0.05, 0.5, 0.95)) -> pd.DataFrame:
    """
    Prédit q05/q50/q95 de r(t+1) pour chaque actif.
    Retour: DataFrame index=tickers, colonnes: q5, q50, q95
    """
    tickers = list(models.keys())
    out = {"q5": [], "q50": [], "q95": []}

    for t in tickers:
        q5 = float(models[t][0.05].predict(x_last[t])[0])
        q50 = float(models[t][0.5].predict(x_last[t])[0])
        q95 = float(models[t][0.95].predict(x_last[t])[0])
        out["q5"].append(q5)
        out["q50"].append(q50)
        out["q95"].append(q95)

    return pd.DataFrame(out, index=tickers)
