print("=== MAIN_ML.PY (IA -> optimisation sous contraintes) ===")

import os
import numpy as np
import matplotlib.pyplot as plt

from data import load_prices, returns_from_prices
from ml_module import predict_mu_ml
from optimize_cvxpy import markowitz_with_turnover


def main():
    # Universe
    raw_tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "JPM", "XOM", "JNJ", "NVDA", "PG"]

    # 1) Data
    prices = load_prices(raw_tickers, start="2021-01-01")
    rets = returns_from_prices(prices)
    rets = rets.dropna(axis=1, how="any").dropna()

    if rets.shape[1] < 2 or len(rets) < 300:
        raise RuntimeError(f"Not enough clean data. cols={rets.shape[1]} rows={len(rets)}")

    tickers = list(rets.columns)

    # 2) IA: predict expected returns mu
    mu_ml, info = predict_mu_ml(
        returns=rets,
        train_window=252,
        n_lags=5,
        min_rows=120,
        clip_mu=0.02,
    )

    print("\nML diagnostics (R2 can be low/negative in finance):")
    for t in tickers:
        d = info["per_asset"][t]
        print(f"- {t}: trained={d['trained']} rows={d['rows']} r2={d['r2']} fallback={d['fallback']}")

    # 3) Risk model (covariance on same last window)
    window = rets.tail(min(252, len(rets)))
    Sigma = window.cov().values

    # 4) Realistic constraints: turnover penalty from previous portfolio
    # For a "first allocation", previous portfolio can be equal-weight.
    w_old = np.ones(len(tickers)) / len(tickers)

    w, obj = markowitz_with_turnover(
        mu=mu_ml,
        Sigma=Sigma,
        w_old=w_old,
        lam=10.0,       # risk aversion
        gamma_tc=0.2,   # transaction-cost proxy (turnover penalty)
        w_max=0.60,     # concentration limit
    )

    print("\nOptimised weights (sorted):")
    for t, wi in sorted(zip(tickers, w), key=lambda x: -x[1]):
        print(f"{t}: {wi:.4f}")

    print(f"\nObjective value: {obj:.6f}")
    print(f"Turnover (L1): {np.abs(w - w_old).sum():.4f}")
    print(f"Sum weights: {w.sum():.6f}")
    print(f"Max weight: {w.max():.4f}")

    # 5) Plot (optional: may overlap if turnover dominates, and that's OK)
    port = (rets @ w).add(1).cumprod()
    ew = (rets.mean(axis=1)).add(1).cumprod()

    plt.figure()
    plt.plot(ew.index, ew.values, label="Equal-weight")
    plt.plot(port.index, port.values, label="Optimized (ML mu + cvxpy)")
    plt.legend()
    plt.title("Cumulative performance (in-sample)")
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), "performance_ml.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
