print("=== MAIN_COMPARE.PY (CVXPY baseline vs CP-SAT constraints) ===")

import numpy as np
import matplotlib.pyplot as plt

from data import load_prices, returns_from_prices
from optimize_cvxpy import markowitz_with_turnover
from optimize_cpsat import cpsat_portfolio


def main():
    tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "JPM", "XOM", "JNJ", "NVDA", "PG"]

    # Simple sector mapping (demo)
    sector_map = {
        "AAPL": "Tech", "MSFT": "Tech", "AMZN": "Tech", "GOOGL": "Tech", "META": "Tech", "NVDA": "Tech",
        "JPM": "Financials",
        "XOM": "Energy",
        "JNJ": "Healthcare",
        "PG": "Consumer",
    }
    sectors = [sector_map[t] for t in tickers]

    # Sector bounds (ex: max 60% Tech, min 10% Healthcare)
    sector_bounds = {
        "Tech": (0.0, 0.60),
        "Healthcare": (0.10, 0.50),
        "Financials": (0.0, 0.40),
        "Energy": (0.0, 0.40),
        "Consumer": (0.0, 0.40),
    }

    prices = load_prices(tickers, start="2021-01-01")
    rets = returns_from_prices(prices).dropna()

    window = rets.tail(252)
    mu = window.mean().values
    Sigma = window.cov().values

    # ---------- (1) CVXPY baseline ----------
    w_old = np.ones(len(tickers)) / len(tickers)
    w_cvx, _ = markowitz_with_turnover(mu, Sigma, w_old=w_old, lam=10.0, gamma_tc=0.2, w_max=0.60)

    # ---------- (2) CP-SAT (cardinality + lots + sectors + TC) ----------
    # Convert weights -> lots
    Q = 1000
    q_old = (w_old * Q).round().astype(int)

    w_cpsat, q_cpsat, status = cpsat_portfolio(
        mu=mu,
        sectors=sectors,
        sector_bounds=sector_bounds,
        K=5,              # max 5 assets
        Q=Q,
        q_old=q_old,
        tc_per_lot=1,     # transaction cost per lot
        q_min=10,         # minimum 1% if selected (10 lots out of 1000)
        q_max=None,
        time_limit_s=10.0,
    )

    print("\nCVXPY weights:")
    for t, wi in sorted(zip(tickers, w_cvx), key=lambda x: -x[1]):
        print(f"{t}: {wi:.4f}")

    print(f"\nCP-SAT status: {status}")
    if w_cpsat is not None:
        print("CP-SAT weights (cardinality/sector/lot constrained):")
        for t, wi in sorted(zip(tickers, w_cpsat), key=lambda x: -x[1]):
            if wi > 1e-6:
                print(f"{t}: {wi:.4f}")

    # Quick performance plot (in-sample)
    port_cvx = (rets @ w_cvx).add(1).cumprod()
    plt.figure()
    plt.plot(port_cvx.index, port_cvx.values, label="CVXPY (baseline)")

    if w_cpsat is not None:
        port_sat = (rets @ w_cpsat).add(1).cumprod()
        plt.plot(port_sat.index, port_sat.values, label="CP-SAT (real constraints)")

    ew = (rets.mean(axis=1)).add(1).cumprod()
    plt.plot(ew.index, ew.values, label="Equal-weight")

    plt.legend()
    plt.title("Cumulative performance (in-sample)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
