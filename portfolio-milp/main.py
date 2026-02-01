import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import load_prices, returns_from_prices
from optimize_cvxpy import markowitz_with_turnover
from optimize_cpsat import cpsat_cvar_portfolio
from ml_module import fit_quantile_models, build_scenarios_from_quantiles
from backtest import annualised_metrics


def main():
    print("=== MAIN.PY (Full #40 project: ML quantiles -> scenarios -> CP-SAT CVaR + constraints, OOS walk-forward) ===")

    # -------------------------
    # PARAMETERS (tweak here)
    # -------------------------
    TICKERS = ["AAPL","MSFT","AMZN","GOOGL","META","JPM","XOM","JNJ","NVDA","PG"]

    START_DATE = "2021-01-01"
    TRAIN_WINDOW = 252
    REBALANCE_STEP = 21
    SEED = 42

    # Realistic constraints
    K_MAX = 5
    W_MAX = 0.60

    # Discrete lots
    LOT_SIZE = 1000
    BUDGET_Q = 1000
    Q_MIN = 1

    # Transaction costs
    TC_PER_LOT = 5

    # CVaR + scenarios
    S_SCENARIOS = 100
    BETA_CVAR = 0.95
    LAMBDA_CVAR = 1.0
    TIME_LIMIT_S = 10.0

    # CVXPY baseline parameters (set to avoid "stuck equal-weight")
    LAM_CVXPY = 2.0
    GAMMA_TC_CVXPY = 0.02

    # ML settings
    LAGS = 5
    LOOKBACK_ML = TRAIN_WINDOW

    # Sector mapping (demo, but realistic in spirit)
    sector_map = {
        "AAPL": "Tech", "MSFT": "Tech", "AMZN": "ConsDisc", "GOOGL": "Tech", "META": "Tech",
        "JPM": "Finance", "XOM": "Energy", "JNJ": "Health", "NVDA": "Tech", "PG": "ConsStap",
    }
    sectors = [sector_map[t] for t in TICKERS]

    # Sector bounds (in weight)
    sector_bounds = {
        "Tech": (0.10, 0.70),
        "Finance": (0.00, 0.30),
        "Energy": (0.00, 0.30),
        "Health": (0.00, 0.40),
        "ConsStap": (0.00, 0.40),
        "ConsDisc": (0.00, 0.40),
    }

    # -------------------------
    # DATA
    # -------------------------
    prices = load_prices(TICKERS, start=START_DATE)
    rets = returns_from_prices(prices)
    rets = rets.dropna()

    n = rets.shape[1]
    assert n == len(TICKERS)

    # Baselines initial weights
    w_prev_cvx = np.ones(n) / n
    # CP-SAT state in lots
    q_prev = np.round((np.ones(n) / n) * BUDGET_Q).astype(int)
    # adjust to sum exactly Q
    q_prev[-1] += (BUDGET_Q - q_prev.sum())

    # Store weights at each rebalance
    weights_cvx = {}
    weights_cps = {}
    weights_ew = {}

    # Store OOS daily returns
    oos_cvx = []
    oos_cps = []
    oos_ew = []
    oos_dates = []

    # -------------------------
    # WALK-FORWARD
    # -------------------------
    rng = np.random.default_rng(SEED)
    t_start = time.time()

    for t0 in range(TRAIN_WINDOW, len(rets) - REBALANCE_STEP, REBALANCE_STEP):
        # progress
        # print(f"[rebalance] {t0}/{len(rets)} date={rets.index[t0].date()}", flush=True)

        train = rets.iloc[t0 - TRAIN_WINDOW : t0]
        test = rets.iloc[t0 : t0 + REBALANCE_STEP]

        # equal-weight
        w_ew = np.ones(n) / n

        # -------------------------
        # CVXPY baseline (convex)
        # -------------------------
        mu_hist = train.mean().values
        Sigma = train.cov().values
        Sigma = 0.5 * (Sigma + Sigma.T) + 1e-8 * np.eye(n)

        w_cvx, _ = markowitz_with_turnover(
            mu_hist, Sigma, w_old=w_prev_cvx, lam=LAM_CVXPY, gamma_tc=GAMMA_TC_CVXPY, w_max=W_MAX
        )

        # -------------------------
        # ML quantiles -> scenarios -> CP-SAT CVaR
        # -------------------------
        models = fit_quantile_models(train, lags=LAGS, lookback=LOOKBACK_ML)

        # Build last-lags features per ticker for prediction at t0
        feats = {}
        for j, tkr in enumerate(TICKERS):
            # last LAGS returns from train
            x = train[tkr].values[-LAGS:]
            feats[tkr] = x

        # predict quantiles with small helper
        # (we keep it simple: call model.predict directly)
        q05, q50, q95 = {}, {}, {}
        for tkr in TICKERS:
            pack = models[tkr]
            fb = pack["fallback"]
            x = np.asarray(feats[tkr]).reshape(1, -1)
            if not pack["trained"]:
                q05[tkr], q50[tkr], q95[tkr] = fb, fb, fb
            else:
                q05[tkr] = float(pack[0.05].predict(x)[0])
                q50[tkr] = float(pack[0.50].predict(x)[0])
                q95[tkr] = float(pack[0.95].predict(x)[0])

        tickers_s, R = build_scenarios_from_quantiles(q05, q50, q95, S=S_SCENARIOS, seed=int(rng.integers(0, 1_000_000)))
        assert tickers_s == TICKERS

        # scenario mean as mu proxy
        mu_scen = R.mean(axis=0)

        w_cps, q_sol, status = cpsat_cvar_portfolio(
            scenarios_returns=R,
            mu=mu_scen,
            sectors=sectors,
            sector_bounds=sector_bounds,
            q_old=q_prev,
            Q=BUDGET_Q,
            K=K_MAX,
            q_min=Q_MIN,
            w_max=W_MAX,
            tc_per_lot=TC_PER_LOT,
            beta=BETA_CVAR,
            lambda_cvar=LAMBDA_CVAR,
            time_limit_s=TIME_LIMIT_S,
            seed=int(rng.integers(0, 1_000_000)),
        )

        # If CP-SAT fails, fallback to equal-weight
        if w_cps is None:
            w_cps = w_ew.copy()
            q_sol = q_prev.copy()

        # store weights
        weights_ew[t0] = w_ew
        weights_cvx[t0] = w_cvx
        weights_cps[t0] = w_cps

        # OOS apply (daily returns)
        oos_dates.extend(list(test.index))
        oos_ew.extend(list(test.values @ w_ew))
        oos_cvx.extend(list(test.values @ w_cvx))
        oos_cps.extend(list(test.values @ w_cps))

        # update states
        w_prev_cvx = w_cvx.copy()
        q_prev = q_sol.copy()

    # Convert OOS series
    oos_dates = pd.to_datetime(oos_dates)
    s_ew = pd.Series(oos_ew, index=oos_dates, name="Equal-weight (OOS)")
    s_cvx = pd.Series(oos_cvx, index=oos_dates, name="CVXPY (OOS)")
    s_cps = pd.Series(oos_cps, index=oos_dates, name="CP-SAT CVaR (OOS)")

    # Metrics
    m_ew = annualised_metrics(s_ew)
    m_cvx = annualised_metrics(s_cvx)
    m_cps = annualised_metrics(s_cps)

    print("\n=== OOS Summary (annualised approx) ===")
    print(f"Equal-weight : mean={m_ew['mean']:.3f} vol={m_ew['vol']:.3f} sharpe={m_ew['sharpe']:.3f} maxDD={m_ew['maxDD']:.3f}")
    print(f"CVXPY        : mean={m_cvx['mean']:.3f} vol={m_cvx['vol']:.3f} sharpe={m_cvx['sharpe']:.3f} maxDD={m_cvx['maxDD']:.3f}")
    print(f"CP-SAT CVaR  : mean={m_cps['mean']:.3f} vol={m_cps['vol']:.3f} sharpe={m_cps['sharpe']:.3f} maxDD={m_cps['maxDD']:.3f}")

    # Diagnostics
    # CVXPY turnover in weights
    cvx_turn = []
    prev = np.ones(n) / n
    for t0, w in weights_cvx.items():
        cvx_turn.append(float(np.abs(w - prev).sum()))
        prev = w
    # CP-SAT turnover in lots approx from stored weights
    cps_turn = []
    prev_q = np.round((np.ones(n) / n) * BUDGET_Q).astype(int)
    prev_q[-1] += (BUDGET_Q - prev_q.sum())
    avg_k = []
    for t0, w in weights_cps.items():
        q = np.round(w * BUDGET_Q).astype(int)
        q[-1] += (BUDGET_Q - q.sum())
        cps_turn.append(float(np.abs(q - prev_q).sum()))
        avg_k.append(int((q > 0).sum()))
        prev_q = q

    print("\n=== Trading / Constraints diagnostics ===")
    print(f"Avg turnover CVXPY (L1 weights): {np.mean(cvx_turn):.4f}")
    print(f"Avg turnover CP-SAT (lots)     : {np.mean(cps_turn):.2f}")
    print(f"Avg selected assets CP-SAT     : {np.mean(avg_k):.2f} (max K={K_MAX})")

    # Plot cumulative
    cum_ew = (1.0 + s_ew).cumprod()
    cum_cvx = (1.0 + s_cvx).cumprod()
    cum_cps = (1.0 + s_cps).cumprod()

    plt.figure(figsize=(10, 6))
    plt.plot(cum_ew.index, cum_ew.values, label="Equal-weight (OOS)")
    plt.plot(cum_cvx.index, cum_cvx.values, label="CVXPY mean-variance+turnover (OOS)")
    plt.plot(cum_cps.index, cum_cps.values, label="IA: ML scenarios + CP-SAT CVaR (OOS)")
    plt.title("Cumulative performance (OUT-OF-SAMPLE walk-forward)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative return")
    plt.legend()
    plt.tight_layout()

    # Save artifacts
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fig_path = "performance_oos.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved figure: {fig_path}")

    # Save run json + metrics csv
    run = {
        "timestamp_utc": ts,
        "tickers": TICKERS,
        "params": {
            "START_DATE": START_DATE,
            "TRAIN_WINDOW": TRAIN_WINDOW,
            "REBALANCE_STEP": REBALANCE_STEP,
            "K_MAX": K_MAX,
            "W_MAX": W_MAX,
            "LOT_SIZE": LOT_SIZE,
            "BUDGET_Q": BUDGET_Q,
            "Q_MIN": Q_MIN,
            "TC_PER_LOT": TC_PER_LOT,
            "S_SCENARIOS": S_SCENARIOS,
            "BETA_CVAR": BETA_CVAR,
            "LAMBDA_CVAR": LAMBDA_CVAR,
            "TIME_LIMIT_S": TIME_LIMIT_S,
            "LAM_CVXPY": LAM_CVXPY,
            "GAMMA_TC_CVXPY": GAMMA_TC_CVXPY,
            "LAGS": LAGS,
            "LOOKBACK_ML": LOOKBACK_ML,
        },
        "metrics": {
            "equal_weight": m_ew,
            "cvxpy": m_cvx,
            "cpsat_cvar": m_cps,
        },
        "diagnostics": {
            "avg_turnover_cvx": float(np.mean(cvx_turn)) if len(cvx_turn) else None,
            "avg_turnover_cps_lots": float(np.mean(cps_turn)) if len(cps_turn) else None,
            "avg_selected_assets_cps": float(np.mean(avg_k)) if len(avg_k) else None,
        },
    }

    import os
    os.makedirs("runs", exist_ok=True)

    run_json_path = f"runs/run_{ts}.json"
    with open(run_json_path, "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2)

    metrics_df = pd.DataFrame([
        {"strategy": "Equal-weight", **m_ew},
        {"strategy": "CVXPY", **m_cvx},
        {"strategy": "CP-SAT CVaR", **m_cps},
    ])
    metrics_csv_path = f"runs/metrics_{ts}.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)

    print(f"Saved run json: {os.path.abspath(run_json_path)}")
    print(f"Saved metrics : {os.path.abspath(metrics_csv_path)}")

    plt.show()

    print(f"\nDone in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
 