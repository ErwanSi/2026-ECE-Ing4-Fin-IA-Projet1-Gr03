import os
from datetime import datetime

import numpy as np

from runner import RunConfig, load_returns
from scenarios import bootstrap_scenarios
from optimize_cpsat import cpsat_cvar_portfolio
from optimize_milp import milp_cvar_portfolio


def main():
    tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "JPM", "XOM", "JNJ", "NVDA", "PG"]
    sector_map = {
        "AAPL": "TECH", "MSFT": "TECH", "AMZN": "CONS", "GOOGL": "TECH", "META": "TECH",
        "NVDA": "TECH", "JPM": "FIN", "XOM": "ENERGY", "JNJ": "HEALTH", "PG": "CONS",
    }
    sectors = [sector_map[t] for t in tickers]
    sector_bounds = {
        "TECH": (0.00, 0.70),
        "FIN": (0.00, 0.30),
        "ENERGY": (0.00, 0.30),
        "HEALTH": (0.00, 0.30),
        "CONS": (0.00, 0.50),
        "OTHER": (0.00, 0.30),
    }

    cfg = RunConfig(
        tickers=tickers,
        sectors=sectors,
        sector_bounds=sector_bounds,
        lambda_cvar=0.3,
        turnover_cap_lots=400,
        cap_disabled_first_window=False,  # compare on an already-initialised window
        cost_rate=0.001,
        cps_time_limit=10.0,
        seed=42,
        n_scenarios=200,
    )

    rets = load_returns(cfg)

    # take one training window
    train = rets.iloc[: cfg.train_len]
    mu = train.mean().values
    scenarios = bootstrap_scenarios(train.values, n_scenarios=cfg.n_scenarios, seed=cfg.seed)

    n = len(cfg.tickers)
    Q = cfg.Q

    # start from equal-weight lots
    q_old = np.full(n, Q // n, dtype=int)
    q_old[: (Q - q_old.sum())] += 1

    cps = cpsat_cvar_portfolio(
        scenarios_returns=scenarios,
        mu=mu,
        sectors=cfg.sectors,
        sector_bounds=cfg.sector_bounds,
        q_old=q_old,
        Q=cfg.Q,
        K=cfg.K,
        q_min=cfg.q_min,
        w_max=cfg.w_max,
        cost_rate=cfg.cost_rate,
        beta=cfg.beta,
        lambda_cvar=cfg.lambda_cvar,
        turnover_cap_lots=cfg.turnover_cap_lots,
        time_limit_s=cfg.cps_time_limit,
        seed=cfg.seed,
    )

    milp = milp_cvar_portfolio(
        scenarios_returns=scenarios,
        mu=mu,
        sectors=cfg.sectors,
        sector_bounds=cfg.sector_bounds,
        q_old=q_old,
        Q=cfg.Q,
        K=cfg.K,
        q_min=cfg.q_min,
        w_max=cfg.w_max,
        cost_rate=cfg.cost_rate,
        beta=cfg.beta,
        lambda_cvar=cfg.lambda_cvar,
        turnover_cap_lots=cfg.turnover_cap_lots,
        time_limit_s=cfg.cps_time_limit,
    )

    os.makedirs("runs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("runs", f"compare_cpsat_milp_{ts}.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("=== CP-SAT vs MILP comparison (single window) ===\n")
        f.write(f"tickers={tickers}\n")
        f.write(f"Q={cfg.Q}, K={cfg.K}, w_max={cfg.w_max}, cap={cfg.turnover_cap_lots}, lambda={cfg.lambda_cvar}, beta={cfg.beta}\n\n")

        if isinstance(cps, tuple):
            f.write(f"CP-SAT: status={cps[2]}\n")
        else:
            f.write(f"CP-SAT: status={cps.status}, solve_time={cps.solve_time_s:.3f}s, selected={int((cps.q_lots>0).sum())}, turnover={int(np.abs(cps.q_lots-q_old).sum())}\n")

        if isinstance(milp, tuple):
            f.write(f"MILP : status={milp[2]}\n")
        else:
            f.write(f"MILP : status={milp.status}, solve_time={milp.solve_time_s:.3f}s, selected={int((milp.q_lots>0).sum())}, turnover={int(np.abs(milp.q_lots-q_old).sum())}\n")

    print(f"Saved comparison: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()