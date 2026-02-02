from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple, List, Any

import numpy as np
import pandas as pd

from data import download_prices
from backtest import perf_metrics, wealth_curve
from scenarios import bootstrap_scenarios
from optimize_cpsat import cpsat_cvar_portfolio


@dataclass
class RunConfig:
    tickers: List[str]
    sectors: List[str]
    sector_bounds: Dict[str, Tuple[float, float]]

    start: str = "2018-01-01"
    train_len: int = 252 * 2
    test_len: int = 21
    step: int = 21

    Q: int = 1000
    K: int = 5
    q_min: int = 1
    w_max: float = 0.60

    beta: float = 0.95
    lambda_cvar: float = 0.3

    turnover_cap_lots: Optional[int] = 400
    cap_disabled_first_window: bool = True

    cost_rate: float = 0.001
    cps_time_limit: float = 10.0
    seed: int = 42

    # Scenarios: choose S multiple of 20 to keep denom integer for beta=0.95
    n_scenarios: int = 200


def load_returns(cfg: RunConfig) -> pd.DataFrame:
    prices = download_prices(cfg.tickers, start=cfg.start)
    prices = prices.loc[:, [t for t in cfg.tickers if t in prices.columns]].copy()
    rets = prices.pct_change().dropna(how="any")

    if rets.shape[1] != len(cfg.tickers):
        raise ValueError("Some tickers missing from returns after download.")
    return rets


def init_equal_weight_lots(Q: int, n: int) -> np.ndarray:
    base = Q // n
    rem = Q - base * n
    q0 = np.full(n, base, dtype=int)
    for i in range(rem):
        q0[i] += 1
    if int(q0.sum()) != int(Q):
        raise ValueError("init_equal_weight_lots: does not sum to Q")
    return q0


def turnover_weight_from_lots(turnover_lots: float, Q: int) -> float:
    return 0.5 * float(turnover_lots) / float(Q)


def run_walkforward(cfg: RunConfig, rets: Optional[pd.DataFrame] = None, save_curves: bool = False) -> Dict[str, Any]:
    """
    Returns dict with metrics, diagnostics, and optionally wealth curves.
    """
    if rets is None:
        rets = load_returns(cfg)

    if len(rets) < cfg.train_len + cfg.test_len + 5:
        raise ValueError("Not enough data for train/test lengths.")

    n = len(cfg.tickers)
    if len(cfg.sectors) != n:
        raise ValueError("sectors must match tickers length.")

    # OOS series
    oos_dates = []
    oos_ret_ew = []
    oos_ret_cps = []

    # diagnostics
    turnovers_lots = []
    turnovers_w = []
    costs = []
    selected = []
    statuses = []
    solve_times = []

    # portfolios
    w_ew = np.ones(n) / n
    q_old = init_equal_weight_lots(cfg.Q, n)
    w_prev = q_old / float(cfg.Q)

    for start_i in range(0, len(rets) - cfg.train_len - cfg.test_len + 1, cfg.step):
        train = rets.iloc[start_i: start_i + cfg.train_len]
        test = rets.iloc[start_i + cfg.train_len: start_i + cfg.train_len + cfg.test_len]

        mu = train.mean().values
        scenarios = bootstrap_scenarios(train.values, n_scenarios=min(cfg.n_scenarios, len(train)), seed=cfg.seed)

        cap_to_use = cfg.turnover_cap_lots
        if cfg.cap_disabled_first_window and start_i == 0:
            cap_to_use = None

        res = cpsat_cvar_portfolio(
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
            turnover_cap_lots=cap_to_use,
            time_limit_s=cfg.cps_time_limit,
            seed=cfg.seed,
        )

        # Handle infeasible (should be rare after tuning)
        if isinstance(res, tuple):
            w_new = w_prev.copy()
            q_sol = q_old.copy()
            status_str = res[2]
            solve_t = 0.0
        else:
            w_new = res.weights
            q_sol = res.q_lots
            status_str = res.status
            solve_t = res.solve_time_s

        statuses.append(status_str)
        solve_times.append(float(solve_t))

        turnover_lots = float(np.sum(np.abs(q_sol - q_old)))
        tw = turnover_weight_from_lots(turnover_lots, cfg.Q)
        # cost deducted once at rebalance date
        cost = cfg.cost_rate * tw

        turnovers_lots.append(turnover_lots)
        turnovers_w.append(tw)
        costs.append(cost)
        selected.append(int(np.sum(q_sol > 0)))

        # update
        q_old = q_sol.copy()
        w_prev = w_new.copy()

        # apply to test period, cost once at first day
        first = True
        for dt, row in test.iterrows():
            rvec = row.values.astype(float)
            oos_dates.append(dt)

            oos_ret_ew.append(float(np.dot(w_ew, rvec)))

            r_cps = float(np.dot(w_new, rvec))
            if first:
                r_cps -= cost
                first = False
            oos_ret_cps.append(r_cps)

    idx = pd.to_datetime(oos_dates)
    s_ew = pd.Series(oos_ret_ew, index=idx, name="EW")
    s_cps = pd.Series(oos_ret_cps, index=idx, name="CP-SAT")

    out = {
        "config": asdict(cfg),
        "metrics": {
            "equal_weight": perf_metrics(s_ew),
            "cpsat": perf_metrics(s_cps),
        },
        "diagnostics": {
            "avg_turnover_lots": float(np.mean(turnovers_lots)) if turnovers_lots else 0.0,
            "avg_turnover_weight": float(np.mean(turnovers_w)) if turnovers_w else 0.0,
            "avg_cost": float(np.mean(costs)) if costs else 0.0,
            "avg_selected_assets": float(np.mean(selected)) if selected else 0.0,
            "pct_optimal": float(np.mean([("OPTIMAL" in s) for s in statuses])) if statuses else 0.0,
            "avg_solve_time_s": float(np.mean(solve_times)) if solve_times else 0.0,
            "statuses_tail": statuses[-10:],
        },
    }

    if save_curves:
        out["curves"] = {"ew": wealth_curve(s_ew), "cps": wealth_curve(s_cps)}

    return out
