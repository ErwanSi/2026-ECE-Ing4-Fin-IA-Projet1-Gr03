from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import numpy as np
from ortools.sat.python import cp_model


@dataclass
class CPSatResult:
    weights: np.ndarray
    q_lots: np.ndarray
    status: str
    solve_time_s: float


def _sector_bounds_to_lots(sector_bounds: Dict[str, Tuple[float, float]], Q: int) -> Dict[str, Tuple[int, int]]:
    out = {}
    for sec, (lb, ub) in sector_bounds.items():
        lb_lots = int(np.ceil(lb * Q - 1e-12))
        ub_lots = int(np.floor(ub * Q + 1e-12))
        out[sec] = (max(0, lb_lots), min(Q, ub_lots))
    return out


def cpsat_cvar_portfolio(
    scenarios_returns: np.ndarray,
    mu: np.ndarray,
    sectors: List[str],
    sector_bounds: Dict[str, Tuple[float, float]],
    q_old: np.ndarray,
    Q: int,
    K: int,
    q_min: int,
    w_max: float,
    cost_rate: float,
    beta: float,
    lambda_cvar: float,
    turnover_cap_lots: Optional[int],
    time_limit_s: float = 10.0,
    seed: int = 42,
    scale: int = 1_000_000,
) -> CPSatResult | tuple[None, None, str]:
    """
    Maximise: expected return - lambda * CVaR(loss) - transaction_cost(turnover)

    - q_i integer lots, sum q_i = Q
    - cardinality: sum z_i <= K and q_min*z_i <= q_i <= q_max*z_i
    - sector bounds: for each sector, lb*Q <= sum q_i <= ub*Q
    - turnover: t_i >= |q_i - q_old_i|, optional cap sum t_i <= turnover_cap_lots
    - cost: penalty proportional to turnover_lots (one-way), aligned with OOS cost model
    - CVaR on loss with scenarios: loss_s = - sum_i (q_i/Q) * r_{s,i}
      Implemented as integer scaling: r_int = round(r*scale)
      loss_int = - sum_i q_i * r_int  (units: lots*scale)
      CVaR linearisation: u_s >= loss_int - eta, u_s >= 0
      CVaR proxy objective uses denom = (1-beta)*S ; choose S such that denom integer.
    """
    if scenarios_returns.ndim != 2:
        raise ValueError("scenarios_returns must be 2D (S, N).")
    S, N = scenarios_returns.shape
    if len(mu) != N or len(sectors) != N or len(q_old) != N:
        raise ValueError("Dimension mismatch between mu/sectors/q_old and scenarios_returns.")
    if not (0.0 < beta < 1.0):
        raise ValueError("beta must be in (0,1).")

    # Choose denom integer by requiring S multiple of 20 when beta=0.95.
    # If not, we still proceed but denom becomes rounded; for stability we recommend S=200 in runner.
    alpha = 1.0 - beta
    denom = int(round(alpha * S))
    if denom <= 0:
        return (None, None, "CpSolverStatus.INFEASIBLE")

    q_max = int(np.floor(w_max * Q + 1e-12))
    if q_max <= 0:
        return (None, None, "CpSolverStatus.INFEASIBLE")

    sector_bounds_lots = _sector_bounds_to_lots(sector_bounds, Q)
    unique_sectors = sorted(set(sectors))

    # Scale returns to int for CP-SAT
    R_int = np.rint(scenarios_returns * scale).astype(int)  # (S,N)
    mu_int = np.rint(mu * scale).astype(int)               # (N,)

    model = cp_model.CpModel()

    # Decision vars
    q = [model.NewIntVar(0, q_max, f"q[{i}]") for i in range(N)]
    z = [model.NewBoolVar(f"z[{i}]") for i in range(N)]

    # Budget
    model.Add(sum(q) == Q)

    # Cardinality and linking
    model.Add(sum(z) <= K)
    for i in range(N):
        model.Add(q[i] >= q_min * z[i])
        model.Add(q[i] <= q_max * z[i])

    # Sector constraints (in lots)
    for sec in unique_sectors:
        lb_lots, ub_lots = sector_bounds_lots.get(sec, (0, Q))
        idx = [i for i, s in enumerate(sectors) if s == sec]
        if len(idx) == 0:
            continue
        model.Add(sum(q[i] for i in idx) >= lb_lots)
        model.Add(sum(q[i] for i in idx) <= ub_lots)

    # Turnover absolute values
    # t_i integer lots >= |q_i - q_old_i|
    t = [model.NewIntVar(0, Q, f"t[{i}]") for i in range(N)]
    for i in range(N):
        model.Add(t[i] >= q[i] - int(q_old[i]))
        model.Add(t[i] >= int(q_old[i]) - q[i])

    total_turnover = model.NewIntVar(0, N * Q, "total_turnover")
    model.Add(total_turnover == sum(t))

    if turnover_cap_lots is not None:
        model.Add(total_turnover <= int(turnover_cap_lots))

    # CVaR variables
    # loss_int(s) = - sum_i q_i * R_int[s,i]
    # eta: can be negative; bound it safely
    # Rough bounds: q up to 600, r_int maybe +-200k -> sum about +-1.2e8. Use wider.
    eta = model.NewIntVar(-10**12, 10**12, "eta")
    u = [model.NewIntVar(0, 10**12, f"u[{s}]") for s in range(S)]

    for s in range(S):
        loss_expr = []
        for i in range(N):
            # coefficient is integer; CP-SAT handles large ints in linear expressions (64-bit)
            loss_expr.append(q[i] * int(-R_int[s, i]))
        loss_s = sum(loss_expr)
        model.Add(u[s] >= loss_s - eta)
        # u[s] >= 0 implicit

    # Objective scaling
    # Expected return term in same "lots*scale" units: sum q_i * mu_int[i]
    # CVaR proxy: denom*eta + sum u
    # Transaction cost penalty in same units:
    # portfolio_cost (in return units) ~ cost_rate * 0.5 * sum|Δq|/Q
    # Multiply by Q*scale -> scale*cost_rate*0.5 * sum|Δq|
    tc_coeff = int(round(scale * cost_rate * 0.5))

    exp_ret = sum(q[i] * int(mu_int[i]) for i in range(N))
    cvar_proxy = denom * eta + sum(u)
    tc_proxy = tc_coeff * total_turnover

    # To keep terms comparable across denom, scale exp_ret and tc_proxy by denom
    obj = denom * exp_ret - int(round(lambda_cvar)) * cvar_proxy - denom * tc_proxy

    model.Maximize(obj)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.random_seed = int(seed)
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)
    status_str = solver.StatusName(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return (None, None, status_str)

    q_sol = np.array([int(solver.Value(q[i])) for i in range(N)], dtype=int)
    w_sol = q_sol / float(Q)
    return CPSatResult(weights=w_sol, q_lots=q_sol, status=status_str, solve_time_s=float(solver.WallTime()))
