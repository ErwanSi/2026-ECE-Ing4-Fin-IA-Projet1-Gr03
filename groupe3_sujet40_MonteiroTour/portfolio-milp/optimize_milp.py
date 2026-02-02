from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import numpy as np
from ortools.linear_solver import pywraplp


@dataclass
class MILPResult:
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


def milp_cvar_portfolio(
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
) -> MILPResult | tuple[None, None, str]:
    """
    MILP variant using OR-Tools CBC (MIP).
    Uses float coefficients (w = q/Q) for losses; q and z are integer/binary.
    """
    if scenarios_returns.ndim != 2:
        raise ValueError("scenarios_returns must be 2D (S, N).")
    S, N = scenarios_returns.shape
    if len(mu) != N or len(sectors) != N or len(q_old) != N:
        raise ValueError("Dimension mismatch.")
    if not (0.0 < beta < 1.0):
        raise ValueError("beta must be in (0,1).")

    alpha = 1.0 - beta
    denom = alpha * S
    if denom <= 1e-12:
        return (None, None, "INFEASIBLE")

    q_max = int(np.floor(w_max * Q + 1e-12))
    if q_max <= 0:
        return (None, None, "INFEASIBLE")

    sector_bounds_lots = _sector_bounds_to_lots(sector_bounds, Q)
    unique_sectors = sorted(set(sectors))

    solver = pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")
    if solver is None:
        raise RuntimeError("CBC solver not available in this OR-Tools build.")

    solver.SetTimeLimit(int(time_limit_s * 1000))

    # Vars
    q = [solver.IntVar(0, q_max, f"q[{i}]") for i in range(N)]
    z = [solver.BoolVar(f"z[{i}]") for i in range(N)]
    t = [solver.NumVar(0.0, float(Q), f"t[{i}]") for i in range(N)]

    # Budget
    solver.Add(sum(q) == Q)

    # Cardinality + linking
    solver.Add(sum(z) <= K)
    for i in range(N):
        solver.Add(q[i] >= q_min * z[i])
        solver.Add(q[i] <= q_max * z[i])

    # Sector bounds
    for sec in unique_sectors:
        lb_lots, ub_lots = sector_bounds_lots.get(sec, (0, Q))
        idx = [i for i, s in enumerate(sectors) if s == sec]
        if len(idx) == 0:
            continue
        solver.Add(sum(q[i] for i in idx) >= lb_lots)
        solver.Add(sum(q[i] for i in idx) <= ub_lots)

    # Turnover abs
    for i in range(N):
        solver.Add(t[i] >= q[i] - float(q_old[i]))
        solver.Add(t[i] >= float(q_old[i]) - q[i])

    total_turnover = solver.NumVar(0.0, float(N * Q), "total_turnover")
    solver.Add(total_turnover == sum(t))

    if turnover_cap_lots is not None:
        solver.Add(total_turnover <= float(turnover_cap_lots))

    # CVaR: loss_s = - sum_i (q_i/Q)*r_{s,i}
    eta = solver.NumVar(-10.0, 10.0, "eta")  # daily loss typically within [-10,10] is safe
    u = [solver.NumVar(0.0, 10.0, f"u[{s}]") for s in range(S)]

    for s in range(S):
        loss_s = solver.Sum([(-scenarios_returns[s, i] / float(Q)) * q[i] for i in range(N)])
        solver.Add(u[s] >= loss_s - eta)

    # Objective:
    # maximize: mu·w - lambda * (eta + (1/denom)*sum u) - cost_rate*0.5*(sum|Δq|/Q)
    exp_ret = solver.Sum([(mu[i] / float(Q)) * q[i] for i in range(N)])
    cvar = eta + (1.0 / denom) * solver.Sum(u)
    tc = (cost_rate * 0.5 / float(Q)) * total_turnover

    solver.Maximize(exp_ret - lambda_cvar * cvar - tc)

    status = solver.Solve()
    status_str = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }.get(status, str(status))

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return (None, None, status_str)

    q_sol = np.array([int(round(q[i].solution_value())) for i in range(N)], dtype=int)
    w_sol = q_sol / float(Q)

    return MILPResult(weights=w_sol, q_lots=q_sol, status=status_str, solve_time_s=float(solver.WallTime()) / 1000.0)
