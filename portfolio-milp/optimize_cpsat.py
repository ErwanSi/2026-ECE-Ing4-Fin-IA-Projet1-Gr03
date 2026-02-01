from ortools.sat.python import cp_model
import numpy as np


def cpsat_cvar_portfolio(
    scenarios_returns: np.ndarray,
    mu: np.ndarray,
    sectors: list[str],
    sector_bounds: dict,
    q_old: np.ndarray | None,
    Q: int = 1000,
    K: int = 5,
    q_min: int = 1,
    w_max: float = 0.60,
    tc_per_lot: int = 5,
    beta: float = 0.95,
    lambda_cvar: float = 1.0,
    time_limit_s: float = 10.0,
    seed: int = 42,
):
    """
    MILP/CSP with CP-SAT:
    - discrete lots q_i integers, sum q_i = Q
    - cardinality via z_i
    - sector bounds (in lots)
    - transaction costs via abs(q_i - q_old_i)
    - CVaR risk on scenario losses

    Objective (scaled to integers):
        maximize  E[port_return] - lambda_cvar * CVaR_beta(loss) - tc_cost
    """
    R = np.asarray(scenarios_returns, dtype=float)
    S, n = R.shape
    mu = np.asarray(mu, dtype=float).reshape(-1)
    assert len(mu) == n

    if q_old is None:
        q_old = np.zeros(n, dtype=int)
    else:
        q_old = np.asarray(q_old, dtype=int).reshape(-1)

    # Scaling
    R_SCALE = 1_000_000  # scenario returns scaled
    MU_SCALE = 1_000_000

    R_int = np.round(R * R_SCALE).astype(int)
    mu_int = np.round(mu * MU_SCALE).astype(int)

    # CVaR denom: (1-beta)*S must be integer >= 1
    denom = int(round((1.0 - beta) * S))
    denom = max(1, denom)

    model = cp_model.CpModel()

    q_max = int(np.floor(w_max * Q + 1e-9))

    q = [model.NewIntVar(0, q_max, f"q_{i}") for i in range(n)]
    z = [model.NewBoolVar(f"z_{i}") for i in range(n)]
    t = [model.NewIntVar(0, Q, f"t_{i}") for i in range(n)]  # turnover abs lots

    # budget
    model.Add(sum(q) == Q)

    # cardinality + linking
    model.Add(sum(z) <= int(K))
    for i in range(n):
        model.Add(q[i] <= q_max * z[i])
        model.Add(q[i] >= int(q_min) * z[i])

        diff = model.NewIntVar(-Q, Q, f"diff_{i}")
        model.Add(diff == q[i] - int(q_old[i]))
        model.AddAbsEquality(t[i], diff)

    # sector constraints
    for sec, (Lw, Uw) in sector_bounds.items():
        idx = [i for i, s in enumerate(sectors) if s == sec]
        if not idx:
            continue
        L = int(round(Lw * Q))
        U = int(round(Uw * Q))
        model.Add(sum(q[i] for i in idx) >= L)
        model.Add(sum(q[i] for i in idx) <= U)

    # Scenario portfolio return (scaled integer): pr_s = sum_i q_i * R_int[s,i]
    pr = []
    for s in range(S):
        pr_s = model.NewIntVar(-10**12, 10**12, f"pr_{s}")
        model.Add(pr_s == sum(q[i] * int(R_int[s, i]) for i in range(n)))
        pr.append(pr_s)

    # CVaR on losses: loss_s = -pr_s
    eta = model.NewIntVar(-10**12, 10**12, "eta")
    u = [model.NewIntVar(0, 10**12, f"u_{s}") for s in range(S)]
    for s in range(S):
        loss_s = model.NewIntVar(-10**12, 10**12, f"loss_{s}")
        model.Add(loss_s == -pr[s])
        # u_s >= loss_s - eta, u_s >= 0
        model.Add(u[s] >= loss_s - eta)
        model.Add(u[s] >= 0)

    # Expected return term (use mu_int and q directly for stability)
    exp_ret = model.NewIntVar(-10**12, 10**12, "exp_ret")
    model.Add(exp_ret == sum(q[i] * int(mu_int[i]) for i in range(n)))

    # Transaction costs
    tc = model.NewIntVar(0, 10**12, "tc")
    model.Add(tc == int(tc_per_lot) * sum(t))

    # Objective scaling:
    # maximize exp_ret - lambda*( eta + (1/denom)*sum u ) - tc
    # => multiply by denom:
    # maximize denom*exp_ret - lambda*(denom*eta + sum u) - denom*tc
    # lambda_cvar may be float => scale it
    LAMBDA_SCALE = 1000
    lam_int = int(round(lambda_cvar * LAMBDA_SCALE))

    obj = (
        denom * exp_ret
        - lam_int * (denom * eta + sum(u)) // LAMBDA_SCALE
        - denom * tc
    )

    model.Maximize(obj)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = 8
    solver.parameters.random_seed = int(seed)

    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, None, status

    q_sol = np.array([solver.Value(q[i]) for i in range(n)], dtype=int)
    w_sol = q_sol / float(Q)
    return w_sol, q_sol, status
