import numpy as np
import cvxpy as cp


def markowitz_with_turnover(mu, Sigma, w_old=None, lam=2.0, gamma_tc=0.02, w_max=0.60):
    """
    Mean-variance with L1 turnover penalty.

    max_w   mu^T w - lam * w^T Sigma w - gamma_tc * ||w - w_old||_1
    s.t.    sum(w)=1, 0 <= w <= w_max

    Notes:
    - This is convex.
    - If gamma_tc too large, optimizer sticks to w_old (often equal-weight).
    """
    mu = np.asarray(mu).reshape(-1)
    n = mu.shape[0]

    Sigma = np.asarray(Sigma)
    Sigma = 0.5 * (Sigma + Sigma.T)  # enforce symmetry
    Sigma = Sigma + 1e-8 * np.eye(n)  # small ridge

    if w_old is None:
        w_old = np.ones(n) / n
    w_old = np.asarray(w_old).reshape(-1)

    w = cp.Variable(n)
    risk = cp.quad_form(w, Sigma)
    turnover = cp.norm1(w - w_old)

    obj = cp.Maximize(mu @ w - lam * risk - gamma_tc * turnover)
    constraints = [cp.sum(w) == 1, w >= 0, w <= float(w_max)]

    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.OSQP, verbose=False)

    if w.value is None:
        # fallback safe
        return w_old, float("nan")

    w_sol = np.asarray(w.value).reshape(-1)
    # clip numerical noise + renormalize
    w_sol = np.clip(w_sol, 0.0, float(w_max))
    s = w_sol.sum()
    if s <= 1e-12:
        w_sol = np.ones(n) / n
    else:
        w_sol = w_sol / s

    return w_sol, prob.value
