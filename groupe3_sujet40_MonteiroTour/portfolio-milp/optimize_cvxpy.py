import numpy as np
import cvxpy as cp


def markowitz_with_turnover(mu, Sigma, w_old=None, lam=0.3, gamma_tc=0.05, w_max=0.60):
    """
    Convex baseline:
      maximize mu^T w - lam*||w - w_old||_1 - gamma_tc*||w||_2^2 - risk
    with:
      sum(w)=1, 0<=w<=w_max
    """
    mu = np.asarray(mu, dtype=float).reshape(-1)
    n = len(mu)

    Sigma = np.asarray(Sigma, dtype=float)
    Sigma = 0.5 * (Sigma + Sigma.T)  # enforce symmetry
    Sigma = Sigma + 1e-8 * np.eye(n)

    if w_old is None:
        w_old = np.ones(n) / n
    w_old = np.asarray(w_old, dtype=float).reshape(-1)

    w = cp.Variable(n)

    risk = cp.quad_form(w, Sigma)
    turnover = cp.norm1(w - w_old)
    reg = cp.sum_squares(w)

    obj = cp.Maximize(mu @ w - risk - lam * turnover - gamma_tc * reg)
    constraints = [cp.sum(w) == 1, w >= 0, w <= float(w_max)]

    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    if w.value is None:
        return w_old, None

    w_sol = np.array(w.value).reshape(-1)
    w_sol = np.clip(w_sol, 0.0, float(w_max))
    s = float(w_sol.sum())
    if s > 1e-12:
        w_sol /= s
    else:
        w_sol = w_old

    return w_sol, prob.value
