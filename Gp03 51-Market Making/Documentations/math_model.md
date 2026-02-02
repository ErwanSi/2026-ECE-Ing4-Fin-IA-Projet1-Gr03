# Mathematical Model: Optimal Market Making with Inventory Risk

This document details the mathematical framework used in `src/models/inventory_hjb.py`, `src/solvers/hjb_solver.py`, and the experimental `src/solvers/csp_solver.py`.

## 1. Market Dynamics

### Price Process
We assume a mid-price $S_t$ following an arithmetic Brownian motion:
$$ dS_t = \sigma dW_t $$
where $\sigma$ is the volatility.

### Quotes
The Market Maker (MM) controls the bid and ask quotes $\delta_t^b$ and $\delta_t^a$, which are distances from the mid-price:
$$ P_t^b = S_t - \delta_t^b $$
$$ P_t^a = S_t + \delta_t^a $$

### Order Flow
The arrival of market orders is modeled by Poisson processes $N_t^b$ (sell orders hitting our bid) and $N_t^a$ (buy orders hitting our ask) with intensities exponentially decaying with spread:
$$ \Lambda^b(\delta^b) = A e^{-k \delta^b} $$
$$ \Lambda^a(\delta^a) = A e^{-k \delta^a} $$
- **$A$**: Base intensity parameter.
- **$k$**: Market depth parameter (sensitivity to spread).

## 2. Optimization Problem (HJB)

The MM maximizes the expected utility of terminal wealth over horizon $[0, T]$. We use an Exponential Utility (CARA) function:
$$ U(x) = -e^{-\gamma x} $$
where $\gamma$ is the risk aversion parameter.

### Value Function
The value function $u(t, x, q, S)$ satisfies the Hamilton-Jacobi-Bellman (HJB) equation.
Using the change of variables from **Guéant et al. (2013)**, we use the ansatz:
$$ u(t, x, q, S) = -e^{-\gamma(x + qS)} v_q(t)^{-\frac{\gamma}{k}} $$

This reduces the problem to solving a linear system of ODEs for $v_q(t)$.

### System of ODEs
For inventory levels $q \in \{-Q, \dots, Q\}$:
$$ \dot{v}_q(t) = \alpha q^2 v_q(t) - \eta \left( v_{q-1}(t) + v_{q+1}(t) \right) $$

where:
- $\alpha = \frac{k^2 \sigma^2 \gamma}{2}$ (Inventory Risk penalty)
- $\eta = A \left( 1 + \frac{\gamma}{k} \right)^{-1}$ (Liquidity intensity)

Terminal Condition: $v_q(T) = 1$.

In `hjb_solver.py`, this system is solved using the **Matrix Exponential** method:
$$ V(t) = \exp(-M (T-t)) V(T) $$
where $M$ is the tridiagonal matrix representing the ODE system.

## 3. Optimal Quotes and Interpretation

Once $v_q(t)$ is computed, the optimal quotes are:

$$ \delta_t^{b*} = \frac{1}{k} \ln \left( \frac{v_q(t)}{v_{q+1}(t)} \right) + \text{const} $$
$$ \delta_t^{a*} = \frac{1}{k} \ln \left( \frac{v_q(t)}{v_{q-1}(t)} \right) + \text{const} $$
with $\text{const} = \frac{1}{\gamma} \ln \left( 1 + \frac{\gamma}{k} \right)$.

### Interpretation
1. **Base Spread:** The constant term ensures positive spread to capture the bid-ask bounce.
2. **Inventory Skew:** The ratio $\frac{v_q}{v_{q+1}}$ adjusts quotes based on inventory:
   - If $q > 0$ (long), we lower $\delta^a$ (sell cheaper) and raise $\delta^b$ (buy lower) to dump inventory.
   - If $q < 0$ (short), we do the reverse to buy back.
3. **Time Decay:** As $t \to T$, the urgency to close inventory increases (if a penalty is applied, though classic Guéant just maximizes utility).

## 4. Inventory Constraints

### Hard Constraints
The inventory is rigidly bounded:
$$ q \in \{-Q_{max}, \ldots, Q_{max}\} $$

### Boundary Conditions
At the boundaries, the market maker cannot extend further:
- If $q = Q_{max}$, $\delta^b \to \infty$ (Stop buying).
- If $q = -Q_{max}$, $\delta^a \to \infty$ (Stop selling).

These boundaries are naturally handled in the linear ODE system by assuming $v_{Q+1}(t)$ is effectively zero or the transition rate is zero.

## 5. Alternative: CSP Formulation (Experimental)

A Constraint Satisfaction Problem (CSP) approach is implemented in `csp_solver.py` using **OR-Tools**.

### Formulation
Instead of stochastic control, we treat the problem as a deterministic scheduling problem (or solve for specific scenarios):
- **Variables**: Action at each step (Buy, Sell, Hold).
- **Constraints**: 
  - Inventory bounds: $-Q_{max} \leq q_t \leq Q_{max}$
  - Cash constraints.
  - Optional: Value at Risk (VaR) or Drawdown limits.
- **Objective**: Maximize Total Wealth.

This serves as a benchmark or for solving complex constrained scenarios where HJB is intractable.

## 6. Metrics & Risk Measures

The strategy performance is evaluated using:
- **PnL**: $X_t + q_t S_t - X_0$ (Mark-to-Market PnL).
- **Sharpe Ratio**: $\frac{E[\text{Returns}]}{\sqrt{\text{Var}[\text{Returns}]}}$.
- **Max Drawdown**: Maximum peak-to-trough decline in PnL.
- **Inventory Distribution**: Histogram of time spent at each inventory level $q$.
