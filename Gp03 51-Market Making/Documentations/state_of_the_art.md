# State of the Art: Optimal Market Making with Inventory Risk

## Table of Contents

1. [Introduction](#introduction)
2. [Foundational Papers](#foundational-papers)
3. [Modern Approaches](#modern-approaches)
4. [Reinforcement Learning Methods](#reinforcement-learning-methods)
5. [Comparative Analysis](#comparative-analysis)
6. [Project Direction & Implementation](#project-direction--implementation)
7. [Open Challenges](#open-challenges)

---

## Introduction

Market making is a fundamental problem in quantitative finance where an agent continuously provides liquidity by posting bid and ask quotes. The core challenge lies in balancing two competing objectives:

1. **Profit maximization** through bid-ask spreads
2. **Risk management** of inventory exposure to price movements

This document provides a comprehensive review of the state-of-the-art approaches to optimal market making with inventory constraints.

---

## Foundational Papers

### 1. Guéant, Lehalle, and Fernandez-Tapia (2013)
**Title:** "Dealing with the Inventory Risk. A solution to the market making problem"  
**Source:** Mathematical Finance, 2013

#### Summary
This seminal paper extends the Avellaneda-Stoikov model by introducing explicit inventory constraints. The authors formulate the market making problem as a stochastic optimal control problem where a market maker maximizes expected utility of P&L over a finite time horizon.

#### Key Contributions

1. **Hamilton-Jacobi-Bellman (HJB) Transformation**
   - The HJB equations are transformed into a system of linear Ordinary Differential Equations (ODEs)
   - This transformation enables efficient numerical solution (Matrix Exponential).

2. **Inventory Constraints**
   - Explicit bounds on inventory position: $q \in [-Q_{max}, Q_{max}]$
   - Terminal inventory liquidation cost

3. **Closed-Form Approximations**
   - Asymptotic analysis for large time horizons
   - Spectral characterization of optimal quotes

#### Key Equations

**Optimal Quotes (Asymptotic):**
$$\delta^b(q) \approx \frac{1}{k} \ln\left(1 + \frac{\gamma}{k}\right) + \frac{2q+1}{2} \frac{\gamma \sigma^2}{A k}$$

Where:
- $\delta^b, \delta^a$: bid/ask half-spreads
- $k$: market depth parameter
- $\gamma$: risk aversion coefficient
- $q$: inventory position

---

### 2. Avellaneda and Stoikov (2008)
**Title:** "High-frequency trading in a limit order book"

#### Summary
The foundational paper that first formulated market making as a stochastic control problem. Introduced the concept of optimal quoting based on inventory and time horizon.

#### Limitations
- No explicit inventory bounds (soft penalty via utility).
- Assumes simple exponential decay of fill probabilities.
- Does not account for adverse selection directly.

---

## Modern Approaches

### 3. Stochastic Control with Alpha Signals
**Source:** Cartea et al. (2014) / Stanford Framework (Rao, 2020)

#### Summary
Modern frameworks extend the HJB equations to include short-term alpha signals (price drift).

#### Key Features
1. **Alpha Integration**: Adjusting the "fair price" or "reservation price" based on short-term factors (momentum, order book imbalance).
   $$ r(s, q, t) = s + \text{Alpha} - q \gamma \sigma^2 (T-t) $$
2. **Numerical Methods**: Finite difference schemes for solving HJB in complex state spaces (e.g., stochastic volatility).

---

## Reinforcement Learning Methods

### 4. Adaptive Market Making (SIAM 2024 / Deep RL)
**Source:** Recent literature (Zhang 2024, Gu 2024)

#### Summary
Recent advances apply Deep Reinforcement Learning (DRL) to learn quoting policies without assuming a specific market model (model-free).

#### Key Features
1. **Deep Q-Networks (DQN) & PPO**: Agents learn to map states (Price, Inventory, Volatility, Order Book Imbalance) to actions (Spread, Bias).
2. **Adaptability**: These models can theoretically adapt to regime changes (e.g., sudden volatility spikes) better than static HJB parameters.
3. **Challenges**: High sample complexity (needs millions of steps) and "Sim-to-Real" gap.

---

## Comparative Analysis

| Method | Inventory Constraints | Computational Complexity | Adaptability | Theoretical Guarantees |
|--------|---------------------|-------------------------|--------------|----------------------|
| Avellaneda-Stoikov | Soft | Low | Low | High |
| **Guéant et al.** | **Hard** | **Medium** | **Low** | **High** |
| Stochastic Control | Yes | High | Low | High |
| Deep RL (PPO) | Learned | High | High | Low |

---

## Project Direction & Implementation

For this project (`market-making-inventory`), we adopt a hybrid approach to ensure both rigor and experimental discovery.

### Baseline: Guéant et al. (Analytical)
We implement the **Guéant et al.** approach as the core mathematical model.
- **Why?** It is tractable, robust, and provides a mathematically guaranteed optimal strategy under its assumptions.
- **Implementation:** `src/solvers/hjb_solver.py` uses the Matrix Exponential method to solve the linear ODEs exactly.

### Experimental: Reinforcement Learning (Agent)
We compare the analytical baseline against a **Deep RL Agent (PPO)**.
- **Why?** To see if an AI agent can discover superior strategies, particularly in complex scenarios where the Poisson assumption fails or when incorporating real market data.
- **Implementation:** `src/rl_env/` contains a Gymnasium environment compatible with Stable Baselines 3.

---

## Open Challenges

1. **Multi-Asset Market Making**: Correlated inventory management and cross-asset hedging.
2. **Adverse Selection**: Detecting informed traders (toxic flow) and widening spreads accordingly.
3. **Real-World Implementation**: Handling latency and microstructure noise.

---

## Conclusion

The field has evolved from the early Avellaneda-Stoikov model to modern Deep RL. Our project bridges this gap by providing a codebase that implements the rigorous HJB solution as a ground truth, while enabling RL experimentation.
