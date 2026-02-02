# Gp03 - 51. Market Making   Hamza OUADOUDI/Erwan SIMON/Guillaume Orsini

A scientifically rigorous implementation of **Optimal Market Making with Inventory Risk**, based on the frameworks of **Avellaneda-Stoikov (2008)** and **Guéant et al. (2013)**.

## Project Goal
To provide a modular, reproducible, and extensible codebase for researching and testing market making strategies that optimize PnL while managing inventory constraints.

## Features
- **Theoretical Models**: Analytical solvers for the HJB equation (Matrix Exponential method).
- **Simulation**: High-fidelity Order Book simulator with Poisson arrival rates ($A e^{-k\delta}$) and Brownian price dynamics.
- **Reinforcement Learning**: Gymnasium environment compatible with Stable Baselines 3 (PPO/DQN).
- **Experimentation**: Backtesting pipeline with automated plotting of PnL, Inventory, and Sharpe Ratios.
- **Real Data Backtest**: Integration with Binance API (via `ccxt`) to test strategies on historical market data.
- **Constraint Programming**: Experimental OR-Tools solver for optimal execution paths.

## Architecture
```
market-making/
├── src/
│   ├── models/       # Analytical models (Avellaneda-Stoikov)
│   ├── solvers/      # HJB (SciPy) and CSP (OR-Tools) solvers
│   ├── data/         # Simulators and Price generators
│   └── rl_env/       # Gymnasium MarketMakingEnv
├── Experiments/      # Backtesting scripts
├── Documentations/             # Mathematical documentation and State of the Art
└── requirements.txt
```

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Running Backtests](#running-backtests)
5. [Training RL Agents](#training-rl-agents)
6. [Reproducing Results](#reproducing-results)
7. [Key Results](#key-results)
8. [References](#references)

---

## Installation

### Prerequisites
- Python 3.10+
- pip (virtualenv recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/ErwanSi/2026-ECE-Ing4-Fin-IA-Projet1-Gr03.git
cd 2026-ECE-Ing4-Fin-IA-Projet1-Gr03
```

### Step 2: Create Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\activate   # Windows
# source venv/bin/activate # Linux/Mac
```

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```
*Note: This installs numpy, pandas, matplotlib, stable-baselines3, ccxt, etc.*

---

## Quick Start

### Running Your First Backtest
The easiest way to see the project in action is to run the standard simulation script. This compares the **Naive** strategy against the **Optimal HJB** strategy.

```powershell
python experiments/run_backtest.py
```

**Results:**
- Check `experiments/results/` for plots (`pnl.png`, `inventory_dist.png`).
- Confirms that HJB manages inventory risk better than Naive.

### Using the Python API
```python
from src.models.inventory_hjb import InventoryHJBModel, MarketParameters
from src.data.simulator import OrderBookSimulator, SimulationConfig

# 1. Setup Model
params = MarketParameters(sigma=0.2, k=1.5, A=140.0)
model = InventoryHJBModel(params)

# 2. Setup Simulator
sim_config = SimulationConfig(sigma=0.2, T=1.0, dt=0.001)
simulator = OrderBookSimulator(sim_config)

# 3. Define Policy
def hjb_policy(inventory, time_left):
    d_b, d_a = model.get_quotes(inventory, time_left)
    return d_b, d_a

# 4. Run
results = simulator.run_simulation(hjb_policy)
print(f"Final PnL: {results['pnl'].iloc[-1]:.2f}")
```

---

## Configuration

### Model Parameters
The [`MarketParameters`](src/models/inventory_hjb.py) class controls the mathematical model:

| Parameter | Default | Description |
|-----------|----------|-------------|
| `sigma` | 0.1 | Volatility of the asset (annualized) |
| `A` | 140.0 | Base order arrival intensity |
| `k` | 1.5 | Market depth (sensitivity to spread) |
| `gamma` | 0.1 | Risk aversion coefficient |
| `dt` | 1/252 | Time step for internal model dynamics |

### Simulator Parameters
The [`SimulationConfig`](src/data/simulator.py) class controls the simulation environment:

| Parameter | Default | Description |
|-----------|----------|-------------|
| `S0` | 100.0 | Initial price |
| `T` | 1.0 | Simulation horizon (years) |
| `dt` | 1/2520 | Time resolution |
| `maker_fee` | 0.0 | Transaction costs |
| `allow_sniping` | False | Enable toxic flow (latency arbitrage) |

---

## Running Backtests

We provide several pre-built experiments in the `experiments/` folder:

### 1. Standard Simulation
Performs a comparative backtest on synthetic data (Brownian Motion).
```powershell
python experiments/run_backtest.py
```

### 2. Real Data (Crypto)
Tests strategies on real Bitcoin price history (downloaded via Binance API).
```powershell
python experiments/run_real_data_backtest.py
```
*Features:*
- Downloads 10000m of BTC/USDT 1m candles.
- Estimates volatility from real data.
- Replays the exact price path.

### 3. Realistic Scenarios (Toxic Flow)
Tests resilience against "snipers" (latency arbitrageurs) and transaction fees.
```powershell
python experiments/run_realistic_backtest.py
```

---

## Training RL Agents

Train a Deep Reinforcement Learning agent (PPO) to discover optimal strategies without knowing the math model.

### Train
```powershell
python src/rl_env/train_rl.py
```
This script:
1. Creates a `MarketMakingEnv` (Gymnasium).
2. Trains a PPO agent (Stable Baselines 3) for ~1M steps.
3. Saves models to `models/ppo_market_maker`.

### Evaluate
The training script automatically evaluates the agent periodically. You can modify it to load the model and run a dedicated test episode.

---

## Reproducing Results

To reproduce the classic Guéant et al. (2013) results showing inventory mean-reversion:

```python
from src.models.inventory_hjb import InventoryHJBModel, MarketParameters

# Parameters closer to the paper's assumptions
params = MarketParameters(
    sigma=0.3,
    gamma=0.1,
    k=0.3,    # Lower liquidity
    A=0.5     # Lower intensity
)
model = InventoryHJBModel(params)

# Check quotes for long inventory (q=5) vs short inventory (q=-5)
# Expect: Bid drops (don't buy), Ask drops (sell aggressive) for q=5
b_long, a_long = model.get_quotes(inventory_q=5)
b_short, a_short = model.get_quotes(inventory_q=-5)

print(f"Long Inventory (q=5): Bid_skew={b_long:.4f}, Ask_skew={a_long:.4f}")
print(f"Short Inventory (q=-5): Bid_skew={b_short:.4f}, Ask_skew={a_short:.4f}")
```

---

## Key Results
The analytical HJB strategy consistently outperforms naive fixed-spread strategies by skewing quotes to mean-revert inventory, thus avoiding toxic inventory buildup during trends.

## References
See `docs/references.md`.
