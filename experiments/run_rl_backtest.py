import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.simulator import OrderBookSimulator, SimulationConfig
from src.models.inventory_hjb import InventoryHJBModel, MarketParameters
from src.solvers.hjb_solver import HJBSolver

def run_rl_comparison():
    print("Starting Backtest: RL vs HJB vs Naive...")
    
    # 1. Load RL Model
    model_path = "models/ppo_market_maker.zip"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run 'python src/rl_env/train_rl.py' first.")
        return

    print("Loading PPO Model...")
    model = PPO.load(model_path)
    
    # Configuration (Must match training config roughly)
    config = SimulationConfig(T=1.0, dt=1/2520, sigma=0.5)
    params = MarketParameters(sigma=0.5, gamma=0.1, k=1.5, A=140.0)
    
    # Initialize HJB
    hjb_model = InventoryHJBModel(params)
    solver_hjb = HJBSolver(params, max_inventory_Q=20)
    
    # --- STRATEGIES ---
    
    def strategy_naive(q, t_left):
        fixed_spread = 0.05
        return fixed_spread, fixed_spread
        
    def strategy_hjb_exact(q, t_left):
        return solver_hjb.get_optimal_quotes(q, t_left)

    def strategy_rl(q, t_left):
        # Map state to observation
        # Obs: [Inventory, Normalized Time]
        # Normalized Time = (T - t_left) / T ? No, env uses current_step / max_steps
        # t_left goes from T to 0. current_time goes from 0 to T.
        # current_time = T - t_left
        # norm_time = (T - t_left) / T
        
        current_time_norm = (config.T - t_left) / config.T
        obs = np.array([float(q), current_time_norm], dtype=np.float32)
        
        # Predict action
        action, _ = model.predict(obs, deterministic=True)
        return action[0], action[1]

    # --- SIMULATIONS ---
    results = {}
    strategies = {
        'Naive': strategy_naive,
        'HJB_Exact': strategy_hjb_exact,
        'RL_PPO': strategy_rl
    }
    
    # Use SAME seed/price path for fair comparison
    from src.data.simulator import generate_price_path
    price_path = generate_price_path(config, int(config.T/config.dt))
    
    for name, strat in strategies.items():
        print(f"Running {name}...")
        sim = OrderBookSimulator(config, price_path=price_path)
        df = sim.run_simulation(strat)
        results[name] = df
        
        print(f"{name} Final PnL: {df['pnl'].iloc[-1]:.2f}")
        
    plot_results(results)
    print("Comparison Complete. See experiments/results/rl_comparison.png")

def plot_results(results):
    os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)
    
    # PnL
    plt.figure(figsize=(10, 6))
    for name, df in results.items():
        plt.plot(df['pnl'], label=name)
    plt.title('PnL Comparison: RL vs Maths')
    plt.xlabel('Steps')
    plt.ylabel('PnL')
    plt.legend()
    plt.grid(True)
    plt.savefig('experiments/results/rl_comparison.png')
    plt.close()
    
    # Inventory
    plt.figure(figsize=(10, 6))
    for name, df in results.items():
        plt.plot(df['inventory'], alpha=0.7, label=name)
    plt.title('Inventory Management')
    plt.legend()
    plt.savefig('experiments/results/rl_inventory.png')
    plt.close()

if __name__ == "__main__":
    run_rl_comparison()
