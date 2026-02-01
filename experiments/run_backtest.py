import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.simulator import OrderBookSimulator, SimulationConfig
from src.models.inventory_hjb import InventoryHJBModel, MarketParameters
from src.solvers.hjb_solver import HJBSolver

def run_backtest():
    print("Starting Backtest...")
    
    # Configuration
    config = SimulationConfig(T=1.0, dt=1/2520, sigma=0.5) # Higher sigma to see effects
    params = MarketParameters(sigma=0.5, gamma=0.1, k=1.5, A=140.0)
    
    # Initialize Models
    hjb_model = InventoryHJBModel(params)
    solver_hjb = HJBSolver(params, max_inventory_Q=20)
    
    # Strategies definitions
    def strategy_naive(q, t_left):
        # Fixed symmetric spread
        fixed_spread = 0.05
        return fixed_spread, fixed_spread
        
    def strategy_hjb_approx(q, t_left):
        # Analytical Approximation
        return hjb_model.get_quotes(q, t_left)

    def strategy_hjb_exact(q, t_left):
        # Matrix Exponential exact solution
        return solver_hjb.get_optimal_quotes(q, t_left)

    # Run Simulations
    results = {}
    strategies = {
        'Naive': strategy_naive,
        'HJB_Approx': strategy_hjb_approx,
        'HJB_Exact': strategy_hjb_exact
    }
    
    for name, strat in strategies.items():
        print(f"Running {name}...")
        sim = OrderBookSimulator(config)
        df = sim.run_simulation(strat)
        results[name] = df
        
        print(f"{name} Final PnL: {df['pnl'].iloc[-1]:.2f}")
        
    # Plotting
    plot_results(results)
    # Plotting
    plot_results(results)
    
    # --- CSP (Benchmark) ---
    # Calculate optimal hindsight performance on the SAME price path
    if 'Naive' in results and 'src.solvers.csp_solver' in sys.modules:
        try:
            from src.solvers.csp_solver import CSPSolver
            print("\nRunning CSP Benchmark (Theoretical Upper Bound)...")
            
            df_ref = results['Naive']
            full_price_path = df_ref['price'].values
            
            # --- Downsample for Performance ---
            # CSP is slow (NP-hardish), so we solve for a smaller grid
            # e.g., 50 steps representing the path
            n_steps_csp = 50
            indices = np.linspace(0, len(full_price_path)-1, n_steps_csp, dtype=int)
            price_path_ds = full_price_path[indices]
            
            # --- Perfect Liquidity Assumption ---
            # Assume we CAN trade at any of these steps (Upper Bound)
            buy_arrivals = [True] * n_steps_csp
            sell_arrivals = [True] * n_steps_csp
            
            csp = CSPSolver(horizon_steps=n_steps_csp, max_inventory=10)
            opt_wealth, actions = csp.solve_trajectory(price_path_ds, buy_arrivals, sell_arrivals)
            
            if opt_wealth is not None:
                print(f"CSP Optimal PnL (Approx Upper Bound): {opt_wealth:.2f}")
                
                # Plot simple horizontal line or point for ref
                # Since time scale is different, we just print it.
            else:
                print("CSP Status: Infeasible/Failed")
            
        except ImportError:
            pass
        except Exception as e:
            print(f"CSP Failed: {e}")

    print("\nBacktest Complete. Results saved in experiments/results/")

def plot_results(results):
    os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)
    
    # 1. PnL Trajectories
    plt.figure(figsize=(10, 6))
    for name, df in results.items():
        plt.plot(df['time'], df['pnl'], label=name)
    plt.title('PnL Trajectories')
    plt.xlabel('Time')
    plt.ylabel('PnL')
    plt.legend()
    plt.grid(True)
    plt.savefig('experiments/results/pnl.png')
    plt.close()
    
    # 2. Inventory Distribution
    plt.figure(figsize=(10, 6))
    for name, df in results.items():
        plt.hist(df['inventory'], bins=20, alpha=0.5, label=name, density=True)
    plt.title('Inventory Distribution')
    plt.xlabel('Inventory')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig('experiments/results/inventory_dist.png')
    plt.close()
    
    # 3. Inventory vs Time (for HJB Exact) (Detail)
    if 'HJB_Exact' in results:
        df = results['HJB_Exact']
        plt.figure(figsize=(10, 6))
        plt.plot(df['time'], df['inventory'], color='purple')
        plt.title('HJB Exact: Inventory Management')
        plt.xlabel('Time')
        plt.ylabel('Inventory')
        plt.grid(True)
        plt.savefig('experiments/results/hjb_inventory.png')
        plt.close()

if __name__ == "__main__":
    run_backtest()
