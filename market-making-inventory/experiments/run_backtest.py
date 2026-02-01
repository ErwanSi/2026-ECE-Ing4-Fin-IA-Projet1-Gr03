"""
Market Making Backtest Runner

This script runs backtests for different market making strategies
and generates performance plots and statistics.

Strategies supported:
- HJB optimal strategy (Guéant-Lehalle-Fernandez-Tapia)
- Constant spread baseline
- RL-based strategy (if trained)
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models import InventoryHJB, create_default_model
from data import MarketMakingSimulator, SimulatorParameters, create_default_simulator
from solvers import create_default_solver, create_default_csp_solver


class BacktestResults:
    """Container for backtest results."""
    
    def __init__(self, name: str):
        self.name = name
        self.pnl_history: List[float] = []
        self.inventory_history: List[int] = []
        self.price_history: List[float] = []
        self.time_history: List[float] = []
        self.trades: List = []
        self.final_pnl: float = 0.0
        self.final_inventory: int = 0
        self.n_trades: int = 0
    
    def add_results(self, results: Dict) -> None:
        """Add simulation results."""
        self.pnl_history = results['pnl_history'].tolist()
        self.inventory_history = results['inventory_history'].tolist()
        self.price_history = results['price_history'].tolist()
        self.time_history = results['time_history'].tolist()
        self.trades = results['trades']
        self.final_pnl = results['final_pnl']
        self.final_inventory = results['final_inventory']
        self.n_trades = results['n_trades']
    
    def get_statistics(self) -> Dict:
        """Compute performance statistics."""
        pnl_array = np.array(self.pnl_history)
        returns = np.diff(pnl_array)
        
        # Sharpe ratio
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns)
        else:
            sharpe = 0.0
        
        # Maximum drawdown
        running_max = np.maximum.accumulate(pnl_array)
        drawdown = (pnl_array - running_max) / (running_max + 1e-10)
        max_drawdown = np.min(drawdown)
        
        # Average inventory
        avg_inventory = np.mean(np.abs(self.inventory_history))
        
        return {
            'final_pnl': self.final_pnl,
            'final_inventory': self.final_inventory,
            'n_trades': self.n_trades,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'avg_inventory': avg_inventory,
            'max_inventory': np.max(np.abs(self.inventory_history))
        }


def hjb_policy(model: InventoryHJB) -> callable:
    """
    Create a policy function from HJB model.
    
    Args:
        model: HJB model
        
    Returns:
        Policy function
    """
    def policy(t: float, q: int, S: float) -> Tuple[float, float]:
        delta_b, delta_a = model.get_optimal_spreads(t, q)
        return S - delta_b, S + delta_a
    
    return policy


def constant_spread_policy(spread: float = 0.02) -> callable:
    """
    Create a constant spread policy.
    
    Args:
        spread: Constant bid-ask spread
        
    Returns:
        Policy function
    """
    def policy(t: float, q: int, S: float) -> Tuple[float, float]:
        return S - spread/2, S + spread/2
    
    return policy


def inventory_aware_policy(
    base_spread: float = 0.02,
    inventory_adjustment: float = 0.001
) -> callable:
    """
    Create an inventory-aware policy.
    
    Args:
        base_spread: Base spread
        inventory_adjustment: Adjustment per unit of inventory
        
    Returns:
        Policy function
    """
    def policy(t: float, q: int, S: float) -> Tuple[float, float]:
        # Widen spread on the side we want to reduce inventory
        delta_b = base_spread/2 + inventory_adjustment * q
        delta_a = base_spread/2 - inventory_adjustment * q
        
        # Ensure positive spreads
        delta_b = max(delta_b, 0.001)
        delta_a = max(delta_a, 0.001)
        
        return S - delta_b, S + delta_a
    
    return policy


def run_backtest(
    simulator: MarketMakingSimulator,
    policy: callable,
    name: str
) -> BacktestResults:
    """
    Run a single backtest.
    
    Args:
        simulator: Market making simulator
        policy: Policy function
        name: Strategy name
        
    Returns:
        BacktestResults object
    """
    print(f"\nRunning backtest for {name}...")
    results = simulator.run_simulation(policy)
    
    backtest = BacktestResults(name)
    backtest.add_results(results)
    
    stats = backtest.get_statistics()
    print(f"  Final PnL: {stats['final_pnl']:.4f}")
    print(f"  Final Inventory: {stats['final_inventory']}")
    print(f"  Number of Trades: {stats['n_trades']}")
    print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown: {stats['max_drawdown']:.4f}")
    
    return backtest


def plot_results(
    backtests: List[BacktestResults],
    output_dir: str = "results"
) -> None:
    """
    Plot backtest results.
    
    Args:
        backtests: List of backtest results
        output_dir: Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Market Making Strategy Comparison', fontsize=16)
    
    # Plot 1: PnL over time
    ax = axes[0, 0]
    for bt in backtests:
        ax.plot(bt.time_history, bt.pnl_history, label=bt.name, linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('PnL')
    ax.set_title('PnL Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Inventory over time
    ax = axes[0, 1]
    for bt in backtests:
        ax.plot(bt.time_history, bt.inventory_history, label=bt.name, linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Inventory')
    ax.set_title('Inventory Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Plot 3: Price over time
    ax = axes[1, 0]
    for bt in backtests:
        ax.plot(bt.time_history, bt.price_history, label=bt.name, linewidth=2, alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Midprice')
    ax.set_title('Price Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: PnL distribution
    ax = axes[1, 1]
    for bt in backtests:
        returns = np.diff(bt.pnl_history)
        ax.hist(returns, bins=50, alpha=0.5, label=bt.name, density=True)
    ax.set_xlabel('Return')
    ax.set_ylabel('Density')
    ax.set_title('Return Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'backtest_comparison.png'), dpi=150)
    print(f"\nPlot saved to {output_dir}/backtest_comparison.png")
    
    # Create statistics table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    data = []
    columns = ['Strategy', 'Final PnL', 'Final Inv', 'Trades', 'Sharpe', 'Max DD', 'Avg |Inv|']
    
    for bt in backtests:
        stats = bt.get_statistics()
        data.append([
            bt.name,
            f"{stats['final_pnl']:.4f}",
            f"{stats['final_inventory']}",
            f"{stats['n_trades']}",
            f"{stats['sharpe_ratio']:.4f}",
            f"{stats['max_drawdown']:.4f}",
            f"{stats['avg_inventory']:.2f}"
        ])
    
    table = ax.table(cellText=data, colLabels=columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    plt.title('Performance Statistics Comparison', fontsize=14, pad=20)
    plt.savefig(os.path.join(output_dir, 'statistics_table.png'), dpi=150, bbox_inches='tight')
    print(f"Statistics table saved to {output_dir}/statistics_table.png")
    
    plt.close('all')


def main():
    """Main function to run backtests."""
    parser = argparse.ArgumentParser(description='Run market making backtests')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n-runs', type=int, default=1, help='Number of simulation runs')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--strategies', type=str, nargs='+', 
                       default=['hjb', 'constant', 'inventory_aware'],
                       help='Strategies to test')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Market Making Backtest Runner")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize simulator
    print(f"\nInitializing simulator with seed={args.seed}...")
    simulator = create_default_simulator(seed=args.seed)
    
    # Initialize strategies
    backtests = []
    
    # HJB strategy
    if 'hjb' in args.strategies:
        print("\nInitializing HJB model...")
        hjb_model = create_default_model()
        policy = hjb_policy(hjb_model)
        bt = run_backtest(simulator, policy, "HJB Optimal")
        backtests.append(bt)
    
    # Constant spread strategy
    if 'constant' in args.strategies:
        policy = constant_spread_policy(spread=0.02)
        bt = run_backtest(simulator, policy, "Constant Spread")
        backtests.append(bt)
    
    # Inventory-aware strategy
    if 'inventory_aware' in args.strategies:
        policy = inventory_aware_policy(base_spread=0.02, inventory_adjustment=0.001)
        bt = run_backtest(simulator, policy, "Inventory Aware")
        backtests.append(bt)
    
    # Plot results
    if backtests:
        plot_results(backtests, args.output_dir)
        
        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"{'Strategy':<20} {'PnL':>10} {'Sharpe':>10} {'Trades':>10}")
        print("-" * 60)
        for bt in backtests:
            stats = bt.get_statistics()
            print(f"{bt.name:<20} {stats['final_pnl']:>10.4f} "
                  f"{stats['sharpe_ratio']:>10.4f} {stats['n_trades']:>10}")
        print("=" * 60)
    else:
        print("\nNo strategies selected for backtesting!")


if __name__ == "__main__":
    main() 
