import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class SimulationConfig:
    S0: float = 100.0       # Initial price
    T: float = 1.0          # Total time (years)
    dt: float = 1/2520      # Time step (e.g. 10 steps per day if 1/252 is daily)
    sigma: float = 0.2      # Volatility
    seed: int = 42
    k: float = 1.5          # Liquidity parameter (decay)
    A: float = 140.0        # Order arrival intensity
    maker_fee: float = 0.0  # Transaction fee ratio (e.g. 0.0001 for 1bp)
    allow_sniping: bool = False # If True, allows "toxic" fills when price jumps

class OrderBookSimulator:
    """
    Simulates a simplified Limit Order Book with:
    - Mid-price following Arithmetic Brownian Motion
    - Probability of Fill based on quote distance (Poisson intensity)
    - Optional: Transaction Fees and Toxic Flow (Sniping)
    """
    
    
    def __init__(self, config: SimulationConfig, price_path: Optional[np.ndarray] = None):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.external_price_path = price_path
        self.reset()
        
    def reset(self):
        if self.external_price_path is not None:
             self.current_price = self.external_price_path[0]
        else:
             self.current_price = self.config.S0
        
        self.current_time = 0.0
        self.current_idx = 0 # For replaying path
        self.cash = 0.0
        self.inventory = 0
        self.history = {
            'time': [],
            'price': [],
            'inventory': [],
            'cash': [],
            'pnl': []
        }
    
    def step(self, delta_b: float, delta_a: float) -> Tuple[bool, bool, float]:
        """
        Advance one time step and determine if orders are filled.
        """
        dt = self.config.dt
        sigma = self.config.sigma
        old_price = self.current_price
        
        # 1. Update Price
        if self.external_price_path is not None:
            # Replay Mode
            self.current_idx += 1
            if self.current_idx < len(self.external_price_path):
                self.current_price = self.external_price_path[self.current_idx]
            else:
                pass
        else:
            # Simulation Mode (Brownian)
            dW = self.rng.standard_normal() * np.sqrt(dt)
            self.current_price += sigma * dW 
        
        self.current_time += dt
        
        # 2. Determine Fills
        # Use config parameters
        A = self.config.A
        k = self.config.k
        
        # Standard Poisson Fills
        lambda_b = A * np.exp(-k * delta_b)
        lambda_a = A * np.exp(-k * delta_a)
        
        prob_b = 1 - np.exp(-lambda_b * dt)
        prob_a = 1 - np.exp(-lambda_a * dt)
        
        filled_buy = self.rng.random() < prob_b
        filled_sell = self.rng.random() < prob_a
        
        # --- 3. TOXIC FLOW / SNIPING Logic ---
        # If price moved MORE than the spread, you get picked off by latency arbitrageurs.
        # This overrides the probabilistic fill (you WILL be filled).
        if self.config.allow_sniping:
            price_move = self.current_price - old_price
            
            # If Price jumped UP > Ask spread -> We sold too cheap (Sniper bought from us)
            if price_move > delta_a:
                filled_sell = True
                
            # If Price crashed DOWN > Bid spread -> We bought too expensive (Sniper sold to us)
            if price_move < -delta_b:
                filled_buy = True

        # Update Inventory & Cash
        # Apply Fees if any
        fee_rate = self.config.maker_fee
        
        if filled_buy:
            exec_price = old_price - delta_b # We buy at our bid (relative to OLD price effectively in sim)
            # Actually, in this discrete step, if we were sniped, we traded at old_price - delta_b
            # while market is at current_price.
            
            # Correction: In standard step, we trade against flow.
            # If using replay, current_price is the CLOSE of the candle/step?
            # Let's assume trade happens at order price.
            
            transaction_cost = exec_price * fee_rate
            self.inventory += 1
            self.cash -= (exec_price + transaction_cost)
            
        if filled_sell:
            exec_price = old_price + delta_a
            transaction_cost = exec_price * fee_rate
            self.inventory -= 1
            self.cash += (exec_price - transaction_cost)
            
        # Log state
        self._log_state()
        
        return filled_buy, filled_sell, self.current_price

    def _log_state(self):
        # Mark-to-market PnL
        pnl = self.cash + self.inventory * self.current_price
        
        self.history['time'].append(self.current_time)
        self.history['price'].append(self.current_price)
        self.history['inventory'].append(self.inventory)
        self.history['cash'].append(self.cash)
        self.history['pnl'].append(pnl)

    def run_simulation(self, strategy_fn):
        """
        Run full simulation using a strategy function.
        strategy_fn: (inventory, time) -> (delta_b, delta_a)
        """
        steps = int(self.config.T / self.config.dt)
        for _ in range(steps):
            delta_b, delta_a = strategy_fn(self.inventory, self.config.T - self.current_time)
            self.step(delta_b, delta_a)
        return pd.DataFrame(self.history)

def generate_price_path(config: SimulationConfig, steps: int) -> np.ndarray:
    """
    Generate pre-computed price path for reproducible testing.
    """
    rng = np.random.default_rng(config.seed + 1)
    dW = rng.standard_normal(steps) * np.sqrt(config.dt)
    path = np.zeros(steps + 1)
    path[0] = config.S0
    # Cumulative sum for Arithmetic Brownian Motion
    path[1:] = config.S0 + np.cumsum(config.sigma * dW)
    return path
