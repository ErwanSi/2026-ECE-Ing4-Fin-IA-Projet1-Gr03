import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ..data.simulator import OrderBookSimulator, SimulationConfig

class MarketMakingEnv(gym.Env):
    """
    Gymnasium Environment for Optimal Market Making.
    
    Observation:
        - Inventory (normalized)
        - Time elapsed (normalized)
        
    Action:
        - Spread Bid (continuous)
        - Spread Ask (continuous)
        
    Reward:
        - Change in PnL
        - Inventory Penalty (optional, explicit or implicit in utility)
    """
    
    def __init__(self, config: SimulationConfig = None):
        super(MarketMakingEnv, self).__init__()
        
        if config is None:
            config = SimulationConfig()
        self.config = config
        self.simulator = OrderBookSimulator(config)
        
        # Action space: [delta_b, delta_a]
        # We assume spreads are within [0, 5] (reasonable range for simulation)
        self.action_space = spaces.Box(low=0.0, high=5.0, shape=(2,), dtype=np.float32)
        
        # Observation space: [Inventory, Normalized Time]
        # Inventory range approx -20 to 20
        self.observation_space = spaces.Box(
            low=np.array([-50, 0.0]), 
            high=np.array([50, 1.0]), 
            dtype=np.float32
        )
        
        self.max_steps = int(config.T / config.dt)
        self.current_step = 0
        self.prev_pnl = 0.0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.simulator.config.seed = seed if seed else np.random.randint(0, 10000)
        self.simulator.rng = np.random.default_rng(self.simulator.config.seed)
        self.simulator.reset()
        
        self.current_step = 0
        self.prev_pnl = 0.0
        
        return self._get_obs(), {}
        
    def step(self, action):
        delta_b, delta_a = action
        
        # Run simulator step
        # Ideally, RL actions are usually "fixed" for certain dt
        # Here we map 1 Env step = 1 Sim step
        filled_buy, filled_sell, new_price = self.simulator.step(delta_b, delta_a)
        
        self.current_step += 1
        
        # Calculate Reward
        # Reward = New PnL - Old PnL
        # Note: Using pure PnL reward leads to risk-neutrality. 
        # To induce risk aversion, we must penalize inventory or use Utility reward.
        
        current_pnl = self.simulator.cash + self.simulator.inventory * new_price
        step_pnl = current_pnl - self.prev_pnl
        self.prev_pnl = current_pnl
        
        # Penalized Reward: PnL - gamma * Inventory^2 * factor
        # Using running inventory penalty helps convergence
        inventory_penalty = 0.01 * (self.simulator.inventory ** 2)
        reward = step_pnl - inventory_penalty
        
        terminated = self.current_step >= self.max_steps
        truncated = False # Can implement inventory stop-out
        
        if abs(self.simulator.inventory) > 50:
            truncated = True
            reward -= 100 # Large penalty for breaking limits
            
        return self._get_obs(), reward, terminated, truncated, {}
    
    def _get_obs(self):
        return np.array([
            float(self.simulator.inventory),
            self.current_step / self.max_steps
        ], dtype=np.float32)
