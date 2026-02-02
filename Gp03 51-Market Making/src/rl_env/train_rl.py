import sys
import os

# Add project root to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from src.rl_env.market_making_env import MarketMakingEnv

def train_agent(total_timesteps=100000):
    """
    Train a PPO agent on the MarketMakingEnv.
    """
    # Create env
    env = make_vec_env(lambda: MarketMakingEnv(), n_envs=4)
    
    # Instantiate the agent
    model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.01)
    
    print("Starting training...")
    model.learn(total_timesteps=total_timesteps)
    print("Training finished.")
    
    # Save
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_market_maker")
    return model

if __name__ == "__main__":
    train_agent()
