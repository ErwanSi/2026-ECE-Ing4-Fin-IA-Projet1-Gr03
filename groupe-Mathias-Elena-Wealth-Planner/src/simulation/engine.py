import numpy as np
import pandas as pd
from typing import List, Dict, Any
from src.core.model import InvestmentMDP
from src.solvers.base import BaseSolver

class SimulationEngine:
    """
    Moteur de simulation Monte Carlo pour évaluer les politiques d'investissement.
    """
    
    def __init__(self, mdp: InvestmentMDP):
        self.mdp = mdp

    def run_simulation(self, solver: BaseSolver, n_trajectories: int = 100) -> pd.DataFrame:
        """
        Exécute n simulations et retourne les résultats sous forme de DataFrame.
        """
        results = []
        T = self.mdp.i_cfg.horizon
        
        for i in range(n_trajectories):
            wealth = self.mdp.i_cfg.initial_wealth
            wealth_history = [wealth]
            allocation_history = []
            previous_weights = None
            
            for t in range(T):
                # Obtenir l'allocation du solveur
                weights = solver.get_policy(wealth, t)
                allocation_history.append(weights)
                
                # Générer un rendement aléatoire pour cette étape
                returns_sample = self.mdp.generate_returns_sample(1)[0]
                
                # Transition avec frais de transaction et pénalités
                wealth = self.mdp.transition(wealth, t, weights, returns_sample, previous_weights)
                wealth_history.append(wealth)
                previous_weights = weights
                
                if wealth <= 0:
                    for _ in range(t + 1, T):
                        wealth_history.append(0.0)
                        allocation_history.append(np.zeros_like(weights))
                    break
            
            # Stockage des résultats de la trajectoire
            for t, (w, a) in enumerate(zip(wealth_history, allocation_history + [np.nan])):
                inflation_factor = (1 + self.mdp.i_cfg.inflation_rate) ** t
                wealth_real = w / inflation_factor
                
                res_dict = {
                    "trajectory": i,
                    "time": t,
                    "wealth": w,
                    "wealth_real": wealth_real
                }
                
                if isinstance(a, np.ndarray):
                    # Calcul de la liquidité (Cash + Obligations)
                    cash_idx = self.mdp.m_cfg.assets.index("Cash")
                    bond_idx = self.mdp.m_cfg.assets.index("Obligations")
                    liquidity = w * (a[cash_idx] + a[bond_idx])
                    res_dict["liquidity"] = liquidity
                    
                    for j, asset in enumerate(self.mdp.m_cfg.assets):
                        res_dict[f"alloc_{asset.lower()}"] = a[j]
                else:
                    res_dict["liquidity"] = 0.0
                    for asset in self.mdp.m_cfg.assets:
                        res_dict[f"alloc_{asset.lower()}"] = np.nan
                results.append(res_dict)
                
        return pd.DataFrame(results)

    def calculate_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        stats = df.groupby("time")["wealth"].agg([
            "mean", "std", "min", "max",
            lambda x: np.percentile(x, 5),
            lambda x: np.percentile(x, 50),
            lambda x: np.percentile(x, 95)
        ]).rename(columns={
            "<lambda_0>": "p5",
            "<lambda_1>": "median",
            "<lambda_2>": "p95"
        })
        return stats

    def run_stress_test(self, solver: BaseSolver, n_trajectories: int = 100) -> pd.DataFrame:
        """
        Exécute n simulations en mode stress test (scénario de crise).
        
        Scénario de crise :
        - Rendement moyen des actions : -5% (au lieu de 8%)
        - Volatilité doublée pour tous les actifs
        """
        # Sauvegarde des paramètres originaux
        original_means = self.mdp.means.copy()
        original_vols = self.mdp.vols.copy()
        original_cov = self.mdp.cov.copy()
        
        # Modification des paramètres pour le scénario de crise
        # Actions (index 0) : rendement moyen à -5%
        self.mdp.means[0] = -0.05
        
        # Volatilité doublée pour tous les actifs
        self.mdp.vols = self.mdp.vols * 2.0
        
        # Recalcul de la matrice de covariance
        self.mdp.cov = np.diag(self.mdp.vols) @ self.mdp.corr @ np.diag(self.mdp.vols)
        
        try:
            # Exécution de la simulation avec les paramètres modifiés
            results = []
            T = self.mdp.i_cfg.horizon
            
            for i in range(n_trajectories):
                wealth = self.mdp.i_cfg.initial_wealth
                wealth_history = [wealth]
                allocation_history = []
                previous_weights = None
                
                for t in range(T):
                    # Obtenir l'allocation du solveur
                    weights = solver.get_policy(wealth, t)
                    allocation_history.append(weights)
                    
                    # Générer un rendement aléatoire pour cette étape (avec paramètres de crise)
                    returns_sample = self.mdp.generate_returns_sample(1)[0]
                    
                    # Transition avec frais de transaction et pénalités
                    wealth = self.mdp.transition(wealth, t, weights, returns_sample, previous_weights)
                    wealth_history.append(wealth)
                    previous_weights = weights
                    
                    if wealth <= 0:
                        for _ in range(t + 1, T):
                            wealth_history.append(0.0)
                            allocation_history.append(np.zeros_like(weights))
                        break
                
                # Stockage des résultats de la trajectoire
                for t, (w, a) in enumerate(zip(wealth_history, allocation_history + [np.nan])):
                    inflation_factor = (1 + self.mdp.i_cfg.inflation_rate) ** t
                    wealth_real = w / inflation_factor
                    
                    res_dict = {
                        "trajectory": i,
                        "time": t,
                        "wealth": w,
                        "wealth_real": wealth_real
                    }
                    
                    if isinstance(a, np.ndarray):
                        # Calcul de la liquidité (Cash + Obligations)
                        cash_idx = self.mdp.m_cfg.assets.index("Cash")
                        bond_idx = self.mdp.m_cfg.assets.index("Obligations")
                        liquidity = w * (a[cash_idx] + a[bond_idx])
                        res_dict["liquidity"] = liquidity
                        
                        for j, asset in enumerate(self.mdp.m_cfg.assets):
                            res_dict[f"alloc_{asset.lower()}"] = a[j]
                    else:
                        res_dict["liquidity"] = 0.0
                        for asset in self.mdp.m_cfg.assets:
                            res_dict[f"alloc_{asset.lower()}"] = np.nan
                    results.append(res_dict)
                    
            return pd.DataFrame(results)
            
        finally:
            # Restauration des paramètres originaux
            self.mdp.means = original_means
            self.mdp.vols = original_vols
            self.mdp.cov = original_cov
