import os
import sys
import time
import pandas as pd

# Ajouter le chemin du projet au sys.path pour les imports
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from src.core.config import MarketConfig, InvestmentConfig, SolverConfig
from src.core.model import InvestmentMDP
from src.solvers.dp_solver import DPSolver
try:
    from src.solvers.rl_solver import RLSolver, TORCH_AVAILABLE
except (ImportError, OSError):
    TORCH_AVAILABLE = False
from src.solvers.ortools_solver import ORToolsSolver
from src.simulation.engine import SimulationEngine


def benchmark_stress_solver(name, solver, mdp, sim_engine, n_trajs=200):
    """
    Évalue un solveur en mode stress test (scénario de crise).
    
    Scénario de crise :
    - Rendement moyen des actions : -5%
    - Volatilité doublée pour tous les actifs
    """
    print(f"\n--- Évaluation du solveur en mode STRESS : {name} ---")
    start_time = time.perf_counter()
    solver.solve()
    solve_time = time.perf_counter() - start_time
    print(f"Temps de résolution : {solve_time:.4f}s")
    
    # Exécution du stress test
    results_df = sim_engine.run_stress_test(solver, n_trajectories=n_trajs)
    results_df['solver'] = name
    
    # Calcul des statistiques sur la richesse finale
    final_wealths = results_df[results_df['time'] == mdp.i_cfg.horizon]['wealth']
    mean_wealth = final_wealths.mean()
    std_wealth = final_wealths.std()
    
    # Calcul du Sharpe Ratio (simplifié)
    returns = final_wealths / mdp.i_cfg.initial_wealth - 1
    sharpe = returns.mean() / (returns.std() + 1e-8)
    
    # Calcul du taux de ruine (richesse finale <= 0)
    ruin_rate = (final_wealths <= 0).sum() / len(final_wealths)
    
    print(f"Richesse finale moyenne : {mean_wealth:.2f} k€")
    print(f"Écart-type de la richesse finale : {std_wealth:.2f} k€")
    print(f"Sharpe Ratio : {sharpe:.2f}")
    print(f"Taux de ruine : {ruin_rate:.2%}")
    
    return results_df, solve_time, mean_wealth, std_wealth, sharpe, ruin_rate


def run_stress_analysis(output_dir: str = "output", n_trajectories: int = 200):
    """
    Lance les 3 solveurs (DPSolver, ORToolsSolver, RLSolver) sur le scénario de crise
    et enregistre les résultats dans output/stress_results.csv.
    """
    # 1. Configuration
    market_cfg = MarketConfig()
    invest_cfg = InvestmentConfig()
    solver_cfg = SolverConfig()
    
    # 2. Initialisation du MDP et du moteur de simulation
    mdp = InvestmentMDP(market_cfg, invest_cfg)
    sim_engine = SimulationEngine(mdp)
    
    # 3. Définition des solveurs à tester
    solvers = [
        ("DP", DPSolver(mdp, solver_cfg)),
        ("OR-Tools", ORToolsSolver(mdp, solver_cfg))
    ]
    
    if TORCH_AVAILABLE:
        solvers.append(("RL", RLSolver(mdp, solver_cfg)))
    else:
        print("RL désactivé : Stable-Baselines3 ou PyTorch non disponible.")
    
    # 4. Exécution des stress tests
    all_results = []
    summary_data = []
    
    for name, solver in solvers:
        res_df, s_time, m_wealth, s_wealth, sharpe, ruin = benchmark_stress_solver(
            name, solver, mdp, sim_engine, n_trajs=n_trajectories
        )
        all_results.append(res_df)
        summary_data.append({
            "Solver": name,
            "Time (s)": s_time,
            "Mean Wealth (k€)": m_wealth,
            "Std Wealth (k€)": s_wealth,
            "Sharpe Ratio": sharpe,
            "Ruin Rate": ruin
        })
    
    # 5. Sauvegarde des résultats
    # Création sécurisée du dossier output
    os.makedirs(output_dir, exist_ok=True)
    
    # Fusion de tous les résultats
    comparison_df = pd.concat(all_results)
    
    # Sauvegarde du fichier CSV détaillé
    stress_results_path = os.path.join(output_dir, "stress_results.csv")
    comparison_df.to_csv(stress_results_path, index=False)
    print(f"\nRésultats détaillés sauvegardés dans : {stress_results_path}")
    
    # Sauvegarde du résumé
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, "stress_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Résumé sauvegardé dans : {summary_path}")
    
    # Affichage du résumé
    print("\n" + "="*60)
    print("RÉSUMÉ DES RÉSULTATS STRESS TEST")
    print("="*60)
    print(summary_df.to_string(index=False))
    print("="*60)
    
    return comparison_df, summary_df


if __name__ == "__main__":
    # Définition du dossier de base (basé sur le dossier de main.py)
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(project_dir, "output")
    
    print(f"Dossier de sortie : {output_dir}")
    print("Lancement de l'analyse de robustesse (Stress Test)...")
    print("Scénario de crise : Rendement actions = -5%, Volatilité doublée")
    
    run_stress_analysis(output_dir=output_dir, n_trajectories=200)
    
    print("\nAnalyse de robustesse terminée.")
