import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def get_output_dir():
    """Fonction utilitaire pour localiser le dossier output de manière robuste."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Si le script est dans src/analysis, on remonte de deux crans
    if 'src' in base_dir and 'analysis' in base_dir:
        # Remonte de src/analysis vers la racine du projet
        project_root = os.path.dirname(os.path.dirname(base_dir))
        return os.path.join(project_root, 'output')
    
    # Cas par défaut (si exécuté depuis la racine)
    return os.path.join(base_dir, 'output')


def load_data():
    """Charge les données des fichiers CSV avec gestion cohérente des unités."""
    output_dir = get_output_dir()
    print(f"Dossier cible détecté : {output_dir}")
    
    normal_path = os.path.join(output_dir, 'summary.csv')
    stress_path = os.path.join(output_dir, 'stress_summary.csv')
    
    if not os.path.exists(normal_path):
        # Fallback simple pour débogage local
        normal_path = 'output/summary.csv'
        stress_path = 'output/stress_summary.csv'
    
    # Chargement
    try:
        normal_df = pd.read_csv(normal_path)
        stress_df = pd.read_csv(stress_path)
    except FileNotFoundError:
        print(f"ERREUR CRITIQUE : Impossible de trouver les fichiers dans {output_dir}")
        print("Vérifiez que vous avez bien lancé les simulations (main.py et stress_test) avant.")
        return pd.DataFrame(), pd.DataFrame()
    
    # Harmonisation des noms de colonnes
    for df in [normal_df, stress_df]:
        cols = [c for c in df.columns if 'Mean Wealth' in c]
        if cols:
            df.rename(columns={cols[0]: 'Mean Wealth'}, inplace=True)
    
    return normal_df, stress_df


def create_robustness_comparison(normal_df, stress_df):
    if normal_df.empty or stress_df.empty:
        return
    
    solvers = ['DP', 'OR-Tools', 'RL']
    
    # Extraction sécurisée
    def get_val(df, solver):
        if 'Solver' not in df.columns:
            return 0.0
        val = df.loc[df['Solver'] == solver, 'Mean Wealth']
        return val.values[0] if not val.empty else 0.0
    
    normal_vals = [get_val(normal_df, s) for s in solvers]
    stress_vals = [get_val(stress_df, s) for s in solvers]
    
    # Création du DataFrame pour Seaborn
    df_plot = pd.DataFrame({
        'Solver': solvers * 2,
        'Condition': ['Marché Normal'] * 3 + ['Stress Test (Crise)'] * 3,
        'Richesse': np.concatenate([normal_vals, stress_vals])
    })
    
    # Graphique
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    
    ax = sns.barplot(
        data=df_plot,
        x='Solver',
        y='Richesse',
        hue='Condition',
        palette={'Marché Normal': '#2ecc71', 'Stress Test (Crise)': '#e74c3c'},
        edgecolor='black',
        alpha=0.9
    )
    
    # Annotations
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(p.get_x() + p.get_width()/2., height + 1,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontweight='bold')
    
    plt.title("Robustesse des Stratégies : Impact d'un Krach Boursier",
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel("Richesse Finale Moyenne")
    plt.tight_layout()
    
    # --- CORRECTION ICI : Récupération propre du chemin ---
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'robustness_comparison.png')
    
    plt.savefig(save_path, dpi=300)
    print(f"Graphique sauvegardé avec succès : {save_path}")
    
    # Tableau de vérification
    print("\n--- VÉRIFICATION DES VALEURS ---")
    print(f"{'Solveur':<10} | {'Normal':<10} | {'Stress':<10}")
    for i, s in enumerate(solvers):
        print(f"{s:<10} | {normal_vals[i]:<10.1f} | {stress_vals[i]:<10.1f}")


if __name__ == '__main__':
    normal, stress = load_data()
    create_robustness_comparison(normal, stress)
