import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import time

from src.core.config import MarketConfig, InvestmentConfig, SolverConfig
from src.core.model import InvestmentMDP
from src.solvers.dp_solver import DPSolver
from src.solvers.ortools_solver import ORToolsSolver
try:
    from src.solvers.rl_solver import RLSolver, TORCH_AVAILABLE
except (ImportError, OSError):
    TORCH_AVAILABLE = False
from src.simulation.engine import SimulationEngine
from src.utils.plotting import ASSET_COLORS

# Configuration de la page
st.set_page_config(page_title="Wealth Planner AI", page_icon="💰", layout="wide")

st.title("💰 Wealth Planner AI : Optimisation d'Investissement")
st.markdown("""
Cette application utilise la **Programmation Dynamique**, l'**Optimisation Linéaire** et le **Reinforcement Learning** pour concevoir votre stratégie d'investissement optimale sur mesure.
""")

# --- SIDEBAR : PARAMÈTRES GÉNÉRAUX ---
st.sidebar.header("👤 Profil & Paramètres")

initial_wealth = st.sidebar.number_input("Capital Initial (k€)", min_value=0.0, value=200.0, step=10.0)
monthly_savings = st.sidebar.number_input("Épargne Mensuelle (k€)", min_value=0.0, value=1.0, step=0.1)
current_age = st.sidebar.slider("Âge Actuel", 18, 80, 35)
retirement_age = st.sidebar.slider("Âge de Retraite (Horizon)", current_age + 5, 100, 65)
horizon = retirement_age - current_age

risk_profile = st.sidebar.selectbox(
    "Profil de Risque",
    ["Prudent", "Équilibré", "Dynamique"],
    index=1
)

# Ajustement de l'aversion au risque selon le profil
risk_aversion_map = {
    "Prudent": 4.0,
    "Équilibré": 2.0,
    "Dynamique": 1.0
}
risk_aversion = risk_aversion_map[risk_profile]

# --- ZONE PRINCIPALE : PLAN DE VIE ---
st.header("📅 Votre Plan de Vie")
st.subheader("Événements de cash-flow (Sorties de capital)")

# Initialisation des événements par défaut
if 'events_df' not in st.session_state:
    st.session_state.events_df = pd.DataFrame([
        {"Nom": "Achat Voiture", "Année": 5, "Montant (k€)": 20.0},
        {"Nom": "Apport Immobilier", "Année": 12, "Montant (k€)": 80.0},
        {"Nom": "Études Enfants", "Année": 20, "Montant (k€)": 30.0}
    ])

edited_events = st.data_editor(
    st.session_state.events_df,
    num_rows="dynamic",
    column_config={
        "Année": st.column_config.NumberColumn(min_value=1, max_value=horizon),
        "Montant (k€)": st.column_config.NumberColumn(min_value=0.0)
    },
    key="events_editor"
)

# --- CALCULS ---
if st.button("🚀 Calculer la Stratégie Optimale", type="primary"):
    with st.spinner("Calcul des stratégies en cours (DP, OR-Tools, RL)..."):
        # 1. Préparation de la configuration
        market_cfg = MarketConfig()
        
        # Transformation des événements
        life_events = {}
        event_names = {}
        for _, row in edited_events.iterrows():
            year = int(row["Année"])
            amount = float(row["Montant (k€)"])
            life_events[year] = life_events.get(year, 0) + amount
            event_names[year] = row["Nom"]

        invest_cfg = InvestmentConfig(
            initial_wealth=initial_wealth,
            horizon=horizon,
            monthly_savings=monthly_savings,
            life_events=life_events,
            event_names=event_names,
            risk_aversion=risk_aversion
        )
        
        solver_cfg = SolverConfig(
            wealth_grid_size=40,
            max_wealth=initial_wealth * 5 + monthly_savings * 12 * horizon,
            total_timesteps=20000 # Réduit pour le dashboard
        )
        
        # 2. Initialisation du MDP et Moteur
        mdp = InvestmentMDP(market_cfg, invest_cfg)
        sim_engine = SimulationEngine(mdp)
        
        all_results_list = []
        
        # 3. Résolution DP
        dp_solver = DPSolver(mdp, solver_cfg)
        dp_solver.solve()
        results_dp = sim_engine.run_simulation(dp_solver, n_trajectories=100)
        results_dp['solver'] = 'DP'
        all_results_list.append(results_dp)
        
        # 4. Résolution OR-Tools
        ort_solver = ORToolsSolver(mdp, solver_cfg)
        results_ort = sim_engine.run_simulation(ort_solver, n_trajectories=100)
        results_ort['solver'] = 'OR-Tools'
        all_results_list.append(results_ort)
        
        # 5. Résolution RL (si disponible)
        if TORCH_AVAILABLE:
            rl_solver = RLSolver(mdp, solver_cfg)
            rl_solver.solve()
            results_rl = sim_engine.run_simulation(rl_solver, n_trajectories=100)
            results_rl['solver'] = 'RL'
            all_results_list.append(results_rl)
        else:
            st.warning("Le solveur RL est désactivé (PyTorch non disponible).")
        
        st.session_state.all_results = pd.concat(all_results_list)
        st.session_state.life_events = life_events
        st.session_state.horizon = horizon
        st.session_state.retirement_age = retirement_age
        
        st.success("Calcul terminé !")

# --- AFFICHAGE DES RÉSULTATS ---
if 'all_results' in st.session_state:
    all_results = st.session_state.all_results
    life_events = st.session_state.life_events
    horizon = st.session_state.horizon
    retirement_age = st.session_state.retirement_age
    
    # KPIs pour le solveur sélectionné par défaut (DP)
    available_solvers = all_results['solver'].unique()
    
    st.divider()
    
    # --- GRAPHIQUES INTERACTIFS (PLOTLY) ---
    tab1, tab2, tab3 = st.tabs(["📈 Richesse", "📊 Allocation", "⚖️ Comparaison"])
    
    with tab1:
        selected_solver_w = st.selectbox("Choisir le solveur pour la richesse", available_solvers, key="w_solver")
        df_w = all_results[all_results['solver'] == selected_solver_w]
        
        st.subheader(f"Convergence de la Richesse ({selected_solver_w})")
        stats = df_w.groupby('time')['wealth'].agg(['mean', 'std', lambda x: np.percentile(x, 5), lambda x: np.percentile(x, 95)])
        stats.columns = ['mean', 'std', 'p5', 'p95']
        
        fig_wealth = go.Figure()
        
        # Zone d'ombre p5-p95
        fig_wealth.add_trace(go.Scatter(
            x=stats.index, y=stats['p95'],
            mode='lines', line=dict(width=0),
            showlegend=False, name='p95'
        ))
        fig_wealth.add_trace(go.Scatter(
            x=stats.index, y=stats['p5'],
            mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor='rgba(31, 119, 180, 0.1)',
            showlegend=False, name='p5'
        ))
        
        # Ligne moyenne
        fig_wealth.add_trace(go.Scatter(
            x=stats.index, y=stats['mean'],
            mode='lines', line=dict(color='#1f77b4', width=4),
            name='Richesse Moyenne'
        ))
        
        # Événements
        for year, amount in life_events.items():
            fig_wealth.add_vline(x=year, line_dash="dash", line_color="red", opacity=0.5)
            fig_wealth.add_annotation(x=year, y=stats['mean'].max(), text=f"-{amount}k€", showarrow=True, arrowhead=1)

        fig_wealth.update_layout(
            title=f"Évolution de la Richesse au cours du temps ({selected_solver_w})",
            xaxis_title="Années",
            yaxis_title="Valeur (k€)",
            hovermode="x unified",
            template="plotly_white"
        )
        st.plotly_chart(fig_wealth, use_container_width=True)

    with tab2:
        selected_solver_a = st.selectbox("Choisir le solveur pour l'allocation", available_solvers, key="a_solver")
        df_a = all_results[all_results['solver'] == selected_solver_a]
        
        st.subheader(f"Composition du Portefeuille ({selected_solver_a})")
        alloc_cols = [c for c in df_a.columns if c.startswith('alloc_')]
        avg_alloc = df_a.groupby('time')[alloc_cols].mean()
        
        # Garantir des valeurs positives pour le stacked area chart
        avg_alloc = avg_alloc.clip(lower=0)
        avg_alloc = avg_alloc.div(avg_alloc.sum(axis=1), axis=0)
        
        fig_alloc = go.Figure()
        for col in alloc_cols:
            asset_name = col.replace('alloc_', '').capitalize()
            fig_alloc.add_trace(go.Scatter(
                x=avg_alloc.index, y=avg_alloc[col],
                mode='lines',
                stackgroup='one',
                name=asset_name,
                line=dict(color=ASSET_COLORS.get(asset_name, None))
            ))
        
        fig_alloc.update_layout(
            title=f"Allocation d'Actifs Optimale ({selected_solver_a})",
            xaxis_title="Années",
            yaxis_title="Poids (%)",
            yaxis=dict(range=[0, 1]),
            template="plotly_white"
        )
        st.plotly_chart(fig_alloc, use_container_width=True)
        
    with tab3:
        st.subheader("Comparaison des Performances")
        
        # Violin plot incluant tous les solveurs disponibles
        fig_comp = px.violin(
            all_results[all_results['time'] == horizon],
            x="solver", y="wealth", color="solver",
            box=True, points="all",
            title="Distribution de la Richesse Finale (Tous les solveurs)"
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # Tableau récapitulatif
        summary_list = []
        for s in available_solvers:
            final_w = all_results[(all_results['solver'] == s) & (all_results['time'] == horizon)]['wealth']
            summary_list.append({
                "Solveur": s,
                "Richesse Moyenne": f"{final_w.mean():.1f} k€",
                "Écart-type": f"{final_w.std():.1f} k€",
                "Probabilité de Succès": f"{(final_w > 0).mean()*100:.1f} %"
            })
        st.table(pd.DataFrame(summary_list))

else:
    st.info("Configurez vos paramètres dans la barre latérale et cliquez sur 'Calculer' pour voir votre stratégie.")

# Footer
st.markdown("---")
st.caption("Développé par Roo Code Expert - 2026")
