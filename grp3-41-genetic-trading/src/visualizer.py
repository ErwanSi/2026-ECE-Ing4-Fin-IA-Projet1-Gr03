"""
Module de visualisation pour les stratégies de trading par algorithmes génétiques.

Ce module contient la classe Visualizer qui permet de créer des graphiques
pour analyser les performances des stratégies de trading.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour sauvegarder les graphiques
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os

from backtester import BacktestResult
from trading_strategy import TradingStrategy


class Visualizer:
    """
    Classe pour visualiser les résultats des stratégies de trading.
    
    Permet de créer des graphiques pour :
    - Courbe d'équité (equity curve)
    - Drawdown au fil du temps
    - Évolution du fitness par génération
    - Comparaison des performances entre ensembles
    """
    
    def __init__(self, output_dir: str = "plots"):
        """
        Initialise le visualiseur.
        
        Args:
            output_dir: Répertoire de sortie pour les graphiques
        """
        self.output_dir = output_dir
        self._ensure_output_dir()
        
        # Configuration du style
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            plt.style.use('seaborn-darkgrid')
        self.colors = {
            'train': '#2ecc71',
            'validation': '#3498db',
            'test': '#e74c3c',
            'benchmark': '#95a5a6',
            'equity': '#27ae60',
            'drawdown': '#c0392b',
            'fitness': '#8e44ad',
            'avg_fitness': '#f39c12'
        }
    
    def _ensure_output_dir(self):
        """Crée le répertoire de sortie s'il n'existe pas."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def plot_equity_curve(
        self,
        train_result: BacktestResult,
        validation_result: BacktestResult,
        test_result: BacktestResult,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Trace la courbe d'équité pour l'entraînement, validation et test.
        
        Args:
            train_result: Résultat du backtesting sur l'entraînement
            validation_result: Résultat du backtesting sur la validation
            test_result: Résultat du backtesting sur le test
            save_path: Chemin pour sauvegarder le graphique
            show: Afficher le graphique
            
        Returns:
            Figure matplotlib
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Courbe d'équité d'entraînement
        train_equity = train_result.equity_curve
        ax.plot(train_equity.index, train_equity.values, 
                color=self.colors['train'], linewidth=2, label='Entraînement')
        
        # Courbe d'équité de validation
        val_equity = validation_result.equity_curve
        ax.plot(val_equity.index, val_equity.values, 
                color=self.colors['validation'], linewidth=2, label='Validation')
        
        # Courbe d'équité de test
        test_equity = test_result.equity_curve
        ax.plot(test_equity.index, test_equity.values, 
                color=self.colors['test'], linewidth=2, label='Test')
        
        # Ligne de capital initial
        initial_capital = train_result.equity_curve.iloc[0]
        ax.axhline(y=initial_capital, color=self.colors['benchmark'], 
                   linestyle='--', alpha=0.5, label='Capital initial')
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Capital (€)', fontsize=12, fontweight='bold')
        ax.set_title('Courbe d\'Équité - Comparaison Entraînement/Validation/Test', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Formatage des dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_drawdown(
        self,
        train_result: BacktestResult,
        validation_result: BacktestResult,
        test_result: BacktestResult,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Trace le drawdown au fil du temps pour chaque ensemble.
        
        Args:
            train_result: Résultat du backtesting sur l'entraînement
            validation_result: Résultat du backtesting sur la validation
            test_result: Résultat du backtesting sur le test
            save_path: Chemin pour sauvegarder le graphique
            show: Afficher le graphique
            
        Returns:
            Figure matplotlib
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        def calculate_drawdown(equity_curve: pd.Series) -> pd.Series:
            """Calcule le drawdown à partir de la courbe d'équité."""
            rolling_max = equity_curve.expanding().max()
            drawdown = (equity_curve - rolling_max) / rolling_max * 100
            return drawdown
        
        # Drawdown d'entraînement
        train_dd = calculate_drawdown(train_result.equity_curve)
        ax.fill_between(train_dd.index, train_dd.values, 0, 
                        color=self.colors['train'], alpha=0.3, label='Entraînement')
        ax.plot(train_dd.index, train_dd.values, 
                color=self.colors['train'], linewidth=1)
        
        # Drawdown de validation
        val_dd = calculate_drawdown(validation_result.equity_curve)
        ax.fill_between(val_dd.index, val_dd.values, 0, 
                        color=self.colors['validation'], alpha=0.3, label='Validation')
        ax.plot(val_dd.index, val_dd.values, 
                color=self.colors['validation'], linewidth=1)
        
        # Drawdown de test
        test_dd = calculate_drawdown(test_result.equity_curve)
        ax.fill_between(test_dd.index, test_dd.values, 0, 
                        color=self.colors['test'], alpha=0.3, label='Test')
        ax.plot(test_dd.index, test_dd.values, 
                color=self.colors['test'], linewidth=1)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
        ax.set_title('Drawdown au Fil du Temps', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Formatage des dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_fitness_evolution(
        self,
        history: List[Dict],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Trace l'évolution du fitness par génération.
        
        Args:
            history: Historique de l'algorithme génétique
            save_path: Chemin pour sauvegarder le graphique
            show: Afficher le graphique
            
        Returns:
            Figure matplotlib
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        generations = [h['generation'] for h in history]
        best_fitness = [h['best_fitness'] for h in history]
        avg_fitness = [h['avg_fitness'] for h in history]
        worst_fitness = [h['worst_fitness'] for h in history]
        
        ax.plot(generations, best_fitness, 
                color=self.colors['fitness'], linewidth=2, label='Meilleur fitness')
        ax.plot(generations, avg_fitness, 
                color=self.colors['avg_fitness'], linewidth=2, label='Fitness moyen')
        ax.plot(generations, worst_fitness, 
                color=self.colors['drawdown'], linewidth=1, linestyle='--', label='Pire fitness')
        
        ax.fill_between(generations, worst_fitness, best_fitness, 
                        alpha=0.2, color=self.colors['fitness'])
        
        ax.set_xlabel('Génération', fontsize=12, fontweight='bold')
        ax.set_ylabel('Fitness', fontsize=12, fontweight='bold')
        ax.set_title('Évolution du Fitness par Génération', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_performance_comparison(
        self,
        train_fitness: float,
        validation_fitness: float,
        test_fitness: float,
        validation_components: Dict,
        test_components: Dict,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Trace un graphique comparatif des performances entre ensembles.
        
        Args:
            train_fitness: Fitness sur l'entraînement
            validation_fitness: Fitness sur la validation
            test_fitness: Fitness sur le test
            validation_components: Composantes du fitness de validation
            test_components: Composantes du fitness de test
            save_path: Chemin pour sauvegarder le graphique
            show: Afficher le graphique
            
        Returns:
            Figure matplotlib
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Comparaison des fitness
        ax1 = axes[0, 0]
        labels = ['Entraînement', 'Validation', 'Test']
        fitness_values = [train_fitness, validation_fitness, test_fitness]
        colors = [self.colors['train'], self.colors['validation'], self.colors['test']]
        
        bars = ax1.bar(labels, fitness_values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Fitness', fontsize=11, fontweight='bold')
        ax1.set_title('Comparaison des Fitness', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Ajouter les valeurs sur les barres
        for bar, value in zip(bars, fitness_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Comparaison des rendements
        ax2 = axes[0, 1]
        returns = [validation_components['total_return'], test_components['total_return']]
        colors2 = [self.colors['validation'], self.colors['test']]
        
        bars2 = ax2.bar(['Validation', 'Test'], returns, color=colors2, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Rendement (%)', fontsize=11, fontweight='bold')
        ax2.set_title('Comparaison des Rendements', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        for bar, value in zip(bars2, returns):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. Comparaison des Sharpe ratios
        ax3 = axes[1, 0]
        sharpe = [validation_components['sharpe_ratio'], test_components['sharpe_ratio']]
        
        bars3 = ax3.bar(['Validation', 'Test'], sharpe, color=colors2, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Sharpe Ratio', fontsize=11, fontweight='bold')
        ax3.set_title('Comparaison des Sharpe Ratios', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Sharpe = 1')
        ax3.legend()
        
        for bar, value in zip(bars3, sharpe):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Comparaison des drawdowns
        ax4 = axes[1, 1]
        drawdowns = [validation_components['max_drawdown'], test_components['max_drawdown']]
        
        bars4 = ax4.bar(['Validation', 'Test'], drawdowns, color=colors2, alpha=0.7, edgecolor='black')
        ax4.set_ylabel('Max Drawdown (%)', fontsize=11, fontweight='bold')
        ax4.set_title('Comparaison des Drawdowns', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        for bar, value in zip(bars4, drawdowns):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 1,
                    f'{value:.2f}%', ha='center', va='top', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_walk_forward_results(
        self,
        wf_results: List[BacktestResult],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Trace les résultats du walk-forward testing.
        
        Args:
            wf_results: Liste des résultats du walk-forward testing
            save_path: Chemin pour sauvegarder le graphique
            show: Afficher le graphique
            
        Returns:
            Figure matplotlib
        """
        if not wf_results:
            print("Aucun résultat de walk-forward testing à afficher.")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extraire les métriques
        returns = [r.total_return for r in wf_results]
        sharpe_ratios = [r.sharpe_ratio for r in wf_results]
        drawdowns = [r.max_drawdown for r in wf_results]
        win_rates = [r.win_rate for r in wf_results]
        
        window_numbers = list(range(1, len(wf_results) + 1))
        
        # 1. Rendements par fenêtre
        ax1 = axes[0, 0]
        ax1.bar(window_numbers, returns, color=self.colors['fitness'], alpha=0.7, edgecolor='black')
        ax1.axhline(y=np.mean(returns), color='red', linestyle='--', 
                   label=f'Moyenne: {np.mean(returns):.2f}%')
        ax1.set_xlabel('Fenêtre', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Rendement (%)', fontsize=11, fontweight='bold')
        ax1.set_title('Rendements par Fenêtre (Walk-Forward)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.legend()
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # 2. Sharpe ratios par fenêtre
        ax2 = axes[0, 1]
        ax2.bar(window_numbers, sharpe_ratios, color=self.colors['avg_fitness'], alpha=0.7, edgecolor='black')
        ax2.axhline(y=np.mean(sharpe_ratios), color='red', linestyle='--', 
                   label=f'Moyenne: {np.mean(sharpe_ratios):.2f}')
        ax2.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Sharpe = 1')
        ax2.set_xlabel('Fenêtre', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Sharpe Ratio', fontsize=11, fontweight='bold')
        ax2.set_title('Sharpe Ratios par Fenêtre', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend()
        
        # 3. Drawdowns par fenêtre
        ax3 = axes[1, 0]
        ax3.bar(window_numbers, drawdowns, color=self.colors['drawdown'], alpha=0.7, edgecolor='black')
        ax3.axhline(y=np.mean(drawdowns), color='red', linestyle='--', 
                   label=f'Moyenne: {np.mean(drawdowns):.2f}%')
        ax3.set_xlabel('Fenêtre', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Max Drawdown (%)', fontsize=11, fontweight='bold')
        ax3.set_title('Drawdowns par Fenêtre', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.legend()
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # 4. Win rates par fenêtre
        ax4 = axes[1, 1]
        ax4.bar(window_numbers, win_rates, color=self.colors['train'], alpha=0.7, edgecolor='black')
        ax4.axhline(y=np.mean(win_rates), color='red', linestyle='--', 
                   label=f'Moyenne: {np.mean(win_rates):.2f}%')
        ax4.axhline(y=50, color='green', linestyle='--', alpha=0.5, label='50%')
        ax4.set_xlabel('Fenêtre', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Win Rate (%)', fontsize=11, fontweight='bold')
        ax4.set_title('Win Rates par Fenêtre', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_strategy_parameters(
        self,
        strategy: TradingStrategy,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Visualise les paramètres de la stratégie.
        
        Args:
            strategy: Stratégie de trading
            save_path: Chemin pour sauvegarder le graphique
            show: Afficher le graphique
            
        Returns:
            Figure matplotlib
        """
        params = strategy.get_parameters()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Créer un radar chart pour les paramètres
        categories = list(params.keys())
        values = list(params.values())
        
        # Normaliser les valeurs pour l'affichage
        max_values = {
            'sma_short_period': 50,
            'sma_long_period': 200,
            'ema_period': 50,
            'rsi_period': 30,
            'rsi_oversold': 40,
            'rsi_overbought': 90,
            'macd_fast': 20,
            'macd_slow': 50,
            'macd_signal': 15,
            'stop_loss_pct': 20
        }
        
        normalized_values = [values[i] / max_values[cat] for i, cat in enumerate(categories)]
        
        # Créer le bar plot
        bars = ax.barh(categories, values, color=self.colors['fitness'], alpha=0.7, edgecolor='black')
        
        # Ajouter les valeurs sur les barres
        for bar, value in zip(bars, values):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                   f'{value:.1f}', ha='left', va='center', fontweight='bold')
        
        ax.set_xlabel('Valeur', fontsize=12, fontweight='bold')
        ax.set_title('Paramètres de la Stratégie', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_comprehensive_dashboard(
        self,
        train_result: BacktestResult,
        validation_result: BacktestResult,
        test_result: BacktestResult,
        history: List[Dict],
        train_fitness: float,
        validation_fitness: float,
        test_fitness: float,
        validation_components: Dict,
        test_components: Dict,
        strategy: TradingStrategy,
        wf_results: Optional[List[BacktestResult]] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Crée un tableau de bord complet avec tous les graphiques.
        
        Args:
            train_result: Résultat du backtesting sur l'entraînement
            validation_result: Résultat du backtesting sur la validation
            test_result: Résultat du backtesting sur le test
            history: Historique de l'algorithme génétique
            train_fitness: Fitness sur l'entraînement
            validation_fitness: Fitness sur la validation
            test_fitness: Fitness sur le test
            validation_components: Composantes du fitness de validation
            test_components: Composantes du fitness de test
            strategy: Stratégie de trading
            wf_results: Résultats du walk-forward testing
            save_path: Chemin pour sauvegarder le graphique
            show: Afficher le graphique
            
        Returns:
            Figure matplotlib
        """
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Courbe d'équité
        ax1 = plt.subplot(3, 3, 1)
        train_equity = train_result.equity_curve
        val_equity = validation_result.equity_curve
        test_equity = test_result.equity_curve
        
        ax1.plot(train_equity.index, train_equity.values, 
                color=self.colors['train'], linewidth=1.5, label='Entraînement')
        ax1.plot(val_equity.index, val_equity.values, 
                color=self.colors['validation'], linewidth=1.5, label='Validation')
        ax1.plot(test_equity.index, test_equity.values, 
                color=self.colors['test'], linewidth=1.5, label='Test')
        ax1.set_title('Courbe d\'Équité', fontsize=11, fontweight='bold')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Drawdown
        ax2 = plt.subplot(3, 3, 2)
        def calculate_drawdown(equity_curve: pd.Series) -> pd.Series:
            rolling_max = equity_curve.expanding().max()
            return (equity_curve - rolling_max) / rolling_max * 100
        
        train_dd = calculate_drawdown(train_result.equity_curve)
        val_dd = calculate_drawdown(validation_result.equity_curve)
        test_dd = calculate_drawdown(test_result.equity_curve)
        
        ax2.fill_between(train_dd.index, train_dd.values, 0, 
                        color=self.colors['train'], alpha=0.3)
        ax2.fill_between(val_dd.index, val_dd.values, 0, 
                        color=self.colors['validation'], alpha=0.3)
        ax2.fill_between(test_dd.index, test_dd.values, 0, 
                        color=self.colors['test'], alpha=0.3)
        ax2.set_title('Drawdown', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Évolution du fitness
        ax3 = plt.subplot(3, 3, 3)
        generations = [h['generation'] for h in history]
        best_fitness = [h['best_fitness'] for h in history]
        avg_fitness = [h['avg_fitness'] for h in history]
        
        ax3.plot(generations, best_fitness, color=self.colors['fitness'], linewidth=1.5, label='Meilleur')
        ax3.plot(generations, avg_fitness, color=self.colors['avg_fitness'], linewidth=1.5, label='Moyen')
        ax3.set_title('Évolution du Fitness', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Génération')
        ax3.legend(loc='best', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. Comparaison des fitness
        ax4 = plt.subplot(3, 3, 4)
        labels = ['Entraînement', 'Validation', 'Test']
        fitness_values = [train_fitness, validation_fitness, test_fitness]
        colors = [self.colors['train'], self.colors['validation'], self.colors['test']]
        
        bars = ax4.bar(labels, fitness_values, color=colors, alpha=0.7, edgecolor='black')
        ax4.set_title('Comparaison des Fitness', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Fitness')
        ax4.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars, fitness_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 5. Comparaison des rendements
        ax5 = plt.subplot(3, 3, 5)
        returns = [validation_components['total_return'], test_components['total_return']]
        colors2 = [self.colors['validation'], self.colors['test']]
        
        bars2 = ax5.bar(['Validation', 'Test'], returns, color=colors2, alpha=0.7, edgecolor='black')
        ax5.set_title('Rendements', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Rendement (%)')
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        for bar, value in zip(bars2, returns):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 6. Comparaison des Sharpe ratios
        ax6 = plt.subplot(3, 3, 6)
        sharpe = [validation_components['sharpe_ratio'], test_components['sharpe_ratio']]
        
        bars3 = ax6.bar(['Validation', 'Test'], sharpe, color=colors2, alpha=0.7, edgecolor='black')
        ax6.set_title('Sharpe Ratios', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Sharpe Ratio')
        ax6.grid(True, alpha=0.3, axis='y')
        ax6.axhline(y=1, color='green', linestyle='--', alpha=0.5)
        
        for bar, value in zip(bars3, sharpe):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 7. Comparaison des drawdowns
        ax7 = plt.subplot(3, 3, 7)
        drawdowns = [validation_components['max_drawdown'], test_components['max_drawdown']]
        
        bars4 = ax7.bar(['Validation', 'Test'], drawdowns, color=colors2, alpha=0.7, edgecolor='black')
        ax7.set_title('Max Drawdowns', fontsize=11, fontweight='bold')
        ax7.set_ylabel('Drawdown (%)')
        ax7.grid(True, alpha=0.3, axis='y')
        ax7.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        for bar, value in zip(bars4, drawdowns):
            ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 1,
                    f'{value:.2f}%', ha='center', va='top', fontweight='bold', fontsize=9)
        
        # 8. Win rates
        ax8 = plt.subplot(3, 3, 8)
        win_rates = [validation_components['win_rate'], test_components['win_rate']]
        
        bars5 = ax8.bar(['Validation', 'Test'], win_rates, color=colors2, alpha=0.7, edgecolor='black')
        ax8.set_title('Win Rates', fontsize=11, fontweight='bold')
        ax8.set_ylabel('Win Rate (%)')
        ax8.grid(True, alpha=0.3, axis='y')
        ax8.axhline(y=50, color='green', linestyle='--', alpha=0.5)
        
        for bar, value in zip(bars5, win_rates):
            ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 9. Paramètres de la stratégie
        ax9 = plt.subplot(3, 3, 9)
        params = strategy.get_parameters()
        param_names = list(params.keys())
        param_values = list(params.values())
        
        bars6 = ax9.barh(param_names, param_values, color=self.colors['fitness'], alpha=0.7, edgecolor='black')
        ax9.set_title('Paramètres de la Stratégie', fontsize=11, fontweight='bold')
        ax9.set_xlabel('Valeur')
        ax9.grid(True, alpha=0.3, axis='x')
        
        for bar, value in zip(bars6, param_values):
            ax9.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{value:.1f}', ha='left', va='center', fontweight='bold', fontsize=8)
        
        plt.suptitle('Tableau de Bord Complet - Stratégie de Trading par Algorithme Génétique', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def close_all(self):
        """Ferme toutes les figures matplotlib."""
        plt.close('all')
