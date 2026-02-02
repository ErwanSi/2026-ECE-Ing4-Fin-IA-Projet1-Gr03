"""
Module de calcul du fitness multi-objectifs pour les stratégies de trading.

Ce module contient la classe FitnessCalculator qui combine plusieurs métriques
de performance (rendement, Sharpe ratio, max drawdown) en un score unique.

Améliorations anti-overfitting :
- Validation croisée (cross-validation)
- Pénalité pour la complexité de la stratégie
- Régularisation L2
- Pondération ajustée pour favoriser la généralisation
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit

from trading_strategy import TradingStrategy
from backtester import Backtester, BacktestResult


@dataclass
class FitnessWeights:
    """Pondérations pour les différents objectifs du fitness."""
    return_weight: float = 0.25
    sharpe_weight: float = 0.25
    drawdown_weight: float = 0.20
    stability_weight: float = 0.15
    generalization_weight: float = 0.10
    complexity_penalty_weight: float = 0.05
    
    def normalize(self):
        """Normalise les pondérations pour que leur somme soit 1."""
        total = (self.return_weight + self.sharpe_weight +
                 self.drawdown_weight + self.stability_weight +
                 self.generalization_weight + self.complexity_penalty_weight)
        if total > 0:
            self.return_weight /= total
            self.sharpe_weight /= total
            self.drawdown_weight /= total
            self.stability_weight /= total
            self.generalization_weight /= total
            self.complexity_penalty_weight /= total


class FitnessCalculator:
    """
    Calculateur de fitness multi-objectifs pour les stratégies de trading.
    
    Combine plusieurs métriques de performance en un score unique :
    - Rendement total
    - Sharpe ratio
    - Maximum drawdown (inversé)
    - Stabilité (variance des rendements)
    - Score de généralisation (cross-validation)
    - Pénalité de complexité
    
    Améliorations anti-overfitting :
    - Validation croisée temporelle (TimeSeriesSplit)
    - Pénalité pour la complexité de la stratégie
    - Régularisation L2 sur les paramètres
    - Score de généralisation basé sur la variance des folds
    """
    
    def __init__(
        self,
        weights: FitnessWeights = None,
        backtester: Backtester = None,
        data: pd.DataFrame = None,
        use_cross_validation: bool = True,
        cv_folds: int = 3,
        complexity_penalty: float = 0.01,
        l2_regularization: float = 0.001
    ):
        """
        Initialise le calculateur de fitness.
        
        Args:
            weights: Pondérations des différents objectifs
            backtester: Instance de Backtester pour évaluer les stratégies
            data: Données historiques pour le backtesting
            use_cross_validation: Utiliser la validation croisée
            cv_folds: Nombre de folds pour la cross-validation
            complexity_penalty: Coefficient de pénalité pour la complexité
            l2_regularization: Coefficient de régularisation L2
        """
        self.weights = weights if weights else FitnessWeights()
        self.weights.normalize()
        
        self.backtester = backtester if backtester else Backtester()
        self.data = data
        
        # Paramètres anti-overfitting
        self.use_cross_validation = use_cross_validation
        self.cv_folds = cv_folds
        self.complexity_penalty = complexity_penalty
        self.l2_regularization = l2_regularization
        
        # Stockage des résultats pour analyse
        self.last_result: Optional[BacktestResult] = None
        self.last_strategy: Optional[TradingStrategy] = None
        self.last_cv_scores: Optional[List[float]] = None
        self.last_generalization_score: Optional[float] = None
    
    def set_data(self, data: pd.DataFrame):
        """
        Définit les données à utiliser pour le backtesting.
        
        Args:
            data: DataFrame contenant les données OHLCV
        """
        self.data = data
    
    def calculate_fitness(self, genes: np.ndarray) -> float:
        """
        Calcule le fitness d'un chromosome avec mécanismes anti-overfitting.
        
        Args:
            genes: Chromosome représentant une stratégie
            
        Returns:
            Score de fitness
        """
        if self.data is None:
            raise ValueError("Les données de backtesting ne sont pas définies.")
        
        # Créer la stratégie à partir des gènes
        strategy = TradingStrategy(genes=genes)
        
        # Utiliser la validation croisée si activée
        if self.use_cross_validation and len(self.data) > 100:
            result, generalization_score, cv_scores = self._cross_validate_fitness(strategy)
            self.last_cv_scores = cv_scores
            self.last_generalization_score = generalization_score
        else:
            result = self.backtester.run_backtest(self.data, strategy)
            generalization_score = 0.5
            self.last_cv_scores = None
            self.last_generalization_score = None
        
        # Stocker pour analyse
        self.last_result = result
        self.last_strategy = strategy
        
        # Calculer le fitness multi-objectifs avec pénalités
        return self._multi_objective_fitness_with_penalties(result, genes, generalization_score)
    
    def _multi_objective_fitness_with_penalties(
        self,
        result: BacktestResult,
        genes: np.ndarray,
        generalization_score: float
    ) -> float:
        """
        Calcule le fitness multi-objectifs avec pénalités anti-overfitting.
        
        Args:
            result: Résultat du backtesting
            genes: Chromosome représentant la stratégie
            generalization_score: Score de généralisation de la cross-validation
            
        Returns:
            Score de fitness combiné
        """
        # Score de rendement (normalisé entre 0 et 1)
        return_score = self._normalize_return(result.total_return)
        
        # Score de Sharpe (normalisé entre 0 et 1)
        sharpe_score = self._normalize_sharpe(result.sharpe_ratio)
        
        # Score de drawdown (inversé, normalisé entre 0 et 1)
        drawdown_score = self._normalize_drawdown(result.max_drawdown)
        
        # Score de stabilité (basé sur la variance de la courbe d'équité)
        stability_score = self._calculate_stability(result.equity_curve)
        
        # Score de généralisation (basé sur la cross-validation)
        generalization_score_normalized = self._normalize_generalization(generalization_score)
        
        # Pénalité de complexité (favorise les stratégies simples)
        complexity_penalty = self._calculate_complexity_penalty(genes)
        
        # Pénalité L2 (régularisation sur les paramètres)
        l2_penalty = self._calculate_l2_penalty(genes)
        
        # Combinaison pondérée
        fitness = (
            self.weights.return_weight * return_score +
            self.weights.sharpe_weight * sharpe_score +
            self.weights.drawdown_weight * drawdown_score +
            self.weights.stability_weight * stability_score +
            self.weights.generalization_weight * generalization_score_normalized -
            self.weights.complexity_penalty_weight * complexity_penalty -
            self.weights.complexity_penalty_weight * l2_penalty
        )
        
        return max(0.0, fitness)  # Assurer que le fitness est non-négatif
    
    def _multi_objective_fitness(self, result: BacktestResult) -> float:
        """
        Calcule le fitness multi-objectifs à partir des résultats (sans pénalités).
        
        Args:
            result: Résultat du backtesting
            
        Returns:
            Score de fitness combiné
        """
        # Score de rendement (normalisé entre 0 et 1)
        return_score = self._normalize_return(result.total_return)
        
        # Score de Sharpe (normalisé entre 0 et 1)
        sharpe_score = self._normalize_sharpe(result.sharpe_ratio)
        
        # Score de drawdown (inversé, normalisé entre 0 et 1)
        drawdown_score = self._normalize_drawdown(result.max_drawdown)
        
        # Score de stabilité (basé sur la variance de la courbe d'équité)
        stability_score = self._calculate_stability(result.equity_curve)
        
        # Combinaison pondérée
        fitness = (
            self.weights.return_weight * return_score +
            self.weights.sharpe_weight * sharpe_score +
            self.weights.drawdown_weight * drawdown_score +
            self.weights.stability_weight * stability_score
        )
        
        return fitness
    
    def _normalize_return(self, total_return: float) -> float:
        """
        Normalise le rendement total.
        
        Args:
            total_return: Rendement total en pourcentage
            
        Returns:
            Score normalisé entre 0 et 1
        """
        # Fonction sigmoïde pour normaliser
        # 0% -> 0.5, 50% -> ~0.73, 100% -> ~0.88
        return 1 / (1 + np.exp(-0.05 * (total_return - 20)))
    
    def _normalize_sharpe(self, sharpe_ratio: float) -> float:
        """
        Normalise le Sharpe ratio.
        
        Args:
            sharpe_ratio: Sharpe ratio
            
        Returns:
            Score normalisé entre 0 et 1
        """
        # Sharpe < 0 -> 0, Sharpe = 1 -> 0.5, Sharpe = 3 -> ~0.95
        if sharpe_ratio < 0:
            return 0.0
        return 1 / (1 + np.exp(-1.5 * (sharpe_ratio - 1)))
    
    def _normalize_drawdown(self, max_drawdown: float) -> float:
        """
        Normalise le maximum drawdown (inversé).
        
        Args:
            max_drawdown: Maximum drawdown en pourcentage (négatif)
            
        Returns:
            Score normalisé entre 0 et 1
        """
        # Plus le drawdown est faible (proche de 0), meilleur est le score
        # 0% -> 1, -10% -> 0.73, -20% -> 0.5, -50% -> 0.12
        return 1 / (1 + np.exp(0.1 * (max_drawdown + 10)))
    
    def _calculate_stability(self, equity_curve: pd.Series) -> float:
        """
        Calcule un score de stabilité basé sur la variance de la courbe d'équité.
        
        Args:
            equity_curve: Courbe d'équité
            
        Returns:
            Score de stabilité entre 0 et 1
        """
        if len(equity_curve) < 2:
            return 0.5
        
        # Calculer les rendements quotidiens
        returns = equity_curve.pct_change().dropna()
        
        if len(returns) == 0:
            return 0.5
        
        # Coefficient de variation (écart-type / moyenne)
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0:
            return 1.0
        
        cv = abs(std_return / mean_return) if mean_return != 0 else float('inf')
        
        # Normaliser le CV (plus c'est bas, meilleur c'est)
        # CV = 0 -> 1, CV = 1 -> 0.5, CV = 5 -> 0.12
        return 1 / (1 + cv)
    
    def _cross_validate_fitness(
        self,
        strategy: TradingStrategy
    ) -> Tuple[BacktestResult, float, List[float]]:
        """
        Effectue une validation croisée temporelle pour évaluer la généralisation.
        
        Args:
            strategy: Stratégie de trading à évaluer
            
        Returns:
            Tuple (résultat moyen, score de généralisation, scores par fold)
        """
        if len(self.data) < 50:
            # Pas assez de données pour la cross-validation
            result = self.backtester.run_backtest(self.data, strategy)
            return result, 0.5, [self._multi_objective_fitness(result)]
        
        # Utiliser TimeSeriesSplit pour la validation croisée temporelle
        tscv = TimeSeriesSplit(n_splits=min(self.cv_folds, len(self.data) // 20))
        
        fold_scores = []
        fold_results = []
        
        for train_idx, test_idx in tscv.split(self.data):
            train_data = self.data.iloc[train_idx]
            test_data = self.data.iloc[test_idx]
            
            if len(test_data) < 10:
                continue
            
            # Backtesting sur le fold de test
            result = self.backtester.run_backtest(test_data, strategy)
            fold_results.append(result)
            fold_scores.append(self._multi_objective_fitness(result))
        
        if not fold_scores:
            result = self.backtester.run_backtest(self.data, strategy)
            return result, 0.5, [self._multi_objective_fitness(result)]
        
        # Calculer le résultat moyen
        avg_result = self._average_backtest_results(fold_results)
        
        # Score de généralisation : inverse de la variance des scores
        # Plus la variance est faible, meilleure est la généralisation
        if len(fold_scores) > 1:
            variance = np.var(fold_scores)
            generalization_score = 1.0 / (1.0 + 10.0 * variance)
        else:
            generalization_score = 0.5
        
        return avg_result, generalization_score, fold_scores
    
    def _average_backtest_results(self, results: List[BacktestResult]) -> BacktestResult:
        """
        Calcule la moyenne des résultats de backtesting.
        
        Args:
            results: Liste des résultats de backtesting
            
        Returns:
            Résultat moyen
        """
        if not results:
            return BacktestResult(
                total_return=0.0,
                annualized_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                avg_trade_return=0.0,
                best_trade=0.0,
                worst_trade=0.0,
                trades=[],
                equity_curve=pd.Series()
            )
        
        avg_result = BacktestResult(
            total_return=np.mean([r.total_return for r in results]),
            annualized_return=np.mean([r.annualized_return for r in results]),
            sharpe_ratio=np.mean([r.sharpe_ratio for r in results]),
            max_drawdown=np.mean([r.max_drawdown for r in results]),
            win_rate=np.mean([r.win_rate for r in results]),
            profit_factor=np.mean([r.profit_factor for r in results]),
            total_trades=int(np.mean([r.total_trades for r in results])),
            winning_trades=int(np.mean([r.winning_trades for r in results])),
            losing_trades=int(np.mean([r.losing_trades for r in results])),
            avg_trade_return=np.mean([r.avg_trade_return for r in results]),
            best_trade=np.mean([r.best_trade for r in results]),
            worst_trade=np.mean([r.worst_trade for r in results]),
                trades=[],
            equity_curve=pd.Series()
        )
        
        return avg_result
    
    def _normalize_generalization(self, generalization_score: float) -> float:
        """
        Normalise le score de généralisation.
        
        Args:
            generalization_score: Score de généralisation
            
        Returns:
            Score normalisé entre 0 et 1
        """
        # Le score est déjà entre 0 et 1, mais on peut appliquer une transformation
        # pour favoriser les scores élevés
        return generalization_score
    
    def _calculate_complexity_penalty(self, genes: np.ndarray) -> float:
        """
        Calcule une pénalité basée sur la complexité de la stratégie.
        
        Favorise les stratégies avec des paramètres plus simples et proches des valeurs par défaut.
        
        Args:
            genes: Chromosome représentant la stratégie
            
        Returns:
            Pénalité de complexité (entre 0 et 1)
        """
        # Valeurs par défaut pour chaque paramètre
        default_genes = np.array([
            10.0,   # SMA court
            30.0,   # SMA long
            20.0,   # EMA période
            14.0,   # RSI période
            30.0,   # RSI survente
            70.0,   # RSI surachat
            12.0,   # MACD rapide
            26.0,   # MACD lent
            9.0,    # MACD signal
            5.0     # Stop loss %
        ])
        
        # Normaliser les gènes et les valeurs par défaut
        max_values = np.array([50.0, 200.0, 50.0, 30.0, 40.0, 90.0, 20.0, 50.0, 15.0, 20.0])
        
        normalized_genes = genes / max_values
        normalized_defaults = default_genes / max_values
        
        # Calculer la distance euclidienne normalisée
        distance = np.sqrt(np.sum((normalized_genes - normalized_defaults) ** 2))
        
        # Pénalité basée sur la distance (plus c'est loin des valeurs par défaut, plus c'est pénalisé)
        penalty = self.complexity_penalty * distance
        
        return min(1.0, penalty)
    
    def _calculate_l2_penalty(self, genes: np.ndarray) -> float:
        """
        Calcule une pénalité L2 sur les paramètres de la stratégie.
        
        Favorise les paramètres plus petits (régularisation).
        
        Args:
            genes: Chromosome représentant la stratégie
            
        Returns:
            Pénalité L2 (entre 0 et 1)
        """
        # Normaliser les gènes
        max_values = np.array([50.0, 200.0, 50.0, 30.0, 40.0, 90.0, 20.0, 50.0, 15.0, 20.0])
        normalized_genes = genes / max_values
        
        # Calculer la norme L2
        l2_norm = np.sqrt(np.sum(normalized_genes ** 2))
        
        # Pénalité basée sur la norme L2
        penalty = self.l2_regularization * l2_norm
        
        return min(1.0, penalty)
    
    def calculate_fitness_from_result(self, result: BacktestResult) -> float:
        """
        Calcule le fitness directement à partir d'un résultat de backtesting.
        
        Args:
            result: Résultat du backtesting
            
        Returns:
            Score de fitness
        """
        return self._multi_objective_fitness(result)
    
    def get_fitness_components(self, genes: np.ndarray) -> Dict[str, float]:
        """
        Retourne les composantes individuelles du fitness.
        
        Args:
            genes: Chromosome représentant une stratégie
            
        Returns:
            Dictionnaire des composantes du fitness
        """
        if self.data is None:
            raise ValueError("Les données de backtesting ne sont pas définies.")
        
        strategy = TradingStrategy(genes=genes)
        result = self.backtester.run_backtest(self.data, strategy)
        
        return {
            'total_return': result.total_return,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'win_rate': result.win_rate,
            'profit_factor': result.profit_factor,
            'total_trades': result.total_trades,
            'return_score': self._normalize_return(result.total_return),
            'sharpe_score': self._normalize_sharpe(result.sharpe_ratio),
            'drawdown_score': self._normalize_drawdown(result.max_drawdown),
            'stability_score': self._calculate_stability(result.equity_curve),
            'fitness': self._multi_objective_fitness(result)
        }
    
    def set_weights(self, **kwargs):
        """
        Modifie les pondérations du fitness.
        
        Args:
            **kwargs: Nouvelles pondérations (return_weight, sharpe_weight, etc.)
        """
        for key, value in kwargs.items():
            if hasattr(self.weights, key):
                setattr(self.weights, key, value)
        self.weights.normalize()
    
    def get_weights(self) -> Dict[str, float]:
        """
        Retourne les pondérations actuelles.
        
        Returns:
            Dictionnaire des pondérations
        """
        return {
            'return_weight': self.weights.return_weight,
            'sharpe_weight': self.weights.sharpe_weight,
            'drawdown_weight': self.weights.drawdown_weight,
            'stability_weight': self.weights.stability_weight,
            'generalization_weight': self.weights.generalization_weight,
            'complexity_penalty_weight': self.weights.complexity_penalty_weight
        }
    
    def get_cv_info(self) -> Dict[str, any]:
        """
        Retourne les informations de la dernière cross-validation.
        
        Returns:
            Dictionnaire contenant les informations de cross-validation
        """
        return {
            'cv_scores': self.last_cv_scores,
            'generalization_score': self.last_generalization_score,
            'use_cross_validation': self.use_cross_validation,
            'cv_folds': self.cv_folds
        }
    
    def set_anti_overfitting_params(
        self,
        use_cross_validation: Optional[bool] = None,
        cv_folds: Optional[int] = None,
        complexity_penalty: Optional[float] = None,
        l2_regularization: Optional[float] = None
    ):
        """
        Modifie les paramètres anti-overfitting.
        
        Args:
            use_cross_validation: Utiliser la validation croisée
            cv_folds: Nombre de folds pour la cross-validation
            complexity_penalty: Coefficient de pénalité pour la complexité
            l2_regularization: Coefficient de régularisation L2
        """
        if use_cross_validation is not None:
            self.use_cross_validation = use_cross_validation
        if cv_folds is not None:
            self.cv_folds = max(2, min(10, cv_folds))
        if complexity_penalty is not None:
            self.complexity_penalty = max(0.0, min(1.0, complexity_penalty))
        if l2_regularization is not None:
            self.l2_regularization = max(0.0, min(1.0, l2_regularization))
