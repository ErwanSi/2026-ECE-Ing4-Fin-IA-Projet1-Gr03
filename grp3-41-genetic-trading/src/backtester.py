"""
Module de backtesting des stratégies de trading.

Ce module contient la classe Backtester qui permet de tester les stratégies
sur des données historiques et de calculer les métriques de performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

from trading_strategy import TradingStrategy, Signal


@dataclass
class Trade:
    """Représente un trade effectué."""
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    quantity: int
    signal: Signal
    pnl: float
    pnl_pct: float


@dataclass
class BacktestResult:
    """Résultats du backtesting."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade_return: float
    best_trade: float
    worst_trade: float
    trades: List[Trade]
    equity_curve: pd.Series


class Backtester:
    """
    Classe pour effectuer le backtesting des stratégies de trading.
    
    Permet de simuler l'exécution d'une stratégie sur des données historiques
    et de calculer les métriques de performance.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0001,
        position_size_pct: float = 0.95,
        max_drawdown_pct: float = 20.0
    ):
        """
        Initialise le backtester.
        
        Args:
            initial_capital: Capital initial
            commission: Commission par trade (en pourcentage)
            slippage: Slippage par trade (en pourcentage)
            position_size_pct: Pourcentage du capital investi par trade (0-1)
            max_drawdown_pct: Drawdown maximum autorisé avant d'arrêter le trading
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_size_pct = position_size_pct
        self.max_drawdown_pct = max_drawdown_pct
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        strategy: TradingStrategy,
        use_stop_loss: bool = True,
        allow_short: bool = False
    ) -> BacktestResult:
        """
        Exécute le backtesting d'une stratégie.
        
        Args:
            df: DataFrame contenant les données OHLCV
            strategy: Stratégie de trading à tester
            use_stop_loss: Utiliser le stop loss
            allow_short: Autoriser les positions courtes
            
        Returns:
            BacktestResult contenant les résultats
        """
        # Générer les signaux
        df = strategy.generate_signals(df)
        
        # Initialiser les variables
        capital = self.initial_capital
        position = 0  # 0 = pas de position, >0 = long, <0 = short
        entry_price = 0.0
        entry_date = None
        quantity = 0
        stop_loss_price = 0.0
        peak_equity = self.initial_capital
        trading_stopped = False
        last_signal = None  # Garder une trace du dernier signal (None = pas de signal précédent)
        
        trades = []
        equity_curve = []
        
        for i, row in df.iterrows():
            current_price = row['Close']
            
            # Calculer la valeur du portefeuille correctement
            if position > 0:
                # Position longue : cash + valeur de la position
                portfolio_value = capital + (current_price - entry_price) * quantity
            elif position < 0:
                # Position courte : cash + profit/perte de la position courte
                portfolio_value = capital + (entry_price - current_price) * quantity
            else:
                portfolio_value = capital
            
            equity_curve.append(portfolio_value)
            
            # Mettre à jour le pic d'équité
            peak_equity = max(peak_equity, portfolio_value)
            
            # Vérifier le drawdown maximum et arrêter le trading si dépassé
            current_drawdown = (peak_equity - portfolio_value) / peak_equity * 100
            if current_drawdown > self.max_drawdown_pct and not trading_stopped:
                trading_stopped = True
                # Fermer la position existante
                if position != 0:
                    if position > 0:
                        exit_price = current_price * (1 - self.slippage)
                        pnl = (exit_price - entry_price) * quantity
                        capital += exit_price * quantity * (1 - self.commission)
                        trades.append(Trade(
                            entry_date=entry_date,
                            exit_date=i,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            quantity=quantity,
                            signal=Signal.BUY,
                            pnl=pnl,
                            pnl_pct=(exit_price - entry_price) / entry_price * 100
                        ))
                    else:
                        exit_price = current_price * (1 + self.slippage)
                        pnl = (entry_price - exit_price) * quantity
                        # Pour une position courte : on rachète pour fermer
                        capital += entry_price * quantity * (1 - self.commission) - exit_price * quantity * (1 + self.commission)
                        trades.append(Trade(
                            entry_date=entry_date,
                            exit_date=i,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            quantity=quantity,
                            signal=Signal.SELL,
                            pnl=pnl,
                            pnl_pct=(entry_price - exit_price) / entry_price * 100
                        ))
                    position = 0
                    entry_price = 0.0
                    entry_date = None
                    quantity = 0
                    stop_loss_price = 0.0
                continue
            
            # Vérifier le stop loss
            if use_stop_loss and position != 0 and not trading_stopped:
                if position > 0 and current_price <= stop_loss_price:
                    # Stop loss pour position longue
                    exit_price = current_price * (1 - self.slippage)
                    pnl = (exit_price - entry_price) * quantity
                    capital += exit_price * quantity * (1 - self.commission)
                    trades.append(Trade(
                        entry_date=entry_date,
                        exit_date=i,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        quantity=quantity,
                        signal=Signal.BUY,
                        pnl=pnl,
                        pnl_pct=(exit_price - entry_price) / entry_price * 100
                    ))
                    position = 0
                    entry_price = 0.0
                    entry_date = None
                    quantity = 0
                    stop_loss_price = 0.0
                    continue
                elif position < 0 and current_price >= stop_loss_price:
                    # Stop loss pour position courte
                    exit_price = current_price * (1 + self.slippage)
                    pnl = (entry_price - exit_price) * quantity
                    # Pour une position courte : on rachète pour fermer
                    capital += entry_price * quantity * (1 - self.commission) - exit_price * quantity * (1 + self.commission)
                    trades.append(Trade(
                        entry_date=entry_date,
                        exit_date=i,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        quantity=quantity,
                        signal=Signal.SELL,
                        pnl=pnl,
                        pnl_pct=(entry_price - exit_price) / entry_price * 100
                    ))
                    position = 0
                    entry_price = 0.0
                    entry_date = None
                    quantity = 0
                    stop_loss_price = 0.0
                    continue
            
            # Gérer les signaux - ne traiter que les transitions de signaux
            signal = row['signal']
            
            # Ne traiter que si le signal a changé et n'est pas HOLD
            if signal != last_signal and signal != Signal.HOLD.value and not trading_stopped:
                last_signal = signal
                
                if signal == Signal.BUY.value and position == 0:
                    # Entrée en position longue avec position sizing
                    entry_price = current_price * (1 + self.slippage)
                    position_value = capital * self.position_size_pct
                    quantity = int(position_value / entry_price)
                    if quantity > 0:
                        capital -= entry_price * quantity * (1 + self.commission)
                        position = 1
                        entry_date = i
                        stop_loss_price = entry_price * (1 - strategy.stop_loss_pct / 100)
                    
                elif signal == Signal.SELL.value and position == 0 and allow_short:
                    # Entrée en position courte avec position sizing
                    entry_price = current_price * (1 - self.slippage)
                    position_value = capital * self.position_size_pct
                    quantity = int(position_value / entry_price)
                    if quantity > 0:
                        # Pour une vente à découvert : on reçoit le cash de la vente
                        capital += entry_price * quantity * (1 - self.commission)
                        position = -1
                        entry_date = i
                        stop_loss_price = entry_price * (1 + strategy.stop_loss_pct / 100)
                    
                elif signal == Signal.SELL.value and position > 0:
                    # Sortie de position longue
                    exit_price = current_price * (1 - self.slippage)
                    pnl = (exit_price - entry_price) * quantity
                    capital += exit_price * quantity * (1 - self.commission)
                    trades.append(Trade(
                        entry_date=entry_date,
                        exit_date=i,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        quantity=quantity,
                        signal=Signal.BUY,
                        pnl=pnl,
                        pnl_pct=(exit_price - entry_price) / entry_price * 100
                    ))
                    position = 0
                    entry_price = 0.0
                    entry_date = None
                    quantity = 0
                    stop_loss_price = 0.0
                    
                elif signal == Signal.BUY.value and position < 0:
                    # Sortie de position courte
                    exit_price = current_price * (1 + self.slippage)
                    pnl = (entry_price - exit_price) * quantity
                    # Pour fermer une position courte : on rachète
                    capital += entry_price * quantity * (1 - self.commission) - exit_price * quantity * (1 + self.commission)
                    trades.append(Trade(
                        entry_date=entry_date,
                        exit_date=i,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        quantity=quantity,
                        signal=Signal.SELL,
                        pnl=pnl,
                        pnl_pct=(entry_price - exit_price) / entry_price * 100
                    ))
                    position = 0
                    entry_price = 0.0
                    entry_date = None
                    quantity = 0
                    stop_loss_price = 0.0
        
        # Fermer la position restante
        if position != 0:
            last_price = df['Close'].iloc[-1]
            if position > 0:
                exit_price = last_price * (1 - self.slippage)
                pnl = (exit_price - entry_price) * quantity
                capital += exit_price * quantity * (1 - self.commission)
                trades.append(Trade(
                    entry_date=entry_date,
                    exit_date=df.index[-1],
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=quantity,
                    signal=Signal.BUY,
                    pnl=pnl,
                    pnl_pct=(exit_price - entry_price) / entry_price * 100
                ))
            else:
                exit_price = last_price * (1 + self.slippage)
                pnl = (entry_price - exit_price) * quantity
                # Fermer position courte
                capital += entry_price * quantity * (1 - self.commission) - exit_price * quantity * (1 + self.commission)
                trades.append(Trade(
                    entry_date=entry_date,
                    exit_date=df.index[-1],
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=quantity,
                    signal=Signal.SELL,
                    pnl=pnl,
                    pnl_pct=(entry_price - exit_price) / entry_price * 100
                ))
        
        # Créer la série de courbe d'équité
        equity_series = pd.Series(equity_curve, index=df.index)
        
        # Calculer les métriques
        return self._calculate_metrics(trades, equity_series)
    
    def _calculate_metrics(self, trades: List[Trade], equity_curve: pd.Series) -> BacktestResult:
        """
        Calcule les métriques de performance.
        
        Args:
            trades: Liste des trades effectués
            equity_curve: Courbe d'équité
            
        Returns:
            BacktestResult avec les métriques calculées
        """
        if not trades:
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
                equity_curve=equity_curve
            )
        
        # Rendement total
        total_return = (equity_curve.iloc[-1] - self.initial_capital) / self.initial_capital * 100
        
        # Rendement annualisé
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        if days > 0 and self.initial_capital > 0 and equity_curve.iloc[-1] > 0:
            annualized_return = ((equity_curve.iloc[-1] / self.initial_capital) ** (365 / days) - 1) * 100
        else:
            annualized_return = 0.0
        
        # Sharpe Ratio
        returns = equity_curve.pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Maximum Drawdown
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
        
        # Statistiques des trades
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0.0
        
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        avg_trade_return = np.mean([t.pnl_pct for t in trades]) if trades else 0.0
        best_trade = max([t.pnl_pct for t in trades]) if trades else 0.0
        worst_trade = min([t.pnl_pct for t in trades]) if trades else 0.0
        
        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_trade_return=avg_trade_return,
            best_trade=best_trade,
            worst_trade=worst_trade,
            trades=trades,
            equity_curve=equity_curve
        )
    
    def walk_forward_test(
        self,
        df: pd.DataFrame,
        strategy: TradingStrategy,
        train_size: int = 252,
        test_size: int = 63,
        step_size: int = 21
    ) -> List[BacktestResult]:
        """
        Effectue un walk-forward testing pour éviter le curve-fitting.
        
        Args:
            df: DataFrame contenant les données OHLCV
            strategy: Stratégie de trading à tester
            train_size: Taille de la période d'entraînement (en jours)
            test_size: Taille de la période de test (en jours)
            step_size: Pas entre chaque fenêtre (en jours)
            
        Returns:
            Liste des résultats pour chaque fenêtre
        """
        results = []
        total_length = len(df)
        
        start_idx = train_size
        
        while start_idx + test_size < total_length:
            # Période d'entraînement
            train_data = df.iloc[start_idx - train_size:start_idx]
            
            # Période de test
            test_data = df.iloc[start_idx:start_idx + test_size]
            
            # Backtesting sur la période de test
            result = self.run_backtest(test_data, strategy)
            results.append(result)
            
            # Avancer la fenêtre
            start_idx += step_size
        
        return results
    
    def generate_report(self, result: BacktestResult) -> str:
        """
        Génère un rapport de performance.
        
        Args:
            result: Résultat du backtesting
            
        Returns:
            Rapport sous forme de chaîne de caractères
        """
        report = f"""
{'='*60}
RAPPORT DE BACKTESTING
{'='*60}

Capital Initial: {self.initial_capital:,.2f} €
Capital Final: {result.equity_curve.iloc[-1]:,.2f} €

PERFORMANCE
-----------
Rendement Total: {result.total_return:.2f}%
Rendement Annualisé: {result.annualized_return:.2f}%
Sharpe Ratio: {result.sharpe_ratio:.2f}
Maximum Drawdown: {result.max_drawdown:.2f}%

TRADES
------
Nombre Total de Trades: {result.total_trades}
Trades Gagnants: {result.winning_trades}
Trades Perdants: {result.losing_trades}
Taux de Réussite: {result.win_rate:.2f}%
Profit Factor: {result.profit_factor:.2f}

Rendement Moyen par Trade: {result.avg_trade_return:.2f}%
Meilleur Trade: {result.best_trade:.2f}%
Pire Trade: {result.worst_trade:.2f}%

{'='*60}
"""
        return report
