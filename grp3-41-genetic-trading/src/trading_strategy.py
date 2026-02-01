"""
Module de définition des stratégies de trading.

Ce module contient la classe TradingStrategy qui encode les indicateurs techniques,
les règles de décision et les paramètres de trading sous forme de chromosome.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from enum import Enum


class Signal(Enum):
    """Signaux de trading."""
    BUY = 1
    SELL = -1
    HOLD = 0


class TradingStrategy:
    """
    Classe représentant une stratégie de trading encodée comme chromosome.
    
    Le chromosome contient les paramètres suivants :
    - Gènes 0-2 : Périodes des moyennes mobiles (SMA court, SMA long, EMA)
    - Gènes 3-4 : Paramètres RSI (période, seuil de survente, seuil de surachat)
    - Gènes 5-6 : Paramètres MACD (période rapide, période lente, période signal)
    - Gènes 7-8 : Seuils de décision (seuil achat, seuil vente)
    - Gène 9 : Stop loss en pourcentage
    """
    
    def __init__(self, genes: np.ndarray = None):
        """
        Initialise une stratégie de trading.
        
        Args:
            genes: Tableau numpy contenant les paramètres de la stratégie
        """
        if genes is None:
            # Valeurs par défaut
            self.genes = np.array([
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
        else:
            self.genes = genes.copy()
        
        self._validate_genes()
    
    def _validate_genes(self):
        """Valide et corrige les gènes pour qu'ils soient dans les bornes acceptables."""
        # Bornes pour chaque gène
        bounds = [
            (5, 50),    # SMA court
            (10, 200),  # SMA long
            (5, 50),    # EMA période
            (5, 30),    # RSI période
            (15, 35),   # RSI survente (augmenté pour plus de signaux)
            (65, 85),   # RSI surachat (réduit pour plus de signaux)
            (5, 20),    # MACD rapide
            (15, 50),   # MACD lent
            (5, 15),    # MACD signal
            (2, 15)     # Stop loss % (réduit pour moins de pertes)
        ]
        
        for i, (min_val, max_val) in enumerate(bounds):
            if i < len(self.genes):
                self.genes[i] = max(min_val, min(max_val, self.genes[i]))
                # Arrondir les périodes à l'entier le plus proche
                if i < 9:
                    self.genes[i] = round(self.genes[i])
        
        # S'assurer que SMA court < SMA long
        if self.genes[0] >= self.genes[1]:
            self.genes[0] = max(5, int(self.genes[1] * 0.5))
        
        # S'assurer que MACD rapide < MACD lent
        if self.genes[6] >= self.genes[7]:
            self.genes[6] = max(5, int(self.genes[7] * 0.5))
    
    @property
    def sma_short_period(self) -> int:
        """Période de la SMA courte."""
        return int(self.genes[0])
    
    @property
    def sma_long_period(self) -> int:
        """Période de la SMA longue."""
        return int(self.genes[1])
    
    @property
    def ema_period(self) -> int:
        """Période de l'EMA."""
        return int(self.genes[2])
    
    @property
    def rsi_period(self) -> int:
        """Période du RSI."""
        return int(self.genes[3])
    
    @property
    def rsi_oversold(self) -> float:
        """Seuil de survente RSI."""
        return self.genes[4]
    
    @property
    def rsi_overbought(self) -> float:
        """Seuil de surachat RSI."""
        return self.genes[5]
    
    @property
    def macd_fast(self) -> int:
        """Période rapide du MACD."""
        return int(self.genes[6])
    
    @property
    def macd_slow(self) -> int:
        """Période lente du MACD."""
        return int(self.genes[7])
    
    @property
    def macd_signal(self) -> int:
        """Période du signal MACD."""
        return int(self.genes[8])
    
    @property
    def stop_loss_pct(self) -> float:
        """Pourcentage de stop loss."""
        return self.genes[9]
    
    def calculate_sma(self, data: pd.Series, period: int) -> pd.Series:
        """
        Calcule la moyenne mobile simple (SMA).
        
        Args:
            data: Série de prix
            period: Période de la moyenne
            
        Returns:
            Série contenant la SMA
        """
        return data.rolling(window=period).mean()
    
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """
        Calcule la moyenne mobile exponentielle (EMA).
        
        Args:
            data: Série de prix
            period: Période de la moyenne
            
        Returns:
            Série contenant l'EMA
        """
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, data: pd.Series, period: int) -> pd.Series:
        """
        Calcule l'indice de force relative (RSI).
        
        Args:
            data: Série de prix
            period: Période du RSI
            
        Returns:
            Série contenant le RSI
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, data: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calcule le MACD (Moving Average Convergence Divergence).
        
        Args:
            data: Série de prix
            
        Returns:
            Tuple (MACD, Signal, Histogramme)
        """
        ema_fast = self.calculate_ema(data, self.macd_fast)
        ema_slow = self.calculate_ema(data, self.macd_slow)
        
        macd = ema_fast - ema_slow
        signal = self.calculate_ema(macd, self.macd_signal)
        histogram = macd - signal
        
        return macd, signal, histogram
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Génère les signaux de trading basés sur les indicateurs.
        
        Args:
            df: DataFrame contenant les données OHLCV
            
        Returns:
            DataFrame avec les indicateurs et signaux
        """
        df = df.copy()
        
        # Calcul des indicateurs
        df['sma_short'] = self.calculate_sma(df['Close'], self.sma_short_period)
        df['sma_long'] = self.calculate_sma(df['Close'], self.sma_long_period)
        df['ema'] = self.calculate_ema(df['Close'], self.ema_period)
        df['rsi'] = self.calculate_rsi(df['Close'], self.rsi_period)
        
        macd, signal, histogram = self.calculate_macd(df['Close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = histogram
        
        # Génération des signaux
        df['signal'] = Signal.HOLD.value
        
        # Signal basé sur le croisement SMA (seulement si les SMA sont valides)
        sma_cross_up = (df['sma_short'] > df['sma_long']) & (df['sma_short'].shift(1) <= df['sma_long'].shift(1))
        sma_cross_down = (df['sma_short'] < df['sma_long']) & (df['sma_short'].shift(1) >= df['sma_long'].shift(1))
        
        # Signal basé sur le RSI (seulement si RSI est valide)
        rsi_buy = (df['rsi'] < self.rsi_oversold) & (df['rsi'].shift(1) >= self.rsi_oversold)
        rsi_sell = (df['rsi'] > self.rsi_overbought) & (df['rsi'].shift(1) <= self.rsi_overbought)
        
        # Signal basé sur le MACD (seulement si MACD est valide)
        macd_cross_up = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        macd_cross_down = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        
        # Filtre de tendance : prix au-dessus de la SMA longue = tendance haussière
        uptrend = df['Close'] > df['sma_long']
        downtrend = df['Close'] < df['sma_long']
        
        # Combinaison des signaux avec filtre de tendance plus flexible
        # Achat : croisement SMA haussier (sans filtre de tendance) OU (RSI survente en tendance haussière) OU (MACD croise haussier)
        df.loc[sma_cross_up, 'signal'] = Signal.BUY.value
        df.loc[rsi_buy & uptrend, 'signal'] = Signal.BUY.value
        df.loc[macd_cross_up, 'signal'] = Signal.BUY.value
        
        # Vente : croisement SMA baissier (sans filtre de tendance) OU (RSI surachat en tendance baissière) OU (MACD croise baissier)
        df.loc[sma_cross_down, 'signal'] = Signal.SELL.value
        df.loc[rsi_sell & downtrend, 'signal'] = Signal.SELL.value
        df.loc[macd_cross_down, 'signal'] = Signal.SELL.value
        
        # Après un signal BUY ou SELL, mettre HOLD pour permettre de ré-entrer
        # Cela évite le problème où le signal reste le même et empêche de nouvelles positions
        signal_values = df['signal'].values.copy()
        for i in range(1, len(df)):
            if signal_values[i] != Signal.HOLD.value:
                # Mettre HOLD sur les jours suivants jusqu'au prochain signal
                j = i + 1
                while j < len(df) and signal_values[j] == signal_values[i]:
                    signal_values[j] = Signal.HOLD.value
                    j += 1
        df['signal'] = signal_values
        
        return df
    
    def get_parameters(self) -> Dict:
        """
        Retourne les paramètres de la stratégie sous forme de dictionnaire.
        
        Returns:
            Dictionnaire des paramètres
        """
        return {
            'sma_short_period': self.sma_short_period,
            'sma_long_period': self.sma_long_period,
            'ema_period': self.ema_period,
            'rsi_period': self.rsi_period,
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought,
            'macd_fast': self.macd_fast,
            'macd_slow': self.macd_slow,
            'macd_signal': self.macd_signal,
            'stop_loss_pct': self.stop_loss_pct
        }
    
    def __repr__(self) -> str:
        """Représentation textuelle de la stratégie."""
        params = self.get_parameters()
        return f"TradingStrategy({params})"
    
    @classmethod
    def from_genes(cls, genes: np.ndarray) -> 'TradingStrategy':
        """
        Crée une stratégie à partir d'un chromosome.
        
        Args:
            genes: Chromosome contenant les paramètres
            
        Returns:
            Instance de TradingStrategy
        """
        return cls(genes=genes)
    
    def to_genes(self) -> np.ndarray:
        """
        Retourne le chromosome représentant la stratégie.
        
        Returns:
            Tableau numpy des gènes
        """
        return self.genes.copy()
