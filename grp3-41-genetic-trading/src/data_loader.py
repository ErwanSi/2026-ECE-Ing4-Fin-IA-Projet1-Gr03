"""
Module de chargement des données financières.

Ce module contient la classe DataLoader qui permet de charger des données
historiques depuis différentes sources (yfinance, fichiers CSV).
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from datetime import datetime, timedelta
import os


class DataLoader:
    """
    Classe pour charger et prétraiter les données financières.
    
    Supporte le chargement depuis :
    - API Yahoo Finance (yfinance)
    - Fichiers CSV locaux
    """
    
    def __init__(self):
        """Initialise le loader de données."""
        self.data: Optional[pd.DataFrame] = None
        self.symbol: Optional[str] = None
    
    def load_from_yfinance(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "2y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Charge les données depuis Yahoo Finance.
        
        Args:
            symbol: Symbole de l'actif (ex: 'AAPL', '^GSPC')
            start_date: Date de début (format 'YYYY-MM-DD')
            end_date: Date de fin (format 'YYYY-MM-DD')
            period: Période prédéfinie ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Intervalle des données ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            
        Returns:
            DataFrame contenant les données OHLCV
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "Le package 'yfinance' est requis. "
                "Installez-le avec: pip install yfinance"
            )
        
        # Définir les dates si non spécifiées
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Télécharger les données
        ticker = yf.Ticker(symbol)
        
        if start_date and end_date:
            df = ticker.history(start=start_date, end=end_date, interval=interval)
        else:
            df = ticker.history(period=period, interval=interval)
        
        # Nettoyer et formater les données
        df = self._clean_data(df)
        
        self.data = df
        self.symbol = symbol
        
        return df
    
    def load_from_csv(
        self,
        filepath: str,
        date_column: str = "Date",
        date_format: str = "%Y-%m-%d"
    ) -> pd.DataFrame:
        """
        Charge les données depuis un fichier CSV.
        
        Args:
            filepath: Chemin vers le fichier CSV
            date_column: Nom de la colonne contenant les dates
            date_format: Format des dates
            
        Returns:
            DataFrame contenant les données OHLCV
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Le fichier {filepath} n'existe pas.")
        
        # Charger le CSV
        df = pd.read_csv(filepath)
        
        # Convertir la colonne de date
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column], format=date_format)
            df.set_index(date_column, inplace=True)
        
        # Nettoyer et formater les données
        df = self._clean_data(df)
        
        self.data = df
        self.symbol = os.path.basename(filepath).split('.')[0]
        
        return df
    
    def load_sample_data(self, symbol: str = "SAMPLE") -> pd.DataFrame:
        """
        Génère des données de test simulées.
        
        Args:
            symbol: Nom du symbole simulé
            
        Returns:
            DataFrame contenant des données OHLCV simulées
        """
        # Paramètres de simulation
        np.random.seed(42)
        n_days = 500
        start_date = datetime.now() - timedelta(days=n_days)
        
        dates = pd.date_range(start=start_date, periods=n_days, freq='D')
        
        # Simulation d'un mouvement brownien géométrique
        dt = 1/252  # Pas de temps (jours de trading)
        mu = 0.1    # Drift annuel
        sigma = 0.2 # Volatilité annuelle
        
        # Prix initial
        S0 = 100
        
        # Génération des rendements
        returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), n_days)
        prices = S0 * np.exp(np.cumsum(returns))
        
        # Création des OHLC
        df = pd.DataFrame(index=dates)
        df['Close'] = prices
        
        # Simulation des High/Low/Open
        daily_range = prices * np.random.uniform(0.01, 0.03, n_days)
        df['High'] = prices + daily_range / 2
        df['Low'] = prices - daily_range / 2
        df['Open'] = df['Close'].shift(1).fillna(S0)
        
        # Volume simulé
        df['Volume'] = np.random.randint(1000000, 10000000, n_days)
        
        # Nettoyer
        df = self._clean_data(df)
        
        self.data = df
        self.symbol = symbol
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie et formate les données.
        
        Args:
            df: DataFrame brut
            
        Returns:
            DataFrame nettoyé
        """
        # Renommer les colonnes si nécessaire
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'adj close': 'Adj Close',
            'volume': 'Volume'
        }
        df.columns = [column_mapping.get(col.lower(), col) for col in df.columns]
        
        # S'assurer que les colonnes requises existent
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Colonne manquante: {col}")
        
        # Supprimer les lignes avec des valeurs manquantes
        df = df.dropna()
        
        # S'assurer que l'index est un datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                raise ValueError(f"Impossible de convertir l'index en datetime: {e}")
        
        # Trier par date
        df = df.sort_index()
        
        return df
    
    def get_data(self) -> pd.DataFrame:
        """
        Retourne les données chargées.
        
        Returns:
            DataFrame contenant les données
            
        Raises:
            ValueError: Si aucune donnée n'a été chargée
        """
        if self.data is None:
            raise ValueError("Aucune donnée chargée. Utilisez d'abord load_from_yfinance ou load_from_csv.")
        return self.data.copy()
    
    def split_train_test(
        self,
        train_ratio: float = 0.7,
        validation_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Divise les données en ensembles d'entraînement, validation et test.
        
        Args:
            train_ratio: Proportion pour l'entraînement
            validation_ratio: Proportion pour la validation
            
        Returns:
            Tuple (train_data, validation_data, test_data)
        """
        if self.data is None:
            raise ValueError("Aucune donnée chargée.")
        
        n = len(self.data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + validation_ratio))
        
        train_data = self.data.iloc[:train_end]
        validation_data = self.data.iloc[train_end:val_end]
        test_data = self.data.iloc[val_end:]
        
        return train_data, validation_data, test_data
    
    def get_info(self) -> dict:
        """
        Retourne des informations sur les données chargées.
        
        Returns:
            Dictionnaire contenant les informations
        """
        if self.data is None:
            return {}
        
        return {
            'symbol': self.symbol,
            'start_date': self.data.index[0].strftime('%Y-%m-%d'),
            'end_date': self.data.index[-1].strftime('%Y-%m-%d'),
            'total_days': len(self.data),
            'columns': list(self.data.columns),
            'price_range': {
                'min': float(self.data['Low'].min()),
                'max': float(self.data['High'].max()),
                'current': float(self.data['Close'].iloc[-1])
            }
        }
    
    def resample(self, freq: str = 'W') -> pd.DataFrame:
        """
        Rééchantillonne les données à une fréquence différente.
        
        Args:
            freq: Fréquence de rééchantillonnage ('D', 'W', 'M', 'Q', 'Y')
            
        Returns:
            DataFrame rééchantillonné
        """
        if self.data is None:
            raise ValueError("Aucune donnée chargée.")
        
        agg_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }
        
        return self.data.resample(freq).agg(agg_dict).dropna()
    
    def add_features(self) -> pd.DataFrame:
        """
        Ajoute des caractéristiques techniques aux données.
        
        Returns:
            DataFrame avec les caractéristiques ajoutées
        """
        if self.data is None:
            raise ValueError("Aucune donnée chargée.")
        
        df = self.data.copy()
        
        # Rendements
        df['Returns'] = df['Close'].pct_change()
        
        # Rendements logarithmiques
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Volatilité glissante
        df['Volatility_20'] = df['Returns'].rolling(window=20).std()
        
        # Range (High - Low)
        df['Range'] = df['High'] - df['Low']
        df['Range_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        
        # Volume moyen
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
        
        return df.dropna()
