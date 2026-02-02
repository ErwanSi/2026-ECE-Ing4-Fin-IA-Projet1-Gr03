"""
Package pour les stratégies de trading par algorithmes génétiques.

Projet ECE - Groupe 3 - Sujet n°41
"""

__version__ = "1.0.0"
__author__ = "Groupe 3"

from genetic_algorithm import GeneticAlgorithm
from trading_strategy import TradingStrategy
from backtester import Backtester
from fitness import FitnessCalculator
from data_loader import DataLoader
from visualizer import Visualizer

__all__ = [
    "GeneticAlgorithm",
    "TradingStrategy",
    "Backtester",
    "FitnessCalculator",
    "DataLoader",
]
