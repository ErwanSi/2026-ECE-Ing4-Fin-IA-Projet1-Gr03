"""
Configuration pour l'API FastAPI du solveur Wordle CSP.

Ce module contient les paramètres de configuration pour:
- CORS (Cross-Origin Resource Sharing)
- Chemins des ressources (dictionnaire)
- Métadonnées de l'API
"""

from pathlib import Path
import os

# Chemins
PROJECT_ROOT = Path(__file__).parent.parent
DICTIONARY_PATH = os.getenv("DICTIONARY_PATH", str(PROJECT_ROOT / "data" / "mots_fr_5.txt"))

# CORS
CORS_ORIGINS = [
    "http://localhost:3000",  # React dev
    "http://localhost:5173",  # Vite dev
    "http://localhost:8080",  # Vue dev
]

# API
API_TITLE = "Wordle CSP Solver API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
API REST pour le solveur Wordle basé sur CSP (Constraint Satisfaction Problem).

Fonctionnalités:
- Résolution de Wordle par filtrage CSP
- Stratégies heuristiques (naive, frequency, entropy, mixed)
- Suggestions de mots basées sur l'état actuel
"""
