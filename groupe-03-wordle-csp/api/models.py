"""
Modèles Pydantic pour l'API Wordle CSP.

Ce module contient les schémas de validation pour les requêtes et réponses de l'API.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Literal
from datetime import datetime


class HealthResponse(BaseModel):
    """Réponse du endpoint /health pour vérifier l'état de l'API."""
    
    status: str = Field(..., description="État de l'API (ok ou error)")
    version: str = Field(..., description="Version de l'API")
    timestamp: datetime = Field(..., description="Timestamp de la vérification")
    dictionary_loaded: bool = Field(..., description="Indique si le dictionnaire est chargé")
    word_count: int = Field(..., description="Nombre de mots dans le dictionnaire")


class GameCreateRequest(BaseModel):
    """Requête de création de partie."""
    strategy: Literal["naive", "frequency", "entropy", "mixed"] = "mixed"
    solver: Literal["filtering", "cpsat"] = "filtering"  # cpsat sera ajouté plus tard

class GameCreateResponse(BaseModel):
    """Réponse création de partie."""
    game_id: str
    strategy: str
    solver: str
    total_words: int
    candidates_count: int
    first_suggestion: Optional[str] = None

class ConstraintRequest(BaseModel):
    """Requête d'ajout de contrainte."""
    guess: str = Field(..., min_length=5, max_length=5)
    feedback: str = Field(..., min_length=5, max_length=5)
    
    @validator("guess")
    def guess_uppercase(cls, v):
        return v.upper()
    
    @validator("feedback")
    def feedback_valid(cls, v):
        v = v.upper()
        if not all(c in "GYB" for c in v):
            raise ValueError("Le feedback doit contenir uniquement G, Y, B")
        return v

class ConstraintResponse(BaseModel):
    """Réponse après ajout de contrainte."""
    success: bool
    candidates_remaining: int
    solved: bool
    solution: Optional[str] = None
    top_candidates: List[str] = []

class SuggestRequest(BaseModel):
    """Requête de suggestion."""
    limit: int = Field(default=1, ge=1, le=10)

class SuggestResponse(BaseModel):
    """Réponse suggestion."""
    suggestions: List[str]
    candidates_count: int
    strategy_used: str

class AutoSimulateRequest(BaseModel):
    """Requête simulation automatique."""
    secret: str = Field(..., min_length=5, max_length=5)
    max_turns: int = Field(default=6, ge=1, le=20)
    
    @validator("secret")
    def secret_uppercase(cls, v):
        return v.upper()

class Turn(BaseModel):
    """Un tour de jeu."""
    turn_number: int
    guess: str
    feedback: str
    candidates_remaining: int

class AutoSimulateResponse(BaseModel):
    """Réponse simulation automatique."""
    success: bool
    secret: str
    turns: List[Turn]
    total_turns: int
    solved: bool
    final_guess: Optional[str] = None

class GameStateResponse(BaseModel):
    """État actuel d'une partie."""
    game_id: str
    strategy: str
    solver: str
    candidates_count: int
    constraints_count: int
    solved: bool
    solution: Optional[str] = None
