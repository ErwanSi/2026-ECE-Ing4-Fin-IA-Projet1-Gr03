"""
Application FastAPI principale pour le solveur Wordle CSP.

Ce module contient:
- Configuration de l'application FastAPI
- Middleware CORS
- Endpoints système (/health, /)
- Chargement du dictionnaire au démarrage
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from datetime import datetime
from pathlib import Path
import logging

from .config import CORS_ORIGINS, API_TITLE, API_VERSION, API_DESCRIPTION, DICTIONARY_PATH
from .models import (
    HealthResponse,
    GameCreateRequest, GameCreateResponse,
    ConstraintRequest, ConstraintResponse,
    SuggestRequest, SuggestResponse,
    AutoSimulateRequest, AutoSimulateResponse, Turn,
    GameStateResponse
)
from .services import GameSessionManager, GameSession
from src.wordle_feedback import compute_feedback

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Application FastAPI
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# État global
dictionary_words = set()
session_manager = None


@app.on_event("startup")
async def startup_event():
    """Charger le dictionnaire et initialiser le gestionnaire de sessions au démarrage."""
    global dictionary_words, session_manager
    try:
        with open(DICTIONARY_PATH, 'r', encoding='utf-8') as f:
            dictionary_words = set(line.strip().upper() for line in f if len(line.strip()) == 5)
        logger.info(f"Dictionnaire chargé: {len(dictionary_words)} mots depuis {DICTIONARY_PATH}")
        
        # Initialiser le gestionnaire de sessions
        session_manager = GameSessionManager(DICTIONARY_PATH)
        logger.info("Gestionnaire de sessions initialisé")
    except Exception as e:
        logger.error(f"Erreur chargement dictionnaire: {e}")
        dictionary_words = set()


@app.get("/", include_in_schema=False)
async def root():
    """Rediriger vers la documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Vérifier l'état de santé de l'API."""
    return HealthResponse(
        status="ok",
        version=API_VERSION,
        timestamp=datetime.now(),
        dictionary_loaded=len(dictionary_words) > 0,
        word_count=len(dictionary_words)
    )


@app.post("/game/new", response_model=GameCreateResponse, tags=["Game"])
async def create_game(request: GameCreateRequest):
    """Créer une nouvelle partie."""
    session = session_manager.create_session(
        strategy=request.strategy,
        solver=request.solver
    )
    
    # Obtenir la première suggestion
    suggestion_data = session.suggest(limit=1)
    first_suggestion = suggestion_data["suggestions"][0] if suggestion_data["suggestions"] else None
    
    return GameCreateResponse(
        game_id=session.game_id,
        strategy=session.strategy_name,
        solver=session.solver_type,
        total_words=session.total_words,
        candidates_count=session.solver.get_candidate_count(),
        first_suggestion=first_suggestion
    )


@app.post("/game/{game_id}/constraint", response_model=ConstraintResponse, tags=["Game"])
async def add_constraint(game_id: str, request: ConstraintRequest):
    """Ajouter une contrainte (guess + feedback) à une partie."""
    session = session_manager.get_session(game_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Partie {game_id} non trouvée")
    
    result = session.add_constraint(request.guess, request.feedback)
    return ConstraintResponse(**result)


@app.get("/game/{game_id}/suggest", response_model=SuggestResponse, tags=["Game"])
async def get_suggestion(game_id: str, limit: int = 1):
    """Obtenir une suggestion de mot pour une partie."""
    session = session_manager.get_session(game_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Partie {game_id} non trouvée")
    
    result = session.suggest(limit=limit)
    return SuggestResponse(**result)


@app.get("/game/{game_id}/state", response_model=GameStateResponse, tags=["Game"])
async def get_game_state(game_id: str):
    """Obtenir l'état actuel d'une partie."""
    session = session_manager.get_session(game_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Partie {game_id} non trouvée")
    
    state = session.get_state()
    return GameStateResponse(**state)


@app.post("/game/{game_id}/reset", tags=["Game"])
async def reset_game(game_id: str):
    """Réinitialiser une partie."""
    session = session_manager.get_session(game_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Partie {game_id} non trouvée")
    
    session.reset()
    return {"success": True, "message": "Partie réinitialisée"}


@app.delete("/game/{game_id}", tags=["Game"])
async def delete_game(game_id: str):
    """Supprimer une partie."""
    session_manager.delete_session(game_id)
    return {"success": True, "message": "Partie supprimée"}


@app.post("/simulate", response_model=AutoSimulateResponse, tags=["Simulation"])
async def auto_simulate(request: AutoSimulateRequest):
    """Simuler une partie automatique avec un mot secret donné."""
    # Créer une session temporaire pour la simulation
    session = session_manager.create_session(strategy="mixed", solver="filtering")
    
    turns = []
    solved = False
    final_guess = None
    
    for turn_num in range(1, request.max_turns + 1):
        # Obtenir une suggestion
        suggestion_data = session.suggest(limit=1)
        if not suggestion_data["suggestions"]:
            break
        
        guess = suggestion_data["suggestions"][0]
        final_guess = guess
        
        # Calculer le feedback
        feedback = compute_feedback(guess, request.secret)
        
        # Ajouter la contrainte
        result = session.add_constraint(guess, feedback)
        
        turns.append(Turn(
            turn_number=turn_num,
            guess=guess,
            feedback=feedback,
            candidates_remaining=result["candidates_remaining"]
        ))
        
        # Vérifier si résolu
        if guess == request.secret:
            solved = True
            break
    
    # Nettoyer la session temporaire
    session_manager.delete_session(session.game_id)
    
    return AutoSimulateResponse(
        success=solved,
        secret=request.secret,
        turns=turns,
        total_turns=len(turns),
        solved=solved,
        final_guess=final_guess
    )
