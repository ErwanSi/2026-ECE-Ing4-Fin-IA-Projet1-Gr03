"""Service de gestion des sessions de jeu."""
import uuid
from typing import Dict, Optional, Set
from pathlib import Path
import sys

# Import des modules existants
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.csp_solver import WordleCSPSolver
from src.ortools_cpsat_solver import WordleORToolsSolver
from src.strategy import get_strategy
from src.wordle_feedback import compute_feedback

class GameSession:
    """Session de jeu Wordle avec solveur CSP."""
    
    def __init__(self, game_id: str, dictionary_path: str, strategy_name: str, solver_type: str = "filtering"):
        self.game_id = game_id
        self.strategy_name = strategy_name
        self.solver_type = solver_type
        
        # Initialiser le solveur selon le type choisi
        if solver_type == "cpsat":
            self.solver = WordleORToolsSolver(dictionary_path)
        elif solver_type == "filtering":
            self.solver = WordleCSPSolver(dictionary_path)
        else:
            raise ValueError(f"Type de solveur inconnu: {solver_type}. Utilisez 'filtering' ou 'cpsat'.")
        
        # Initialiser la stratégie
        self.strategy = get_strategy(strategy_name)
        
        # Historique
        self.constraints = []
        self.total_words = len(self.solver.initial_candidates)
    
    def add_constraint(self, guess: str, feedback: str) -> Dict:
        """Ajouter une contrainte et retourner l'état."""
        remaining = self.solver.add_constraint(guess, feedback)
        self.constraints.append((guess, feedback))
        
        candidates = self.solver.get_candidates()
        solved = self.solver.is_solved()
        solution = self.solver.get_solution() if solved else None
        
        return {
            "success": True,
            "candidates_remaining": remaining,
            "solved": solved,
            "solution": solution,
            "top_candidates": candidates[:20] if not solved else []
        }
    
    def suggest(self, limit: int = 1) -> Dict:
        """Suggérer le(s) prochain(s) mot(s)."""
        candidates = self.solver.get_candidates()
        all_words = self.solver.initial_candidates
        
        if not candidates:
            return {
                "suggestions": [],
                "candidates_count": 0,
                "strategy_used": self.strategy_name
            }
        
        # Utiliser la stratégie pour sélectionner
        suggested = self.strategy.select_word(candidates, all_words)
        
        # Si limit > 1, prendre les premiers candidats
        suggestions = [suggested]
        if limit > 1 and len(candidates) > 1:
            for cand in candidates[:limit]:
                if cand not in suggestions:
                    suggestions.append(cand)
                if len(suggestions) >= limit:
                    break
        
        return {
            "suggestions": suggestions,
            "candidates_count": len(candidates),
            "strategy_used": self.strategy_name
        }
    
    def get_state(self) -> Dict:
        """Obtenir l'état actuel."""
        return {
            "game_id": self.game_id,
            "strategy": self.strategy_name,
            "solver": self.solver_type,
            "candidates_count": self.solver.get_candidate_count(),
            "constraints_count": len(self.constraints),
            "solved": self.solver.is_solved(),
            "solution": self.solver.get_solution() if self.solver.is_solved() else None
        }
    
    def reset(self):
        """Réinitialiser la session."""
        self.solver.reset()
        self.constraints.clear()


class GameSessionManager:
    """Gestionnaire de sessions de jeu (en mémoire pour simplicité)."""
    
    def __init__(self, dictionary_path: str):
        self.dictionary_path = dictionary_path
        self.sessions: Dict[str, GameSession] = {}
    
    def create_session(self, strategy: str = "mixed", solver: str = "filtering") -> GameSession:
        """Créer une nouvelle session."""
        game_id = str(uuid.uuid4())
        session = GameSession(game_id, self.dictionary_path, strategy, solver)
        self.sessions[game_id] = session
        return session
    
    def get_session(self, game_id: str) -> Optional[GameSession]:
        """Récupérer une session."""
        return self.sessions.get(game_id)
    
    def delete_session(self, game_id: str):
        """Supprimer une session."""
        if game_id in self.sessions:
            del self.sessions[game_id]
    
    def cleanup_old_sessions(self, max_sessions: int = 1000):
        """Nettoyer les vieilles sessions (simple: garder les N dernières)."""
        if len(self.sessions) > max_sessions:
            # Supprimer les plus anciennes
            to_remove = len(self.sessions) - max_sessions
            for game_id in list(self.sessions.keys())[:to_remove]:
                del self.sessions[game_id]
