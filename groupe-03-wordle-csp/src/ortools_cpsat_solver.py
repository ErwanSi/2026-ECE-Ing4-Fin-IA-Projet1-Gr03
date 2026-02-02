"""
Solveur Wordle basé sur OR-Tools CP-SAT.
Alternative au filtrage CSP pour comparaison de performances.
"""

from typing import Set, List, Dict, Tuple, Optional
from pathlib import Path
from ortools.sat.python import cp_model
from .constraints import feedback_to_constraints, merge_constraints


class WordleORToolsSolver:
    """
    Solveur Wordle utilisant Google OR-Tools CP-SAT.
    
    Interface compatible avec WordleCSPSolver pour interchangeabilité.
    """
    
    def __init__(self, dictionary_path: str):
        """
        Initialise le solveur avec un dictionnaire.
        
        Args:
            dictionary_path: Chemin vers le fichier dictionnaire (1 mot par ligne, 5 lettres)
        """
        self.dictionary_path = Path(dictionary_path)
        self.initial_candidates = self._load_dictionary()
        self.candidates = self.initial_candidates.copy()
        self.constraints_history = []  # Liste de (guess, feedback)
        
    def _load_dictionary(self) -> Set[str]:
        """Charge le dictionnaire depuis le fichier."""
        words = set()
        with open(self.dictionary_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().upper()
                if len(word) == 5 and word.isalpha():
                    words.add(word)
        return words
    
    def add_constraint(self, guess: str, feedback: str) -> int:
        """
        Ajoute une contrainte et filtre les candidats.
        
        Args:
            guess: Mot proposé (5 lettres)
            feedback: Feedback (5 caractères G/Y/B)
        
        Returns:
            Nombre de candidats restants
        
        Raises:
            ValueError: Si le guess ou feedback est invalide
        """
        guess = guess.upper()
        feedback = feedback.upper()
        
        if len(guess) != 5 or not guess.isalpha():
            raise ValueError(f"Le guess '{guess}' doit contenir exactement 5 lettres")
        
        if len(feedback) != 5 or not all(c in 'GYB' for c in feedback):
            raise ValueError(f"Le feedback '{feedback}' doit contenir 5 caractères G/Y/B")
        
        # Ajouter à l'historique
        self.constraints_history.append((guess, feedback))
        
        # Extraire les contraintes
        new_constraints = feedback_to_constraints(guess, feedback)
        
        # Fusionner avec les contraintes existantes
        all_constraints = [feedback_to_constraints(g, f) for g, f in self.constraints_history]
        merged = merge_constraints(all_constraints)
        
        # Filtrer les candidats qui satisfont toutes les contraintes
        self.candidates = {
            word for word in self.candidates
            if self._satisfies_constraints(word, merged)
        }
        
        return len(self.candidates)
    
    def _satisfies_constraints(self, word: str, constraints: Dict) -> bool:
        """
        Vérifie si un mot satisfait toutes les contraintes.
        
        Args:
            word: Mot à vérifier (5 lettres, majuscules)
            constraints: Dictionnaire de contraintes
        
        Returns:
            True si le mot satisfait toutes les contraintes
        """
        from collections import Counter
        
        word = word.upper()
        letter_count = Counter(word)
        
        # 1. Vérifier positions exactes (vertes)
        for pos, letter in constraints.get('exact', {}).items():
            if word[pos] != letter:
                return False
        
        # 2. Vérifier lettres absentes (grises)
        for letter in constraints.get('absent', set()):
            if letter in word:
                return False
        
        # 3. Vérifier lettres présentes (jaunes)
        for letter, forbidden_positions in constraints.get('present', {}).items():
            # La lettre doit être dans le mot
            if letter not in word:
                return False
            # Mais pas aux positions interdites
            for pos in forbidden_positions:
                if word[pos] == letter:
                    return False
        
        # 4. Vérifier nombre minimum d'occurrences
        for letter, min_count in constraints.get('min_count', {}).items():
            if letter_count.get(letter, 0) < min_count:
                return False
        
        # 5. Vérifier nombre maximum d'occurrences
        for letter, max_count in constraints.get('max_count', {}).items():
            if letter_count.get(letter, 0) > max_count:
                return False
        
        return True
    
    def get_candidates(self) -> List[str]:
        """Retourne la liste triée des candidats restants."""
        return sorted(list(self.candidates))
    
    def get_candidate_count(self) -> int:
        """Retourne le nombre de candidats restants."""
        return len(self.candidates)
    
    def is_solved(self) -> bool:
        """Retourne True si un seul candidat reste."""
        return len(self.candidates) == 1
    
    def get_solution(self) -> Optional[str]:
        """
        Retourne la solution si unique, None sinon.
        
        Returns:
            Le mot solution si unique, None sinon
        """
        if self.is_solved():
            return list(self.candidates)[0]
        return None
    
    def reset(self):
        """Réinitialise le solveur à l'état initial."""
        self.candidates = self.initial_candidates.copy()
        self.constraints_history.clear()
    
    def get_statistics(self) -> Dict:
        """
        Retourne des statistiques sur l'état actuel.
        
        Returns:
            Dictionnaire avec:
                - total_words: Nombre de mots dans le dictionnaire
                - candidates_remaining: Nombre de candidats restants
                - constraints_applied: Nombre de contraintes appliquées
                - solved: True si résolu
        """
        return {
            'total_words': len(self.initial_candidates),
            'candidates_remaining': len(self.candidates),
            'constraints_applied': len(self.constraints_history),
            'solved': self.is_solved(),
            'solver_type': 'ortools_cpsat'
        }
