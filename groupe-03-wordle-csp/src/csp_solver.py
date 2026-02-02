"""
Solveur CSP pour Wordle

Ce module implémente la modélisation CSP (Constraint Satisfaction Problem)
du jeu Wordle. Le CSP est formulé comme suit:

VARIABLES:
- Approche choisie: variable unique dont le domaine est l'ensemble des mots candidats
- Alternative: 5 variables (pos_0, pos_1, ..., pos_4) où chaque variable a pour
  domaine l'ensemble des lettres possibles à cette position

DOMAINE:
- Initialement: tous les mots du dictionnaire français (5 lettres)
- Après chaque feedback: domaine réduit par propagation des contraintes

CONTRAINTES:
1. Lettres exactes (vertes): la lettre i doit être X à la position p
2. Lettres présentes (jaunes): la lettre X doit être présente mais PAS à la position p
3. Lettres absentes (grises): la lettre X ne doit apparaître nulle part
4. Contrainte de multiplicité: gestion rigoureuse des lettres répétées

MÉTHODE DE RÉSOLUTION:
- Filtrage (arc-consistency) plutôt que backtracking
- Approche efficace pour Wordle: le domaine se réduit rapidement
"""

from typing import List, Set, Dict
from .wordle_feedback import is_compatible
import logging

logger = logging.getLogger(__name__)


class WordleCSPSolver:
    """
    Solveur CSP pour Wordle.

    Le solveur maintient un ensemble de mots candidats et applique
    des contraintes à chaque itération pour réduire l'espace de recherche.
    """

    def __init__(self, dictionary_path: str):
        """
        Initialise le solveur avec un dictionnaire.

        Args:
            dictionary_path: chemin vers le fichier de mots (un mot par ligne)
        """
        self.dictionary_path = dictionary_path
        self.initial_candidates = self._load_dictionary()
        self.candidates = self.initial_candidates.copy()
        self.constraints = []  # Liste de (guess, feedback)

        logger.info(f"Solveur CSP initialisé avec {len(self.initial_candidates)} mots")

    def _load_dictionary(self) -> Set[str]:
        """
        Charge le dictionnaire depuis le fichier.

        Returns:
            Ensemble de mots normalisés (majuscules, 5 lettres)
        """
        try:
            with open(self.dictionary_path, 'r', encoding='utf-8') as f:
                words = {line.strip().upper() for line in f if len(line.strip()) == 5}
            return words
        except FileNotFoundError:
            logger.error(f"Dictionnaire non trouvé: {self.dictionary_path}")
            raise
        except Exception as e:
            logger.error(f"Erreur lors du chargement du dictionnaire: {e}")
            raise

    def reset(self):
        """Réinitialise le solveur à l'état initial."""
        self.candidates = self.initial_candidates.copy()
        self.constraints = []
        logger.info("Solveur réinitialisé")

    def add_constraint(self, guess: str, feedback: str) -> int:
        """
        Ajoute une contrainte et filtre les candidats.

        Cette méthode implémente la propagation de contraintes (arc-consistency).
        Chaque mot candidat est testé: s'il était le secret, produirait-il
        le même feedback pour 'guess' ?

        Args:
            guess: mot proposé
            feedback: feedback reçu (ex: 'GGYBB')

        Returns:
            Nombre de candidats restants après filtrage

        Raises:
            ValueError: si le feedback est invalide
        """
        guess = guess.upper()
        feedback = feedback.upper()

        if len(guess) != 5 or len(feedback) != 5:
            raise ValueError("Le mot et le feedback doivent faire 5 caractères")

        if not all(c in 'GYB' for c in feedback):
            raise ValueError("Le feedback doit contenir uniquement G, Y, B")

        # Ajouter la contrainte à l'historique
        self.constraints.append((guess, feedback))

        # Filtrer: ne garder que les mots compatibles
        previous_count = len(self.candidates)
        self.candidates = {
            word for word in self.candidates
            if is_compatible(word, guess, feedback)
        }
        new_count = len(self.candidates)

        logger.info(
            f"Contrainte ajoutée: {guess} -> {feedback}. "
            f"Candidats: {previous_count} -> {new_count}"
        )

        return new_count

    def get_candidates(self) -> List[str]:
        """
        Retourne la liste des mots candidats actuels.

        Returns:
            Liste triée des candidats
        """
        return sorted(self.candidates)

    def get_candidate_count(self) -> int:
        """
        Retourne le nombre de candidats restants.

        Returns:
            Nombre de candidats
        """
        return len(self.candidates)

    def is_solved(self) -> bool:
        """
        Vérifie si le problème est résolu (un seul candidat).

        Returns:
            True si un seul mot est possible
        """
        return len(self.candidates) == 1

    def get_solution(self) -> str:
        """
        Retourne la solution si le problème est résolu.

        Returns:
            Le mot solution

        Raises:
            ValueError: si le problème n'est pas résolu ou incohérent
        """
        if len(self.candidates) == 0:
            raise ValueError("Aucun candidat restant: contraintes incohérentes")
        if len(self.candidates) > 1:
            raise ValueError(f"{len(self.candidates)} candidats restants: problème non résolu")

        return list(self.candidates)[0]

    def get_statistics(self) -> Dict:
        """
        Retourne des statistiques sur l'état du solveur.

        Returns:
            Dictionnaire avec les statistiques
        """
        return {
            'initial_candidates': len(self.initial_candidates),
            'current_candidates': len(self.candidates),
            'reduction_rate': 1 - (len(self.candidates) / len(self.initial_candidates)),
            'constraints_applied': len(self.constraints),
            'is_solved': self.is_solved()
        }


def analyze_constraint_strength(dictionary_path: str, guess: str, feedback: str) -> Dict:
    """
    Analyse la force d'une contrainte (combien de mots elle élimine).

    Fonction utilitaire pour l'analyse théorique.

    Args:
        dictionary_path: chemin du dictionnaire
        guess: mot proposé
        feedback: feedback donné

    Returns:
        Statistiques sur l'efficacité de cette contrainte
    """
    solver = WordleCSPSolver(dictionary_path)
    initial_count = solver.get_candidate_count()
    solver.add_constraint(guess, feedback)
    final_count = solver.get_candidate_count()

    return {
        'guess': guess,
        'feedback': feedback,
        'initial_count': initial_count,
        'final_count': final_count,
        'eliminated': initial_count - final_count,
        'elimination_rate': (initial_count - final_count) / initial_count if initial_count > 0 else 0
    }
