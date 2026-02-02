"""
Stratégies heuristiques pour le solveur Wordle

Ce module implémente plusieurs heuristiques d'IA EXPLORATOIRE
pour choisir le meilleur mot à proposer à chaque étape.

Heuristiques implémentées:
1. Baseline naïve: premier mot du dictionnaire
2. Fréquence de lettres: maximise les lettres les plus fréquentes
3. Entropie (information gain): maximise l'information gagnée

Ces heuristiques permettent de minimiser le nombre de coups
nécessaires pour trouver le mot secret.
"""

from typing import List, Dict, Set
from collections import Counter
import math
import logging

from .wordle_feedback import compute_feedback

logger = logging.getLogger(__name__)


class Strategy:
    """Classe de base pour les stratégies."""

    def __init__(self, name: str):
        self.name = name

    def select_word(self, candidates: List[str], all_words: Set[str]) -> str:
        """
        Sélectionne le meilleur mot selon la stratégie.

        Args:
            candidates: mots candidats restants
            all_words: dictionnaire complet (pour hard mode)

        Returns:
            Mot sélectionné
        """
        raise NotImplementedError


class NaiveStrategy(Strategy):
    """
    Stratégie naïve: retourne le premier mot alphabétiquement.

    Baseline pour comparaison.
    """

    def __init__(self):
        super().__init__("Naive")

    def select_word(self, candidates: List[str], all_words: Set[str]) -> str:
        """Retourne le premier mot alphabétiquement."""
        if not candidates:
            raise ValueError("Aucun candidat disponible")
        return sorted(candidates)[0]


class FrequencyStrategy(Strategy):
    """
    Stratégie basée sur la fréquence des lettres.

    Principe:
    - Calculer la fréquence de chaque lettre dans les candidats restants
    - Choisir le mot qui contient les lettres les plus fréquentes
    - Bonus pour la diversité des lettres (éviter les doublons)

    Complexité: O(n * m) où n = nombre de candidats, m = longueur du mot
    """

    def __init__(self):
        super().__init__("Frequency")

    def select_word(self, candidates: List[str], all_words: Set[str]) -> str:
        """Sélectionne le mot avec le meilleur score de fréquence."""
        if not candidates:
            raise ValueError("Aucun candidat disponible")

        # Calculer les fréquences des lettres dans les candidats
        letter_freq = self._compute_letter_frequencies(candidates)

        # Scorer chaque mot
        best_word = None
        best_score = -1

        for word in candidates:
            score = self._score_word(word, letter_freq)
            if score > best_score:
                best_score = score
                best_word = word

        logger.debug(f"Frequency strategy: {best_word} (score: {best_score:.2f})")
        return best_word

    def _compute_letter_frequencies(self, words: List[str]) -> Dict[str, float]:
        """
        Calcule la fréquence de chaque lettre.

        Returns:
            Dictionnaire {lettre: fréquence normalisée}
        """
        counter = Counter()
        for word in words:
            # Compter chaque lettre unique dans le mot
            for letter in set(word):
                counter[letter] += 1

        total = sum(counter.values())
        return {letter: count / total for letter, count in counter.items()}

    def _score_word(self, word: str, freq: Dict[str, float]) -> float:
        """
        Calcule le score d'un mot basé sur les fréquences.

        Score = somme des fréquences des lettres uniques
        Bonus: +10% si toutes les lettres sont différentes
        """
        unique_letters = set(word)
        score = sum(freq.get(letter, 0) for letter in unique_letters)

        # Bonus diversité
        if len(unique_letters) == 5:
            score *= 1.1

        return score


class EntropyStrategy(Strategy):
    """
    Stratégie basée sur l'entropie (gain d'information).

    Principe:
    - Pour chaque mot possible, simuler tous les feedbacks possibles
    - Calculer l'entropie: H = -Σ p(feedback) * log2(p(feedback))
    - Choisir le mot qui maximise l'entropie moyenne

    L'entropie mesure "l'information gagnée" par un coup.
    Plus l'entropie est élevée, plus le coup est informatif.

    Complexité: O(n² * m) où n = nombre de candidats
    ATTENTION: Coûteux en calcul, utiliser avec parcimonie ou sampling
    """

    def __init__(self, max_candidates_for_full_search: int = 100):
        super().__init__("Entropy")
        self.max_candidates_for_full_search = max_candidates_for_full_search

    def select_word(self, candidates: List[str], all_words: Set[str]) -> str:
        """Sélectionne le mot avec la meilleure entropie."""
        if not candidates:
            raise ValueError("Aucun candidat disponible")

        # Si trop de candidats, utiliser un échantillon ou fallback
        if len(candidates) > self.max_candidates_for_full_search:
            logger.warning(
                f"Trop de candidats ({len(candidates)}). "
                f"Utilisation de la stratégie fréquence."
            )
            # Fallback sur fréquence
            return FrequencyStrategy().select_word(candidates, all_words)

        best_word = None
        best_entropy = -1

        # Tester chaque mot candidat
        for word in candidates:
            entropy = self._compute_entropy(word, candidates)
            if entropy > best_entropy:
                best_entropy = entropy
                best_word = word

        logger.debug(f"Entropy strategy: {best_word} (entropy: {best_entropy:.2f})")
        return best_word

    def _compute_entropy(self, guess: str, candidates: List[str]) -> float:
        """
        Calcule l'entropie attendue pour un mot.

        Pour chaque candidat secret possible:
        1. Calculer le feedback que produirait 'guess'
        2. Grouper par feedback
        3. Calculer H = -Σ p_i * log2(p_i)

        Args:
            guess: mot à tester
            candidates: liste des secrets possibles

        Returns:
            Entropie en bits
        """
        # Grouper les candidats par feedback
        feedback_groups = {}

        for secret in candidates:
            feedback = compute_feedback(guess, secret)
            if feedback not in feedback_groups:
                feedback_groups[feedback] = 0
            feedback_groups[feedback] += 1

        # Calculer l'entropie
        total = len(candidates)
        entropy = 0.0

        for count in feedback_groups.values():
            probability = count / total
            if probability > 0:
                entropy -= probability * math.log2(probability)

        return entropy


class MixedStrategy(Strategy):
    """
    Stratégie mixte: combine fréquence et entropie.

    Utilise l'entropie si peu de candidats, sinon fréquence.
    """

    def __init__(self, entropy_threshold: int = 50):
        super().__init__("Mixed")
        self.entropy_threshold = entropy_threshold
        self.entropy_strategy = EntropyStrategy()
        self.frequency_strategy = FrequencyStrategy()

    def select_word(self, candidates: List[str], all_words: Set[str]) -> str:
        """Sélectionne la stratégie appropriée selon le contexte."""
        if len(candidates) <= self.entropy_threshold:
            logger.debug(f"Utilisation de l'entropie ({len(candidates)} candidats)")
            return self.entropy_strategy.select_word(candidates, all_words)
        else:
            logger.debug(f"Utilisation de la fréquence ({len(candidates)} candidats)")
            return self.frequency_strategy.select_word(candidates, all_words)


# Dictionnaire des stratégies disponibles
STRATEGIES = {
    'naive': NaiveStrategy(),
    'frequency': FrequencyStrategy(),
    'entropy': EntropyStrategy(),
    'mixed': MixedStrategy()
}


def get_strategy(name: str) -> Strategy:
    """
    Retourne une stratégie par son nom.

    Args:
        name: nom de la stratégie ('naive', 'frequency', 'entropy', 'mixed')

    Returns:
        Instance de Strategy

    Raises:
        ValueError: si le nom est invalide
    """
    if name not in STRATEGIES:
        raise ValueError(
            f"Stratégie inconnue: {name}. "
            f"Disponibles: {', '.join(STRATEGIES.keys())}"
        )
    return STRATEGIES[name]


def suggest_first_word() -> str:
    """
    Suggère un bon premier mot (analyse pré-calculée).

    Mots optimaux pour le français:
    - AROSE, STARE (anglais)
    - AIDER, ARBRE, CARTE (français)

    Returns:
        Premier mot recommandé
    """
    # Mot choisi empiriquement: bonnes lettres fréquentes, diversité
    return "AROSE"
