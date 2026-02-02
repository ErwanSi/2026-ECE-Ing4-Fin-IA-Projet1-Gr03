"""
Calcul du feedback Wordle (vert/jaune/gris)

Ce module implémente la logique canonique de Wordle pour calculer
le feedback d'une proposition par rapport à un mot secret.

Codes de feedback:
- 'G' (Green/Vert): lettre correcte à la bonne position
- 'Y' (Yellow/Jaune): lettre correcte mais mal placée
- 'B' (Black/Gris): lettre absente ou déjà utilisée

ATTENTION: Gestion rigoureuse des lettres répétées.
"""

from typing import List
from collections import Counter


def compute_feedback(guess: str, secret: str) -> str:
    """
    Calcule le feedback Wordle pour une proposition.

    Algorithme (conforme aux règles officielles Wordle):
    1. D'abord, identifier toutes les lettres exactes (vertes)
    2. Ensuite, pour chaque lettre non-verte, vérifier si elle peut être jaune
       en tenant compte du nombre d'occurrences restantes dans le secret
    3. Sinon, la lettre est grise

    Args:
        guess: mot proposé (5 lettres)
        secret: mot secret (5 lettres)

    Returns:
        String de 5 caractères parmi 'G', 'Y', 'B'

    Exemple:
        >>> compute_feedback("GREVE", "GERER")
        'GBGYG'
        # G bien placé, R déjà utilisé (gris), E mal placé, V absent, E bien placé
    """
    if len(guess) != 5 or len(secret) != 5:
        raise ValueError("Les mots doivent faire exactement 5 lettres")

    guess = guess.upper()
    secret = secret.upper()

    feedback = ['B'] * 5  # Par défaut: toutes grises
    secret_counts = Counter(secret)  # Compte les occurrences de chaque lettre

    # Phase 1: Marquer les lettres exactes (vertes)
    for i in range(5):
        if guess[i] == secret[i]:
            feedback[i] = 'G'
            secret_counts[guess[i]] -= 1  # Consommer cette occurrence

    # Phase 2: Marquer les lettres mal placées (jaunes)
    for i in range(5):
        if feedback[i] == 'B':  # Pas déjà verte
            if guess[i] in secret_counts and secret_counts[guess[i]] > 0:
                feedback[i] = 'Y'
                secret_counts[guess[i]] -= 1  # Consommer cette occurrence

    return ''.join(feedback)


def is_compatible(word: str, guess: str, feedback: str) -> bool:
    """
    Vérifie si un mot est compatible avec un couple (guess, feedback).

    Cette fonction est utilisée pour filtrer les candidats possibles
    après avoir reçu un feedback.

    Args:
        word: mot candidat à tester
        guess: mot qui a été proposé
        feedback: feedback reçu ('GGYBB' par exemple)

    Returns:
        True si le mot est compatible avec cette contrainte

    Exemple:
        >>> is_compatible("GERER", "GREVE", "GBGYG")
        True
    """
    if len(word) != 5 or len(guess) != 5 or len(feedback) != 5:
        return False

    # Le feedback que produirait 'word' en tant que secret doit être identique
    return compute_feedback(guess, word) == feedback


def feedback_to_constraints(guess: str, feedback: str) -> dict:
    """
    Convertit un feedback en contraintes exploitables par le CSP.

    Retourne un dictionnaire avec:
    - 'exact': dict {position: lettre} pour les lettres vertes
    - 'present': dict {lettre: [positions_interdites]} pour les lettres jaunes
    - 'absent': set des lettres grises (absentes)

    Args:
        guess: mot proposé
        feedback: feedback reçu

    Returns:
        Dictionnaire de contraintes structurées

    Exemple:
        >>> feedback_to_constraints("ARBRE", "BGYGB")
        {
            'exact': {2: 'B'},
            'present': {'R': [1]},
            'absent': {'A', 'E'}
        }
    """
    guess = guess.upper()
    feedback = feedback.upper()

    exact = {}      # position -> lettre
    present = {}    # lettre -> liste de positions où elle N'est PAS
    absent = set()  # lettres absentes

    # Lettres avec feedback vert ou jaune (présentes dans le secret)
    letters_in_secret = set()

    for i, (letter, fb) in enumerate(zip(guess, feedback)):
        if fb == 'G':
            exact[i] = letter
            letters_in_secret.add(letter)
        elif fb == 'Y':
            if letter not in present:
                present[letter] = []
            present[letter].append(i)
            letters_in_secret.add(letter)
        elif fb == 'B':
            # Attention: une lettre peut être grise ET présente ailleurs
            # On ne l'ajoute à absent que si elle n'est pas déjà présente
            if letter not in letters_in_secret:
                absent.add(letter)

    return {
        'exact': exact,
        'present': present,
        'absent': absent
    }


def format_feedback_display(guess: str, feedback: str) -> str:
    """
    Formate joliment un feedback pour affichage terminal.

    Utilise des couleurs ANSI si possible.
    """
    # Codes couleur ANSI
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    GRAY = '\033[90m'
    RESET = '\033[0m'

    result = []
    for letter, fb in zip(guess.upper(), feedback):
        if fb == 'G':
            result.append(f"{GREEN}{letter}{RESET}")
        elif fb == 'Y':
            result.append(f"{YELLOW}{letter}{RESET}")
        else:
            result.append(f"{GRAY}{letter}{RESET}")

    return ' '.join(result)
