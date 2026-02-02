"""
Module de dérivation des contraintes depuis les feedbacks Wordle.
Utilisé par le solveur OR-Tools CP-SAT.
"""

from typing import Dict, Set, List, Tuple
from collections import Counter


def feedback_to_constraints(guess: str, feedback: str) -> Dict[str, any]:
    """
    Convertit un feedback Wordle en contraintes structurées pour CP-SAT.
    
    Args:
        guess: Mot proposé (5 lettres, majuscules)
        feedback: Feedback (5 caractères G/Y/B)
            G (Green/Vert): Lettre correcte à la bonne position
            Y (Yellow/Jaune): Lettre correcte à la mauvaise position
            B (Black/Gris): Lettre absente
    
    Returns:
        Dictionnaire de contraintes:
        {
            'exact': {position: lettre},  # Positions exactes (vertes)
            'present': {lettre: [positions_interdites]},  # Lettres présentes mais mal placées
            'absent': {lettres},  # Lettres totalement absentes
            'min_count': {lettre: count},  # Nombre minimum d'occurrences
            'max_count': {lettre: count}   # Nombre maximum d'occurrences
        }
    
    Exemple:
        guess = "TARES"
        feedback = "BYGBB"
        -> {'exact': {}, 'present': {'A': [1]}, 'absent': {'T', 'R', 'E', 'S'}, ...}
    """
    guess = guess.upper()
    feedback = feedback.upper()
    
    if len(guess) != 5 or len(feedback) != 5:
        raise ValueError("Le guess et le feedback doivent contenir 5 caractères")
    
    if not all(c in 'GYB' for c in feedback):
        raise ValueError("Le feedback doit contenir uniquement G, Y ou B")
    
    constraints = {
        'exact': {},           # {position: lettre}
        'present': {},         # {lettre: [positions_interdites]}
        'absent': set(),       # {lettres}
        'min_count': {},       # {lettre: count}
        'max_count': {}        # {lettre: count}
    }
    
    # Compter les occurrences de feedback pour chaque lettre
    letter_feedback = {}  # {lettre: {'green': [positions], 'yellow': [positions], 'black': [positions]}}
    
    for i, (letter, fb) in enumerate(zip(guess, feedback)):
        if letter not in letter_feedback:
            letter_feedback[letter] = {'green': [], 'yellow': [], 'black': []}
        
        if fb == 'G':
            letter_feedback[letter]['green'].append(i)
        elif fb == 'Y':
            letter_feedback[letter]['yellow'].append(i)
        else:  # B
            letter_feedback[letter]['black'].append(i)
    
    # Analyser chaque lettre pour dériver les contraintes
    for letter, feedbacks in letter_feedback.items():
        green_count = len(feedbacks['green'])
        yellow_count = len(feedbacks['yellow'])
        black_count = len(feedbacks['black'])
        
        # Positions exactes (vertes)
        for pos in feedbacks['green']:
            constraints['exact'][pos] = letter
        
        # Lettres présentes mais mal placées (jaunes)
        if yellow_count > 0:
            if letter not in constraints['present']:
                constraints['present'][letter] = []
            constraints['present'][letter].extend(feedbacks['yellow'])
        
        # Contraintes de comptage
        if green_count > 0 or yellow_count > 0:
            # La lettre apparaît au moins green_count + yellow_count fois
            min_occurrences = green_count + yellow_count
            constraints['min_count'][letter] = min_occurrences
            
            # Si la lettre a aussi du gris, elle apparaît exactement min_occurrences fois
            if black_count > 0:
                constraints['max_count'][letter] = min_occurrences
        else:
            # Toutes les occurrences sont grises -> lettre absente
            constraints['absent'].add(letter)
    
    return constraints


def merge_constraints(constraints_list: List[Dict]) -> Dict:
    """
    Fusionne plusieurs ensembles de contraintes en un seul.
    
    Args:
        constraints_list: Liste de dictionnaires de contraintes
    
    Returns:
        Dictionnaire de contraintes fusionnées
    """
    merged = {
        'exact': {},
        'present': {},
        'absent': set(),
        'min_count': {},
        'max_count': {}
    }
    
    for constraints in constraints_list:
        # Exact positions (union)
        merged['exact'].update(constraints.get('exact', {}))
        
        # Present letters (union des positions interdites)
        for letter, positions in constraints.get('present', {}).items():
            if letter not in merged['present']:
                merged['present'][letter] = []
            merged['present'][letter].extend(positions)
            # Supprimer les doublons
            merged['present'][letter] = list(set(merged['present'][letter]))
        
        # Absent letters (union)
        merged['absent'].update(constraints.get('absent', set()))
        
        # Min count (maximum des minimums)
        for letter, count in constraints.get('min_count', {}).items():
            merged['min_count'][letter] = max(merged['min_count'].get(letter, 0), count)
        
        # Max count (minimum des maximums)
        for letter, count in constraints.get('max_count', {}).items():
            if letter in merged['max_count']:
                merged['max_count'][letter] = min(merged['max_count'][letter], count)
            else:
                merged['max_count'][letter] = count
    
    return merged
