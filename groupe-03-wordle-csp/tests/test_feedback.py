"""
Tests unitaires pour le module wordle_feedback

Ces tests vérifient:
- Le calcul correct du feedback Wordle
- La gestion des lettres répétées (cas critique)
- La compatibilité mot/feedback
- La conversion en contraintes
"""

import pytest
from src.wordle_feedback import (
    compute_feedback,
    is_compatible,
    feedback_to_constraints
)


class TestComputeFeedback:
    """Tests du calcul de feedback."""

    def test_feedback_all_correct(self):
        """Test: toutes les lettres correctes."""
        assert compute_feedback("ARBRE", "ARBRE") == "GGGGG"
        assert compute_feedback("GERER", "GERER") == "GGGGG"

    def test_feedback_all_wrong(self):
        """Test: aucune lettre commune."""
        assert compute_feedback("ARBRE", "SOTTE") == "BBBBG"
        assert compute_feedback("PLUME", "BATON") == "BBBBB"

    def test_feedback_mixed(self):
        """Test: mélange de vert/jaune/gris."""
        # A correct, R mal placé, B absent, R correct, E correct
        assert compute_feedback("ARBRE", "AUTRE") == "GBBGG"

    def test_feedback_repeated_letters_case1(self):
        """
        Test critique: lettres répétées.

        Cas: GERER avec guess ARBRE
        - A: absent (gris)
        - R: mal placé (jaune) - le R en position 1 de GERER
        - B: absent (gris)
        - R: bien placé (vert) - le R en position 2 de GERER
        - E: mal placé (jaune) - un des E de GERER
        """
        result = compute_feedback("ARBRE", "GERER")
        assert result == "BYBYY"

    def test_feedback_repeated_letters_case2(self):
        """
        Test: deux lettres identiques dans guess, une seule dans secret.

        Exemple: guess TERRE, secret ARBRE
        - T: absent
        - E: mal placé (E en position 4 du secret)
        - R: bien placé
        - R: bien placé
        - E: absent (déjà utilisé)
        """
        result = compute_feedback("TERRE", "ARBRE")
        assert result == "BBYGG"

    def test_feedback_repeated_letters_case3(self):
        """
        Test: une lettre dans guess, plusieurs dans secret.

        Exemple: guess ARBRE, secretERRE
        - A: absent
        - R: mal placé (un des R de ERRE)
        - B: absent
        - R: bien placé
        - E: bien placé
        """
        result = compute_feedback("ARBRE", "ERRES")
        # A absent, R mal placé (existe ailleurs), B absent, R correct, E correct
        assert result == "BGBYY"

    def test_feedback_invalid_length(self):
        """Test: mots de longueur invalide."""
        with pytest.raises(ValueError):
            compute_feedback("ABC", "ARBRE")

        with pytest.raises(ValueError):
            compute_feedback("ARBRE", "ABCDEF")

    def test_feedback_case_insensitive(self):
        """Test: insensible à la casse."""
        assert compute_feedback("arbre", "ARBRE") == "GGGGG"
        assert compute_feedback("ArBrE", "aRbRe") == "GGGGG"


class TestIsCompatible:
    """Tests de compatibilité mot/feedback."""

    def test_compatible_exact_match(self):
        """Test: mot exact compatible."""
        assert is_compatible("ARBRE", "ARBRE", "GGGGG") is True

    def test_compatible_mixed(self):
        """Test: compatibilité mixte."""
        # Si on propose ARBRE et qu'on a "BYBYY", est-ce que GERER est compatible?
        assert is_compatible("GERER", "ARBRE", "BYBYY") is True

    def test_not_compatible(self):
        """Test: incompatibilité."""
        # SOTTE ne peut pas donner GGGGG pour ARBRE
        assert is_compatible("SOTTE", "ARBRE", "GGGGG") is False

    def test_compatible_with_repeated_letters(self):
        """Test: compatibilité avec lettres répétées."""
        # GERER devrait être compatible avec guess ARBRE et feedback "BYBYY"
        assert is_compatible("GERER", "ARBRE", "BYBYY") is True

        # TERRE ne devrait PAS être compatible avec guess ARBRE et feedback "BYBYY"
        assert is_compatible("TERRE", "ARBRE", "BYBYY") is False


class TestFeedbackToConstraints:
    """Tests de conversion feedback -> contraintes."""

    def test_constraints_all_green(self):
        """Test: toutes lettres vertes."""
        constraints = feedback_to_constraints("ARBRE", "GGGGG")

        assert constraints['exact'] == {0: 'A', 1: 'R', 2: 'B', 3: 'R', 4: 'E'}
        assert constraints['present'] == {}
        assert constraints['absent'] == set()

    def test_constraints_mixed(self):
        """Test: contraintes mixtes."""
        # ARBRE -> "BYBYY"
        # A: absent, R: jaune (pos 1), B: absent, R: vert (pos 3), E: jaune (pos 4)
        constraints = feedback_to_constraints("ARBRE", "BYBYY")

        assert constraints['exact'] == {}
        assert constraints['present'] == {'R': [1, 3], 'E': [4]}
        assert constraints['absent'] == {'A', 'B'}

    def test_constraints_repeated_in_secret(self):
        """Test: lettre répétée avec un vert et un jaune."""
        # TERRE -> BYGGB (secret ARBRE)
        constraints = feedback_to_constraints("TERRE", "BYGGB")

        assert constraints['exact'] == {2: 'R', 3: 'R'}
        assert constraints['present'] == {'E': [1]}
        assert 'T' in constraints['absent']
        # Le deuxième E est gris mais E ne doit pas être dans absent car il y a un E jaune
        assert 'E' not in constraints['absent']


class TestEdgeCases:
    """Tests des cas limites."""

    def test_all_same_letter_guess(self):
        """Test: guess avec la même lettre répétée."""
        # AAAAA vs ARBRE -> seule la première position est verte
        result = compute_feedback("AAAAA", "ARBRE")
        assert result == "GBBBB"

    def test_all_same_letter_secret(self):
        """Test: secret avec la même lettre répétée."""
        result = compute_feedback("ARBRE", "EEEEE")
        # Seul le E en dernière position est vert
        assert result == "BBBBG"

    def test_empty_intersection(self):
        """Test: aucune lettre en commun."""
        result = compute_feedback("ABCDE", "FGHIJ")
        assert result == "BBBBB"

    def test_anagram(self):
        """Test: anagrammes."""
        # ARBRE vs BARRE
        result = compute_feedback("ARBRE", "BARRE")
        # A: mal placé (existe en pos 1 de BARRE)
        # R: mal placé (existe en pos 2 de BARRE)
        # B: mal placé (existe en pos 0 de BARRE)
        # R: mal placé (existe en pos 3 de BARRE)
        # E: bien placé
        assert result == "YYYGG"
