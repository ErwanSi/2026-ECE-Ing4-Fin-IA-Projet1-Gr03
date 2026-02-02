"""
Tests unitaires pour le filtrage CSP

Ces tests vérifient:
- Le chargement du dictionnaire
- L'ajout de contraintes
- Le filtrage correct des candidats
- Les cas limites (0 candidat, contraintes incohérentes)
"""

import pytest
import tempfile
import os
from pathlib import Path

from src.csp_solver import WordleCSPSolver


@pytest.fixture
def temp_dictionary():
    """Crée un dictionnaire temporaire pour les tests."""
    words = [
        "ARBRE",
        "AUTRE",
        "AITRE",
        "GERER",
        "TERRE",
        "SOTTE",
        "BARRE",
        "CARRE",
        "TARTE"
    ]

    # Créer un fichier temporaire
    fd, path = tempfile.mkstemp(suffix='.txt')
    with os.fdopen(fd, 'w') as f:
        f.write('\n'.join(words))

    yield path

    # Nettoyer
    os.unlink(path)


class TestSolverInitialization:
    """Tests d'initialisation du solveur."""

    def test_load_dictionary(self, temp_dictionary):
        """Test: chargement correct du dictionnaire."""
        solver = WordleCSPSolver(temp_dictionary)
        assert solver.get_candidate_count() == 9

    def test_load_nonexistent_dictionary(self):
        """Test: erreur si dictionnaire inexistant."""
        with pytest.raises(FileNotFoundError):
            WordleCSPSolver("/path/to/nonexistent/dictionary.txt")

    def test_initial_state(self, temp_dictionary):
        """Test: état initial du solveur."""
        solver = WordleCSPSolver(temp_dictionary)
        assert len(solver.constraints) == 0
        assert not solver.is_solved()


class TestConstraintFiltering:
    """Tests du filtrage par contraintes."""

    def test_add_constraint_all_green(self, temp_dictionary):
        """Test: feedback tout vert réduit à 1 candidat."""
        solver = WordleCSPSolver(temp_dictionary)
        remaining = solver.add_constraint("ARBRE", "GGGGG")

        assert remaining == 1
        assert solver.is_solved()
        assert solver.get_solution() == "ARBRE"

    def test_add_constraint_filters_correctly(self, temp_dictionary):
        """Test: filtrage correct des candidats."""
        solver = WordleCSPSolver(temp_dictionary)

        # Après ARBRE -> BYBYY, seul GERER devrait rester
        # (R en position 3, E quelque part, pas de A ni B)
        remaining = solver.add_constraint("ARBRE", "BYBYY")

        candidates = solver.get_candidates()
        assert "GERER" in candidates

    def test_multiple_constraints(self, temp_dictionary):
        """Test: application de plusieurs contraintes."""
        solver = WordleCSPSolver(temp_dictionary)

        # Première contrainte
        solver.add_constraint("ARBRE", "BYBBB")
        count1 = solver.get_candidate_count()

        # Deuxième contrainte
        solver.add_constraint("SOTTE", "BBBBB")
        count2 = solver.get_candidate_count()

        # Le nombre de candidats doit diminuer
        assert count2 <= count1

    def test_constraint_eliminates_all(self, temp_dictionary):
        """Test: contrainte qui élimine tous les candidats."""
        solver = WordleCSPSolver(temp_dictionary)

        # Contrainte impossible: toutes les lettres de ZZZZZ seraient vertes
        # mais ZZZZZ n'est pas dans le dictionnaire
        remaining = solver.add_constraint("ZZZZZ", "GGGGG")

        assert remaining == 0

    def test_invalid_feedback_raises_error(self, temp_dictionary):
        """Test: feedback invalide lève une erreur."""
        solver = WordleCSPSolver(temp_dictionary)

        with pytest.raises(ValueError):
            solver.add_constraint("ARBRE", "GGGX")  # X invalide

        with pytest.raises(ValueError):
            solver.add_constraint("ARBRE", "GGG")  # Trop court


class TestSolverReset:
    """Tests de réinitialisation."""

    def test_reset(self, temp_dictionary):
        """Test: reset restaure l'état initial."""
        solver = WordleCSPSolver(temp_dictionary)
        initial_count = solver.get_candidate_count()

        # Ajouter des contraintes
        solver.add_constraint("ARBRE", "BYBBB")
        assert solver.get_candidate_count() < initial_count

        # Reset
        solver.reset()
        assert solver.get_candidate_count() == initial_count
        assert len(solver.constraints) == 0


class TestSolverStatistics:
    """Tests des statistiques."""

    def test_get_statistics(self, temp_dictionary):
        """Test: récupération des statistiques."""
        solver = WordleCSPSolver(temp_dictionary)
        solver.add_constraint("ARBRE", "BYBBB")

        stats = solver.get_statistics()

        assert 'initial_candidates' in stats
        assert 'current_candidates' in stats
        assert 'reduction_rate' in stats
        assert 'constraints_applied' in stats
        assert stats['constraints_applied'] == 1

    def test_reduction_rate(self, temp_dictionary):
        """Test: calcul du taux de réduction."""
        solver = WordleCSPSolver(temp_dictionary)
        initial = solver.get_candidate_count()

        solver.add_constraint("ARBRE", "BBBBB")  # Élimine ARBRE

        stats = solver.get_statistics()
        expected_rate = 1 - (solver.get_candidate_count() / initial)

        assert stats['reduction_rate'] == pytest.approx(expected_rate)


class TestEdgeCases:
    """Tests des cas limites."""

    def test_get_solution_with_multiple_candidates(self, temp_dictionary):
        """Test: get_solution lève une erreur si >1 candidat."""
        solver = WordleCSPSolver(temp_dictionary)

        with pytest.raises(ValueError, match="problème non résolu"):
            solver.get_solution()

    def test_get_solution_with_no_candidates(self, temp_dictionary):
        """Test: get_solution lève une erreur si 0 candidat."""
        solver = WordleCSPSolver(temp_dictionary)
        solver.add_constraint("ZZZZZ", "GGGGG")  # Élimine tout

        with pytest.raises(ValueError, match="contraintes incohérentes"):
            solver.get_solution()

    def test_case_insensitivity(self, temp_dictionary):
        """Test: insensibilité à la casse."""
        solver = WordleCSPSolver(temp_dictionary)

        # Minuscules et majuscules doivent donner le même résultat
        solver.add_constraint("arbre", "bybbb")
        count1 = solver.get_candidate_count()

        solver.reset()
        solver.add_constraint("ARBRE", "BYBBB")
        count2 = solver.get_candidate_count()

        assert count1 == count2


from src.wordle_feedback import compute_feedback

class TestRealWorldScenario:
    """Tests de scénarios réalistes."""

    def test_typical_game(self, temp_dictionary):
        """Test: partie typique de Wordle (feedback généré automatiquement)."""
        solver = WordleCSPSolver(temp_dictionary)

        # On fixe un secret présent dans le mini-dictionnaire de test
        secret = "GERER"

        # Tour 1
        guess1 = "ARBRE"
        fb1 = compute_feedback(guess1, secret)
        solver.add_constraint(guess1, fb1)
        assert solver.get_candidate_count() > 0

        # Tour 2 : on prend un candidat (ou un mot standard) et on rejoue
        candidates = solver.get_candidates()
        guess2 = candidates[0] if candidates else "TERRE"
        fb2 = compute_feedback(guess2, secret)
        solver.add_constraint(guess2, fb2)
        assert solver.get_candidate_count() > 0

        # Tour 3 : on tente de converger
        candidates = solver.get_candidates()
        guess3 = candidates[0] if candidates else "GERER"
        fb3 = compute_feedback(guess3, secret)
        solver.add_constraint(guess3, fb3)

        # À la fin, le secret doit être compatible
        assert "GERER" in solver.get_candidates()
