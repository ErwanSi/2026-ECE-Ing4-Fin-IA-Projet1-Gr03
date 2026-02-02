"""
Tests unitaires pour le solveur OR-Tools CP-SAT.
"""

import pytest
from pathlib import Path
from src.ortools_cpsat_solver import WordleORToolsSolver
from src.constraints import feedback_to_constraints, merge_constraints
from src.wordle_feedback import compute_feedback


@pytest.fixture
def mock_dictionary(tmp_path):
    """Crée un dictionnaire mock pour les tests."""
    dict_file = tmp_path / "test_dict.txt"
    words = ["MARDI", "MARLI", "MARIE", "PARTI", "TARES", "ARBRE", "GRAIN"]
    dict_file.write_text("\n".join(words), encoding='utf-8')
    return str(dict_file)


@pytest.fixture
def solver(mock_dictionary):
    """Crée une instance du solveur pour les tests."""
    return WordleORToolsSolver(mock_dictionary)


class TestConstraintDerivation:
    """Tests pour l'extraction de contraintes."""
    
    def test_feedback_to_constraints_all_green(self):
        """Test contraintes avec feedback tout vert."""
        constraints = feedback_to_constraints("MARDI", "GGGGG")
        assert constraints['exact'] == {0: 'M', 1: 'A', 2: 'R', 3: 'D', 4: 'I'}
        assert len(constraints['absent']) == 0
        assert len(constraints['present']) == 0
    
    def test_feedback_to_constraints_all_black(self):
        """Test contraintes avec feedback tout gris."""
        constraints = feedback_to_constraints("TARES", "BBBBB")
        assert constraints['absent'] == {'T', 'A', 'R', 'E', 'S'}
        assert len(constraints['exact']) == 0
        assert len(constraints['present']) == 0
    
    def test_feedback_to_constraints_mixed(self):
        """Test contraintes avec feedback mixte."""
        constraints = feedback_to_constraints("TARES", "BYGBB")
        assert constraints['exact'] == {2: 'R'}  # R est vert en position 2
        assert 'A' in constraints['present']  # A est jaune
        assert 1 in constraints['present']['A']  # Position 1 interdite
        assert constraints['absent'] == {'T', 'E', 'S'}  # R n'est plus absent car il est vert
    
    def test_feedback_to_constraints_repeated_letters(self):
        """Test contraintes avec lettres répétées."""
        # "ABACA": A en pos 0 vert (G), B en pos 1 jaune (Y), reste gris (BBB)
        constraints = feedback_to_constraints("ABACA", "GYBBB")
        assert constraints['exact'][0] == 'A'
        assert 'B' in constraints['present']  # B est jaune, pas A
        assert 1 in constraints['present']['B']  # Position 1 interdite pour B
        assert constraints['min_count']['A'] == 1  # Au moins 1 A (le vert)
        assert constraints['min_count']['B'] == 1  # Au moins 1 B (le jaune)
    
    def test_feedback_to_constraints_invalid_input(self):
        """Test validation des entrées."""
        with pytest.raises(ValueError):
            feedback_to_constraints("ABC", "GGGGG")
        
        with pytest.raises(ValueError):
            feedback_to_constraints("ABCDE", "XYZ")
    
    def test_merge_constraints(self):
        """Test fusion de contraintes multiples."""
        c1 = feedback_to_constraints("TARES", "BYGBB")
        c2 = feedback_to_constraints("MARDI", "GGBBB")
        
        merged = merge_constraints([c1, c2])
        
        # Vérifier que les positions exactes sont combinées
        assert 0 in merged['exact']
        assert 1 in merged['exact']
        
        # Vérifier que les lettres absentes sont combinées
        assert 'T' in merged['absent']
        assert 'R' in merged['absent']


class TestORToolsSolver:
    """Tests pour le solveur OR-Tools CP-SAT."""
    
    def test_solver_initialization(self, solver):
        """Test initialisation du solveur."""
        assert len(solver.initial_candidates) == 7
        assert 'MARDI' in solver.initial_candidates
    
    def test_add_constraint_reduces_candidates(self, solver):
        """Test que l'ajout de contrainte réduit les candidats."""
        initial_count = solver.get_candidate_count()
        # Éliminer les mots avec T, A, R, E, S -> garde MARDI, MARLI, MARIE, PARTI, GRAIN (5 éliminés: aucun, car MARDI et autres contiennent ces lettres)
        # Utilisons un feedback qui élimine vraiment: Z n'existe pas dans ces mots
        # Utilisons "ZZZBB" pour éliminer les mots contenant E et S à la fin
        # Simplement: tous gris pour un mot qui n'existe pas dans les candidats
        remaining = solver.add_constraint("BOUGE", "BBBBB")
        assert remaining < initial_count
        assert remaining >= 0  # Peut être 0 si toutes les lettres sont éliminées
    
    def test_solver_finds_solution(self, solver):
        """Test résolution complète."""
        secret = "MARDI"
        max_turns = 10
        
        for turn in range(max_turns):
            candidates = solver.get_candidates()
            if not candidates:
                break
            
            guess = candidates[0]  # Stratégie naive
            feedback = compute_feedback(guess, secret)
            solver.add_constraint(guess, feedback)
            
            if solver.is_solved() and solver.get_solution() == secret:
                break
        
        assert solver.is_solved()
        assert solver.get_solution() == secret
    
    def test_solver_reset(self, solver):
        """Test réinitialisation du solveur."""
        solver.add_constraint("TARES", "BYGBB")
        assert solver.get_candidate_count() < len(solver.initial_candidates)
        
        solver.reset()
        assert solver.get_candidate_count() == len(solver.initial_candidates)
        assert len(solver.constraints_history) == 0
    
    def test_invalid_guess_raises_error(self, solver):
        """Test qu'un guess invalide lève une erreur."""
        with pytest.raises(ValueError):
            solver.add_constraint("ABC", "GGGGG")
        
        with pytest.raises(ValueError):
            solver.add_constraint("ABCDE", "GGG")
    
    def test_get_statistics(self, solver):
        """Test récupération des statistiques."""
        stats = solver.get_statistics()
        
        assert stats['total_words'] == 7
        assert stats['candidates_remaining'] == 7
        assert stats['constraints_applied'] == 0
        assert stats['solved'] == False
        assert stats['solver_type'] == 'ortools_cpsat'
        
        solver.add_constraint("TARES", "BYGBB")
        stats = solver.get_statistics()
        assert stats['constraints_applied'] == 1
    
    def test_exact_position_constraint(self, solver):
        """Test contrainte position exacte (verte)."""
        # M en position 0 -> garde MARDI, MARLI, MARIE
        solver.add_constraint("MARDI", "GBBBB")
        candidates = solver.get_candidates()
        
        for word in candidates:
            assert word[0] == 'M'
    
    def test_present_letter_constraint(self, solver):
        """Test contrainte lettre présente (jaune)."""
        # A présent mais pas en position 1
        solver.add_constraint("TARES", "BYGBB")
        candidates = solver.get_candidates()
        
        for word in candidates:
            assert 'A' in word
            assert word[1] != 'A'
    
    def test_absent_letter_constraint(self, solver):
        """Test contrainte lettre absente (grise)."""
        solver.add_constraint("TARES", "BBBBB")
        candidates = solver.get_candidates()
        
        for word in candidates:
            for letter in 'TARES':
                assert letter not in word


class TestCompatibilityWithFiltering:
    """Tests de compatibilité entre CP-SAT et filtrage."""
    
    def test_same_results_as_filtering(self, mock_dictionary):
        """Test que CP-SAT et filtrage donnent les mêmes résultats."""
        from src.csp_solver import WordleCSPSolver
        
        cpsat_solver = WordleORToolsSolver(mock_dictionary)
        filtering_solver = WordleCSPSolver(mock_dictionary)
        
        # Appliquer les mêmes contraintes
        constraints = [
            ("TARES", "BYGBB"),
            ("MARDI", "GGGGG")
        ]
        
        for guess, feedback in constraints:
            cpsat_solver.add_constraint(guess, feedback)
            filtering_solver.add_constraint(guess, feedback)
        
        # Vérifier même résultat
        cpsat_candidates = set(cpsat_solver.get_candidates())
        filtering_candidates = set(filtering_solver.get_candidates())
        assert cpsat_candidates == filtering_candidates
    
    def test_multiple_constraints_compatibility(self, mock_dictionary):
        """Test compatibilité avec plusieurs contraintes."""
        from src.csp_solver import WordleCSPSolver
        
        cpsat_solver = WordleORToolsSolver(mock_dictionary)
        filtering_solver = WordleCSPSolver(mock_dictionary)
        
        # Séquence de contraintes
        constraints = [
            ("GRAIN", "BYBBB"),  # R présent, pas en position 1
            ("ARBRE", "BYGBB"),  # R présent, pas en position 1
        ]
        
        for guess, feedback in constraints:
            cpsat_solver.add_constraint(guess, feedback)
            filtering_solver.add_constraint(guess, feedback)
        
        cpsat_candidates = set(cpsat_solver.get_candidates())
        filtering_candidates = set(filtering_solver.get_candidates())
        assert cpsat_candidates == filtering_candidates


class TestEdgeCases:
    """Tests des cas limites."""
    
    def test_no_candidates_left(self, solver):
        """Test quand aucun candidat ne reste (contraintes impossibles)."""
        # Appliquer des contraintes qui éliminent tous les candidats
        solver.add_constraint("ZZZZZ", "GGGGG")  # Aucun mot ne correspond
        
        assert solver.get_candidate_count() == 0
        assert not solver.is_solved()
        assert solver.get_solution() is None
    
    def test_single_candidate_is_solved(self, solver):
        """Test qu'un seul candidat = résolu."""
        # Appliquer des contraintes jusqu'à avoir un seul candidat
        solver.add_constraint("MARDI", "GGGGG")
        
        assert solver.is_solved()
        assert solver.get_solution() == "MARDI"
    
    def test_repeated_letters_constraint(self, mock_dictionary):
        """Test gestion des lettres répétées."""
        # Créer un dictionnaire avec des mots à lettres répétées
        from tempfile import NamedTemporaryFile
        
        with NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("ABACA\nELEVE\nPAPER\n")
            dict_path = f.name
        
        try:
            solver = WordleORToolsSolver(dict_path)
            
            # Test: 2 'A' dans le mot, un vert, un jaune
            solver.add_constraint("ABACA", "GYBBB")
            candidates = solver.get_candidates()
            
            # Le candidat doit avoir 'A' en position 0 et au moins un autre 'A'
            for word in candidates:
                assert word[0] == 'A'
                assert word.count('A') >= 2
        
        finally:
            import os
            os.unlink(dict_path)
