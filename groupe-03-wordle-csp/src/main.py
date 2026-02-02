"""
Interface en ligne de commande (CLI) pour le solveur Wordle CSP

Modes disponibles:
- interactive: mode interactif pour jouer/résoudre pas à pas
- suggest: suggère le prochain mot basé sur l'historique
- auto: résolution automatique

Usage:
    python -m src.main interactive
    python -m src.main suggest --guesses ARBRE,CRANE --feedbacks BGYBB,GGGBB
    python -m src.main auto --secret GERER
"""

import argparse
import sys
import os
from pathlib import Path
import logging

from .csp_solver import WordleCSPSolver
from .ortools_cpsat_solver import WordleORToolsSolver
from .strategy import get_strategy, suggest_first_word
from .wordle_feedback import compute_feedback, format_feedback_display

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def get_dictionary_path() -> str:
    """Retourne le chemin vers le dictionnaire."""
    # Depuis groupe-03-wordle-csp/src/, aller vers groupe-03-wordle-csp/data/
    base_dir = Path(__file__).parent.parent
    dict_path = base_dir / "data" / "mots_fr_5.txt"

    if not dict_path.exists():
        logger.error(f"Dictionnaire introuvable: {dict_path}")
        sys.exit(1)

    return str(dict_path)


def create_solver(solver_type: str, dictionary_path: str):
    """
    Crée une instance du solveur selon le type spécifié.
    
    Args:
        solver_type: Type de solveur ('filtering' ou 'cpsat')
        dictionary_path: Chemin du dictionnaire
    
    Returns:
        Instance de WordleCSPSolver ou WordleORToolsSolver
    
    Raises:
        ValueError: Si le type de solveur est invalide
    """
    if solver_type == "filtering":
        return WordleCSPSolver(dictionary_path)
    elif solver_type == "cpsat":
        return WordleORToolsSolver(dictionary_path)
    else:
        raise ValueError(f"Type de solveur invalide: {solver_type}. Utilisez 'filtering' ou 'cpsat'.")


def mode_interactive(args):
    """
    Mode interactif: l'utilisateur entre des mots et feedbacks.

    Le programme suggère le prochain meilleur mot à chaque étape.
    """
    print("=" * 60)
    print("SOLVEUR WORDLE CSP - MODE INTERACTIF")
    print("=" * 60)
    print("\nInstructions:")
    print("- Entrez un mot de 5 lettres")
    print("- Entrez le feedback reçu (ex: GGYBB)")
    print("  G = Vert (bien placé)")
    print("  Y = Jaune (mal placé)")
    print("  B = Gris (absent)")
    print("- Tapez 'quit' pour quitter\n")

    dict_path = get_dictionary_path()
    solver = create_solver(args.solver, dict_path)
    strategy = get_strategy(args.strategy)

    print(f"Stratégie: {strategy.name}")
    print(f"Solveur: {args.solver}")
    print(f"Dictionnaire: {solver.get_candidate_count()} mots\n")

    # Suggérer le premier mot
    first_word = suggest_first_word()
    print(f"💡 Suggestion pour le premier mot: {first_word}\n")

    turn = 1
    while True:
        print(f"--- Tour {turn} ---")

        # Demander le mot proposé
        guess = input("Mot proposé: ").strip().upper()

        if guess.lower() == 'quit':
            print("Au revoir!")
            break

        if len(guess) != 5:
            print("❌ Le mot doit faire exactement 5 lettres!")
            continue

        # Demander le feedback
        feedback = input("Feedback (GGGGG si trouvé): ").strip().upper()

        if len(feedback) != 5 or not all(c in 'GYB' for c in feedback):
            print("❌ Le feedback doit faire 5 caractères (G, Y ou B)!")
            continue

        # Afficher le feedback coloré
        print(f"Feedback: {format_feedback_display(guess, feedback)}\n")

        # Cas résolu
        if feedback == 'GGGGG':
            print(f"🎉 Trouvé en {turn} coups: {guess}")
            break

        # Ajouter la contrainte au solveur
        try:
            remaining = solver.add_constraint(guess, feedback)
            print(f"📊 Candidats restants: {remaining}")

            if remaining == 0:
                print("❌ Aucun candidat! Vérifiez vos feedbacks.")
                break

            if remaining <= 10:
                print(f"Candidats: {', '.join(solver.get_candidates())}")

            # Suggérer le prochain mot
            if remaining > 1:
                candidates = solver.get_candidates()
                all_words = solver.initial_candidates
                suggestion = strategy.select_word(candidates, all_words)
                print(f"💡 Suggestion: {suggestion}\n")
            else:
                solution = solver.get_solution()
                print(f"✅ Solution unique: {solution}\n")

        except Exception as e:
            print(f"❌ Erreur: {e}")
            break

        turn += 1


def mode_suggest(args):
    """
    Mode suggestion: donne le meilleur mot suivant basé sur l'historique.

    Usage:
        python -m src.main suggest --guesses ARBRE,CRANE --feedbacks BGYBB,GGGBB
    """
    if not args.guesses or not args.feedbacks:
        print("❌ Vous devez fournir --guesses et --feedbacks")
        sys.exit(1)

    guesses = args.guesses.split(',')
    feedbacks = args.feedbacks.split(',')

    if len(guesses) != len(feedbacks):
        print("❌ Le nombre de guesses et feedbacks doit être identique")
        sys.exit(1)

    dict_path = get_dictionary_path()
    solver = create_solver(args.solver, dict_path)
    strategy = get_strategy(args.strategy)

    print(f"Stratégie: {strategy.name}")
    print(f"Solveur: {args.solver}\n")

    # Appliquer les contraintes
    for guess, feedback in zip(guesses, feedbacks):
        guess = guess.strip().upper()
        feedback = feedback.strip().upper()

        print(f"{guess} -> {feedback}")
        remaining = solver.add_constraint(guess, feedback)
        print(f"Candidats restants: {remaining}\n")

    # Suggérer le prochain mot
    if solver.get_candidate_count() == 0:
        print("❌ Aucun candidat restant!")
    elif solver.get_candidate_count() == 1:
        print(f"✅ Solution: {solver.get_solution()}")
    else:
        candidates = solver.get_candidates()

        if len(candidates) <= 20:
            print(f"Candidats: {', '.join(candidates)}\n")

        suggestion = strategy.select_word(candidates, solver.initial_candidates)
        print(f"💡 Suggestion: {suggestion}")


def mode_auto(args):
    """
    Mode automatique: résout un mot secret automatiquement.

    Usage:
        python -m src.main auto --secret GERER --strategy entropy
    """
    if not args.secret:
        print("❌ Vous devez fournir --secret")
        sys.exit(1)

    secret = args.secret.upper()

    if len(secret) != 5:
        print("❌ Le secret doit faire 5 lettres")
        sys.exit(1)

    dict_path = get_dictionary_path()
    solver = create_solver(args.solver, dict_path)
    strategy = get_strategy(args.strategy)

    print(f"🎯 Résolution automatique de: {secret}")
    print(f"Stratégie: {strategy.name}")
    print(f"Solveur: {args.solver}\n")

    turn = 1
    max_turns = 6

    # Premier mot suggéré
    guess = suggest_first_word()

    while turn <= max_turns:
        print(f"--- Tour {turn} ---")
        print(f"Proposition: {guess}")

        # Calculer le feedback
        feedback = compute_feedback(guess, secret)
        print(f"Feedback: {format_feedback_display(guess, feedback)}")

        if feedback == 'GGGGG':
            print(f"\n🎉 Trouvé en {turn} coups!")
            return turn

        # Mettre à jour le solveur
        solver.add_constraint(guess, feedback)
        remaining = solver.get_candidate_count()
        print(f"Candidats restants: {remaining}\n")

        if remaining == 0:
            print("❌ Erreur: aucun candidat restant!")
            return -1

        # Suggérer le prochain mot
        candidates = solver.get_candidates()
        guess = strategy.select_word(candidates, solver.initial_candidates)

        turn += 1

    print(f"❌ Échec: limite de {max_turns} coups atteinte")
    return -1


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Solveur Wordle CSP - IA Symbolique et Exploratoire"
    )

    subparsers = parser.add_subparsers(dest='mode', help='Mode de fonctionnement')

    # Mode interactif
    parser_interactive = subparsers.add_parser(
        'interactive',
        help='Mode interactif'
    )
    parser_interactive.add_argument(
        '--strategy',
        default='mixed',
        choices=['naive', 'frequency', 'entropy', 'mixed'],
        help='Stratégie à utiliser'
    )
    parser_interactive.add_argument(
        '--solver',
        type=str,
        choices=['filtering', 'cpsat'],
        default='filtering',
        help='Type de solveur CSP (défaut: filtering)'
    )

    # Mode suggestion
    parser_suggest = subparsers.add_parser(
        'suggest',
        help='Suggère le prochain mot'
    )
    parser_suggest.add_argument(
        '--guesses',
        help='Mots proposés séparés par des virgules (ex: ARBRE,CRANE)'
    )
    parser_suggest.add_argument(
        '--feedbacks',
        help='Feedbacks séparés par des virgules (ex: BGYBB,GGGBB)'
    )
    parser_suggest.add_argument(
        '--strategy',
        default='mixed',
        choices=['naive', 'frequency', 'entropy', 'mixed'],
        help='Stratégie à utiliser'
    )
    parser_suggest.add_argument(
        '--solver',
        type=str,
        choices=['filtering', 'cpsat'],
        default='filtering',
        help='Type de solveur CSP (défaut: filtering)'
    )

    # Mode automatique
    parser_auto = subparsers.add_parser(
        'auto',
        help='Résolution automatique'
    )
    parser_auto.add_argument(
        '--secret',
        required=True,
        help='Mot secret à trouver'
    )
    parser_auto.add_argument(
        '--strategy',
        default='mixed',
        choices=['naive', 'frequency', 'entropy', 'mixed'],
        help='Stratégie à utiliser'
    )
    parser_auto.add_argument(
        '--solver',
        type=str,
        choices=['filtering', 'cpsat'],
        default='filtering',
        help='Type de solveur CSP (défaut: filtering)'
    )

    args = parser.parse_args()

    if not args.mode:
        parser.print_help()
        sys.exit(1)

    # Router vers le bon mode
    if args.mode == 'interactive':
        mode_interactive(args)
    elif args.mode == 'suggest':
        mode_suggest(args)
    elif args.mode == 'auto':
        mode_auto(args)


if __name__ == '__main__':
    main()
