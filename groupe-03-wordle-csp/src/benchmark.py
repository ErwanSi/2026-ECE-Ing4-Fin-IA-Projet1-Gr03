"""
Benchmark des stratégies Wordle

Ce module permet de comparer les performances des différentes stratégies
sur un ensemble de mots secrets.

Métriques mesurées:
- Nombre moyen de coups
- Taux de réussite (< 6 coups)
- Temps de calcul
- Distribution des scores

Usage:
    python -m src.benchmark --n 100 --strategies frequency,entropy
"""

import argparse
import random
import time
import statistics
from pathlib import Path
from typing import Dict, List
from collections import Counter

from .csp_solver import WordleCSPSolver
from .strategy import get_strategy, suggest_first_word
from .wordle_feedback import compute_feedback


def get_dictionary_path() -> str:
    """Retourne le chemin vers le dictionnaire."""
    base_dir = Path(__file__).parent.parent
    dict_path = base_dir / "data" / "mots_fr_5.txt"
    return str(dict_path)


def solve_wordle(secret: str, strategy_name: str, max_turns: int = 6, verbose: bool = False) -> Dict:
    """
    Résout un Wordle avec une stratégie donnée.

    Args:
        secret: mot secret
        strategy_name: nom de la stratégie
        max_turns: nombre maximum de tentatives
        verbose: afficher les détails

    Returns:
        Dictionnaire avec les résultats
    """
    dict_path = get_dictionary_path()
    solver = WordleCSPSolver(dict_path)
    strategy = get_strategy(strategy_name)

    secret = secret.upper()
    turn = 1
    guess = suggest_first_word()

    start_time = time.time()

    if verbose:
        print(f"\n🎯 Secret: {secret} | Stratégie: {strategy_name}")

    while turn <= max_turns:
        if verbose:
            print(f"  Tour {turn}: {guess}", end=" ")

        # Calculer le feedback
        feedback = compute_feedback(guess, secret)

        if verbose:
            print(f"-> {feedback}")

        # Victoire
        if feedback == 'GGGGG':
            elapsed = time.time() - start_time
            if verbose:
                print(f"  ✅ Trouvé en {turn} coups ({elapsed:.2f}s)")

            return {
                'success': True,
                'turns': turn,
                'time': elapsed,
                'secret': secret
            }

        # Mettre à jour le solveur
        solver.add_constraint(guess, feedback)
        remaining = solver.get_candidate_count()

        if remaining == 0:
            elapsed = time.time() - start_time
            if verbose:
                print(f"  ❌ Échec: aucun candidat restant")

            return {
                'success': False,
                'turns': turn,
                'time': elapsed,
                'secret': secret,
                'reason': 'no_candidates'
            }

        # Suggérer le prochain mot
        candidates = solver.get_candidates()
        guess = strategy.select_word(candidates, solver.initial_candidates)

        turn += 1

    # Échec: limite atteinte
    elapsed = time.time() - start_time
    if verbose:
        print(f"  ❌ Échec: limite de {max_turns} coups")

    return {
        'success': False,
        'turns': max_turns,
        'time': elapsed,
        'secret': secret,
        'reason': 'max_turns'
    }


def run_benchmark(n_tests: int, strategies: List[str], random_seed: int = 42) -> Dict:
    """
    Lance un benchmark comparatif.

    Args:
        n_tests: nombre de mots à tester
        strategies: liste des stratégies à comparer
        random_seed: graine aléatoire pour reproductibilité

    Returns:
        Dictionnaire avec les résultats
    """
    random.seed(random_seed)

    # Charger le dictionnaire
    dict_path = get_dictionary_path()
    with open(dict_path, 'r', encoding='utf-8') as f:
        all_words = [line.strip().upper() for line in f if len(line.strip()) == 5]

    # Échantillonner n_tests mots
    test_words = random.sample(all_words, min(n_tests, len(all_words)))

    print(f"🏁 BENCHMARK WORDLE CSP")
    print(f"=" * 70)
    print(f"Nombre de tests: {len(test_words)}")
    print(f"Stratégies: {', '.join(strategies)}")
    print(f"=" * 70)

    results = {strategy: [] for strategy in strategies}

    # Tester chaque mot avec chaque stratégie
    for i, secret in enumerate(test_words, 1):
        print(f"\n[{i}/{len(test_words)}] Mot: {secret}")

        for strategy in strategies:
            result = solve_wordle(secret, strategy, verbose=False)
            results[strategy].append(result)

            status = "✅" if result['success'] else "❌"
            print(f"  {strategy:12s}: {status} {result['turns']} coups ({result['time']:.2f}s)")

    # Analyser les résultats
    print(f"\n{'=' * 70}")
    print("📊 RÉSULTATS")
    print(f"{'=' * 70}\n")

    summary = {}

    for strategy in strategies:
        strat_results = results[strategy]

        successes = [r for r in strat_results if r['success']]
        failures = [r for r in strat_results if not r['success']]

        success_rate = len(successes) / len(strat_results) * 100
        avg_turns = statistics.mean([r['turns'] for r in successes]) if successes else 0
        avg_time = statistics.mean([r['time'] for r in strat_results])

        # Distribution des scores
        turns_distribution = Counter([r['turns'] for r in successes])

        summary[strategy] = {
            'success_rate': success_rate,
            'avg_turns': avg_turns,
            'avg_time': avg_time,
            'total_tests': len(strat_results),
            'successes': len(successes),
            'failures': len(failures),
            'turns_distribution': turns_distribution
        }

        print(f"Stratégie: {strategy.upper()}")
        print(f"  Taux de réussite: {success_rate:.1f}% ({len(successes)}/{len(strat_results)})")
        if successes:
            print(f"  Moyenne de coups: {avg_turns:.2f}")
        print(f"  Temps moyen: {avg_time:.2f}s")
        print(f"  Distribution:")
        for turns in sorted(turns_distribution.keys()):
            count = turns_distribution[turns]
            bar = '█' * count
            print(f"    {turns} coups: {bar} ({count})")
        print()

    return summary


def main():
    """Point d'entrée pour le benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark des stratégies Wordle CSP"
    )

    parser.add_argument(
        '--n',
        type=int,
        default=50,
        help='Nombre de mots à tester (défaut: 50)'
    )

    parser.add_argument(
        '--strategies',
        default='naive,frequency,mixed',
        help='Stratégies à comparer, séparées par des virgules (défaut: naive,frequency,mixed)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Graine aléatoire (défaut: 42)'
    )

    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(',')]

    # Valider les stratégies
    valid_strategies = ['naive', 'frequency', 'entropy', 'mixed']
    for strategy in strategies:
        if strategy not in valid_strategies:
            print(f"❌ Stratégie invalide: {strategy}")
            print(f"Stratégies disponibles: {', '.join(valid_strategies)}")
            return

    # Lancer le benchmark
    run_benchmark(args.n, strategies, args.seed)


if __name__ == '__main__':
    main()
