"""
Point d'entrée principal pour le projet de stratégies de trading par algorithmes génétiques.

Projet ECE - Groupe 3 - Sujet n°41

Améliorations :
- Visualisation avec matplotlib
- Mécanismes anti-overfitting
- Validation croisée
- Adaptation dynamique des paramètres
"""

import numpy as np
import pandas as pd
from typing import Optional
from datetime import datetime

from genetic_algorithm import GeneticAlgorithm
from trading_strategy import TradingStrategy
from backtester import Backtester
from fitness import FitnessCalculator, FitnessWeights
from data_loader import DataLoader
from visualizer import Visualizer


# Codes ANSI pour les couleurs (fonctionne sur la plupart des terminaux)
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_header(text: str, width: int = 70):
    """Affiche un en-tête avec des bordures."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * width}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(width)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * width}{Colors.END}\n")


def print_section(number: int, title: str, width: int = 70):
    """Affiche une section numérotée."""
    print(f"{Colors.BOLD}{Colors.BLUE}{'─' * width}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}ÉTAPE {number}: {title}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'─' * width}{Colors.END}\n")


def print_subsection(title: str):
    """Affiche une sous-section."""
    print(f"{Colors.YELLOW}▸ {title}{Colors.END}")


def print_success(text: str):
    """Affiche un message de succès."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_warning(text: str):
    """Affiche un avertissement."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_error(text: str):
    """Affiche une erreur."""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_table(headers: list, rows: list, widths: list = None):
    """Affiche un tableau formaté."""
    if widths is None:
        widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
    
    # Ligne de séparation
    separator = '+' + '+'.join('-' * (w + 2) for w in widths) + '+'
    
    print(separator)
    # En-têtes
    header_row = '|' + '|'.join(f' {Colors.BOLD}{str(headers[i]).ljust(widths[i])}{Colors.END} ' for i in range(len(headers))) + '|'
    print(header_row)
    print(separator)
    
    # Données
    for row in rows:
        data_row = '|' + '|'.join(f' {str(row[i]).ljust(widths[i])} ' for i in range(len(row))) + '|'
        print(data_row)
    
    print(separator)


def print_box(text: str, width: int = 70):
    """Affiche du texte dans une boîte."""
    lines = text.split('\n')
    print(f"╔{'═' * (width - 2)}╗")
    for line in lines:
        print(f"║ {line.ljust(width - 4)} ║")
    print(f"╚{'═' * (width - 2)}╝")


def run_genetic_trading_optimization(
    symbol: str = "^GSPC",
    use_sample_data: bool = False,
    generations: int = 50,
    population_size: int = 30,
    train_ratio: float = 0.7,
    random_seed: int = 42,
    use_cross_validation: bool = True,
    cv_folds: int = 3,
    adaptive_mutation: bool = True,
    create_plots: bool = True,
    show_plots: bool = True
) -> dict:
    """
    Exécute l'optimisation de stratégie de trading par algorithme génétique.
    
    Args:
        symbol: Symbole de l'actif à trader
        use_sample_data: Utiliser des données simulées
        generations: Nombre de générations de l'algorithme génétique
        population_size: Taille de la population
        train_ratio: Ratio des données d'entraînement
        random_seed: Graine pour la reproductibilité
        use_cross_validation: Utiliser la validation croisée
        cv_folds: Nombre de folds pour la cross-validation
        adaptive_mutation: Activer l'adaptation dynamique de la mutation
        create_plots: Créer les graphiques
        show_plots: Afficher les graphiques
        
    Returns:
        Dictionnaire contenant les résultats
    """
    # En-tête principal
    print_header("STRATÉGIES DE TRADING PAR ALGORITHMES GÉNÉTIQUES")
    print(f"{Colors.CYAN}Projet ECE - Groupe 3 - Sujet n°41{Colors.END}")
    print(f"{Colors.YELLOW}Date d'exécution: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}{Colors.END}\n")
    
    # 1. Chargement des données
    print_section(1, "Chargement des données")
    print_subsection("Initialisation du loader de données...")
    loader = DataLoader()
    
    if use_sample_data:
        print_subsection(f"Utilisation de données simulées pour {Colors.BOLD}{symbol}{Colors.END}")
        data = loader.load_sample_data(symbol)
        print_success("Données simulées générées avec succès")
    else:
        print_subsection(f"Téléchargement des données pour {Colors.BOLD}{symbol}{Colors.END}...")
        try:
            data = loader.load_from_yfinance(symbol, period="2y")
            print_success("Données téléchargées avec succès")
        except Exception as e:
            print_error(f"Erreur lors du téléchargement: {e}")
            print_warning("Utilisation de données simulées à la place")
            data = loader.load_sample_data(symbol)
    
    info = loader.get_info()
    print(f"\n{Colors.BOLD}Informations sur les données:{Colors.END}")
    print_table(
        ["Paramètre", "Valeur"],
        [
            ["Symbole", info['symbol']],
            ["Période", f"{info['start_date']} à {info['end_date']}"],
            ["Nombre de jours", str(info['total_days'])],
            ["Prix actuel", f"{info['price_range']['current']:.2f}"],
            ["Prix minimum", f"{info['price_range']['min']:.2f}"],
            ["Prix maximum", f"{info['price_range']['max']:.2f}"]
        ]
    )
    
    # 2. Division des données
    print_section(2, "Division des données")
    print_subsection("Répartition des données en ensembles d'entraînement, validation et test...")
    train_data, validation_data, test_data = loader.split_train_test(
        train_ratio=train_ratio,
        validation_ratio=0.15
    )
    
    total_days = len(data)
    print_table(
        ["Ensemble", "Jours", "Pourcentage"],
        [
            ["Entraînement", str(len(train_data)), f"{len(train_data)/total_days*100:.1f}%"],
            ["Validation", str(len(validation_data)), f"{len(validation_data)/total_days*100:.1f}%"],
            ["Test", str(len(test_data)), f"{len(test_data)/total_days*100:.1f}%"]
        ]
    )
    
    # 3. Configuration de l'algorithme génétique
    print_section(3, "Configuration de l'algorithme génétique")
    print_subsection("Définition des bornes pour les gènes...")
    
    gene_bounds = [
        (5, 50),    # SMA court
        (10, 200),  # SMA long
        (5, 50),    # EMA période
        (5, 30),    # RSI période
        (10, 40),   # RSI survente
        (60, 90),   # RSI surachat
        (5, 20),    # MACD rapide
        (15, 50),   # MACD lent
        (5, 15),    # MACD signal
        (1, 20)     # Stop loss %
    ]
    
    print_subsection("Initialisation de l'algorithme génétique...")
    ga = GeneticAlgorithm(
        population_size=population_size,
        chromosome_length=10,
        gene_bounds=gene_bounds,
        crossover_rate=0.8,
        mutation_rate=0.15,
        mutation_strength=0.3,
        selection_method="tournament",
        tournament_size=3,
        elitism_count=2,
        random_seed=random_seed,
        adaptive_mutation=adaptive_mutation,
        diversity_threshold=0.1,
        early_stopping_patience=15,
        early_stopping_min_delta=0.001,
        use_crowding=True,
        crowding_distance=0.5
    )
    print_success("Algorithme génétique initialisé")
    
    print(f"\n{Colors.BOLD}Paramètres de l'algorithme génétique:{Colors.END}")
    print_table(
        ["Paramètre", "Valeur"],
        [
            ["Taille de la population", str(population_size)],
            ["Nombre de générations", str(generations)],
            ["Taux de croisement", f"{ga.crossover_rate:.2f}"],
            ["Taux de mutation", f"{ga.mutation_rate:.2f}"],
            ["Force de mutation", f"{ga.mutation_strength:.2f}"],
            ["Méthode de sélection", ga.selection_method],
            ["Taille du tournoi", str(ga.tournament_size)],
            ["Élitisme (individus)", str(ga.elitism_count)],
            ["Graine aléatoire", str(random_seed)]
        ]
    )
    
    # 4. Configuration du fitness
    print_section(4, "Configuration de la fonction fitness")
    print_subsection("Initialisation du backtester...")
    backtester = Backtester(initial_capital=100000, commission=0.001)
    print_success("Backtester initialisé")
    
    print_subsection("Définition des pondérations de la fonction fitness...")
    weights = FitnessWeights(
        return_weight=0.25,
        sharpe_weight=0.25,
        drawdown_weight=0.20,
        stability_weight=0.15,
        generalization_weight=0.10,
        complexity_penalty_weight=0.05
    )
    
    fitness_calc = FitnessCalculator(
        weights=weights,
        backtester=backtester,
        data=train_data,
        use_cross_validation=use_cross_validation,
        cv_folds=cv_folds,
        complexity_penalty=0.01,
        l2_regularization=0.001
    )
    print_success("Calculateur de fitness initialisé")
    
    print(f"\n{Colors.BOLD}Pondérations de la fonction fitness:{Colors.END}")
    print_table(
        ["Critère", "Poids", "Description"],
        [
            ["Rendement", f"{weights.return_weight:.2f}", "Profitabilité totale de la stratégie"],
            ["Sharpe Ratio", f"{weights.sharpe_weight:.2f}", "Rendement ajusté au risque"],
            ["Max Drawdown", f"{weights.drawdown_weight:.2f}", "Perte maximale subie"],
            ["Stabilité", f"{weights.stability_weight:.2f}", "Consistance des performances"],
            ["Généralisation", f"{weights.generalization_weight:.2f}", "Score de cross-validation"],
            ["Pénalité complexité", f"{weights.complexity_penalty_weight:.2f}", "Anti-overfitting"]
        ]
    )
    
    print(f"\n{Colors.BOLD}Paramètres anti-overfitting:{Colors.END}")
    print_table(
        ["Paramètre", "Valeur", "Description"],
        [
            ["Cross-validation", "Oui" if use_cross_validation else "Non", "Validation croisée temporelle"],
            ["Nombre de folds", str(cv_folds), "Folds pour la cross-validation"],
            ["Adaptive mutation", "Oui" if adaptive_mutation else "Non", "Adaptation dynamique du taux"],
            ["Crowding", "Oui", "Préservation de la diversité"],
            ["Early stopping", "Oui", "Arrêt si stagnation"]
        ]
    )
    
    # 5. Exécution de l'algorithme génétique
    print_section(5, "Exécution de l'algorithme génétique")
    print_subsection(f"Lancement de l'évolution sur {generations} générations...")
    print(f"{Colors.CYAN}{'─' * 70}{Colors.END}")
    
    best_individual = ga.evolve(
        fitness_func=fitness_calc.calculate_fitness,
        generations=generations
    )
    
    print(f"{Colors.CYAN}{'─' * 70}{Colors.END}")
    print_success("Évolution terminée")
    
    # 6. Analyse des résultats
    print_section(6, "Analyse des résultats de l'optimisation")
    stats = ga.get_statistics()
    
    print(f"\n{Colors.BOLD}Statistiques de l'optimisation:{Colors.END}")
    print_table(
        ["Métrique", "Valeur"],
        [
            ["Nombre de générations", str(stats['total_generations'])],
            ["Meilleur fitness initial", f"{stats['initial_best_fitness']:.4f}"],
            ["Meilleur fitness final", f"{stats['best_fitness']:.4f}"],
            ["Fitness moyen initial", f"{stats['initial_avg_fitness']:.4f}"],
            ["Fitness moyen final", f"{stats['final_avg_fitness']:.4f}"],
            ["Amélioration totale", f"{stats['improvement']:.4f}"],
            ["Taux d'amélioration", f"{stats['improvement_rate']*100:.2f}%"],
            ["Diversité finale", f"{stats['final_diversity']:.4f}"],
            ["Early stopping", "Oui" if stats['early_stopped'] else "Non"]
        ]
    )
    
    # Afficher les informations de cross-validation si activée
    if use_cross_validation:
        cv_info = fitness_calc.get_cv_info()
        if cv_info['cv_scores'] is not None:
            print(f"\n{Colors.BOLD}Résultats de la cross-validation:{Colors.END}")
            print(f"  • Scores par fold: {[f'{s:.4f}' for s in cv_info['cv_scores']]}")
            print(f"  • Score de généralisation: {cv_info['generalization_score']:.4f}")
    
    # 7. Création de la meilleure stratégie
    print_section(7, "Meilleure stratégie trouvée")
    print_subsection("Extraction des paramètres optimaux...")
    best_strategy = TradingStrategy(genes=best_individual.genes)
    params = best_strategy.get_parameters()
    
    print(f"\n{Colors.BOLD}Paramètres de la meilleure stratégie:{Colors.END}")
    print_table(
        ["Paramètre", "Valeur", "Description"],
        [
            ["SMA court", str(params['sma_short_period']), "Période de la moyenne mobile courte"],
            ["SMA long", str(params['sma_long_period']), "Période de la moyenne mobile longue"],
            ["EMA période", str(params['ema_period']), "Période de l'EMA"],
            ["RSI période", str(params['rsi_period']), "Période du RSI"],
            ["RSI survente", str(params['rsi_oversold']), "Seuil de survente RSI"],
            ["RSI surachat", str(params['rsi_overbought']), "Seuil de surachat RSI"],
            ["MACD rapide", str(params['macd_fast']), "Période MACD rapide"],
            ["MACD lent", str(params['macd_slow']), "Période MACD lent"],
            ["MACD signal", str(params['macd_signal']), "Période MACD signal"],
            ["Stop loss %", f"{params['stop_loss_pct']:.1f}%", "Pourcentage de stop loss"]
        ]
    )
    
    # 8. Validation sur les données de validation
    print_section(8, "Validation sur les données de validation")
    print_subsection("Évaluation de la stratégie sur l'ensemble de validation...")
    
    # Sauvegarder le résultat d'entraînement
    train_result = fitness_calc.last_result
    
    # Évaluer sur validation
    fitness_calc.set_data(validation_data)
    validation_fitness = fitness_calc.calculate_fitness(best_individual.genes)
    validation_components = fitness_calc.get_fitness_components(best_individual.genes)
    validation_result = fitness_calc.last_result
    
    print(f"\n{Colors.BOLD}Résultats sur l'ensemble de validation:{Colors.END}")
    print_table(
        ["Métrique", "Valeur"],
        [
            ["Fitness", f"{validation_fitness:.4f}"],
            ["Rendement", f"{validation_components['total_return']:.2f}%"],
            ["Sharpe Ratio", f"{validation_components['sharpe_ratio']:.2f}"],
            ["Max Drawdown", f"{validation_components['max_drawdown']:.2f}%"],
            ["Win Rate", f"{validation_components['win_rate']:.2f}%"],
            ["Profit Factor", f"{validation_components['profit_factor']:.2f}"]
        ]
    )
    
    # 9. Test final sur les données de test
    print_section(9, "Test final sur les données de test")
    print_subsection("Évaluation de la stratégie sur l'ensemble de test...")
    fitness_calc.set_data(test_data)
    test_fitness = fitness_calc.calculate_fitness(best_individual.genes)
    test_components = fitness_calc.get_fitness_components(best_individual.genes)
    test_result = fitness_calc.last_result
    
    print(f"\n{Colors.BOLD}Résultats sur l'ensemble de test:{Colors.END}")
    print_table(
        ["Métrique", "Valeur"],
        [
            ["Fitness", f"{test_fitness:.4f}"],
            ["Rendement", f"{test_components['total_return']:.2f}%"],
            ["Sharpe Ratio", f"{test_components['sharpe_ratio']:.2f}"],
            ["Max Drawdown", f"{test_components['max_drawdown']:.2f}%"],
            ["Win Rate", f"{test_components['win_rate']:.2f}%"],
            ["Profit Factor", f"{test_components['profit_factor']:.2f}"]
        ]
    )
    
    # 10. Rapport détaillé
    print_section(10, "Rapport détaillé du backtesting")
    print_subsection("Génération du rapport complet...")
    
    # Affichage du rapport dans une boîte
    initial_capital = test_result.equity_curve.iloc[0]
    final_capital = test_result.equity_curve.iloc[-1]
    report_lines = [
        f"Capital initial: {initial_capital:,.2f} €",
        f"Capital final:   {final_capital:,.2f} €",
        f"Rendement total: {test_result.total_return:.2f}%",
        f"Rendement annualisé: {test_result.annualized_return:.2f}%",
        "",
        f"Sharpe Ratio: {test_result.sharpe_ratio:.2f}",
        f"Maximum Drawdown: {test_result.max_drawdown:.2f}%",
        f"Win Rate: {test_result.win_rate:.2f}%",
        f"Profit Factor: {test_result.profit_factor:.2f}",
        "",
        f"Total trades: {test_result.total_trades}",
        f"Trades gagnants: {test_result.winning_trades}",
        f"Trades perdants: {test_result.losing_trades}"
    ]
    print_box("\n".join(report_lines))
    
    # 11. Walk-forward testing
    print_section(11, "Walk-forward testing (évitement du curve-fitting)")
    print_subsection("Exécution du walk-forward testing...")
    wf_results = backtester.walk_forward_test(
        df=train_data,
        strategy=best_strategy,
        train_size=126,
        test_size=42,
        step_size=21
    )
    
    if wf_results:
        avg_return = np.mean([r.total_return for r in wf_results])
        avg_sharpe = np.mean([r.sharpe_ratio for r in wf_results])
        avg_dd = np.mean([r.max_drawdown for r in wf_results])
        avg_win_rate = np.mean([r.win_rate for r in wf_results])
        
        print_success(f"Walk-forward testing terminé sur {len(wf_results)} fenêtres")
        print(f"\n{Colors.BOLD}Résultats agrégés du walk-forward testing:{Colors.END}")
        print_table(
            ["Métrique", "Valeur"],
            [
                ["Nombre de fenêtres", str(len(wf_results))],
                ["Rendement moyen", f"{avg_return:.2f}%"],
                ["Sharpe moyen", f"{avg_sharpe:.2f}"],
                ["Drawdown moyen", f"{avg_dd:.2f}%"],
                ["Win Rate moyen", f"{avg_win_rate:.2f}%"]
            ]
        )
    else:
        print_warning("Aucun résultat de walk-forward testing disponible")
    
    # 12. Création des graphiques
    if create_plots:
        print_section(12, "Création des graphiques")
        print_subsection("Génération des visualisations...")
        
        visualizer = Visualizer(output_dir="plots")
        
        # Courbe d'équité
        print_subsection("Courbe d'équité...")
        visualizer.plot_equity_curve(
            train_result=train_result,
            validation_result=validation_result,
            test_result=test_result,
            save_path="equity_curve.png",
            show=show_plots
        )
        print_success("Courbe d'équité générée")
        
        # Drawdown
        print_subsection("Drawdown...")
        visualizer.plot_drawdown(
            train_result=train_result,
            validation_result=validation_result,
            test_result=test_result,
            save_path="drawdown.png",
            show=show_plots
        )
        print_success("Graphique de drawdown généré")
        
        # Évolution du fitness
        print_subsection("Évolution du fitness...")
        visualizer.plot_fitness_evolution(
            history=ga.history,
            save_path="fitness_evolution.png",
            show=show_plots
        )
        print_success("Graphique d'évolution du fitness généré")
        
        # Comparaison des performances
        print_subsection("Comparaison des performances...")
        visualizer.plot_performance_comparison(
            train_fitness=stats['best_fitness'],
            validation_fitness=validation_fitness,
            test_fitness=test_fitness,
            validation_components=validation_components,
            test_components=test_components,
            save_path="performance_comparison.png",
            show=show_plots
        )
        print_success("Graphique de comparaison généré")
        
        # Walk-forward results
        if wf_results:
            print_subsection("Résultats du walk-forward testing...")
            visualizer.plot_walk_forward_results(
                wf_results=wf_results,
                save_path="walk_forward_results.png",
                show=show_plots
            )
            print_success("Graphique de walk-forward généré")
        
        # Paramètres de la stratégie
        print_subsection("Paramètres de la stratégie...")
        visualizer.plot_strategy_parameters(
            strategy=best_strategy,
            save_path="strategy_parameters.png",
            show=show_plots
        )
        print_success("Graphique des paramètres généré")
        
        # Tableau de bord complet
        print_subsection("Tableau de bord complet...")
        visualizer.plot_comprehensive_dashboard(
            train_result=train_result,
            validation_result=validation_result,
            test_result=test_result,
            history=ga.history,
            train_fitness=stats['best_fitness'],
            validation_fitness=validation_fitness,
            test_fitness=test_fitness,
            validation_components=validation_components,
            test_components=test_components,
            strategy=best_strategy,
            wf_results=wf_results,
            save_path="comprehensive_dashboard.png",
            show=show_plots
        )
        print_success("Tableau de bord complet généré")
        
        print_success(f"Tous les graphiques ont été sauvegardés dans le répertoire 'plots'")
    
    # Résultats finaux
    print_header("RÉSUMÉ FINAL")
    
    print(f"{Colors.BOLD}Informations générales:{Colors.END}")
    print_table(
        ["Paramètre", "Valeur"],
        [
            ["Symbole", symbol],
            ["Graine aléatoire", str(random_seed)],
            ["Date d'exécution", datetime.now().strftime('%d/%m/%Y %H:%M:%S')]
        ]
    )
    
    print(f"\n{Colors.BOLD}Comparaison des performances par ensemble:{Colors.END}")
    print_table(
        ["Ensemble", "Fitness", "Rendement", "Sharpe", "Drawdown"],
        [
            ["Entraînement", f"{stats['best_fitness']:.4f}", "-", "-", "-"],
            ["Validation", f"{validation_fitness:.4f}", f"{validation_components['total_return']:.2f}%", f"{validation_components['sharpe_ratio']:.2f}", f"{validation_components['max_drawdown']:.2f}%"],
            ["Test", f"{test_fitness:.4f}", f"{test_components['total_return']:.2f}%", f"{test_components['sharpe_ratio']:.2f}", f"{test_components['max_drawdown']:.2f}%"]
        ]
    )
    
    print(f"\n{Colors.BOLD}Paramètres de la meilleure stratégie:{Colors.END}")
    for key, value in params.items():
        print(f"  {Colors.CYAN}•{Colors.END} {key}: {Colors.GREEN}{value}{Colors.END}")
    
    print(f"\n{Colors.GREEN}{'=' * 70}{Colors.END}")
    print(f"{Colors.GREEN}{'✓ Exécution terminée avec succès !'.center(70)}{Colors.END}")
    print(f"{Colors.GREEN}{'=' * 70}{Colors.END}\n")
    
    return {
        'best_strategy': best_strategy,
        'best_genes': best_individual.genes,
        'training_fitness': stats['best_fitness'],
        'validation_fitness': validation_fitness,
        'test_fitness': test_fitness,
        'test_result': test_result,
        'validation_components': validation_components,
        'test_components': test_components,
        'walk_forward_results': wf_results,
        'ga_history': ga.history,
        'visualizer': visualizer if create_plots else None
    }


def main():
    """Point d'entrée principal du programme."""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}{'╔' + '═' * 68 + '╗'}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.YELLOW}{'║' + ' ' * 68 + '║'}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.YELLOW}{'║' + 'DÉMARRAGE DU PROGRAMME'.center(68) + '║'}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.YELLOW}{'║' + ' ' * 68 + '║'}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.YELLOW}{'╚' + '═' * 68 + '╝'}{Colors.END}\n")
    
    # Exemple d'utilisation avec des données simulées
    print(f"{Colors.CYAN}Configuration de l'exécution:{Colors.END}")
    print(f"  • Symbole: {Colors.GREEN}SP500{Colors.END}")
    print(f"  • Données: {Colors.GREEN}Simulées{Colors.END}")
    print(f"  • Générations: {Colors.GREEN}30{Colors.END}")
    print(f"  • Population: {Colors.GREEN}20{Colors.END}")
    print(f"  • Ratio entraînement: {Colors.GREEN}70%{Colors.END}")
    print(f"  • Graine aléatoire: {Colors.GREEN}42{Colors.END}\n")
    
    results = run_genetic_trading_optimization(
        symbol="SP500",
        use_sample_data=True,
        generations=30,
        population_size=20,
        train_ratio=0.7,
        random_seed=42,
        use_cross_validation=True,
        cv_folds=3,
        adaptive_mutation=True,
        create_plots=True,
        show_plots=True
    )
    
    # Sauvegarder les résultats
    print(f"{Colors.YELLOW}ℹ Résultats sauvegardés dans la variable 'results'.{Colors.END}")
    print(f"{Colors.YELLOW}ℹ Pour accéder aux résultats, utilisez: results['best_strategy'], results['test_fitness'], etc.{Colors.END}\n")
    
    return results


if __name__ == "__main__":
    main()
