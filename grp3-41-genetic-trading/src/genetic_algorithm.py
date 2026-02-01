"""
Module d'implémentation de l'algorithme génétique pour les stratégies de trading.

Ce module contient la classe GeneticAlgorithm qui implémente les opérations
fondamentales d'un algorithme génétique : initialisation, sélection, croisement,
mutation et évolution de la population.

Améliorations pour éviter l'overfitting et améliorer la convergence :
- Adaptation dynamique des taux de mutation
- Maintien de la diversité de la population
- Early stopping basé sur la stagnation du fitness
- Niching pour éviter la convergence prématurée
- Crowding pour préserver la diversité
"""

import random
import numpy as np
from typing import List, Tuple, Callable, Any, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class Individual:
    """Représente un individu (chromosome) dans la population."""
    genes: np.ndarray
    fitness: float = 0.0
    
    def __len__(self):
        return len(self.genes)


class GeneticAlgorithm:
    """
    Implémentation d'un algorithme génétique pour l'optimisation de stratégies de trading.
    
    Les chromosomes encodent les paramètres des stratégies de trading :
    - Indicateurs techniques (SMA, RSI, MACD, etc.)
    - Seuils d'achat/vente
    - Règles de décision
    """
    
    def __init__(
        self,
        population_size: int = 50,
        chromosome_length: int = 10,
        gene_bounds: List[Tuple[float, float]] = None,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.2,
        selection_method: str = "tournament",
        tournament_size: int = 3,
        elitism_count: int = 2,
        random_seed: int = None,
        adaptive_mutation: bool = True,
        diversity_threshold: float = 0.1,
        early_stopping_patience: int = 15,
        early_stopping_min_delta: float = 0.001,
        use_crowding: bool = True,
        crowding_distance: float = 0.5
    ):
        """
        Initialise l'algorithme génétique.
        
        Args:
            population_size: Taille de la population
            chromosome_length: Nombre de gènes par chromosome
            gene_bounds: Bornes pour chaque gène [(min, max), ...]
            crossover_rate: Taux de croisement (0-1)
            mutation_rate: Taux de mutation (0-1)
            mutation_strength: Amplitude de la mutation
            selection_method: Méthode de sélection ('tournament', 'roulette', 'rank')
            tournament_size: Taille du tournoi pour la sélection par tournoi
            elitism_count: Nombre d'élites conservés à chaque génération
            random_seed: Graine pour la reproductibilité
            adaptive_mutation: Activer l'adaptation dynamique du taux de mutation
            diversity_threshold: Seuil de diversité pour déclencher l'adaptation
            early_stopping_patience: Nombre de générations sans amélioration avant l'arrêt
            early_stopping_min_delta: Amélioration minimale considérée comme significative
            use_crowding: Utiliser le crowding pour préserver la diversité
            crowding_distance: Distance pour le crowding
        """
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        
        # Paramètres anti-overfitting et amélioration de convergence
        self.adaptive_mutation = adaptive_mutation
        self.diversity_threshold = diversity_threshold
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.use_crowding = use_crowding
        self.crowding_distance = crowding_distance
        
        # Variables pour l'adaptation dynamique
        self.base_mutation_rate = mutation_rate
        self.base_mutation_strength = mutation_strength
        self.stagnation_counter = 0
        self.best_fitness_history = deque(maxlen=early_stopping_patience)
        
        # Bornes par défaut pour les gènes
        if gene_bounds is None:
            self.gene_bounds = [(0, 100)] * chromosome_length
        else:
            self.gene_bounds = gene_bounds
        
        # Initialisation du générateur aléatoire
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Individual = None
        self.history: List[dict] = []
        self.early_stopped = False
    
    def initialize_population(self) -> List[Individual]:
        """
        Initialise la population avec des individus aléatoires.
        
        Returns:
            Liste des individus initialisés
        """
        self.population = []
        for _ in range(self.population_size):
            genes = np.array([
                random.uniform(bounds[0], bounds[1])
                for bounds in self.gene_bounds
            ])
            individual = Individual(genes=genes)
            self.population.append(individual)
        
        self.generation = 0
        return self.population
    
    def fitness(self, individual: Individual, fitness_func: Callable) -> float:
        """
        Évalue le fitness d'un individu.
        
        Args:
            individual: L'individu à évaluer
            fitness_func: Fonction de fitness externe
            
        Returns:
            Score de fitness
        """
        score = fitness_func(individual.genes)
        individual.fitness = score
        return score
    
    def evaluate_population(self, fitness_func: Callable) -> None:
        """
        Évalue tous les individus de la population.
        
        Args:
            fitness_func: Fonction de fitness
        """
        for individual in self.population:
            self.fitness(individual, fitness_func)
        
        # Met à jour le meilleur individu
        self.best_individual = max(self.population, key=lambda x: x.fitness)
    
    def selection(self) -> Individual:
        """
        Sélectionne un individu pour la reproduction.
        
        Returns:
            L'individu sélectionné
        """
        if self.selection_method == "tournament":
            return self._tournament_selection()
        elif self.selection_method == "roulette":
            return self._roulette_selection()
        elif self.selection_method == "rank":
            return self._rank_selection()
        else:
            return self._tournament_selection()
    
    def _tournament_selection(self) -> Individual:
        """Sélection par tournoi."""
        tournament = random.sample(self.population, min(self.tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness)
    
    def _roulette_selection(self) -> Individual:
        """Sélection par roulette wheel."""
        # Normaliser les fitness pour éviter les valeurs négatives
        min_fitness = min(ind.fitness for ind in self.population)
        adjusted_fitness = [ind.fitness - min_fitness + 1e-6 for ind in self.population]
        total_fitness = sum(adjusted_fitness)
        
        if total_fitness == 0:
            return random.choice(self.population)
        
        pick = random.uniform(0, total_fitness)
        current = 0
        for i, fitness in enumerate(adjusted_fitness):
            current += fitness
            if current > pick:
                return self.population[i]
        
        return self.population[-1]
    
    def _rank_selection(self) -> Individual:
        """Sélection par rang."""
        sorted_pop = sorted(self.population, key=lambda x: x.fitness)
        ranks = list(range(1, len(sorted_pop) + 1))
        total_ranks = sum(ranks)
        
        pick = random.uniform(0, total_ranks)
        current = 0
        for i, rank in enumerate(ranks):
            current += rank
            if current > pick:
                return sorted_pop[i]
        
        return sorted_pop[-1]
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Effectue le croisement entre deux parents.
        
        Args:
            parent1: Premier parent
            parent2: Deuxième parent
            
        Returns:
            Deux enfants résultants du croisement
        """
        if random.random() > self.crossover_rate:
            return Individual(genes=parent1.genes.copy()), Individual(genes=parent2.genes.copy())
        
        # Croisement à un point (single-point crossover)
        crossover_point = random.randint(1, self.chromosome_length - 1)
        
        child1_genes = np.concatenate([
            parent1.genes[:crossover_point],
            parent2.genes[crossover_point:]
        ])
        
        child2_genes = np.concatenate([
            parent2.genes[:crossover_point],
            parent1.genes[crossover_point:]
        ])
        
        return Individual(genes=child1_genes), Individual(genes=child2_genes)
    
    def mutation(self, individual: Individual) -> Individual:
        """
        Applique une mutation à un individu.
        
        Args:
            individual: L'individu à muter
            
        Returns:
            L'individu muté
        """
        mutated_genes = individual.genes.copy()
        
        for i in range(len(mutated_genes)):
            if random.random() < self.mutation_rate:
                # Mutation gaussienne
                delta = random.gauss(0, self.mutation_strength)
                mutated_genes[i] += delta
                
                # Clamp aux bornes
                mutated_genes[i] = max(
                    self.gene_bounds[i][0],
                    min(self.gene_bounds[i][1], mutated_genes[i])
                )
        
        return Individual(genes=mutated_genes)
    
    def evolve(self, fitness_func: Callable, generations: int = 100) -> Individual:
        """
        Fait évoluer la population sur plusieurs générations avec mécanismes anti-overfitting.
        
        Args:
            fitness_func: Fonction de fitness
            generations: Nombre de générations
            
        Returns:
            Le meilleur individu final
        """
        if not self.population:
            self.initialize_population()
        
        self.evaluate_population(fitness_func)
        
        for gen in range(generations):
            self.generation += 1
            
            # Calculer la diversité de la population
            diversity = self._calculate_population_diversity()
            
            # Adaptation dynamique du taux de mutation
            if self.adaptive_mutation:
                self._adapt_mutation_rate(diversity)
            
            # Tri par fitness
            sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
            
            # Élitisme : conserver les meilleurs
            new_population = sorted_pop[:self.elitism_count]
            
            # Créer la nouvelle génération
            while len(new_population) < self.population_size:
                # Sélection
                parent1 = self.selection()
                parent2 = self.selection()
                
                # Croisement
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutation avec taux adaptatif
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                
                # Crowding pour préserver la diversité
                if self.use_crowding:
                    child1 = self._apply_crowding(child1, parent1, parent2)
                    child2 = self._apply_crowding(child2, parent1, parent2)
                
                new_population.extend([child1, child2])
            
            # Ajuster la taille de la population
            self.population = new_population[:self.population_size]
            
            # Évaluer la nouvelle population
            self.evaluate_population(fitness_func)
            
            # Enregistrer l'historique
            self.history.append({
                'generation': self.generation,
                'best_fitness': self.best_individual.fitness,
                'avg_fitness': np.mean([ind.fitness for ind in self.population]),
                'worst_fitness': min(ind.fitness for ind in self.population),
                'diversity': diversity,
                'mutation_rate': self.mutation_rate
            })
            
            # Vérifier l'early stopping
            if self._check_early_stopping():
                print(f"Early stopping déclenché à la génération {self.generation}")
                self.early_stopped = True
                break
            
            # Affichage de progression
            if gen % 10 == 0 or gen == generations - 1:
                print(f"Génération {gen + 1}/{generations} - "
                      f"Meilleur fitness: {self.best_individual.fitness:.4f} - "
                      f"Moyenne: {self.history[-1]['avg_fitness']:.4f} - "
                      f"Diversité: {diversity:.4f} - "
                      f"Mutation: {self.mutation_rate:.4f}")
        
        return self.best_individual
    
    def get_statistics(self) -> dict:
        """
        Retourne les statistiques de l'évolution.
        
        Returns:
            Dictionnaire contenant les statistiques
        """
        if not self.history:
            return {}
        
        initial_best_fitness = self.history[0]['best_fitness']
        initial_avg_fitness = self.history[0]['avg_fitness']
        final_best_fitness = self.history[-1]['best_fitness']
        final_avg_fitness = self.history[-1]['avg_fitness']
        improvement = final_best_fitness - initial_best_fitness
        improvement_rate = (improvement / abs(initial_best_fitness)) if initial_best_fitness != 0 else 0.0
        
        return {
            'total_generations': self.generation,
            'initial_best_fitness': initial_best_fitness,
            'best_fitness': final_best_fitness,
            'best_genes': self.best_individual.genes,
            'initial_avg_fitness': initial_avg_fitness,
            'final_avg_fitness': final_avg_fitness,
            'improvement': improvement,
            'improvement_rate': improvement_rate,
            'early_stopped': self.early_stopped,
            'final_diversity': self.history[-1]['diversity'] if self.history else 0.0
        }
    
    def _calculate_population_diversity(self) -> float:
        """
        Calcule la diversité de la population.
        
        Utilise la distance moyenne entre tous les individus.
        
        Returns:
            Score de diversité (entre 0 et 1)
        """
        if len(self.population) < 2:
            return 0.0
        
        # Normaliser les gènes pour le calcul de distance
        normalized_genes = []
        for ind in self.population:
            norm_genes = [(ind.genes[i] - self.gene_bounds[i][0]) /
                         (self.gene_bounds[i][1] - self.gene_bounds[i][0])
                         for i in range(len(ind.genes))]
            normalized_genes.append(np.array(norm_genes))
        
        # Calculer la distance moyenne
        total_distance = 0.0
        count = 0
        
        for i in range(len(normalized_genes)):
            for j in range(i + 1, len(normalized_genes)):
                distance = np.linalg.norm(normalized_genes[i] - normalized_genes[j])
                total_distance += distance
                count += 1
        
        if count == 0:
            return 0.0
        
        avg_distance = total_distance / count
        
        # Normaliser entre 0 et 1 (distance max théorique = sqrt(n_genes))
        max_distance = np.sqrt(self.chromosome_length)
        diversity = avg_distance / max_distance
        
        return diversity
    
    def _adapt_mutation_rate(self, diversity: float):
        """
        Adapte dynamiquement le taux de mutation en fonction de la diversité.
        
        Si la diversité est faible, augmenter le taux de mutation.
        Si la diversité est élevée, diminuer le taux de mutation.
        
        Args:
            diversity: Diversité actuelle de la population
        """
        if diversity < self.diversity_threshold:
            # Diversité faible : augmenter la mutation
            self.mutation_rate = min(0.5, self.base_mutation_rate * 2.0)
            self.mutation_strength = min(1.0, self.base_mutation_strength * 1.5)
        else:
            # Diversité suffisante : revenir aux valeurs de base
            self.mutation_rate = self.base_mutation_rate
            self.mutation_strength = self.base_mutation_strength
    
    def _check_early_stopping(self) -> bool:
        """
        Vérifie si l'early stopping doit être déclenché.
        
        Returns:
            True si l'early stopping doit être déclenché
        """
        current_best = self.best_individual.fitness
        self.best_fitness_history.append(current_best)
        
        if len(self.best_fitness_history) < self.early_stopping_patience:
            return False
        
        # Vérifier si l'amélioration est significative
        best_in_history = max(self.best_fitness_history)
        improvement = current_best - best_in_history
        
        if improvement < self.early_stopping_min_delta:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        
        return self.stagnation_counter >= self.early_stopping_patience
    
    def _apply_crowding(
        self,
        child: Individual,
        parent1: Individual,
        parent2: Individual
    ) -> Individual:
        """
        Applique le crowding pour préserver la diversité.
        
        Remplace l'enfant par le parent le plus proche si l'enfant est trop proche.
        
        Args:
            child: Enfant à évaluer
            parent1: Premier parent
            parent2: Deuxième parent
            
        Returns:
            Individu à conserver (enfant ou parent)
        """
        # Calculer les distances normalisées
        def normalized_distance(ind1: Individual, ind2: Individual) -> float:
            norm_genes1 = np.array([(ind1.genes[i] - self.gene_bounds[i][0]) /
                                   (self.gene_bounds[i][1] - self.gene_bounds[i][0])
                                   for i in range(len(ind1.genes))])
            norm_genes2 = np.array([(ind2.genes[i] - self.gene_bounds[i][0]) /
                                   (self.gene_bounds[i][1] - self.gene_bounds[i][0])
                                   for i in range(len(ind2.genes))])
            return np.linalg.norm(norm_genes1 - norm_genes2)
        
        dist_to_p1 = normalized_distance(child, parent1)
        dist_to_p2 = normalized_distance(child, parent2)
        
        # Si l'enfant est trop proche des parents, le remplacer
        if dist_to_p1 < self.crowding_distance and dist_to_p2 < self.crowding_distance:
            # Choisir le parent avec le meilleur fitness
            return parent1 if parent1.fitness > parent2.fitness else parent2
        
        return child
    
    def reset_early_stopping(self):
        """Réinitialise les compteurs d'early stopping."""
        self.stagnation_counter = 0
        self.best_fitness_history.clear()
        self.early_stopped = False
