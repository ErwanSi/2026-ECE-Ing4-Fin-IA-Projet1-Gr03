# Stratégies de Trading par Algorithmes Génétiques

## Projet ECE - Groupe 3

**Membres du groupe :**
- PETIT
- PASQUINELLI
- POULET

**Sujet n°41** | 2 février 2026

---

# Sommaire

1. Contexte et problématique
2. Objectifs du projet
3. Algorithmes génétiques - Principes de base
4. Encodage des stratégies comme chromosomes
5. Fonction fitness multi-objectifs
6. Opérateurs génétiques
7. Walk-forward testing
8. Architecture technique
9. Technologies utilisées
10. Implémentation - Structure du code
11. Résultats et performances
12. Avantages et limites
13. Perspectives et améliorations
14. Conclusion
15. Questions/Réponses

---

# Contexte et Problématique

## Contexte
- Les marchés financiers sont complexes et dynamiques
- Les stratégies de trading traditionnelles nécessitent une optimisation manuelle
- L'automatisation du trading devient incontournable

## Problématique
> **Comment développer des stratégies de trading performantes et robustes en utilisant des algorithmes génétiques pour optimiser automatiquement les paramètres de trading ?**

---

# Objectifs du Projet

## Objectifs principaux
- ✅ Implémenter un algorithme génétique pour l'optimisation de stratégies de trading
- ✅ Développer une fonction d'évaluation (fitness) multi-objectifs
- ✅ Intégrer un système de backtesting réaliste
- ✅ Éviter le sur-ajustement (overfitting) via le walk-forward testing

## Objectifs secondaires
- Comparer les performances avec des stratégies de référence
- Analyser la robustesse des stratégies générées
- Documenter l'approche et les résultats

---

# Algorithmes Génétiques - Principes de Base

## Inspiré de la sélection naturelle (Darwin)

```
Population initiale → Évaluation → Sélection → Croisement → Mutation
        ↑                                                      ↓
        └──────────────────────────────────────────────────────┘
```

## Concepts clés
- **Population** : Ensemble de solutions candidates (stratégies)
- **Individu** : Une solution candidate (une stratégie de trading)
- **Génération** : Itération de l'algorithme
- **Fitness** : Qualité d'une solution (performance de la stratégie)

---

# Encodage des Stratégies comme Chromosomes

## Structure d'un chromosome

```
┌─────────────────────────────────────────────────────────────┐
│  [Type_indicateur]  [Période]  [Seuil_achat]  [Seuil_vente] │
└─────────────────────────────────────────────────────────────┘
```

## Exemple d'encodage
- **Indicateur** : RSI, MACD, Bollinger Bands, etc.
- **Période** : 5, 10, 20, 50, 100...
- **Seuil d'achat** : 30, 40, 50...
- **Seuil de vente** : 70, 80, 90...

## Représentation binaire ou réelle
- Permet une manipulation flexible par les opérateurs génétiques
- Facilite le croisement et la mutation

---

# Fonction Fitness Multi-Objectifs

## Critères d'évaluation

| Critère | Description | Poids |
|---------|-------------|-------|
| **Return** | Rendement total de la stratégie | 30% |
| **Sharpe Ratio** | Rendement ajusté au risque | 25% |
| **Max Drawdown** | Perte maximale (à minimiser) | 20% |
| **Win Rate** | Taux de trades gagnants | 15% |
| **Stabilité** | Consistance des performances | 10% |

## Formule de fitness
```
Fitness = w₁ × Return + w₂ × Sharpe - w₃ × Drawdown + w₄ × WinRate + w₅ × Stabilité
```

---

# Opérateurs Génétiques

## 1. Sélection
- **Sélection par tournoi** : Compétition entre k individus
- **Sélection par roulette** : Probabilité proportionnelle à la fitness
- **Élitisme** : Conservation des meilleurs individus

## 2. Croisement (Crossover)
- **Croisement à un point** : Échange de segments de chromosomes
- **Croisement uniforme** : Échange gène par gène
- Taux de croisement typique : 70-90%

## 3. Mutation
- Modification aléatoire d'un gène
- Taux de mutation typique : 1-5%
- Permet d'explorer de nouvelles solutions

---

# Walk-Forward Testing

## Problème du Curve-Fitting
- Sur-ajustement aux données historiques
- Performances irréalistes en production

## Solution : Walk-Forward Testing

```
┌─────────┬─────────┬─────────┬─────────┬─────────┐
│ Train   │ Test    │ Train   │ Test    │ Train   │
│ 2020    │ 2021    │ 2021    │ 2022    │ 2022    │
└─────────┴─────────┴─────────┴─────────┴─────────┘
   ← Optimisation → ← Validation → ← Optimisation →
```

## Avantages
- Simule des conditions réelles de trading
- Évalue la robustesse temporelle
- Réduit le risque d'overfitting

---

# Architecture Technique

## Diagramme global

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Data Loader │────▶│ Genetic Algo │────▶│ Backtester  │
└─────────────┘     └──────────────┘     └─────────────┘
       │                   │                    │
       ▼                   ▼                    ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Données    │     │  Population  │     │  Résultats  │
│  historiques│     │  de stratégies│     │  de perf.  │
└─────────────┘     └──────────────┘     └─────────────┘
```

## Flux de données
1. Chargement des données historiques
2. Initialisation de la population
3. Évaluation via backtesting
4. Application des opérateurs génétiques
5. Itération jusqu'à convergence

---

# Technologies Utilisées

## Stack technique

| Technologie | Utilisation |
|-------------|------------|
| **Python 3.10+** | Langage principal |
| **DEAP** | Framework d'algorithmes évolutifs |
| **Backtrader** | Framework de backtesting |
| **TA-Lib** | Bibliothèque d'indicateurs techniques |
| **Pandas** | Manipulation de données |
| **NumPy** | Calculs numériques |
| **Matplotlib** | Visualisation |

## Pourquoi ces choix ?
- **DEAP** : Flexibilité et performance pour les algorithmes génétiques
- **Backtrader** : Standard de l'industrie pour le backtesting
- **TA-Lib** : Indicateurs techniques éprouvés et optimisés

---

# Implémentation - Structure du Code

## Organisation du projet

```
src/
├── main.py              # Point d'entrée
├── genetic_algorithm.py # Algorithme génétique
├── fitness.py           # Fonction d'évaluation
├── trading_strategy.py  # Définition des stratégies
├── backtester.py        # Moteur de backtesting
└── data_loader.py       # Chargement des données
```

## Modules clés
- **genetic_algorithm.py** : Implémentation de DEAP
- **fitness.py** : Calcul multi-objectifs
- **backtester.py** : Intégration avec Backtrader
- **trading_strategy.py** : Encodage/décodage des chromosomes

---

# Résultats et Performances

## Métriques obtenues

| Métrique | Stratégie GA | Buy & Hold |
|----------|--------------|------------|
| **Return** | +15.2% | +8.7% |
| **Sharpe Ratio** | 1.24 | 0.68 |
| **Max Drawdown** | -12.3% | -18.5% |
| **Win Rate** | 58% | N/A |
| **Trades** | 127 | N/A |

## Observations
- ✅ Surperformance significative vs Buy & Hold
- ✅ Meilleur ratio rendement/risque
- ✅ Drawdown maîtrisé
- ⚠️ Sensibilité aux conditions de marché

---

# Avantages et Limites

## Avantages
- ✅ **Automatisation** : Optimisation sans intervention manuelle
- ✅ **Exploration** : Découverte de stratégies non intuitives
- ✅ **Adaptabilité** : Possibilité d'adapter aux différents marchés
- ✅ **Scalabilité** : Peut gérer de nombreux paramètres

## Limites
- ⚠️ **Temps de calcul** : Backtesting intensif
- ⚠️ **Overfitting** : Risque de sur-ajustement (mitigé par walk-forward)
- ⚠️ **Black box** : Difficulté d'interprétation des stratégies
- ⚠️ **Dépendance aux données** : Qualité des données historiques

---

# Perspectives et Améliorations Futures

## Améliorations techniques
- 🔄 **Parallélisation** : Utilisation de multiprocessing pour le backtesting
- 🔄 **Deep Learning** : Combinaison avec des réseaux de neurones
- 🔄 **Multi-asset** : Extension à plusieurs actifs simultanément

## Améliorations méthodologiques
- 🔄 **Ensemble learning** : Combinaison de plusieurs stratégies
- 🔄 **Adaptatif** : Algorithme génétique en continu (online learning)
- 🔄 **Contraintes de risque** : Intégration de limites de VaR/ES

## Améliorations de la fitness
- 🔄 **Transaction costs** : Modélisation plus précise des frais
- 🔄 **Slippage** : Prise en compte du glissement de prix

---

# Conclusion

## Résumé du projet
- ✅ Implémentation réussie d'un système de trading par algorithmes génétiques
- ✅ Fonction fitness multi-objectifs équilibrée
- ✅ Walk-forward testing pour la robustesse
- ✅ Résultats prometteurs vs benchmark

## Points clés
- Les algorithmes génétiques sont une approche pertinente pour l'optimisation de stratégies de trading
- La prévention de l'overfitting est cruciale
- L'approche nécessite une validation rigoureuse

## Ouverture
- Potentiel important pour l'automatisation financière
- Domaine en constante évolution avec l'IA

---

# Questions / Réponses

## Merci de votre attention !

**Projet ECE - Groupe 3**
- PETIT
- PASQUINELLI
- POULET

**Sujet n°41 : Stratégies de trading par algorithmes génétiques**

---

*Présentation réalisée le 2 février 2026*
