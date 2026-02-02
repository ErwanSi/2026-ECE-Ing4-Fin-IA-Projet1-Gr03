# Stratégies de trading par algorithmes génétiques

**Projet pédagogique ECE - Groupe 3 - Sujet n°41**

---

## 🚀 Guide d'exécution rapide

Pour exécuter le projet, consultez le **[Guide d'Exécution](EXECUTION_GUIDE.md)** qui contient :
- Instructions détaillées pour l'installation des dépendances
- Comment exécuter le script principal (`main.py`)
- Comment exécuter le notebook de démonstration (`demo.ipynb`)
- Description de l'affichage des résultats

**Commande rapide :**
```bash
# Installer les dépendances
pip install -r requirements.txt

# Exécuter le script principal
python src/main.py
```

---

## 📋 Description du problème et contexte

L'optimisation de stratégies de trading algorithmique nécessite d'explorer un espace combinatoire immense de règles et paramètres. Les algorithmes génétiques permettent d'évoluer des populations de stratégies, combinant indicateurs techniques et règles de décision, tout en évitant le surapprentissage grâce à des techniques de validation robustes.

Ce projet vise à développer un système d'optimisation de stratégies de trading basé sur les algorithmes génétiques, capable de générer et d'améliorer automatiquement des stratégies performantes sur les marchés financiers.

---

## 👥 Membres du groupe

- **PETIT**
- **PASQUINELLI**
- **POULET**

---

## 🎯 Objectifs du projet

1. **Encoder les stratégies comme chromosomes** : Représenter les indicateurs techniques, seuils et règles de décision sous forme de chromosomes manipulables par les algorithmes génétiques.

2. **Définir une fonction fitness multi-objectifs** : Évaluer les stratégies selon plusieurs critères de performance (rendement, Sharpe ratio, maximum drawdown).

3. **Implémenter les opérateurs génétiques** : Adapter les mécanismes de sélection, croisement et mutation au domaine financier.

4. **Valider avec le walk-forward testing** : Éviter le curve-fitting en utilisant des techniques de validation robustes sur des périodes de test distinctes.

---

## 🔬 Approches techniques

### Encodage des stratégies
- Représentation des indicateurs techniques (RSI, MACD, Bollinger Bands, etc.)
- Paramétrage des seuils d'achat/vente
- Codage des règles de décision logiques

### Fonction fitness multi-objectifs
- **Rendement** : Profitabilité totale de la stratégie
- **Sharpe Ratio** : Rendement ajusté au risque
- **Maximum Drawdown** : Perte maximale subie
- **Stabilité** : Consistance des performances dans le temps

### Opérateurs génétiques adaptés
- **Sélection** : Tournoi, roulette, ou sélection par rang
- **Croisement** : Recombinaison de règles et paramètres entre stratégies
- **Mutation** : Variation aléatoire des seuils et indicateurs

### Walk-forward testing
- Division des données en périodes d'entraînement et de test
- Validation sur des données non utilisées lors de l'optimisation
- Prévention du surapprentissage (curve-fitting)

---

## 🛠️ Technologies utilisées

### Langage principal
- **Python 3.9+** : Langage de développement principal

### Algorithmes génétiques
- **DEAP** (Distributed Evolutionary Algorithms in Python) : Framework flexible pour les algorithmes évolutifs
- **PyGAD** : Alternative pour les algorithmes génétiques en Python

### Backtesting
- **Backtrader** : Framework de backtesting de stratégies de trading
- **Zipline** : Alternative pour le backtesting (Quantopian)

### Indicateurs techniques
- **TA-Lib** : Bibliothèque d'analyse technique avec plus de 150 indicateurs

### Validation et données
- **QuantConnect** : Plateforme pour validation sur données réelles
- **Pandas** : Manipulation et analyse de données financières
- **NumPy** : Calculs numériques performants

### Visualisation
- **Matplotlib** : Graphiques et visualisations
- **Plotly** : Visualisations interactives

---

## 📁 Structure du projet

```
grp3-41-genetic-trading/
│
├── README.md                 # Documentation principale du projet
├── EXECUTION_GUIDE.md        # Guide d'exécution détaillé ⭐
├── requirements.txt          # Dépendances Python
│
├── src/                      # Code source
│   ├── genetic/              # Implémentation des algorithmes génétiques
│   │   ├── chromosome.py     # Encodage des stratégies
│   │   ├── fitness.py        # Fonction d'évaluation multi-objectifs
│   │   ├── selection.py     # Opérateurs de sélection
│   │   ├── crossover.py      # Opérateurs de croisement
│   │   └── mutation.py       # Opérateurs de mutation
│   │
│   ├── backtesting/          # Module de backtesting
│   │   ├── strategy.py       # Définition des stratégies
│   │   ├── engine.py         # Moteur de backtesting
│   │   └── metrics.py        # Calcul des métriques de performance
│   │
│   ├── indicators/           # Indicateurs techniques
│   │   ├── technical.py      # Indicateurs TA-Lib
│   │   └── custom.py         # Indicateurs personnalisés
│   │
│   ├── validation/           # Validation et walk-forward testing
│   │   ├── walk_forward.py   # Implémentation du walk-forward testing
│   │   └── cross_val.py      # Validation croisée
│   │
│   └── main.py               # Point d'entrée principal
│
├── docs/                     # Documentation technique
│   ├── architecture.md       # Architecture du système
│   ├── algorithms.md         # Documentation des algorithmes
│   └── api.md                # Documentation de l'API
│
├── slides/                   # Support de présentation
│   └── presentation.pptx     # Diapositives pour la soutenance
│
├── data/                     # Données
│   ├── raw/                  # Données brutes
│   ├── processed/            # Données traitées
│   └── results/              # Résultats des expériences
│
└── notebooks/                # Jupyter notebooks
    ├── demo.ipynb               # Démonstration complète ⭐
    ├── 01_exploration.ipynb      # Exploration des données
    ├── 02_indicators.ipynb       # Analyse des indicateurs
    ├── 03_genetic_algo.ipynb     # Tests des algorithmes génétiques
    └── 04_results.ipynb          # Analyse des résultats
```

> ⭐ **Fichiers importants :**
> - [`EXECUTION_GUIDE.md`](EXECUTION_GUIDE.md) : Guide complet pour exécuter le projet
> - [`notebooks/demo.ipynb`](notebooks/demo.ipynb) : Notebook de démonstration avec visualisations

---

## 📦 Procédure d'installation

> 💡 **Note :** Pour des instructions détaillées sur l'exécution du projet, consultez le **[Guide d'Exécution](EXECUTION_GUIDE.md)**.

### Prérequis
- Python 3.9 ou supérieur
- pip (gestionnaire de paquets Python)

### Installation des dépendances

1. Cloner ou télécharger le projet :
```bash
cd grp3-41-genetic-trading
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

> ⚠️ **Important :** Pour des instructions détaillées sur l'installation de TA-Lib (qui peut nécessiter des étapes supplémentaires selon votre système), consultez le **[Guide d'Exécution](EXECUTION_GUIDE.md)**.

### Contenu de requirements.txt
```
# Algorithmes génétiques
deap>=1.3.1
pygad>=2.18.0

# Backtesting
backtrader>=1.9.78.123

# Indicateurs techniques
TA-Lib>=0.4.28

# Manipulation de données
pandas>=1.5.0
numpy>=1.23.0

# Visualisation
matplotlib>=3.6.0
plotly>=5.11.0

# Jupyter
jupyter>=1.0.0
notebook>=6.5.0

# Utilitaires
scipy>=1.9.0
scikit-learn>=1.1.0
tqdm>=4.64.0
```

---

## 🚀 Instructions d'utilisation

> 💡 **Pour des instructions détaillées, consultez le [Guide d'Exécution](EXECUTION_GUIDE.md)**

### Lancer le programme principal

```bash
python src/main.py
```

### Exécuter les notebooks Jupyter

```bash
jupyter notebook notebooks/
```

> 📖 Le notebook [`demo.ipynb`](notebooks/demo.ipynb) contient une démonstration complète avec visualisations interactives.

### Configuration des paramètres

Les paramètres de l'algorithme génétique peuvent être configurés dans `src/main.py` :

```python
# Paramètres de la population
POPULATION_SIZE = 100
GENERATIONS = 50

# Paramètres des opérateurs
CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.2

# Paramètres de la fitness
WEIGHT_RETURN = 0.4
WEIGHT_SHARPE = 0.4
WEIGHT_DRAWDOWN = 0.2
```

### Exemple d'utilisation

```python
from src.genetic import GeneticOptimizer
from src.backtesting import BacktestEngine

# Initialiser l'optimiseur génétique
optimizer = GeneticOptimizer(
    population_size=100,
    generations=50,
    crossover_prob=0.8,
    mutation_prob=0.2
)

# Lancer l'optimisation
best_strategy = optimizer.optimize(data)

# Backtester la meilleure stratégie
engine = BacktestEngine()
results = engine.run(best_strategy, data)

# Afficher les résultats
print(f"Rendement: {results['return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

---

## 🧪 Tests

### Exécuter les tests unitaires

```bash
python -m pytest tests/
```

### Tests disponibles

- `tests/test_chromosome.py` : Tests de l'encodage des chromosomes
- `tests/test_fitness.py` : Tests de la fonction fitness
- `tests/test_operators.py` : Tests des opérateurs génétiques
- `tests/test_backtesting.py` : Tests du moteur de backtesting
- `tests/test_validation.py` : Tests du walk-forward testing

---

## 📊 Résultats attendus

À l'issue du projet, nous devrions obtenir :

1. Une bibliothèque de stratégies de trading optimisées
2. Des métriques de performance comparatives
3. Une analyse de la robustesse des stratégies
4. Une documentation technique complète
5. Une présentation des résultats

---

## 📅 Calendrier du projet

- **Phase 1** : Étude bibliographique et conception
- **Phase 2** : Implémentation des algorithmes génétiques
- **Phase 3** : Intégration avec le backtesting
- **Phase 4** : Validation et optimisation
- **Phase 5** : Analyse des résultats et documentation

---

## 📚 Références

- Holland, J. H. (1992). *Adaptation in Natural and Artificial Systems*
- Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*
- Chan, E. P. (2009). *Quantitative Trading: How to Build Your Own Algorithmic Trading Business*

---

## 📝 Licence

Ce projet est réalisé dans le cadre du cursus ECE Ingénieur 4ème année.

---

**Date de création** : 11 janvier 2026  
**Présentation finale** : 2 février 2026
