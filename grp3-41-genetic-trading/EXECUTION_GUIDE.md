# Guide d'Exécution - Stratégies de Trading par Algorithmes Génétiques

**Projet ECE - Groupe 3 - Sujet n°41**

---

## 📋 Table des matières

1. [Prérequis](#prérequis)
2. [Installation des dépendances](#installation-des-dépendances)
3. [Exécution du script principal](#exécution-du-script-principal)
4. [Exécution du notebook de démonstration](#exécution-du-notebook-de-démonstration)
5. [Affichage des résultats](#affichage-des-résultats)
6. [Personnalisation des paramètres](#personnalisation-des-paramètres)
7. [Résolution des problèmes courants](#résolution-des-problèmes-courants)

---

## Prérequis

Avant d'exécuter le projet, assurez-vous d'avoir les éléments suivants installés sur votre machine :

- **Python 3.9 ou supérieur**
- **pip** (gestionnaire de paquets Python)
- **Git** (optionnel, pour cloner le dépôt)

### Vérification de la version de Python

```bash
python --version
# ou
python3 --version
```

Si Python n'est pas installé, téléchargez-le depuis [python.org](https://www.python.org/downloads/).

---

## Installation des dépendances

### Étape 1 : Naviguer vers le dossier du projet

```bash
cd grp3-41-genetic-trading
```

### Étape 2 : Créer un environnement virtuel (recommandé)

```bash
# Sur Windows
python -m venv venv

# Sur macOS/Linux
python3 -m venv venv
```

### Étape 3 : Activer l'environnement virtuel

```bash
# Sur Windows
venv\Scripts\activate

# Sur macOS/Linux
source venv/bin/activate
```

### Étape 4 : Installer les dépendances

```bash
pip install -r requirements.txt
```

### Contenu de requirements.txt

Le fichier `requirements.txt` contient les dépendances suivantes :

| Catégorie | Package | Version minimale | Description |
|-----------|---------|------------------|-------------|
| Algorithmes génétiques | deap | 1.3.1+ | Framework pour algorithmes évolutifs |
| Algorithmes génétiques | pygad | 2.18.0+ | Alternative pour algorithmes génétiques |
| Backtesting | backtrader | 1.9.78.123+ | Framework de backtesting |
| Indicateurs techniques | TA-Lib | 0.4.28+ | Bibliothèque d'analyse technique |
| Manipulation de données | pandas | 1.5.0+ | Manipulation de données |
| Manipulation de données | numpy | 1.23.0+ | Calculs numériques |
| Visualisation | matplotlib | 3.6.0+ | Graphiques et visualisations |
| Visualisation | plotly | 5.11.0+ | Visualisations interactives |
| Jupyter | jupyter | 1.0.0+ | Interface notebook |
| Jupyter | notebook | 6.5.0+ | Interface notebook |
| Utilitaires | scipy | 1.9.0+ | Calculs scientifiques |
| Utilitaires | scikit-learn | 1.1.0+ | Machine learning |
| Utilitaires | tqdm | 4.64.0+ | Barres de progression |

### Note sur TA-Lib

L'installation de TA-Lib peut nécessiter des étapes supplémentaires selon votre système d'exploitation :

**Windows :**
```bash
pip install TA-Lib
```

**macOS :**
```bash
brew install ta-lib
pip install TA-Lib
```

**Linux (Ubuntu/Debian) :**
```bash
sudo apt-get install ta-lib
pip install TA-Lib
```

Si l'installation de TA-Lib échoue, le projet peut fonctionner sans cette dépendance car les indicateurs techniques sont également implémentés en pur Python.

---

## Exécution du script principal

### Méthode 1 : Exécution directe

```bash
python src/main.py
```

### Méthode 2 : Exécution depuis le dossier src

```bash
cd src
python main.py
```

### Ce qui se passe lors de l'exécution

Le script [`main.py`](src/main.py) exécute les étapes suivantes :

1. **Chargement des données** : Télécharge ou génère des données de prix
2. **Division des données** : Sépare en ensembles d'entraînement, validation et test
3. **Configuration de l'algorithme génétique** : Définit les paramètres de l'AG
4. **Configuration de la fonction fitness** : Définit les critères d'évaluation
5. **Exécution de l'algorithme génétique** : Fait évoluer la population
6. **Analyse des résultats** : Affiche les statistiques de l'optimisation
7. **Validation** : Teste la meilleure stratégie sur les données de validation
8. **Test final** : Teste sur les données de test
9. **Rapport détaillé** : Affiche un rapport complet du backtesting
10. **Walk-forward testing** : Effectue une validation robuste

### Affichage attendu

```
======================================================================
STRATÉGIES DE TRADING PAR ALGORITHMES GÉNÉTIQUES
Projet ECE - Groupe 3 - Sujet n°41
======================================================================

1. Chargement des données...
   Utilisation de données simulées pour SP500
   Période: 2024-01-01 à 2025-12-31
   Nombre de jours: 500
   Prix actuel: 4500.00

2. Division des données...
   Entraînement: 350 jours
   Validation: 75 jours
   Test: 75 jours

3. Configuration de l'algorithme génétique...
   Taille de la population: 20
   Nombre de générations: 30
   Taux de croisement: 0.8
   Taux de mutation: 0.15

4. Configuration de la fonction fitness...
   Pondérations:
     - Rendement: 0.35
     - Sharpe Ratio: 0.30
     - Max Drawdown: 0.25
     - Stabilité: 0.10

5. Exécution de l'algorithme génétique...
----------------------------------------------------------------------
Génération 1/30 | Meilleur: 0.1234 | Moyen: 0.0987 | Pire: 0.0456
Génération 2/30 | Meilleur: 0.1456 | Moyen: 0.1123 | Pire: 0.0567
...
----------------------------------------------------------------------

6. Analyse des résultats...
   Meilleur fitness final: 0.4567
   Amélioration totale: 0.3333

7. Meilleure stratégie trouvée...
   Paramètres:
     - SMA court: 15
     - SMA long: 45
     - EMA période: 20
     - RSI période: 14
     - RSI survente: 30
     - RSI surachat: 70
     - MACD rapide: 12
     - MACD lent: 26
     - MACD signal: 9
     - Stop loss %: 5

8. Validation sur les données de validation...
   Fitness validation: 0.4234
   Rendement: 12.34%
   Sharpe Ratio: 1.23
   Max Drawdown: -8.56%

9. Test final sur les données de test...
   Fitness test: 0.4012
   Rendement: 10.12%
   Sharpe Ratio: 1.15
   Max Drawdown: -9.23%

10. Rapport détaillé du backtesting...
╔══════════════════════════════════════════════════════════════════╗
║                    RAPPORT DE BACKTESTING                         ║
╠══════════════════════════════════════════════════════════════════╣
║  Capital initial: 100,000.00 €                                   ║
║  Capital final:   110,120.00 €                                   ║
║  Rendement total: 10.12%                                         ║
║  Rendement annualisé: 10.45%                                     ║
╠══════════════════════════════════════════════════════════════════╣
║  Sharpe Ratio: 1.15                                               ║
║  Maximum Drawdown: -9.23%                                         ║
║  Win Rate: 58.33%                                                 ║
║  Profit Factor: 1.67                                              ║
╠══════════════════════════════════════════════════════════════════╣
║  Total trades: 24                                                 ║
║  Trades gagnants: 14                                              ║
║  Trades perdants: 10                                              ║
╚══════════════════════════════════════════════════════════════════╝

11. Walk-forward testing (évitement du curve-fitting)...
   Nombre de fenêtres: 5
   Rendement moyen: 8.45%
   Sharpe moyen: 1.02
   Drawdown moyen: -7.89%

======================================================================
RÉSUMÉ FINAL
======================================================================
Symbole: SP500
Fitness entraînement: 0.4567
Fitness validation: 0.4234
Fitness test: 0.4012

Meilleure stratégie:
  SMA court: 15
  SMA long: 45
  EMA période: 20
  RSI période: 14
  RSI survente: 30
  RSI surachat: 70
  MACD rapide: 12
  MACD lent: 26
  MACD signal: 9
  Stop loss %: 5

Résultats sauvegardés dans la variable 'results'.
```

---

## Exécution du notebook de démonstration

### Étape 1 : Installer Jupyter (si pas déjà installé)

```bash
pip install jupyter notebook
```

### Étape 2 : Lancer Jupyter Notebook

```bash
jupyter notebook notebooks/
```

Ou simplement :
```bash
jupyter notebook
```

Puis naviguez vers le dossier `notebooks/` et ouvrez `demo.ipynb`.

### Étape 3 : Exécuter les cellules

Dans le notebook, vous pouvez :
- Exécuter toutes les cellules : `Cell` → `Run All`
- Exécuter cellule par cellule : `Shift + Enter`
- Exécuter une sélection : `Cell` → `Run Selected`

### Contenu du notebook de démonstration

Le notebook [`demo.ipynb`](notebooks/demo.ipynb) contient les sections suivantes :

1. **Importation des modules** : Chargement des bibliothèques nécessaires
2. **Chargement des données** : Import et visualisation des données de prix
3. **Création d'une stratégie de trading** : Définition des paramètres
4. **Backtesting d'une stratégie** : Simulation sur données historiques
5. **Optimisation par algorithme génétique** : Recherche des meilleurs paramètres
6. **Visualisation des résultats** : Graphiques et analyses
7. **Walk-forward testing** : Validation robuste

### Avantages du notebook

- **Visualisation interactive** : Graphiques générés automatiquement
- **Exécution pas à pas** : Comprendre chaque étape
- **Modification facile** : Tester différents paramètres
- **Documentation intégrée** : Explications dans chaque cellule

---

## Affichage des résultats

### Sections de l'affichage

L'affichage est divisé en plusieurs sections clairement identifiées :

#### 1. En-tête du programme
```
======================================================================
STRATÉGIES DE TRADING PAR ALGORITHMES GÉNÉTIQUES
Projet ECE - Groupe 3 - Sujet n°41
======================================================================
```

#### 2. Chargement des données
Affiche les informations sur les données chargées :
- Symbole de l'actif
- Période de temps
- Nombre de jours
- Prix actuel

#### 3. Division des données
Montre la répartition des données :
- Entraînement (70% par défaut)
- Validation (15% par défaut)
- Test (15% par défaut)

#### 4. Configuration de l'algorithme génétique
Affiche les paramètres de l'AG :
- Taille de la population
- Nombre de générations
- Taux de croisement
- Taux de mutation

#### 5. Configuration de la fonction fitness
Montre les pondérations des critères :
- Rendement (35% par défaut)
- Sharpe Ratio (30% par défaut)
- Max Drawdown (25% par défaut)
- Stabilité (10% par défaut)

#### 6. Exécution de l'algorithme génétique
Affiche la progression génération par génération :
```
Génération 1/30 | Meilleur: 0.1234 | Moyen: 0.0987 | Pire: 0.0456
```

#### 7. Analyse des résultats
Statistiques finales de l'optimisation :
- Meilleur fitness final
- Amélioration totale

#### 8. Meilleure stratégie trouvée
Paramètres optimaux de la stratégie :
- SMA court et long
- EMA période
- RSI période, survente, surachat
- MACD rapide, lent, signal
- Stop loss %

#### 9. Validation sur les données de validation
Performance sur l'ensemble de validation :
- Fitness
- Rendement
- Sharpe Ratio
- Max Drawdown

#### 10. Test final sur les données de test
Performance sur l'ensemble de test :
- Fitness
- Rendement
- Sharpe Ratio
- Max Drawdown

#### 11. Rapport détaillé du backtesting
Tableau formaté avec :
- Capital initial et final
- Rendements
- Ratios de performance
- Statistiques de trades

#### 12. Walk-forward testing
Résultats de la validation robuste :
- Nombre de fenêtres
- Rendement moyen
- Sharpe moyen
- Drawdown moyen

#### 13. Résumé final
Synthèse de tous les résultats :
- Fitness sur chaque ensemble
- Paramètres de la meilleure stratégie

### Formatage des résultats

Les résultats sont affichés avec :
- **Séparateurs visuels** : Lignes de 70 caractères
- **Sections numérotées** : Pour suivre la progression
- **Tableaux formatés** : Pour les rapports détaillés
- **Alignement** : Pour une meilleure lisibilité

---

## Personnalisation des paramètres

### Modifier les paramètres dans main.py

Ouvrez le fichier [`src/main.py`](src/main.py) et modifiez les paramètres dans la fonction `main()` :

```python
def main():
    """Point d'entrée principal du programme."""
    # Exemple d'utilisation avec des données simulées
    results = run_genetic_trading_optimization(
        symbol="SP500",           # Symbole de l'actif
        use_sample_data=True,     # True = données simulées, False = données réelles
        generations=30,            # Nombre de générations
        population_size=20,        # Taille de la population
        train_ratio=0.7,          # Ratio d'entraînement
        random_seed=42            # Graine pour la reproductibilité
    )
    
    return results
```

### Paramètres disponibles

| Paramètre | Type | Description | Valeur par défaut |
|-----------|------|-------------|------------------|
| `symbol` | str | Symbole de l'actif (ex: "^GSPC", "AAPL") | "SP500" |
| `use_sample_data` | bool | Utiliser des données simulées | True |
| `generations` | int | Nombre de générations de l'AG | 30 |
| `population_size` | int | Taille de la population | 20 |
| `train_ratio` | float | Ratio des données d'entraînement | 0.7 |
| `random_seed` | int | Graine pour la reproductibilité | 42 |

### Modifier les pondérations de la fitness

Dans la fonction `run_genetic_trading_optimization()`, modifiez les pondérations :

```python
weights = FitnessWeights(
    return_weight=0.35,      # Poids du rendement
    sharpe_weight=0.30,      # Poids du Sharpe Ratio
    drawdown_weight=0.25,   # Poids du Max Drawdown
    stability_weight=0.10   # Poids de la stabilité
)
```

### Modifier les bornes des gènes

Dans la fonction `run_genetic_trading_optimization()`, modifiez les bornes :

```python
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
```

---

## Résolution des problèmes courants

### Problème : ModuleNotFoundError

**Erreur :**
```
ModuleNotFoundError: No module named 'pandas'
```

**Solution :**
```bash
pip install -r requirements.txt
```

### Problème : TA-Lib installation failed

**Erreur :**
```
ERROR: Could not build wheels for ta-lib
```

**Solution :**
- Sur Windows : Téléchargez le fichier `.whl` depuis [Gohlke's repository](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)
- Sur macOS : `brew install ta-lib` puis `pip install TA-Lib`
- Sur Linux : `sudo apt-get install ta-lib` puis `pip install TA-Lib`

### Problème : Données non téléchargeables

**Erreur :**
```
Erreur lors du téléchargement: ...
```

**Solution :**
Le programme utilise automatiquement des données simulées si le téléchargement échoue. Vous pouvez aussi forcer l'utilisation de données simulées :
```python
results = run_genetic_trading_optimization(
    symbol="SP500",
    use_sample_data=True,  # Force l'utilisation de données simulées
    ...
)
```

### Problème : Exécution lente

**Cause :** Trop de générations ou une population trop grande

**Solution :** Réduisez les paramètres :
```python
results = run_genetic_trading_optimization(
    generations=10,      # Réduire le nombre de générations
    population_size=10,  # Réduire la taille de la population
    ...
)
```

### Problème : Pas de graphiques dans le notebook

**Cause :** Matplotlib backend non configuré

**Solution :**
```python
import matplotlib.pyplot as plt
%matplotlib inline
```

### Problème : Mémoire insuffisante

**Cause :** Trop de données ou de population

**Solution :**
- Réduisez la taille de la population
- Utilisez des données simulées
- Réduisez la période de données

---

## 📚 Ressources supplémentaires

- [README.md](README.md) : Documentation principale du projet
- [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md) : Rapport technique détaillé
- [slides/presentation.md](slides/presentation.md) : Support de présentation

---

**Date de création** : 1 février 2026  
**Version** : 1.0
