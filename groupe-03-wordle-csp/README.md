# Solveur Wordle CSP - IA Symbolique et Exploratoire

**Projet universitaire ECE - Ingénieur 4 - Finance**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-Academic-green.svg)

---

## 🚀 Démarrage Rapide (Démo)

### Prérequis
- Python 3.8+
- Node.js 18+
- npm

### Lancement automatique

**Windows (PowerShell):**
```powershell
cd groupe-03-wordle-csp
.\start_demo.ps1
```

**Windows (CMD):**
```batch
cd groupe-03-wordle-csp
start_demo.bat
```

**Manuel (tous OS):**
```bash
# Terminal 1 - Backend
cd groupe-03-wordle-csp
python -m uvicorn api.main:app --reload --port 8000

# Terminal 2 - Frontend
cd groupe-03-wordle-csp/web
npm install  # Première fois seulement
npm run dev
```

### URLs
- **Interface Web (démo):** http://localhost:5173
- **API Documentation:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

📖 **Guide de démonstration complet:** Voir [DEMO.md](DEMO.md)

---

## Table des matières

1. [Présentation](#présentation)
2. [Pourquoi Wordle est un CSP](#pourquoi-wordle-est-un-csp)
3. [IA Symbolique vs IA Exploratoire](#ia-symbolique-vs-ia-exploratoire)
4. [Installation](#installation)
5. [Utilisation](#utilisation)
6. [Tests](#tests)
7. [Structure du projet](#structure-du-projet)
8. [Limites et perspectives](#limites-et-perspectives)
9. [Références](#références)

---

## Présentation

Ce projet implémente un **solveur intelligent pour Wordle** en utilisant des techniques d'**IA symbolique** (CSP - Constraint Satisfaction Problem) et d'**IA exploratoire** (heuristiques de recherche).

### Objectifs pédagogiques

1. Modéliser un problème réel comme un CSP
2. Implémenter des algorithmes de propagation de contraintes
3. Concevoir et comparer des heuristiques de recherche
4. Analyser la complexité algorithmique
5. Évaluer les performances par benchmark

### Fonctionnalités

- ✅ Résolution de Wordle en français (mots de 5 lettres)
- ✅ Gestion rigoureuse des lettres répétées
- ✅ 4 stratégies heuristiques (naive, fréquence, entropie, mixte)
- ✅ Mode interactif + mode automatique + benchmark
- ✅ Tests unitaires avec pytest
- ✅ Documentation technique complète
- ✅ Slides de présentation

---

## Pourquoi Wordle est un CSP

### Définition d'un CSP

Un **Problème de Satisfaction de Contraintes** est défini par :
- **Variables** : éléments à déterminer
- **Domaines** : valeurs possibles pour chaque variable
- **Contraintes** : règles limitant les combinaisons de valeurs

### Wordle comme CSP

**Variable** :
```
mot : variable dont le domaine est l'ensemble des mots du dictionnaire
```

**Domaine initial** :
```
D(mot) = {tous les mots français de 5 lettres} ≈ 5000-8000 mots
```

**Contraintes** (ajoutées après chaque feedback) :

1. **Lettres vertes** (bien placées) :
   ```
   Si feedback[i] = Vert et guess[i] = 'R'
   → mot[i] = 'R'
   ```

2. **Lettres jaunes** (mal placées) :
   ```
   Si feedback[i] = Jaune et guess[i] = 'E'
   → 'E' ∈ mot ET mot[i] ≠ 'E'
   ```

3. **Lettres grises** (absentes) :
   ```
   Si feedback[i] = Gris et guess[i] = 'A'
   → 'A' ∉ mot
   ```

**Résolution** :
Après chaque feedback, le domaine se réduit par **propagation de contraintes** (filtrage arc-consistent).

---

## IA Symbolique vs IA Exploratoire

Ce projet combine deux approches complémentaires :

| Aspect | IA Symbolique | IA Exploratoire |
|--------|---------------|-----------------|
| **Principe** | Raisonnement logique formel | Heuristiques de recherche |
| **Outils** | CSP, logique des prédicats | A*, entropie, fréquence |
| **Question** | "Quels mots sont **possibles**?" | "Quel mot **choisir**?" |
| **Garantie** | Cohérence logique | Optimisation de performance |
| **Dans Wordle** | Filtrage des candidats | Sélection du meilleur mot |

### IA Symbolique : Le CSP

**Rôle** : Maintenir l'ensemble des mots **valides** selon les contraintes.

**Algorithme** : Filtrage (arc-consistency)
```python
def filter_candidates(candidates, guess, feedback):
    return [
        word for word in candidates
        if compute_feedback(guess, word) == feedback
    ]
```

**Garantie** : Tous les mots conservés sont des solutions possibles.

### IA Exploratoire : Les heuristiques

**Rôle** : Choisir le mot qui **minimise le nombre de coups** attendu.

**Heuristiques implémentées** :

1. **Naïve** : Premier mot alphabétiquement (baseline)
2. **Fréquence** : Maximise les lettres les plus fréquentes
3. **Entropie** : Maximise le gain d'information (Shannon)
4. **Mixte** : Combine fréquence (rapide) et entropie (précise)

**Complexité** :
- Fréquence : O(n × m)
- Entropie : O(n² × m)

---

## Installation

### Prérequis

- Python 3.8 ou supérieur
- pip

### Étapes

1. **Cloner le dépôt** (ou télécharger l'archive)

```bash
cd groupe-03-wordle-csp
```

2. **Créer un environnement virtuel** (recommandé)

```bash
python -m venv venv
source venv/bin/activate  # Sur Linux/Mac
# ou
venv\Scripts\activate  # Sur Windows
```

3. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

4. **Vérifier l'installation**

```bash
python -m pytest
```

Tous les tests doivent passer ✅

---

##  Architecture du Projet

```
groupe-03-wordle-csp/
├── api/                    # Backend FastAPI
│   ├── main.py            # Endpoints REST
│   ├── models.py          # Modèles Pydantic
│   ├── services.py        # Gestion sessions de jeu
│   └── config.py          # Configuration CORS
├── web/                    # Frontend React + Vite
│   ├── src/
│   │   ├── components/    # Composants React
│   │   ├── api/          # Client API
│   │   └── styles/       # CSS
│   └── package.json
├── src/                    # Solveur CSP (core)
│   ├── csp_solver.py      # Algorithme CSP
│   ├── wordle_feedback.py # Logique Wordle
│   ├── strategy.py        # Heuristiques
│   └── main.py           # CLI
├── data/
│   └── mots_fr_5.txt     # 5817 mots français
├── tests/                  # Tests unitaires (pytest)
├── docs/                   # Documentation technique
└── slides/                 # Présentation
```

### Technologies utilisées

**Backend:**
- FastAPI (API REST)
- Uvicorn (serveur ASGI)
- Pydantic (validation)

**Frontend:**
- React 18
- Vite (bundler)
- CSS moderne

**Solveur:**
- Python pur (CSP par filtrage)
- Stratégies: naive, fréquence, entropie, mixed

### Flux de fonctionnement

1. **Création de partie:** Frontend → POST /game/new → Backend crée session + retourne suggestion
2. **Ajout contrainte:** Frontend → POST /game/{id}/constraint → Backend filtre candidats
3. **Suggestion:** Frontend → GET /game/{id}/suggest → Backend applique stratégie
4. **Simulation:** Frontend → POST /simulate → Backend résout automatiquement

---

## Utilisation

Le solveur propose plusieurs modes d'utilisation.

### 1. Mode interactif

Interagissez avec le solveur en temps réel.

```bash
python -m src.main interactive --strategy mixed
```

**Déroulement** :
1. Le programme suggère un premier mot
2. Vous proposez un mot
3. Vous entrez le feedback reçu (ex: `GGYBB`)
4. Le programme suggère le prochain mot
5. Répéter jusqu'à trouver le mot

**Exemple** :
```
==========================================================
SOLVEUR WORDLE CSP - MODE INTERACTIF
==========================================================

Stratégie: Mixed
Dictionnaire: 5234 mots

💡 Suggestion pour le premier mot: AROSE

--- Tour 1 ---
Mot proposé: ARBRE
Feedback (GGGGG si trouvé): BYBBB
Feedback: A R B R E
          ⬜🟨⬜⬜⬜

📊 Candidats restants: 127
💡 Suggestion: CRIER
```

### 2. Mode suggestion

Obtenez une suggestion basée sur l'historique.

```bash
python -m src.main suggest \
  --guesses ARBRE,CRANE \
  --feedbacks BGYBB,GGGBB \
  --strategy entropy
```

**Sortie** :
```
Stratégie: Entropy

ARBRE -> BGYBB
Candidats restants: 89

CRANE -> GGGBB
Candidats restants: 3

Candidats: CRABE, CRAPE, CRAVE

💡 Suggestion: CRABE
```

### 3. Mode automatique

Résolution complète d'un mot secret.

```bash
python -m src.main auto --secret GERER --strategy mixed
```

**Sortie** :
```
🎯 Résolution automatique de: GERER
Stratégie: Mixed

--- Tour 1 ---
Proposition: AROSE
Feedback: ⬜⬜⬜🟨🟩
Candidats restants: 156

--- Tour 2 ---
Proposition: CRIER
Feedback: ⬜🟨⬜🟩🟩
Candidats restants: 5

--- Tour 3 ---
Proposition: GERER
Feedback: 🟩🟩🟩🟩🟩

🎉 Trouvé en 3 coups!
```

### 4. Benchmark

Comparez les performances des stratégies.

```bash
python -m src.benchmark --n 100 --strategies naive,frequency,entropy,mixed
```

**Sortie** :
```
🏁 BENCHMARK WORDLE CSP
======================================================================
Nombre de tests: 100
Stratégies: naive, frequency, entropy, mixed
======================================================================

[1/100] Mot: ARBRE
  naive       : ✅ 5 coups (0.08s)
  frequency   : ✅ 4 coups (0.12s)
  entropy     : ✅ 4 coups (1.23s)
  mixed       : ✅ 4 coups (0.18s)

...

======================================================================
📊 RÉSULTATS
======================================================================

Stratégie: NAIVE
  Taux de réussite: 87.0% (87/100)
  Moyenne de coups: 4.8
  Temps moyen: 0.09s
  Distribution:
    3 coups: ███ (3)
    4 coups: ████████████████████ (20)
    5 coups: ██████████████████████████████ (30)
    6 coups: ██████████████████████████████████ (34)

...
```

---

## Tests

Le projet inclut une suite de tests unitaires complète.

### Lancer tous les tests

```bash
pytest
```

### Tests par module

```bash
# Tests du feedback
pytest tests/test_feedback.py

# Tests du CSP
pytest tests/test_filtering.py
```

### Tests avec couverture

```bash
pytest --cov=src tests/
```

### Tests spécifiques

```bash
# Test des lettres répétées (critique)
pytest tests/test_feedback.py::TestComputeFeedback::test_feedback_repeated_letters_case1
```

---

## Structure du projet

```
groupe-03-wordle-csp/
├── README.md                 # Ce fichier
├── requirements.txt          # Dépendances Python
├── data/
│   └── mots_fr_5.txt         # Dictionnaire français (5 lettres)
├── src/
│   ├── __init__.py
│   ├── main.py               # CLI principal
│   ├── wordle_feedback.py    # Calcul du feedback (vert/jaune/gris)
│   ├── csp_solver.py         # Solveur CSP (filtrage)
│   ├── strategy.py           # Heuristiques exploratoires
│   ├── benchmark.py          # Évaluation des performances
│   └── llm_assist.py         # Stub pédagogique LLM
├── tests/
│   ├── __init__.py
│   ├── test_feedback.py      # Tests du feedback
│   └── test_filtering.py     # Tests du filtrage CSP
├── docs/
│   └── technical.md          # Documentation technique détaillée
└── slides/
    └── slides.md             # Slides de présentation
```

### Description des modules

#### `wordle_feedback.py`

Implémente la **logique canonique de Wordle** :
- Calcul du feedback (vert/jaune/gris)
- Gestion correcte des lettres répétées
- Vérification de compatibilité mot/feedback

**Fonctions clés** :
- `compute_feedback(guess, secret) -> str`
- `is_compatible(word, guess, feedback) -> bool`
- `feedback_to_constraints(guess, feedback) -> dict`

#### `csp_solver.py`

Implémente le **solveur CSP** :
- Chargement du dictionnaire
- Filtrage des candidats par propagation de contraintes
- Gestion de l'état (reset, statistiques)

**Classe principale** :
- `WordleCSPSolver`

#### `strategy.py`

Implémente les **heuristiques de recherche** :
- Stratégie naïve (baseline)
- Stratégie fréquence (lettres fréquentes)
- Stratégie entropie (gain d'information)
- Stratégie mixte (hybride)

**Classes** :
- `Strategy` (classe de base)
- `NaiveStrategy`, `FrequencyStrategy`, `EntropyStrategy`, `MixedStrategy`

#### `benchmark.py`

Évalue les **performances comparatives** :
- Simulation de parties sur un corpus
- Métriques : taux de réussite, nombre moyen de coups, temps
- Génération de statistiques détaillées

#### `llm_assist.py`

**Stub pédagogique** pour l'intégration LLM :
- Démonstration d'une approche neuro-symbolique
- Explications en langage naturel
- Perspectives d'amélioration

---

## Limites et perspectives

### Limites actuelles

#### Limites théoriques

1. **Pas d'optimalité garantie** : Les heuristiques sont gloutonnes (greedy)
2. **Pas d'apprentissage** : Le solveur ne s'améliore pas avec l'expérience
3. **Dépendance au dictionnaire** : Si le mot secret n'est pas dans le dico, échec garanti

#### Limites pratiques

1. **Coût de l'entropie** : O(n²) limite l'utilisation sur de grands ensembles
2. **Vision myope** : Optimisation sur 1 coup uniquement (pas de planification multi-coups)
3. **Pas de gestion de l'incertitude** : Suppose que le feedback est toujours correct

### Perspectives d'amélioration

#### 1. Optimisation multi-coups

Utiliser **Minimax** avec élagage alpha-beta pour planifier 2-3 coups en avance.

#### 2. Hybridation neuro-symbolique

Combiner :
- **CSP** (symbolique) : filtre les candidats valides
- **LLM** (neuronal) : score les candidats par pertinence sémantique

```python
# Exemple d'intégration
candidates = csp_solver.get_candidates()  # IA symbolique
llm_scores = llm.score_words(candidates)  # IA neuronale
best_word = combine_scores(candidates, csp_scores, llm_scores)
```

#### 3. Apprentissage par renforcement

Entraîner un agent RL (DQN, Policy Gradient) sur 10 000+ parties pour apprendre des patterns optimaux.

#### 4. Optimisation du premier mot

Pré-calculer le **premier mot universel optimal** par analyse exhaustive.

---

## Références

### Théorie

- **Russell & Norvig** - *Artificial Intelligence: A Modern Approach* (CSP, heuristiques)
- **Shannon, C.** - *A Mathematical Theory of Communication* (entropie)
- **Mackworth, A.** - *Constraint Satisfaction* (arc-consistency)

### Wordle

- **New York Times Wordle** - [https://www.nytimes.com/games/wordle/](https://www.nytimes.com/games/wordle/)
- **Analyses algorithmiques** :
  - Alex Selby - *Optimal Wordle Strategy*
  - 3Blue1Brown - *Solving Wordle using information theory*

### Code et outils

- **Python-constraint** - Bibliothèque CSP
- **pytest** - Framework de tests
- **OR-Tools** - Google Optimization Tools

---

## Auteurs

**Groupe 03**
ECE Paris - Ingénieur 4 - Finance
IA Exploratoire et Symbolique - 2026
Thomas Nassar - Lewis OREL

---

## Licence

Projet académique - ECE Paris

---

## Contact

Pour toute question sur le projet, consultez :
- **Documentation technique** : `docs/technical.md`
- **Slides de présentation** : `slides/slides.md`
- **Tests** : `tests/`

---

**Bonne résolution de Wordle ! 🎯**
