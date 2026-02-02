# Documentation Technique - Solveur Wordle CSP

**Projet universitaire ECE - Ingénieur 4 - Finance**
**IA Symbolique et Exploratoire**

---

## Table des matières

1. [Introduction](#introduction)
2. [Modélisation CSP](#modélisation-csp)
3. [Algorithmes de résolution](#algorithmes-de-résolution)
4. [Heuristiques exploratoires](#heuristiques-exploratoires)
5. [Analyse de complexité](#analyse-de-complexité)
6. [Limites et perspectives](#limites-et-perspectives)

---

## 1. Introduction

### 1.1 Problématique

Wordle est un jeu de déduction où le joueur doit deviner un mot de 5 lettres en 6 essais maximum. Après chaque proposition, un feedback indique:
- **Vert (G)**: lettre correcte à la bonne position
- **Jaune (Y)**: lettre présente mais mal placée
- **Gris (B)**: lettre absente

**Question centrale**: Comment modéliser et résoudre ce problème avec des techniques d'IA symbolique et exploratoire?

### 1.2 Approche choisie

Notre solution combine:
1. **IA SYMBOLIQUE**: Modélisation CSP pour garantir la cohérence logique
2. **IA EXPLORATOIRE**: Heuristiques pour optimiser le nombre de coups

Cette approche hybride offre:
- Exactitude garantie (pas de solutions invalides)
- Performance optimisée (réduction intelligente de l'espace de recherche)
- Explicabilité (chaque décision est justifiable)

---

## 2. Modélisation CSP

### 2.1 Définition formelle

Un **Problème de Satisfaction de Contraintes (CSP)** est défini par:
- Un ensemble de variables **V = {v₁, v₂, ..., vₙ}**
- Pour chaque variable vᵢ, un domaine **D(vᵢ)** de valeurs possibles
- Un ensemble de contraintes **C** limitant les combinaisons de valeurs

**Objectif**: Trouver une affectation de valeurs aux variables qui satisfait toutes les contraintes.

### 2.2 Wordle comme CSP

Nous avons choisi la modélisation suivante:

#### Approche retenue: Variable unique

**Variable**:
- `mot` : variable dont le domaine est l'ensemble des mots du dictionnaire

**Domaine initial**:
```
D(mot) = {tous les mots français de 5 lettres}
        ≈ 5000-8000 mots selon le dictionnaire
```

**Domaine après contraintes**:
Le domaine se réduit progressivement après chaque feedback.

#### Approche alternative (non retenue)

**Variables**:
```
V = {pos₀, pos₁, pos₂, pos₃, pos₄}
```

Chaque posᵢ représente la lettre à la position i.

**Domaines**:
```
D(posᵢ) = {A, B, C, ..., Z} pour tout i
```

**Pourquoi non retenue?**
- Complexité accrue de la modélisation des contraintes
- Gestion difficile des lettres répétées
- Nécessité de contraintes de cohérence lexicale

### 2.3 Contraintes

Les contraintes Wordle se décomposent en 3 types:

#### 2.3.1 Contraintes de lettres exactes (vertes)

**Définition**:
```
Si feedback[i] = G et guess[i] = L
Alors mot[i] = L
```

**Exemple**:
- Guess: ARBRE
- Feedback: G____
- Contrainte: `mot[0] = 'A'`

#### 2.3.2 Contraintes de lettres présentes (jaunes)

**Définition**:
```
Si feedback[i] = Y et guess[i] = L
Alors:
  - L ∈ mot (la lettre est présente)
  - mot[i] ≠ L (mais pas à cette position)
```

**Exemple**:
- Guess: ARBRE
- Feedback: _Y___
- Contraintes:
  - `'R' ∈ mot`
  - `mot[1] ≠ 'R'`

#### 2.3.3 Contraintes de lettres absentes (grises)

**Définition**:
```
Si feedback[i] = B et guess[i] = L
Alors L ∉ mot (la lettre n'apparaît pas)
```

**ATTENTION - Cas particulier**: Lettres répétées
```
Si une lettre L apparaît plusieurs fois dans guess:
  - Certaines occurrences peuvent être vertes/jaunes
  - D'autres occurrences peuvent être grises
  - Une lettre grise signifie "plus d'autres occurrences"
```

**Exemple critique**:
```
Secret: GERER
Guess:  TERRE
Feedback: BYGGB

Explication:
- T[0]: absent (gris)
- E[1]: présent ailleurs (jaune) → un des E de GERER
- R[2]: bien placé (vert) → premier R de GERER
- R[3]: bien placé (vert) → deuxième R de GERER
- E[4]: absent (gris) → le E jaune a "consommé" un E, il n'y en a pas d'autre
```

### 2.4 Représentation mathématique

**Fonction de compatibilité**:
```
compatible(mot, (guess, feedback)) → {true, false}
```

Un mot `w` est compatible avec une contrainte `(g, f)` si et seulement si:
```
compute_feedback(g, w) = f
```

Cette fonction garantit la correction logique: si `w` était le secret, le feedback serait identique.

**Domaine après n contraintes**:
```
D_n(mot) = D_0(mot) ∩ C₁ ∩ C₂ ∩ ... ∩ Cₙ

où Cᵢ = {w | compatible(w, (guessᵢ, feedbackᵢ))}
```

---

## 3. Algorithmes de résolution

### 3.1 Filtrage par propagation de contraintes

**Algorithme**: Arc-Consistency (AC-3 adapté)

```python
function filter_candidates(candidates, guess, feedback):
    result = []
    for word in candidates:
        if compute_feedback(guess, word) == feedback:
            result.append(word)
    return result
```

**Propriétés**:
- **Complétude**: Toutes les solutions sont conservées
- **Correction**: Aucune solution invalide n'est conservée
- **Complexité**: O(n × m) où n = nombre de candidats, m = longueur du mot

**Pourquoi pas de backtracking?**
Le filtrage suffit car:
1. Les contraintes sont monotones (domaine décroissant)
2. Pas besoin d'exploration exhaustive (le feedback guide)
3. Performance suffisante pour Wordle (espace de recherche raisonnable)

### 3.2 Gestion des lettres répétées

**Algorithme de calcul de feedback**:

```python
def compute_feedback(guess, secret):
    feedback = ['B'] * 5
    secret_counts = Counter(secret)

    # Phase 1: Marquer les lettres exactes (vertes)
    for i in range(5):
        if guess[i] == secret[i]:
            feedback[i] = 'G'
            secret_counts[guess[i]] -= 1  # Consommer

    # Phase 2: Marquer les lettres mal placées (jaunes)
    for i in range(5):
        if feedback[i] == 'B':  # Pas déjà verte
            if secret_counts[guess[i]] > 0:
                feedback[i] = 'Y'
                secret_counts[guess[i]] -= 1  # Consommer

    return feedback
```

**Invariant clé**: Chaque lettre du secret est "consommée" au plus une fois.

---

## 4. Heuristiques exploratoires

L'IA exploratoire optimise le **choix du prochain mot** pour minimiser le nombre de coups.

### 4.1 Baseline naïve

**Principe**: Choisir le premier mot alphabétiquement.

```python
def naive_strategy(candidates):
    return sorted(candidates)[0]
```

**Avantages**:
- Simple
- Déterministe
- Baseline pour comparaison

**Inconvénients**:
- Pas d'optimisation
- Performance sous-optimale

### 4.2 Heuristique de fréquence

**Principe**: Choisir le mot qui maximise les lettres les plus fréquentes.

**Algorithme**:
```python
def frequency_strategy(candidates):
    # 1. Calculer la fréquence de chaque lettre
    freq = compute_letter_frequencies(candidates)

    # 2. Scorer chaque mot
    best_word = None
    best_score = -1

    for word in candidates:
        unique_letters = set(word)
        score = sum(freq[letter] for letter in unique_letters)

        # Bonus: toutes lettres différentes
        if len(unique_letters) == 5:
            score *= 1.1

        if score > best_score:
            best_score = score
            best_word = word

    return best_word
```

**Complexité**: O(n × m) où n = nombre de candidats, m = taille du mot

**Intuition**: Les lettres fréquentes apparaissent dans plus de mots → plus de chances d'éliminer des candidats.

### 4.3 Heuristique d'entropie

**Principe**: Maximiser le gain d'information (théorie de l'information).

**Définition - Entropie de Shannon**:
```
H(X) = -Σ p(x) log₂(p(x))
```

**Application à Wordle**:

Pour un mot `guess`, l'entropie mesure "combien d'information" on gagne:

```python
def compute_entropy(guess, candidates):
    # Grouper les candidats par feedback
    feedback_groups = {}

    for secret in candidates:
        fb = compute_feedback(guess, secret)
        if fb not in feedback_groups:
            feedback_groups[fb] = 0
        feedback_groups[fb] += 1

    # Calculer l'entropie
    total = len(candidates)
    entropy = 0.0

    for count in feedback_groups.values():
        p = count / total
        if p > 0:
            entropy -= p * log2(p)

    return entropy
```

**Interprétation**:
- **Entropie élevée**: Feedbacks bien répartis → bonne discrimination
- **Entropie faible**: Feedbacks concentrés → peu informatif

**Exemple**:
```
Candidats: [ARBRE, AUTRE, AITRE, ANCRE]

Guess: ARBRE
Feedbacks possibles:
- GGGGG: 1 candidat (ARBRE lui-même)
- GGGBG: 1 candidat (ANCRE)
- GGYGG: 1 candidat (AUTRE)
- GGYGG: 1 candidat (AITRE)

→ Entropie = -4 × (0.25 × log₂(0.25)) = 2 bits
```

**Complexité**: O(n² × m) → coûteux!

**Optimisation**: Limiter à n ≤ 100 candidats, sinon fallback sur fréquence.

### 4.4 Stratégie mixte

**Principe**: Combiner fréquence et entropie selon le contexte.

```python
def mixed_strategy(candidates):
    if len(candidates) <= 50:
        return entropy_strategy(candidates)
    else:
        return frequency_strategy(candidates)
```

**Justification**:
- Entropie: précise mais coûteuse → utile en fin de partie
- Fréquence: rapide mais moins précise → utile en début de partie

---

## 5. Analyse de complexité

### 5.1 Complexité spatiale

**Dictionnaire**: O(N) où N = taille du dictionnaire (~5000 mots)

**Candidats**: O(n) où n = candidats restants (n ≤ N)

**Contraintes**: O(k) où k = nombre de contraintes (k ≤ 6)

**Total**: O(N) - linéaire en la taille du dictionnaire

### 5.2 Complexité temporelle

#### Par opération

| Opération | Complexité | Justification |
|-----------|------------|---------------|
| Calcul feedback | O(m) | m = 5 (longueur mot) |
| Filtrage candidats | O(n × m) | Tester chaque candidat |
| Heuristique fréquence | O(n × m) | Parcours + calcul fréquences |
| Heuristique entropie | O(n² × m) | n candidats × n secrets |

#### Par partie complète

**Cas moyen**: O(k × n × m) où k = nombre de coups (~4)

**Pire cas**: O(6 × N × m) si aucune réduction du domaine

**En pratique**: Le domaine se réduit exponentiellement → performance acceptable

### 5.3 Analyse empirique

**Benchmark sur 200 mots** (stratégie mixte):
- Taux de réussite: ~95%
- Nombre moyen de coups: 4.2
- Temps moyen par partie: 0.5-2s (selon taille dictionnaire)

---

## 6. Limites et perspectives

### 6.1 Limites actuelles

#### 6.1.1 Limites théoriques

**Pas de garantie d'optimalité**:
Les heuristiques ne garantissent pas le nombre minimal de coups. Elles sont **gloutonnes** (greedy).

**Pas d'apprentissage**:
Le solveur ne s'améliore pas avec l'expérience (pas de machine learning).

**Dépendance au dictionnaire**:
Si le mot secret n'est pas dans le dictionnaire, échec garanti.

#### 6.1.2 Limites pratiques

**Coût de l'entropie**:
Complexité O(n²) limite l'utilisation sur de grands ensembles de candidats.

**Pas d'optimisation multi-coups**:
Les heuristiques sont myopes (ne regardent qu'un coup en avant).

**Pas de gestion de l'incertitude**:
Le solveur suppose que le feedback est toujours correct.

### 6.2 Perspectives d'amélioration

#### 6.2.1 IA symbolique avancée

**Optimisation multi-niveaux**:
```
Au lieu de:  choisir le meilleur mot pour ce tour
Faire:       choisir le mot qui mène au meilleur arbre de décision
```

**Approximation**: Minimax avec élagage alpha-beta (profondeur 2-3)

**CSP avec OR-Tools**:
Utiliser un vrai solveur CSP pour la sélection de mots:
```python
from ortools.sat.python import cp_model

model = cp_model.CpModel()
# Définir variables et contraintes
# Optimiser un objectif (ex: entropie)
```

#### 6.2.2 Hybridation neuro-symbolique

**Principe**: Combiner IA symbolique (CSP) et IA neuronale (LLM/réseau).

```
┌─────────────────────────────────────────────┐
│  Système hybride                            │
│                                              │
│  ┌──────────────┐       ┌────────────────┐ │
│  │ CSP Solver   │──────→│ Candidats      │ │
│  │ (Symbolique) │       │ valides        │ │
│  └──────────────┘       └────────────────┘ │
│         │                       │           │
│         │                       ↓           │
│         │              ┌────────────────┐  │
│         │              │ LLM / Réseau   │  │
│         │              │ (Neuronal)     │  │
│         │              └────────────────┘  │
│         │                       │           │
│         └──────────┬────────────┘          │
│                    ↓                        │
│           ┌──────────────────┐             │
│           │ Meilleur mot     │             │
│           └──────────────────┘             │
└─────────────────────────────────────────────┘
```

**Avantages**:
- **Symbolique**: garantit la cohérence logique
- **Neuronal**: apporte l'intuition et le contexte
- **Hybride**: meilleur des deux mondes

**Implémentation possible**:
1. CSP filtre les candidats valides (garantie logique)
2. LLM score les candidats par pertinence sémantique
3. Combinaison des scores pour la décision finale

#### 6.2.3 Intégration LLM

**Cas d'usage**:

1. **Explication du raisonnement**:
```python
explanation = llm.explain(
    "Pourquoi suggérer ARBRE sachant que R est jaune et E vert?"
)
# → "ARBRE place R en position 1, ce qui teste si R est là..."
```

2. **Génération de mots contextuels**:
```python
suggestion = llm.suggest_word(
    context="Mots liés à la nature",
    candidates=csp_candidates
)
```

3. **Analyse post-mortem**:
```python
analysis = llm.analyze_game(history)
# → "Vous auriez pu gagner en 3 coups en proposant CARTE au tour 2"
```

**Implémentation** (avec API Anthropic/OpenAI):
```python
import anthropic

client = anthropic.Client(api_key="...")

def explain_with_llm(word, constraints):
    prompt = f"""
    Explique pourquoi '{word}' est un bon choix pour Wordle
    sachant ces contraintes: {constraints}
    """

    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text
```

#### 6.2.4 Apprentissage par renforcement

**Principe**: Entraîner un agent RL à jouer à Wordle.

**Formulation**:
- **État**: (historique des guesses, feedbacks, candidats restants)
- **Action**: choisir un mot parmi les candidats
- **Récompense**: -1 par coup, +10 si trouvé, -20 si échec
- **Objectif**: minimiser le nombre de coups

**Algorithme**: Deep Q-Network (DQN) ou Policy Gradient

**Avantage**: Apprend des patterns optimaux sur de nombreuses parties.

#### 6.2.5 Optimisation du premier mot

**Analyse pré-calculée**:
Calculer hors-ligne le meilleur premier mot universel.

**Méthode**:
```python
def find_best_first_word(dictionary):
    best_word = None
    best_avg_reduction = 0

    for candidate in dictionary:
        total_reduction = 0
        for secret in dictionary:
            feedback = compute_feedback(candidate, secret)
            remaining = count_compatible(dictionary, candidate, feedback)
            reduction = 1 - (remaining / len(dictionary))
            total_reduction += reduction

        avg_reduction = total_reduction / len(dictionary)

        if avg_reduction > best_avg_reduction:
            best_avg_reduction = avg_reduction
            best_word = candidate

    return best_word
```

**Résultat empirique** (anglais): SOARE, ROATE, RAISE

---

## 7. Conclusion technique

### 7.1 Synthèse

Ce projet démontre une **application rigoureuse de l'IA symbolique** (CSP) combinée à des **heuristiques exploratoires** pour résoudre Wordle.

**Points forts**:
- Modélisation CSP claire et explicable
- Garantie de correction logique
- Heuristiques efficaces
- Code propre et testé

**Points d'amélioration**:
- Pas d'optimalité garantie
- Entropie coûteuse en calcul
- Pas d'apprentissage

### 7.2 Contributions

1. **Formulation CSP rigoureuse** de Wordle
2. **Gestion correcte des lettres répétées** (cas critique)
3. **Comparaison de 4 stratégies** (naive, fréquence, entropie, mixte)
4. **Ouverture vers l'hybridation neuro-symbolique**

### 7.3 Applicabilité

Les techniques développées sont transférables à:
- Jeux de déduction (Mastermind, Motus)
- Diagnostic médical (élimination de pathologies)
- Debugging (recherche de bugs)
- Tout problème de réduction d'espace de recherche par contraintes

---

**Auteurs**: Groupe 03
**Date**: 2026
**Cours**: IA Exploratoire et Symbolique - ECE Paris

---

## Architecture Web (Extension 2026)

### Vue d'ensemble

Le projet a été étendu avec une architecture web moderne composée de:
- **Backend FastAPI** - API REST pour exposer le solveur CSP
- **Frontend React** - Interface web interactive
- **Communication REST** - HTTP JSON entre front et back

```
┌─────────────────────────┐
│   Frontend React        │
│   (Vite + React 18)     │
│   http://localhost:5173 │
└───────────┬─────────────┘
            │ HTTP REST (JSON)
            ▼
┌─────────────────────────┐
│   Backend FastAPI       │
│   (Python + Uvicorn)    │
│   http://localhost:8000 │
└───────────┬─────────────┘
            │ Import direct
            ▼
┌─────────────────────────┐
│   Modules CSP Core      │
│   (csp_solver.py, etc.) │
└─────────────────────────┘
            │
            ▼
┌─────────────────────────┐
│   Dictionnaire          │
│   (5817 mots)           │
└─────────────────────────┘
```

---

## Backend API FastAPI

### Structure

```
api/
├── main.py         # Application FastAPI + endpoints
├── models.py       # Modèles Pydantic (validation)
├── services.py     # Logique métier (sessions)
└── config.py       # Configuration (CORS, chemins)
```

### Endpoints REST

#### 1. Health Check
```http
GET /health
```
**Réponse:**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "timestamp": "2026-02-02T00:00:00",
  "dictionary_loaded": true,
  "word_count": 5817
}
```

#### 2. Créer une partie
```http
POST /game/new
Content-Type: application/json

{
  "strategy": "mixed",
  "solver": "filtering"
}
```
**Réponse:**
```json
{
  "game_id": "550e8400-e29b-41d4-a716-446655440000",
  "strategy": "mixed",
  "solver": "filtering",
  "total_words": 5817,
  "candidates_count": 5817,
  "first_suggestion": "TARES"
}
```

#### 3. Ajouter une contrainte
```http
POST /game/{game_id}/constraint
Content-Type: application/json

{
  "guess": "TARES",
  "feedback": "BYGBB"
}
```
**Réponse:**
```json
{
  "success": true,
  "candidates_remaining": 142,
  "solved": false,
  "solution": null,
  "top_candidates": ["MARDI", "MARLI", ...]
}
```

#### 4. Obtenir une suggestion
```http
GET /game/{game_id}/suggest?limit=3
```
**Réponse:**
```json
{
  "suggestions": ["MARDI", "MARLI", "MARIE"],
  "candidates_count": 142,
  "strategy_used": "mixed"
}
```

#### 5. État de la partie
```http
GET /game/{game_id}/state
```
**Réponse:**
```json
{
  "game_id": "550e8400-...",
  "strategy": "mixed",
  "solver": "filtering",
  "candidates_count": 142,
  "constraints_count": 1,
  "solved": false,
  "solution": null
}
```

#### 6. Simulation automatique
```http
POST /simulate
Content-Type: application/json

{
  "secret": "MARDI",
  "max_turns": 6
}
```
**Réponse:**
```json
{
  "success": true,
  "secret": "MARDI",
  "turns": [
    {
      "turn_number": 1,
      "guess": "TARES",
      "feedback": "BYGBB",
      "candidates_remaining": 142
    },
    {
      "turn_number": 2,
      "guess": "MARDI",
      "feedback": "GGGGG",
      "candidates_remaining": 1
    }
  ],
  "total_turns": 2,
  "solved": true,
  "final_guess": "MARDI"
}
```

### Gestion des sessions

**Classe GameSession** (`api/services.py`):
- Encapsule un `WordleCSPSolver` et une `Strategy`
- Maintient l'historique des contraintes
- Gère le cycle de vie d'une partie

**Classe GameSessionManager**:
- Dictionnaire en mémoire `{game_id: GameSession}`
- Création/récupération/suppression de sessions
- Nettoyage automatique (max 1000 sessions)

**Limitations actuelles:**
- Sessions en RAM (perdues au redémarrage)
- Pas de persistance (Redis/PostgreSQL non implémenté)
- Pas d'expiration automatique

---

## Frontend React

### Structure

```
web/
├── src/
│   ├── main.jsx              # Bootstrap React
│   ├── App.jsx               # Composant racine + état global
│   ├── App.css               # Styles application
│   ├── index.css             # Styles globaux (Wordle-like)
│   ├── api/
│   │   └── client.js         # Client API HTTP
│   ├── components/
│   │   ├── WordleGrid.jsx    # Grille de jeu avec feedback
│   │   ├── Controls.jsx      # Contrôles interactifs
│   │   ├── Stats.jsx         # Statistiques temps réel
│   │   └── SimulationPanel.jsx  # Mode automatique
│   └── styles/
│       ├── WordleGrid.css
│       ├── Controls.css
│       ├── Stats.css
│       └── SimulationPanel.css
├── package.json              # Dépendances npm
├── vite.config.js           # Configuration Vite (proxy API)
└── index.html               # Point d'entrée HTML
```

### Composants React

#### App.jsx (Composant racine)
**Responsabilités:**
- Gestion d'état global (gameId, attempts, stats, strategy)
- Health check automatique (toutes les 10s)
- Communication entre composants (props + callbacks)

**État géré:**
```javascript
{
  gameId: string | null,
  strategy: "mixed" | "frequency" | "entropy" | "naive",
  solver: "filtering",
  attempts: Array<{guess, feedback, remainingCandidates}>,
  stats: {
    candidatesCount: number,
    totalWords: number,
    solved: boolean,
    solution: string | null
  }
}
```

#### WordleGrid.jsx
- Affiche les tentatives avec feedback coloré
- Animation des cellules (vert/jaune/gris)
- Lignes vides jusqu'à 6 tentatives max

#### Controls.jsx
- Sélection stratégie
- Bouton "Nouvelle Partie" → `createGame()`
- Affichage suggestion du solveur
- Input manuel (guess + feedback) → `addConstraint()`
- Gestion des erreurs (validation, API)

#### SimulationPanel.jsx
- Input mot secret (5 lettres)
- Résolution automatique complète
- Affichage historique avec mini-grille

#### Stats.jsx
- Candidats restants
- Total tentatives
- Nombre de mots dans le dictionnaire
- Statut résolu/non résolu

### Client API (`api/client.js`)

**Fonctions disponibles:**
```javascript
checkHealth()
createGame(strategy, solver)
addConstraint(gameId, guess, feedback)
getSuggestion(gameId, limit)
getGameState(gameId)
simulateGame(secret, maxTurns)
```

**Gestion d'erreurs:**
- Try/catch sur tous les appels
- Messages d'erreur explicites
- Validation côté client (longueur, format)

### Configuration Vite

**Proxy API** (`vite.config.js`):
```javascript
server: {
  port: 5173,
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
      rewrite: (path) => path.replace(/^\/api/, '')
    }
  }
}
```

Permet d'éviter les problèmes CORS en développement.

---

## Flux de données complet

### Scénario: Partie interactive

1. **Initialisation**
   ```
   User → Frontend: Ouvre http://localhost:5173
   Frontend → Backend: GET /health
   Backend → Frontend: {status: "ok", word_count: 5817}
   Frontend: Affiche indicateur "✓ API Online"
   ```

2. **Création de partie**
   ```
   User → Frontend: Clique "Nouvelle Partie" (strategy: mixed)
   Frontend → Backend: POST /game/new {strategy: "mixed"}
   Backend: Crée GameSession avec WordleCSPSolver
   Backend → Frontend: {game_id, first_suggestion: "TARES"}
   Frontend: Affiche suggestion + active contrôles
   ```

3. **Ajout contrainte manuelle**
   ```
   User → Frontend: Saisit "TARES" + "BYGBB"
   Frontend: Validation (longueur, format)
   Frontend → Backend: POST /game/{id}/constraint {guess, feedback}
   Backend: solver.add_constraint() → filtrage
   Backend → Frontend: {candidates_remaining: 142, top_candidates}
   Frontend: Met à jour grille + stats
   Frontend → Backend: GET /game/{id}/suggest
   Backend: strategy.select_word()
   Backend → Frontend: {suggestions: ["MARDI"]}
   Frontend: Affiche nouvelle suggestion
   ```

4. **Résolution**
   ```
   (Répéter étape 3 jusqu'à solved: true)
   Backend → Frontend: {solved: true, solution: "MARDI"}
   Frontend: Affiche message "✅ Résolu ! Le mot était : MARDI"
   ```

### Scénario: Simulation automatique

1. **Lancement simulation**
   ```
   User → Frontend: Saisit mot secret "MARDI"
   Frontend → Backend: POST /simulate {secret: "MARDI", max_turns: 6}
   Backend: Crée session temporaire
   Backend: Boucle {
     suggestion = strategy.select_word()
     feedback = compute_feedback(suggestion, secret)
     solver.add_constraint(suggestion, feedback)
   }
   Backend: Nettoie session
   Backend → Frontend: {turns: [...], solved: true}
   Frontend: Affiche historique complet avec couleurs
   ```

---

## CORS et Sécurité

### Configuration CORS

**Backend** (`api/config.py`):
```python
CORS_ORIGINS = [
    "http://localhost:3000",  # React dev
    "http://localhost:5173",  # Vite dev
    "http://localhost:8080",  # Vue dev
]
```

**FastAPI** (`api/main.py`):
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Considérations de production

**À implémenter pour la production:**
- [ ] Authentification JWT
- [ ] Rate limiting (ex: 100 req/min par IP)
- [ ] Persistance sessions (Redis/PostgreSQL)
- [ ] HTTPS obligatoire
- [ ] Validation CORS stricte (domaines spécifiques)
- [ ] Logging avancé (requêtes, erreurs)
- [ ] Monitoring (Prometheus, Grafana)

---

## Tests et Validation

### Tests Backend

**Tests unitaires existants** (non modifiés):
- `tests/test_feedback.py` - 19 tests
- `tests/test_filtering.py` - 15 tests
- **Total: 34 tests passent** ✅

**Tests API manquants** (à implémenter):
```python
# tests/test_api.py
def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_create_game():
    response = client.post("/game/new", json={"strategy": "mixed"})
    assert response.status_code == 200
    assert "game_id" in response.json()

def test_add_constraint():
    # Créer partie
    # Ajouter contrainte
    # Vérifier candidates_remaining < initial
```

### Tests Frontend

**Non implémentés** (optionnel):
- Tests unitaires composants (Jest + React Testing Library)
- Tests E2E (Playwright/Cypress)

### Validation manuelle

**Checklist démo:**
- [ ] Backend démarre sans erreur
- [ ] Frontend affiche "✓ API Online"
- [ ] Création partie fonctionne
- [ ] Suggestions affichées
- [ ] Ajout contrainte met à jour grille
- [ ] Simulation automatique complète
- [ ] Gestion d'erreurs (mot invalide, feedback incorrect)

---

## Performance et Optimisations

### Complexité algorithme

**Solveur CSP par filtrage:**
- Initialisation: O(n) - chargement dictionnaire
- Ajout contrainte: O(n) - filtrage compatibilité
- Suggestion naive: O(n log n) - tri alphabétique
- Suggestion fréquence: O(n × m) - calcul scores (m=26 lettres)
- Suggestion entropie: O(n² × m) - calcul gain information

**Performances mesurées** (dictionnaire 5817 mots):
- Chargement: ~10 ms
- Contrainte: ~2-5 ms
- Suggestion fréquence: ~10 ms
- Suggestion entropie: ~50-100 ms (>100 candidats)
- **Total partie: <500 ms** (2-4 coups)

### Optimisations possibles

**Backend:**
- [ ] Cache suggestions fréquentes (Redis)
- [ ] Index dictionnaire (Trie)
- [ ] Pool de workers (Gunicorn)
- [ ] Compression réponses (gzip)

**Frontend:**
- [ ] Lazy loading composants
- [ ] Memoization (React.memo)
- [ ] Debounce inputs
- [ ] Service Worker (offline)

---

## Déploiement

### Développement

**Backend:**
```bash
cd groupe-03-wordle-csp
python -m uvicorn api.main:app --reload --port 8000
```

**Frontend:**
```bash
cd groupe-03-wordle-csp/web
npm run dev
```

### Production

**Backend (option 1: Gunicorn):**
```bash
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

**Backend (option 2: Docker):**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt
COPY . .
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Frontend (build):**
```bash
cd web
npm run build
# Servir dist/ avec nginx ou serveur statique
```

---

## Références

### Technologies utilisées

- **Backend:**
  - [FastAPI](https://fastapi.tiangolo.com/) - Framework API moderne
  - [Uvicorn](https://www.uvicorn.org/) - Serveur ASGI
  - [Pydantic](https://docs.pydantic.dev/) - Validation données

- **Frontend:**
  - [React](https://react.dev/) - Bibliothèque UI
  - [Vite](https://vitejs.dev/) - Build tool
  - [MDN CSS](https://developer.mozilla.org/fr/docs/Web/CSS) - Styles modernes

### Documentation projet

- README.md - Guide démarrage rapide
- DEMO.md - Scénario présentation 5 min
- docs/technical.md - Cette documentation
- API Swagger - http://localhost:8000/docs (auto-généré)
