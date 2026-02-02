# Solveur Wordle CSP
## IA Symbolique et Exploratoire

**Groupe 03**
ECE Paris - Ingénieur 4 - Finance
2026
Lewis Orel - Thomas Nassar
---

## Slide 1: Introduction au problème

### Wordle : Qu'est-ce que c'est?

- Jeu de déduction linguistique
- Objectif : Deviner un mot de 5 lettres en 6 essais maximum
- Feedback coloré après chaque proposition :
  - 🟩 **Vert** : Lettre correcte, bonne position
  - 🟨 **Jaune** : Lettre correcte, mauvaise position
  - ⬜ **Gris** : Lettre absente

### Problématique

**Comment résoudre Wordle de manière optimale avec l'IA?**

---

## Slide 2: IA Symbolique vs IA Exploratoire

### Deux approches complémentaires

| **IA Symbolique** | **IA Exploratoire** |
|-------------------|---------------------|
| Raisonnement logique | Heuristiques de recherche |
| Garantie de cohérence | Optimisation de performance |
| CSP, logique formelle | A*, entropie, fréquence |
| "Quels mots sont possibles?" | "Quel mot choisir?" |

### Notre approche : **Hybride**

1. **CSP** filtre les candidats valides (IA symbolique)
2. **Heuristiques** choisissent le meilleur mot (IA exploratoire)

---

## Slide 3: Modélisation CSP

### Définition formelle

Un **CSP** (Constraint Satisfaction Problem) comprend :
- **Variables** : éléments à déterminer
- **Domaines** : valeurs possibles pour chaque variable
- **Contraintes** : règles limitant les combinaisons

### Wordle comme CSP

**Variable** :
- `mot` : variable unique

**Domaine initial** :
- D(mot) = {tous les mots français de 5 lettres} ≈ 5000-8000 mots

**Domaine après contraintes** :
- Réduit progressivement selon les feedbacks

---

## Slide 4: Les contraintes Wordle

### Trois types de contraintes

1. **Lettres exactes (🟩 vertes)**
   ```
   Si feedback[i] = Vert et guess[i] = 'R'
   → mot[i] = 'R'
   ```

2. **Lettres présentes (🟨 jaunes)**
   ```
   Si feedback[i] = Jaune et guess[i] = 'E'
   → 'E' ∈ mot ET mot[i] ≠ 'E'
   ```

3. **Lettres absentes (⬜ grises)**
   ```
   Si feedback[i] = Gris et guess[i] = 'A'
   → 'A' ∉ mot
   ```

### Cas critique : **Lettres répétées**

Gestion rigoureuse du comptage d'occurrences !

---

## Slide 5: Algorithme de filtrage

### Propagation de contraintes (Arc-Consistency)

```python
def filter_candidates(candidates, guess, feedback):
    result = []
    for word in candidates:
        # Tester : si 'word' était le secret,
        # produirait-il le même feedback?
        if compute_feedback(guess, word) == feedback:
            result.append(word)
    return result
```

### Propriétés

- ✅ **Complétude** : toutes les solutions conservées
- ✅ **Correction** : aucune solution invalide
- ⚡ **Complexité** : O(n × m) où n = candidats, m = 5

---

## Slide 6: Heuristiques exploratoires (1/2)

### 1. Baseline naïve

- **Principe** : Premier mot alphabétiquement
- **Avantage** : Simple, déterministe
- **Inconvénient** : Pas d'optimisation

### 2. Heuristique de fréquence

- **Principe** : Maximiser les lettres les plus fréquentes
- **Calcul** :
  ```
  score(mot) = Σ fréquence(lettre) pour lettre unique dans mot
  Bonus +10% si toutes lettres différentes
  ```
- **Complexité** : O(n × m)

---

## Slide 7: Heuristiques exploratoires (2/2)

### 3. Heuristique d'entropie

**Principe** : Maximiser le gain d'information (théorie de Shannon)

```
H(X) = -Σ p(feedback) × log₂(p(feedback))
```

**Intuition** :
- Entropie élevée → feedbacks bien répartis → bonne discrimination
- Entropie faible → feedbacks concentrés → peu informatif

**Complexité** : O(n² × m) → coûteuse !

### 4. Stratégie mixte

- Entropie si n ≤ 50 candidats (précision)
- Fréquence sinon (rapidité)

---

## Slide 8: Architecture du système

```
┌─────────────────────────────────────────────┐
│                 SOLVEUR WORDLE              │
├─────────────────────────────────────────────┤
│                                              │
│  ┌──────────────┐         ┌──────────────┐ │
│  │ Dictionnaire │────────→│  CSP Solver  │ │
│  │  (5000 mots) │         │  (Filtrage)  │ │
│  └──────────────┘         └──────────────┘ │
│                                   │         │
│                                   ↓         │
│                          ┌──────────────┐  │
│                          │  Candidats   │  │
│                          │   valides    │  │
│                          └──────────────┘  │
│                                   │         │
│                                   ↓         │
│                          ┌──────────────┐  │
│                          │ Heuristiques │  │
│                          │ (Fréquence,  │  │
│                          │  Entropie)   │  │
│                          └──────────────┘  │
│                                   │         │
│                                   ↓         │
│                          ┌──────────────┐  │
│                          │ Suggestion   │  │
│                          └──────────────┘  │
└─────────────────────────────────────────────┘
```

---

## Slide 9: Résultats du benchmark

### Expérimentation

- **Corpus** : 200 mots français aléatoires
- **Limite** : 6 coups maximum
- **Métriques** : Taux de réussite, nombre moyen de coups

### Résultats comparatifs

| Stratégie | Taux de réussite | Moy. coups | Temps (s) |
|-----------|------------------|------------|-----------|
| Naïve     | ~85%             | 4.8        | 0.1       |
| Fréquence | ~92%             | 4.3        | 0.2       |
| Entropie  | ~94%             | 4.1        | 1.5       |
| **Mixte** | **~95%**         | **4.2**    | **0.5**   |

### Interprétation

- Stratégie mixte : meilleur compromis performance/temps
- Entropie : précise mais coûteuse
- Baseline naïve : insuffisante

---

## Slide 10: Limites et défis

### Limites théoriques

- ❌ Pas de garantie d'**optimalité** (heuristiques gloutonnes)
- ❌ Pas d'**apprentissage** (pas de ML)
- ❌ **Dépendance au dictionnaire** (mot hors dico = échec)

### Limites pratiques

- ⏱️ Entropie coûteuse : O(n²)
- 🔍 Vision myope : optimisation sur 1 coup uniquement
- 🎯 Pas de gestion de l'incertitude

### Défis résolus

- ✅ Gestion correcte des **lettres répétées**
- ✅ Filtrage exact (pas d'approximation)
- ✅ Performance acceptable (< 2s par partie)

---

## Slide 11: Perspectives d'amélioration

### 1. Optimisation multi-coups

- Minimax avec élagage alpha-beta
- Planification à horizon 2-3 coups

### 2. Hybridation neuro-symbolique

```
CSP (Symbolique) → Candidats valides
         +
LLM (Neuronal) → Scoring sémantique
         ↓
  Meilleure décision
```

### 3. Intégration LLM

- **Explication** du raisonnement
- **Contexte** linguistique (mots liés à un thème)
- **Analyse post-mortem** des parties

### 4. Apprentissage par renforcement

- Entraîner un agent RL sur 10 000+ parties
- Apprendre des patterns optimaux

---

## Slide 12: Démonstration live

### Modes disponibles

1. **Mode interactif**
   ```bash
   python -m src.main interactive
   ```
   - Jouer pas à pas avec l'assistant

2. **Mode suggestion**
   ```bash
   python -m src.main suggest --guesses ARBRE --feedbacks BYBBB
   ```
   - Obtenir le prochain meilleur coup

3. **Mode automatique**
   ```bash
   python -m src.main auto --secret GERER --strategy mixed
   ```
   - Résolution complète automatique

4. **Benchmark**
   ```bash
   python -m src.benchmark --n 50 --strategies naive,frequency,mixed
   ```

---

## Slide 13: Conclusion

### Ce que nous avons appris

1. **IA Symbolique** (CSP) :
   - Modélisation formelle de problèmes de contraintes
   - Propagation de contraintes (arc-consistency)
   - Garantie de correction logique

2. **IA Exploratoire** :
   - Heuristiques de recherche (fréquence, entropie)
   - Compromis performance/temps
   - Théorie de l'information appliquée

3. **Hybridation** :
   - Combiner symbolique + neuronal = meilleur des deux mondes

### Applicabilité

Ces techniques s'appliquent à :
- Jeux de déduction (Mastermind, Motus)
- Diagnostic médical
- Debugging et test logiciel
- Tout problème de réduction d'espace de recherche

---

## Slide 14: Questions?

### Contacts

- **GitHub** : [Lien vers le dépôt]
- **Documentation** : `docs/technical.md`
- **Code source** : `src/`

### Merci de votre attention!

**Groupe 03**
ECE Paris - IA Exploratoire et Symbolique - 2026

---

## Architecture Web 🌐

### Vue d'ensemble

```
┌─────────────────┐
│  Frontend React │  ← Interface utilisateur
│  localhost:5173 │
└────────┬────────┘
         │ REST API (JSON)
         ▼
┌─────────────────┐
│  Backend FastAPI│  ← Serveur API
│  localhost:8000 │
└────────┬────────┘
         │ Import direct
         ▼
┌─────────────────┐
│  Solveur CSP    │  ← Algorithmes
│  (src/)         │
└─────────────────┘
```

**Technologies:**
- Backend: FastAPI + Uvicorn (Python)
- Frontend: React 18 + Vite
- Communication: HTTP REST JSON

---

## Démo Web Interactive 🎮

### Fonctionnalités

**Mode Interactif:**
- ✅ Création de partie avec choix de stratégie
- ✅ Suggestions intelligentes du solveur
- ✅ Ajout manuel de contraintes
- ✅ Feedback visuel en temps réel (🟩🟨⬜)
- ✅ Statistiques dynamiques

**Mode Automatique:**
- ✅ Simulation complète avec mot secret
- ✅ Historique détaillé des coups
- ✅ Résolution en 2-4 tentatives en moyenne

**URL:** http://localhost:5173

---

## Backend API REST 🔌

### Endpoints principaux

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/health` | GET | Vérifier disponibilité |
| `/game/new` | POST | Créer une partie |
| `/game/{id}/constraint` | POST | Ajouter contrainte |
| `/game/{id}/suggest` | GET | Obtenir suggestion |
| `/simulate` | POST | Simulation automatique |

**Documentation:** http://localhost:8000/docs (Swagger)

**Exemple réponse:**
```json
{
  "game_id": "550e8400-...",
  "candidates_count": 5817,
  "first_suggestion": "TARES",
  "strategy": "mixed"
}
```

---

## Frontend React ⚛️

### Composants

```
App.jsx (racine)
├── WordleGrid       ← Grille avec feedback
├── Controls         ← Boutons + inputs
├── Stats            ← Statistiques
└── SimulationPanel  ← Mode automatique
```

**Architecture:**
- État global géré par App.jsx
- Communication API via client.js
- Validation côté client
- Gestion d'erreurs complète

**Styles:** CSS moderne Wordle-like

---

## Gestion des Sessions 🎲

### Cycle de vie

1. **POST /game/new** → Création GameSession
   - Initialise WordleCSPSolver
   - Charge stratégie (mixed/frequency/entropy)
   - Génère UUID unique
   - Retourne première suggestion

2. **POST /game/{id}/constraint** → Filtrage
   - Ajoute contrainte au solver
   - Filtre candidats (O(n) complexité)
   - Met à jour statistiques

3. **GET /game/{id}/suggest** → Génération
   - Applique stratégie heuristique
   - Retourne meilleur(s) mot(s)

4. **DELETE /game/{id}** → Nettoyage
   - Libère mémoire

**Stockage:** En RAM (sessions perdues au redémarrage)

---

## Performance ⚡

### Mesures réelles (5817 mots)

| Opération | Temps | Complexité |
|-----------|-------|------------|
| Chargement dictionnaire | 10 ms | O(n) |
| Ajout contrainte | 2-5 ms | O(n) |
| Suggestion fréquence | 10 ms | O(n×m) |
| Suggestion entropie | 50-100 ms | O(n²×m) |
| **Partie complète** | **<500 ms** | **2-4 coups** |

**Optimisations possibles:**
- Cache suggestions (Redis)
- Index dictionnaire (Trie)
- Pool workers (Gunicorn)

---

## Scripts de Démo 🚀

### Lancement automatique (Windows)

```powershell
cd groupe-03-wordle-csp
.\start_demo.ps1
```

**Le script:**
- ✅ Vérifie prérequis (Python, Node.js)
- ✅ Lance backend (port 8000)
- ✅ Installe dépendances npm (si nécessaire)
- ✅ Lance frontend (port 5173)

**Alternative manuelle:**
```bash
# Terminal 1
python -m uvicorn api.main:app --reload --port 8000

# Terminal 2
cd web && npm run dev
```

---

## Tests et Qualité ✅

### Tests unitaires

**Commande:** `pytest -q`

**Résultats:** 34/34 tests passent ✅
- `test_feedback.py` - 19 tests
- `test_filtering.py` - 15 tests

**Couverture:**
- Calcul feedback (lettres répétées)
- Filtrage CSP (compatibilité)
- Edge cases (mots vides, feedbacks invalides)

**Aucune régression** depuis l'ajout API/Frontend

---

## Scénario de Présentation 📺

### 1. Démarrage (30s)
```powershell
.\start_demo.ps1
```
→ Backend + Frontend démarrent

### 2. Mode Interactif (2min)
- Ouvrir http://localhost:5173
- Créer partie (stratégie Mixed)
- Montrer suggestion: "TARES"
- Ajouter contrainte manuellement
- Observer filtrage en temps réel

### 3. Mode Automatique (1min)
- Saisir mot secret: "MARDI"
- Résoudre automatiquement
- Montrer historique coloré

### 4. API Swagger (1min)
- Ouvrir http://localhost:8000/docs
- Tester POST /simulate
- Montrer réponse JSON

---

## Extensions Futures 🔮

### Fonctionnalités avancées

**Solveur OR-Tools CP-SAT:**
- Approche programmation contraintes
- Comparaison performances filtering vs CP-SAT
- Benchmarks détaillés

**Intégration LLM:**
- Suggestions contextuelles (GPT-4/Claude)
- Explications naturelles
- Approche neuro-symbolique

**Optimisations:**
- Persistance Redis/PostgreSQL
- Authentification JWT
- Rate limiting
- Monitoring (Prometheus)

**UI/UX:**
- Mode multijoueurs
- Statistiques globales
- Thèmes personnalisables
- Support mobile

---

## Conclusion 🎓

### Livrables

✅ **Solveur CSP fonctionnel** (4 stratégies)
✅ **API REST moderne** (FastAPI, 8 endpoints)
✅ **Interface web interactive** (React, animations)
✅ **Scripts de démo** (lancement 1 clic)
✅ **Documentation complète** (README, guides)
✅ **Tests validés** (34/34 passent)

### Compétences démontrées

- Algorithmique (CSP, heuristiques)
- Architecture web (REST API, React)
- Ingénierie logicielle (tests, docs, CI)
- Performance (O(n) filtrage, benchmarks)

**Le projet est prêt pour la présentation finale ECE ING4 ! 🎯**
