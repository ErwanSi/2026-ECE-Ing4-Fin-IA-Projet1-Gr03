# 🎯 Guide de Démonstration - Wordle CSP Solver

## Démarrage rapide (2 minutes)

### Option 1: Script automatique (Recommandé)
```powershell
cd groupe-03-wordle-csp
.\start_demo.ps1
```
cd "C:\Users\lewis\OneDrive\Documents\ECE\ING4\IA Finances\PROJET 1\2026-ECE-Ing4-Fin-IA-Projet1-Gr03\groupe-03-wordle-csp"

### Option 2: Manuel
```bash
# Terminal 1 - Backend
cd groupe-03-wordle-csp
python -m uvicorn api.main:app --reload --port 8000

# Terminal 2 - Frontend
cd groupe-03-wordle-csp/web
npm install  # Première fois seulement
npm run dev
```

## URLs de la démo

- **Frontend:** http://localhost:5173
- **Backend API:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

## Scénario de démonstration (5 minutes)

### 1. Présentation du projet (30 secondes)
"Nous avons développé un solveur Wordle intelligent basé sur CSP (Constraint Satisfaction Problem) avec:
- Une API REST FastAPI
- Une interface web React interactive
- Plusieurs stratégies heuristiques (naive, fréquence, entropie, mixed)"

### 2. Démonstration Mode Interactif (2 minutes)

**Afficher l'interface:**
1. Ouvrir http://localhost:5173
2. Montrer l'indicateur "API Online (5817 mots)"

**Créer une partie:**
1. Sélectionner stratégie "Mixed (Recommandé)"
2. Cliquer "🎮 Nouvelle Partie"
3. Observer: "💡 Suggestion du solveur: TARES" (ou autre)

**Ajouter des contraintes:**
1. Copier la suggestion (ex: TARES)
2. Entrer dans Feedback: "BYGBB" (exemple)
3. Cliquer "➕ Ajouter contrainte"
4. Montrer:
   - Grille mise à jour avec couleurs
   - Candidats restants diminuent
   - Nouvelle suggestion générée

**Répéter 2-3 fois jusqu'à résolution**

### 3. Démonstration Mode Automatique (1 minute)

**Simuler une résolution:**
1. Aller dans "Mode Automatique"
2. Entrer mot secret: "MARDI"
3. Cliquer "🤖 Résoudre automatiquement"
4. Montrer:
   - Historique complet des coups
   - Feedback coloré pour chaque tentative
   - Nombre de candidats à chaque étape
   - Résolution en ~2-4 coups

### 4. Points techniques à mentionner (1 minute)

**Architecture:**
- Frontend React (Vite) + Backend FastAPI
- Gestion de sessions (UUID)
- Communication REST API

**Algorithme CSP:**
- Filtrage par compatibilité arc-consistent
- Pas de backtracking (performance optimale)
- Complexité O(n) par contrainte

**Stratégies:**
- **Naive:** Premier mot alphabétique
- **Fréquence:** Maximise lettres fréquentes
- **Entropie:** Maximise gain d'information
- **Mixed:** Hybride adaptatif (entropie si <50 candidats, sinon fréquence)

### 5. Démonstration API (30 secondes - optionnel)

**Afficher Swagger:**
1. Ouvrir http://localhost:8000/docs
2. Montrer les endpoints disponibles:
   - POST /game/new
   - POST /game/{id}/constraint
   - GET /game/{id}/suggest
   - POST /simulate

**Tester un endpoint:**
1. Cliquer sur "POST /simulate"
2. Try it out
3. Body: `{"secret": "GERER", "max_turns": 6}`
4. Execute
5. Montrer la réponse JSON avec l'historique

## Points forts à souligner

✅ **Interface intuitive:** Wordle-like avec feedback coloré
✅ **Performance:** Résolution en 2-6 coups en moyenne
✅ **Scalabilité:** Architecture API REST moderne
✅ **Extensibilité:** Prêt pour ajout CP-SAT, LLM, benchmarks
✅ **Tests:** 34 tests unitaires (100% pass)
✅ **Documentation:** README complet + docs techniques

## Commandes utiles pour la présentation

```bash
# Relancer les tests
cd groupe-03-wordle-csp
pytest -q

# Utiliser la CLI
python -m src.main interactive --strategy mixed
python -m src.main auto --secret MARDI --strategy entropy

# Benchmark
python -m src.benchmark --n 50 --strategies mixed,entropy
```

## Arrêt de la démo

- Ctrl+C dans les terminaux backend/frontend
- Ou fermer les fenêtres PowerShell/CMD
