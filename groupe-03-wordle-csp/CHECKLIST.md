# ✅ Checklist de Vérification - Projet Wordle CSP Solver

## Pré-démo (avant présentation)

### Infrastructure
- [ ] Python 3.8+ installé et dans le PATH
- [ ] Node.js 18+ installé et dans le PATH
- [ ] npm fonctionnel
- [ ] Git installé (pour versioning)

### Installation Backend
- [ ] cd groupe-03-wordle-csp
- [ ] python -m pip install -r requirements.txt (dépendances de base)
- [ ] python -m pip install -r requirements-api.txt (dépendances API)
- [ ] pytest -q (vérifier 34 tests passent)

### Installation Frontend
- [ ] cd groupe-03-wordle-csp/web
- [ ] npm install (installer dépendances React)
- [ ] Vérifier package.json présent

### Fichiers critiques présents
- [ ] data/mots_fr_5.txt (5817 mots français)
- [ ] api/main.py (backend FastAPI)
- [ ] web/src/App.jsx (frontend React)
- [ ] start_demo.ps1 (script lancement)
- [ ] DEMO.md (guide présentation)

---

## Pendant la démo

### Démarrage
- [ ] Ouvrir PowerShell/CMD dans groupe-03-wordle-csp
- [ ] Exécuter: `.\start_demo.ps1` (ou start_demo.bat)
- [ ] Attendre message "DEMO PRÊTE !"
- [ ] Vérifier backend démarré: http://localhost:8000/health
- [ ] Vérifier frontend démarré: http://localhost:5173

### Test Backend API
- [ ] Ouvrir http://localhost:8000/docs (Swagger)
- [ ] Endpoint /health retourne status "ok"
- [ ] word_count: 5817
- [ ] Tester POST /game/new dans Swagger

### Test Frontend
- [ ] Ouvrir http://localhost:5173
- [ ] Indicateur "✓ API Online (5817 mots)" affiché
- [ ] Bouton "Nouvelle Partie" cliquable
- [ ] Sélecteur stratégie fonctionne

### Mode Interactif
- [ ] Cliquer "🎮 Nouvelle Partie"
- [ ] Suggestion affichée (ex: "TARES")
- [ ] Ajouter contrainte manuellement:
  - [ ] Guess: TARES
  - [ ] Feedback: BYGBB (ou autre)
  - [ ] Cliquer "➕ Ajouter contrainte"
- [ ] Grille mise à jour avec couleurs (vert/jaune/gris)
- [ ] Candidats restants diminuent
- [ ] Nouvelle suggestion générée

### Mode Automatique
- [ ] Aller dans "Mode Automatique"
- [ ] Entrer mot secret: MARDI
- [ ] Cliquer "🤖 Résoudre automatiquement"
- [ ] Historique affiché avec couleurs
- [ ] Message succès "✅ Résolu !"
- [ ] Nombre de tours affiché (2-4 en général)

### Tests CLI (optionnel)
- [ ] Ouvrir nouveau terminal
- [ ] cd groupe-03-wordle-csp
- [ ] python -m src.main interactive --strategy mixed
- [ ] python -m src.main auto --secret MARDI
- [ ] pytest -q (vérifier toujours 34/34)

---

## Post-démo (validation)

### Qualité Code
- [ ] Aucune erreur Python visible
- [ ] Aucune erreur JavaScript console
- [ ] Pas d'avertissements critiques
- [ ] Tests unitaires OK (34/34)

### Documentation
- [ ] README.md à jour avec démarrage rapide
- [ ] DEMO.md présent avec scénario 5 min
- [ ] docs/technical.md complet (architecture)
- [ ] slides/slides.md à jour (25 slides)

### Fichiers Git
- [ ] .gitignore à jour (node_modules, .venv exclus)
- [ ] Pas de fichiers sensibles (.env avec clés API)
- [ ] Structure propre

---

## Troubleshooting

### Backend ne démarre pas
**Symptôme:** Erreur "Module 'fastapi' not found"
**Solution:** `pip install -r requirements-api.txt`

### Frontend ne démarre pas
**Symptôme:** Erreur "Cannot find module 'react'"
**Solution:** `cd web && npm install`

### API Offline dans frontend
**Symptôme:** Indicateur rouge "✗ API Offline"
**Solution:** Vérifier backend démarré sur port 8000

### Port déjà utilisé
**Symptôme:** "Address already in use"
**Solution:** 
- Backend: Tuer processus sur port 8000 (`netstat -ano | findstr :8000`)
- Frontend: Tuer processus sur port 5173

### Tests échouent
**Symptôme:** pytest < 34 tests
**Solution:** Vérifier que src/, tests/ non modifiés

---

## Critères de succès

### Minimum viable (demo doit fonctionner)
✅ Backend démarre sans erreur
✅ Frontend affiche interface
✅ Création partie fonctionne
✅ Ajout contrainte met à jour grille
✅ Simulation automatique complète

### Objectif optimal
✅ Tous les critères minimum
✅ Documentation complète consultable
✅ Tests passent (34/34)
✅ CLI fonctionne en parallèle
✅ Présentation fluide <5 minutes

### Bonus (non obligatoire)
⚪ OR-Tools CP-SAT implémenté
⚪ LLM function calling actif
⚪ Benchmarks comparatifs affichés
⚪ Déploiement en ligne (Vercel/Railway)
