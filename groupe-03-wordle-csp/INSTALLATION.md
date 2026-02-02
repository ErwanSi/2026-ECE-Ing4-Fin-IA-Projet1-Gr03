# 📦 Guide d'Installation Complet - Wordle CSP Solver

## Prérequis Système

### Logiciels requis
- **Python 3.8 ou supérieur**
  - Télécharger: https://www.python.org/downloads/
  - Vérifier: `python --version`
  
- **Node.js 18 ou supérieur**
  - Télécharger: https://nodejs.org/
  - Vérifier: `node --version` et `npm --version`

- **Git** (optionnel, pour clonage)
  - Télécharger: https://git-scm.com/downloads

### Systèmes supportés
- ✅ Windows 10/11 (PowerShell, CMD)
- ✅ macOS 12+ (Terminal, Bash, Zsh)
- ✅ Linux (Ubuntu, Debian, Fedora)

---

## Installation Rapide (Recommandée)

### Windows

**Option 1: Script automatique (PowerShell)**
```powershell
cd groupe-03-wordle-csp
.\start_demo.ps1
```

**Option 2: Script automatique (CMD)**
```batch
cd groupe-03-wordle-csp
start_demo.bat
```

Le script installe automatiquement les dépendances npm si nécessaire.

### macOS / Linux

```bash
cd groupe-03-wordle-csp

# Terminal 1 - Backend
python3 -m pip install -r requirements-api.txt
python3 -m uvicorn api.main:app --reload --port 8000

# Terminal 2 - Frontend
cd web
npm install
npm run dev
```

---

## Installation Manuelle Détaillée

### Étape 1: Cloner le projet (si nécessaire)

```bash
git clone <url-du-repo>
cd 2026-ECE-Ing4-Fin-IA-Projet1-Gr03/groupe-03-wordle-csp
```

### Étape 2: Installer les dépendances Python

**Créer un environnement virtuel (recommandé):**
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

**Installer les dépendances:**
```bash
# Dépendances de base (solveur CSP)
pip install -r requirements.txt

# Dépendances API (backend)
pip install -r requirements-api.txt
```

**Vérifier l'installation:**
```bash
pytest -q
# Attendu: 34 passed
```

### Étape 3: Installer les dépendances Frontend

```bash
cd web
npm install
cd ..
```

**Dépendances installées:**
- react: 18.2.0
- react-dom: 18.2.0
- vite: 5.0.8
- @vitejs/plugin-react: 4.2.1

### Étape 4: Vérifier les fichiers critiques

```bash
# Vérifier dictionnaire
ls data/mots_fr_5.txt

# Vérifier backend
ls api/main.py

# Vérifier frontend
ls web/src/App.jsx
```

---

## Lancement de la Démo

### Méthode 1: Scripts automatiques (Windows)

**PowerShell:**
```powershell
.\start_demo.ps1
```

**CMD:**
```batch
start_demo.bat
```

### Méthode 2: Manuel (tous systèmes)

**Terminal 1 - Backend API:**
```bash
cd groupe-03-wordle-csp
python -m uvicorn api.main:app --reload --port 8000
```

**Terminal 2 - Frontend React:**
```bash
cd groupe-03-wordle-csp/web
npm run dev
```

**Accès:**
- Frontend: http://localhost:5173
- API Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

---

## Utilisation de la CLI (optionnel)

Le projet inclut une CLI fonctionnelle indépendante du web:

### Mode interactif
```bash
python -m src.main interactive --strategy mixed
```

### Mode automatique
```bash
python -m src.main auto --secret MARDI --strategy entropy
```

### Suggestion basée sur historique
```bash
python -m src.main suggest \
  --guesses TARES,MARDI \
  --feedbacks BYGBB,GGGGG \
  --strategy mixed
```

### Benchmark
```bash
python -m src.benchmark --n 50 --strategies naive,frequency,entropy,mixed
```

---

## Tests

### Tests unitaires
```bash
pytest                    # Mode verbose
pytest -q                 # Mode quiet
pytest -v                 # Mode très verbose
pytest --cov=src tests/   # Avec couverture de code
```

**Résultat attendu:** 34 passed

### Tests manuels

**Test backend:**
```bash
curl http://localhost:8000/health
```

**Test frontend:**
Ouvrir http://localhost:5173 dans un navigateur.

---

## Configuration Avancée

### Variables d'environnement (optionnel)

Créer un fichier `.env` dans groupe-03-wordle-csp:

```env
# Backend
DICTIONARY_PATH=./data/mots_fr_5.txt
API_HOST=0.0.0.0
API_PORT=8000

# Frontend (web/.env.local)
VITE_API_URL=http://localhost:8000
```

### Ports personnalisés

**Backend:**
```bash
python -m uvicorn api.main:app --port 5000
```

**Frontend** (modifier `vite.config.js`):
```javascript
server: {
  port: 3000,
  // ...
}
```

---

## Résolution de Problèmes

### Problème: Module 'fastapi' not found

**Cause:** Dépendances API non installées

**Solution:**
```bash
pip install -r requirements-api.txt
```

### Problème: Cannot find module 'react'

**Cause:** Dépendances npm non installées

**Solution:**
```bash
cd web
npm install
```

### Problème: Port already in use (8000 ou 5173)

**Cause:** Processus déjà en cours

**Solution Windows:**
```powershell
# Trouver le processus
netstat -ano | findstr :8000

# Tuer le processus (remplacer PID)
taskkill /PID <PID> /F
```

**Solution macOS/Linux:**
```bash
# Trouver et tuer
lsof -ti:8000 | xargs kill -9
```

### Problème: API Offline dans le frontend

**Cause:** Backend non démarré ou CORS mal configuré

**Solution:**
1. Vérifier backend: http://localhost:8000/health
2. Vérifier logs backend pour erreurs
3. Vérifier CORS dans `api/config.py`

### Problème: Tests échouent

**Cause:** Modifications du code source

**Solution:**
```bash
# Réinitialiser environnement
pip uninstall -y -r requirements.txt
pip install -r requirements.txt
pytest -q
```

---

## Désinstallation

### Supprimer dépendances Python

```bash
# Désactiver venv
deactivate

# Supprimer dossier
rm -rf .venv  # ou rmdir /s .venv sur Windows
```

### Supprimer dépendances npm

```bash
cd web
rm -rf node_modules  # ou rmdir /s node_modules sur Windows
```

---

## Support et Documentation

### Documentation disponible

- **README.md** - Guide de démarrage rapide
- **DEMO.md** - Scénario de présentation (5 min)
- **docs/technical.md** - Documentation technique complète
- **slides/slides.md** - Présentation (25 slides)
- **CHECKLIST.md** - Liste de vérification pré-démo

### Ressources externes

- FastAPI: https://fastapi.tiangolo.com/
- React: https://react.dev/
- Vite: https://vitejs.dev/
- Pytest: https://docs.pytest.org/

---

## Contribution et Développement

### Structure du projet

```
groupe-03-wordle-csp/
├── src/          # Solveur CSP (NE PAS MODIFIER)
├── tests/        # Tests unitaires (NE PAS MODIFIER)
├── api/          # Backend FastAPI (modifiable)
├── web/          # Frontend React (modifiable)
├── data/         # Dictionnaire français
└── docs/         # Documentation
```

### Commandes développement

**Formater le code Python:**
```bash
black src/ api/
flake8 src/ api/
```

**Linter frontend:**
```bash
cd web
npm run lint  # (si configuré)
```

**Build production:**
```bash
# Frontend
cd web
npm run build
# Sortie: web/dist/
```

---

**Date de mise à jour:** 2026-02-02
**Version:** 1.0.0
**Contact:** Groupe 03 - ECE ING4
