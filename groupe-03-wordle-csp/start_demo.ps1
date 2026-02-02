# Script de lancement de la démo Wordle CSP Solver
# Usage: .\start_demo.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Wordle CSP Solver - Démo ECE ING4" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Vérifier que nous sommes dans le bon répertoire
if (-not (Test-Path "api\main.py")) {
    Write-Host "ERREUR: Ce script doit être exécuté depuis le dossier groupe-03-wordle-csp" -ForegroundColor Red
    exit 1
}

Write-Host "[1/3] Vérification de l'environnement..." -ForegroundColor Yellow

# Vérifier Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  ✓ Python détecté: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Python n'est pas installé ou n'est pas dans le PATH" -ForegroundColor Red
    Write-Host "    Installez Python 3.8+ depuis https://www.python.org/" -ForegroundColor Red
    exit 1
}

# Vérifier Node.js
try {
    $nodeVersion = node --version 2>&1
    Write-Host "  ✓ Node.js détecté: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Node.js n'est pas installé ou n'est pas dans le PATH" -ForegroundColor Red
    Write-Host "    Installez Node.js 18+ depuis https://nodejs.org/" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[2/3] Lancement du backend API..." -ForegroundColor Yellow
Write-Host "  Backend démarrera sur http://localhost:8000" -ForegroundColor Cyan
Write-Host "  Documentation: http://localhost:8000/docs" -ForegroundColor Cyan

# Démarrer le backend en arrière-plan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; python -m uvicorn api.main:app --reload --port 8000"

Start-Sleep -Seconds 3

Write-Host ""
Write-Host "[3/3] Lancement du frontend React..." -ForegroundColor Yellow
Write-Host "  Frontend démarrera sur http://localhost:5173" -ForegroundColor Cyan

# Vérifier si node_modules existe
if (-not (Test-Path "web\node_modules")) {
    Write-Host "  Installation des dépendances npm (première fois)..." -ForegroundColor Yellow
    Set-Location web
    npm install
    Set-Location ..
}

# Démarrer le frontend
Set-Location web
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  DEMO PRÊTE !" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Backend API:  http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "Frontend Web: http://localhost:5173" -ForegroundColor Cyan
Write-Host ""
Write-Host "Appuyez sur Ctrl+C pour arrêter le frontend" -ForegroundColor Yellow
Write-Host ""

npm run dev
