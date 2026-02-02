@echo off
chcp 65001 >nul
echo ========================================
echo   Wordle CSP Solver - Démo ECE ING4
echo ========================================
echo.

REM Vérifier que nous sommes dans le bon répertoire
if not exist "api\main.py" (
    echo ERREUR: Ce script doit être exécuté depuis le dossier groupe-03-wordle-csp
    pause
    exit /b 1
)

echo [1/3] Vérification de l'environnement...

REM Vérifier Python
python --version >nul 2>&1
if errorlevel 1 (
    echo   X Python n'est pas installé ou n'est pas dans le PATH
    echo     Installez Python 3.8+ depuis https://www.python.org/
    pause
    exit /b 1
) else (
    echo   √ Python détecté
)

REM Vérifier Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo   X Node.js n'est pas installé ou n'est pas dans le PATH
    echo     Installez Node.js 18+ depuis https://nodejs.org/
    pause
    exit /b 1
) else (
    echo   √ Node.js détecté
)

echo.
echo [2/3] Lancement du backend API...
echo   Backend: http://localhost:8000
echo   Documentation: http://localhost:8000/docs
echo.

REM Démarrer le backend dans une nouvelle fenêtre
start "Backend API" cmd /k "python -m uvicorn api.main:app --reload --port 8000"

timeout /t 3 /nobreak >nul

echo [3/3] Lancement du frontend React...
echo   Frontend: http://localhost:5173
echo.

REM Vérifier si node_modules existe
if not exist "web\node_modules" (
    echo   Installation des dépendances npm (première fois)...
    cd web
    call npm install
    cd ..
)

echo.
echo ========================================
echo   DEMO PRÊTE !
echo ========================================
echo.
echo Backend API:  http://localhost:8000/docs
echo Frontend Web: http://localhost:5173
echo.
echo Appuyez sur Ctrl+C pour arrêter le frontend
echo.

cd web
npm run dev
