# Portfolio Optimization with Real-World Constraints (CSP/MILP)
Projet #40 — Optimisation de portefeuille avec contraintes réelles : lots entiers, cardinalité, diversification sectorielle, coûts de transaction, turnover cap, risque CVaR (scénarios), évaluation walk-forward out-of-sample.

> Point d’entrée “présentation” : `python main.py`  
> Il affiche **uniquement 2 figures** (Wealth OOS + Pareto Sharpe vs Turnover) et imprime un mini-report terminal.

---

## 1) Objectif du projet (contexte sujet)
L’optimisation classique type Markowitz ignore des contraintes de trading réalistes :
- **Quantités entières / lots minimums**
- **Cardinalité** (nombre maximal d’actifs en portefeuille)
- **Contraintes sectorielles** (diversification)
- **Coûts de transaction** et **rééquilibrage** (turnover)
- Contraintes additionnelles possibles (géographie, limites d’exposition, etc.)

Dès qu’on introduit des variables entières (lots) et la cardinalité, le problème devient **combinatoire** (NP-difficile). On le formule donc naturellement en :
- **CSP / CP-SAT** (OR-Tools) pour la version entière (principal)  
- **MILP** (OR-Tools MPSolver + CBC) pour comparaison (même famille de formulation linéaire en entiers)

---

## 2) Fonctionnalités principales
### Optimisation “tradable”
- **Lots entiers** `q_i` et poids `w_i = q_i / Q`
- **Cardinalité** via variables binaires `z_i` et contrainte `sum(z_i) <= K`
- **Bornes de concentration**
  - Par actif : `w_i <= w_max`
  - Minimum d’investissement : `q_i >= q_min * z_i`
- **Contraintes sectorielles**
  - `L_s <= sum_{i∈sector s} w_i <= U_s`
- **Rééquilibrage réaliste**
  - Turnover : `sum_i |q_i - q_i_old|`
  - Option **cap turnover** (hard constraint)
  - **Coûts de transaction** linéaires proportionnels au turnover

### Gestion du risque
- Risque downside via **CVaR (Expected Shortfall)** sur des **scénarios** bootstrapés à partir des rendements historiques de la fenêtre d’entraînement.

### Évaluation out-of-sample
- Backtest **walk-forward** : fenêtre train → optimisation → test OOS → roll.

### Analyse de sensibilité
- Sweep de paramètres (turnover cap, lambda CVaR) et **courbe de compromis** Sharpe vs turnover.

---

## 3) Installation
### Prérequis
- Python 3.10+ recommandé
- Windows/macOS/Linux
- Connexion internet recommandée (pour télécharger les prix via yfinance)

### Installer les dépendances
Dans un venv :
```bash
pip install -r requirements.txt
