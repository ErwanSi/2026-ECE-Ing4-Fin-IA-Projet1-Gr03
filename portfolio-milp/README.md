# Sujet #40 — Optimisation de portefeuille avec contraintes réelles (CSP/MILP)

## Résumé
Projet "IA hybride" :
- IA predictive (ML) : quantile regression pour estimer la distribution conditionnelle des rendements
- IA prescriptive (CSP/MILP) : OR-Tools CP-SAT pour optimiser sous contraintes réelles
- Evaluation : walk-forward OUT-OF-SAMPLE (OOS)

## Contraintes (CP-SAT)
- Lots entiers (q_i)
- Cardinalité max K actifs (z_i binaires)
- Contraintes sectorielles (min/max d'exposition)
- Coûts de transaction via turnover |q - q_old|
- Risque CVaR (linéaire) sur scénarios

## Comparaison
- Equal-weight
- CVXPY (convexe) : mean-variance + pénalité de turnover
- CP-SAT (CSP/MILP) : lots + cardinalité + secteurs + coûts + CVaR

## Exécution
1) Activer venv + installer :
   python -m pip install -r requirements.txt
2) Lancer :
   python main.py

Sorties :
- performance_oos.png
- runs/run_*.json
- runs/metrics_*.csv
