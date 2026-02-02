# Sujet : 50. Optimisation d'exécution d'ordres par contraintes (TWAP/VWAP)
# Membres : Ilhan Taskin  Farhan Morisson
ING4 Groupe 3

------------------------------------------------------------------------------------------
## 1. Contexte et problématique

L’exécution de gros ordres boursiers pose un problème central en finance de marché : comment acheter ou vendre un volume important sans influencer excessivement le prix du marché.

Lorsqu’un ordre est trop concentré dans le temps, il peut créer un impact de marché : le prix évolue défavorablement à cause de la pression d’achat ou de vente. À l’inverse, une exécution trop lente peut exposer le trader à un risque de timing et à une dégradation du prix moyen obtenu.

Deux stratégies de référence sont classiquement utilisées :

- TWAP (Time-Weighted Average Price) : le volume total est réparti uniformément dans le temps.

- VWAP (Volume-Weighted Average Price) : le volume est réparti proportionnellement au volume de marché observé.

Ces approches peuvent être vues comme des cas particuliers d’un problème plus général :

Trouver un planning d’exécution qui minimise l’impact de marché tout en respectant des contraintes de volume, de timing et d’alignement avec la liquidité du marché.

Dans ce projet, nous formulons ce problème comme un problème de satisfaction et d’optimisation sous contraintes (CSP), résolu à l’aide d’un solveur CP-SAT (OR-Tools). Nous comparons cette approche aux baselines TWAP et VWAP, et proposons une ouverture vers des méthodes d’apprentissage par renforcement (Reinforcement Learning).

------------------------------------------------------------------------------------------
## 2. Objectifs du projet

L’objectif de ce projet est de concevoir et d’évaluer un système d’exécution d’ordres boursiers capable de répartir un volume important dans le temps de manière efficace, réaliste et contrôlée.

Plus précisément, nous cherchons à :

Mettre en œuvre des stratégies de référence afin de disposer de points de comparaison clairs. La stratégie TWAP sert de baseline simple et uniforme, tandis que la stratégie VWAP permet de s’aligner sur le profil de liquidité du marché.

Formuler le problème comme une optimisation sous contraintes, dans laquelle le volume total à exécuter, les limites de participation au marché et les bornes par tranche sont explicitement intégrées dans un modèle formel.

Minimiser un proxy d’impact de marché, en favorisant des plannings d’exécution lissés qui évitent les concentrations de volume susceptibles de déplacer les prix.

Contrôler l’écart par rapport à un benchmark VWAP, afin de garantir que l’exécution reste cohérente avec la dynamique de liquidité observée sur le marché.

Analyser le compromis entre ces deux objectifs antagonistes, en mettant en évidence une frontière de Pareto entre discrétion d’exécution (faible impact) et alignement avec le marché (faible erreur de tracking).

Valider l’approche sur données réelles intraday, en utilisant des volumes de marché récents pour construire des plannings d’exécution réalistes.

Ouvrir vers des approches basées sur l’apprentissage par renforcement, dans lesquelles une politique d’exécution pourrait être apprise directement à partir de l’interaction avec un environnement de marché simulé ou réel.

------------------------------------------------------------------------------------------
## 3. Architecture du projet
```
groupe-03-Optimisation-TWAP-VWAP/
├── README.md
├── src/
│   ├── __init__.py
│   ├── cli.py
│   ├── data/
│   │   └── market_data.py
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── twap.py
│   │   ├── vwap.py
│   │   └── constrained_opt_cp.py
│   └── rl/
│       ├── __init__.py
│       ├── env.py
│       └── qlearning.py
├── tests/
│   ├── test_twap.py
│   ├── test_vwap.py
│   └── test_opt_cp.py
├── run_twap.py
├── run_vwap.py
├── run_opt.py
├── run_compare_strat.py
├── run_real_data.py
├── run_rl_train.py
├── run_rl_test.py
├── rl_qtable.pkl            # généré par run_rl_train.py
└── slides/
│    └── presentation.pdf
└── docs/
     └── documentation.pdf

```
------------------------------------------------------------------------------------------
## 4. Installation (Windows)
Prérequis : 
```
Python ≥ 3.9
```

Création de l’environnement virtuel: 
```
python -m venv .venv .\.venv\Scripts\Activate.ps1
```
Installation des dépendances:
```
python -m pip install --upgrade pip
python -m pip install ortools pytest yfinance pandas matplotlib
python -m pip install matplotlib

```

Vérification:
```
python -c "import ortools, yfinance, pandas; print('Dependencies OK')"
```

### Pour MAC
```
python3 -m venv .venv .\.venv\Scripts\Activate.ps1
```
Installation des dépendances:
```
python3 -m pip install --upgrade pip
python3 -m pip install ortools pytest yfinance pandas matplotlib
python3 -m pip install matplotlib

```

Vérification:
```
python3 -c "import ortools, yfinance, pandas; print('Dependencies OK')"
```

------------------------------------------------------------------------------------------
## 5. Utilisation (Windows)
### pour Mac ajouter un '3' après 'python', ex: python3

Entrer dans le fichier :
```
cd groupe-03-Optimisation-TWAP-VWAP
```

Baseline TWAP:
```
python run_twap.py
```

Baseline VWAP:
```
python run_vwap.py
```

Optimisation sous contraintes (CSP / CP-SAT):
```
python run_opt_cp.py
```

Comparaison des stratégies:
```
python run_compare_strat.py
```

Données marché réelles (snapshot intraday):
```
python run_real_data.py
```

Reinforcement Learning Agent:
```
python .\run_rl_train.py
python .\run_rl_test.py
```
ou alors simplement lancer directement les codes avec le bouton 'Run File'

------------------------------------------------------------------------------------------
## 6. Test

**Lancer tous les tests** :
```
cd .\groupe-03-Optimisation-TWAP-VWAP
python -m pytest
```

------------------------------------------------------------------------------------------
 