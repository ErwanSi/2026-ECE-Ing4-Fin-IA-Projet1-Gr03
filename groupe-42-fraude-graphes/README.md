# Détection de Fraude Financière par Graphes

**Projet académique ECE - Groupe 42**

**Auteurs:** Malak El Idrissi et Joe Boueri

---

## Description du Projet

Ce projet vise à détecter des structures suspectes dans les flux transactionnels financiers en utilisant des techniques d'analyse de graphes. Le système identifie trois types principaux de fraudes :

1. **Cycles de blanchiment** : Boucles de transferts masquant l'origine des fonds
2. **Smurfing (Schtroumpfage)** : Fractionnements de montants vers un compte pivot pour éviter les seuils de déclaration
3. **Anomalies de réseaux** : Comportements atypiques dans la structure des transactions (hubs, rafales, communautés isolées)

## Architecture Modulaire

Le projet est organisé selon une architecture modulaire avec séparation des responsabilités :

```
groupe-42-fraude-graphes/
├── README.md                    # Ce fichier
├── DELIVERY.md                  # Fiche de livraison
├── src/
│   ├── __init__.py
│   ├── fraud_detector.py        # Point d'entrée CLI et pipeline principal
│   ├── data/                   # Module de gestion des données
│   │   ├── __init__.py
│   │   ├── loader.py          # Chargement CSV/JSON
│   │   └── generator.py       # Générateur de données synthétiques
│   ├── graph/                  # Module de construction de graphes
│   │   ├── __init__.py
│   │   └── builder.py         # Transformation en nx.DiGraph
│   ├── detection/              # Module de détection de fraude
│   │   ├── __init__.py
│   │   ├── cycle_detector.py  # Détection de boucles
│   │   ├── smurfing_detector.py # Analyse des dépôts fractionnés
│   │   └── network_detector.py # Outliers de centralité
│   └── visualization/          # Module de visualisation
│       ├── __init__.py
│       └── plotter.py         # Visualisation Matplotlib
├── docs/
│   └── technical_report.md      # Rapport technique et recherches théoriques
└── slides/                      # Dossier pour les présentations
    └── presentation.md
```

### Modules Principaux

#### Module `data/`
- **[`loader.py`](src/data/loader.py)** : [`TransactionLoader`](src/data/loader.py:17) - Charge les transactions depuis CSV/JSON avec validation
- **[`generator.py`](src/data/generator.py)** : [`TransactionGenerator`](src/data/generator.py:13) - Génère des données synthétiques avec fraudes intégrées

#### Module `graph/`
- **[`builder.py`](src/graph/builder.py)** : [`GraphBuilder`](src/graph/builder.py:13) - Transforme les transactions en graphe dirigé NetworkX

#### Module `detection/`
- **[`cycle_detector.py`](src/detection/cycle_detector.py)** : [`CycleDetector`](src/detection/cycle_detector.py:87) - Détecte les cycles de blanchiment
- **[`smurfing_detector.py`](src/detection/smurfing_detector.py)** : [`SmurfingDetector`](src/detection/smurfing_detector.py:19) - Détecte les dépôts fractionnés
- **[`network_detector.py`](src/detection/network_detector.py)** : [`NetworkDetector`](src/detection/network_detector.py:17) - Détecte les anomalies de centralité

#### Module `visualization/`
- **[`plotter.py`](src/visualization/plotter.py)** : [`GraphPlotter`](src/visualization/plotter.py:17) - Visualise les graphes avec Matplotlib

#### Point d'Entrée
- **[`fraud_detector.py`](src/fraud_detector.py)** : [`FraudDetectionPipeline`](src/fraud_detector.py:30) - Pipeline principal avec interface CLI

## Prérequis

- **Python 3.10 ou supérieur**
- **pip** (gestionnaire de paquets Python)

## Dépendances

Le projet nécessite les bibliothèques Python suivantes :

```bash
networkx>=3.0
pandas>=2.0
numpy>=1.24
matplotlib>=3.5
```

## Installation

### 1. Cloner ou télécharger le projet

```bash
cd groupe-42-fraude-graphes
```

### 2. Créer un environnement virtuel (recommandé)

```bash
# Sur macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Sur Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install networkx pandas numpy matplotlib
```

## Utilisation

> **⚠️ Important :** Toutes les commandes ci-dessous doivent être exécutées depuis la racine du dossier `groupe-42-fraude-graphes`.

### Interface en Ligne de Commande (CLI)

Le projet fournit une interface CLI complète via le module [`fraud_detector.py`](src/fraud_detector.py):

> **⚠️ Important sur macOS :** Sur macOS, utilisez `python3` au lieu de `python` pour exécuter les commandes.

#### Exemples d'Utilisation

**Génération de base :**
```bash
python3 -m src.fraud_detector --accounts 20 --normal 100 --cycles 1 --smurfing 1 --anomalies 1 --viz-output output/test_graph.png --verbose
```

**Analyse de fichiers CSV :**
```bash
# Petit dataset
python3 -m src.fraud_detector --input data/synthetic/small_dataset.csv --viz-output output/small_test.png --verbose

# Dataset moyen
python3 -m src.fraud_detector --input data/synthetic/medium_dataset.csv --viz-output output/medium_test.png --verbose

# Grand dataset
python3 -m src.fraud_detector --input data/synthetic/large_dataset.csv --viz-output output/large_test.png --verbose
```

**Autres options :**
```bash
# Exécution avec paramètres par défaut
python3 -m src.fraud_detector

# Génération de données personnalisées
python3 -m src.fraud_detector --accounts 200 --normal 2000 --cycles 10 --smurfing 5 --anomalies 5

# Utilisation d'une graine pour la reproductibilité
python3 -m src.fraud_detector --seed 42

# Export des données et du graphe
python3 -m src.fraud_detector --data-output transactions.csv --graph-output graph.gexf

# Mode verbeux
python3 -m src.fraud_detector --verbose

# Aide
python3 -m src.fraud_detector --help
```

#### Arguments CLI

| Argument | Défaut | Description |
|----------|--------|-------------|
| `--accounts` | 100 | Nombre de comptes bancaires |
| `--normal` | 1000 | Nombre de transactions normales |
| `--cycles` | 5 | Nombre de cycles de blanchiment à injecter |
| `--smurfing` | 3 | Nombre de cas de smurfing à injecter |
| `--anomalies` | 3 | Nombre d'anomalies de réseau à injecter |
| `--seed` | None | Graine aléatoire pour la reproductibilité |
| `--data-output` | None | Fichier de sortie CSV pour les données |
| `--graph-output` | None | Fichier de sortie GEXF pour le graphe |
| `--viz-output` | fraud_graph.png | Fichier de sortie pour la visualisation |
| `--max-cycle-length` | 5 | Longueur maximale des cycles à détecter |
| `--smurfing-threshold` | 1000.0 | Seuil de montant pour le smurfing |
| `--pagerank-threshold` | 0.01 | Seuil de PageRank |
| `--betweenness-threshold` | 0.05 | Seuil de betweenness centrality |
| `--verbose` | False | Active le mode verbeux |

### Utilisation Programmatique

#### Exemple 1 : Utilisation du Pipeline Complet

```python
from src.fraud_detector import FraudDetectionPipeline

# Créer le pipeline
pipeline = FraudDetectionPipeline(verbose=True)

# Exécuter le pipeline complet
results = pipeline.run_full_pipeline(
    num_accounts=100,
    num_normal=1000,
    num_cycles=5,
    num_smurfing=3,
    num_anomalies=3,
    seed=42,
    data_output="transactions.csv",
    graph_output="graph.gexf",
    visualization_output="fraud_graph.png"
)

print(f"Cycles détectés : {len(results['cycles'])}")
print(f"Smurfing détecté : {len(results['smurfing'])}")
print(f"Anomalies détectées : {len(results['anomalies'])}")
```

#### Exemple 2 : Chargement de Données Existantes

```python
from src.data.loader import TransactionLoader
from src.graph.builder import GraphBuilder
from src.detection.cycle_detector import CycleDetector

# Charger les transactions depuis un fichier CSV
loader = TransactionLoader()
transactions = loader.load_from_csv("data/transactions.csv")

# Construire le graphe
builder = GraphBuilder()
graph = builder.build_from_transactions(transactions)

# Détecter les cycles
detector = CycleDetector(max_cycle_length=5)
cycles = detector.detect(graph)

print(f"Cycles détectés : {len(cycles)}")
for cycle in cycles:
    print(f"  - Score de risque : {cycle['risk_score']:.2f} ({cycle['risk_level']})")
```

#### Exemple 3 : Génération de Données Synthétiques

```python
from src.data.generator import TransactionGenerator

# Créer le générateur
generator = TransactionGenerator(num_accounts=100, seed=42)

# Générer un jeu de données complet
transactions = generator.generate_complete_dataset(
    num_normal=1000,
    num_cycles=5,
    num_smurfing=3,
    num_anomalies=3
)

# Sauvegarder en CSV
generator.save_to_csv("transactions.csv")

print(f"Transactions générées : {len(transactions)}")
print(f"  - Normales : {len(generator.get_normal_transactions())}")
print(f"  - Frauduleuses : {len(generator.get_fraudulent_transactions())}")
```

#### Exemple 4 : Visualisation

```python
from src.visualization.plotter import GraphPlotter
from src.graph.builder import GraphBuilder

# Charger ou construire le graphe
builder = GraphBuilder()
graph = builder.build_from_transactions(transactions)

# Créer le visualiseur
plotter = GraphPlotter()

# Visualiser le graphe complet
plotter.plot_graph(
    graph,
    output_file="fraud_graph.png",
    highlight_fraud=True,
    layout="spring"
)

# Visualiser une alerte spécifique
if cycles:
    plotter.plot_alert(
        graph,
        cycles[0],
        output_file="alert_cycle.png"
    )

# Visualiser avec heatmap de centralité
plotter.plot_centrality_heatmap(
    graph,
    metric="pagerank",
    output_file="centrality_heatmap.png"
)
```

## Format des Données

### Format CSV

Le fichier CSV doit contenir les colonnes suivantes :

| Colonne | Description | Exemple |
|---------|-------------|---------|
| `sender` / `sender_id` | Identifiant de l'émetteur | `ACC_0001` |
| `receiver` / `receiver_id` | Identifiant du destinataire | `ACC_0002` |
| `amount` | Montant de la transaction | `1500.50` |
| `timestamp` | Horodatage (ISO ou Unix) | `2024-01-15T10:30:00` |
| `transaction_id` | Identifiant unique (optionnel) | `TX_001` |
| `type` | Type de transaction (optionnel) | `normal` |

**Exemple de fichier CSV :**

```csv
sender,receiver,amount,timestamp,transaction_id,type
ACC_0001,ACC_0002,1500.50,2024-01-15T10:30:00,TX_001,normal
ACC_0002,ACC_0003,2500.00,2024-01-15T11:45:00,TX_002,normal
ACC_0003,ACC_0001,1500.00,2024-01-15T14:20:00,TX_003,money_laundering
```

### Format JSON

Le fichier JSON doit contenir un tableau d'objets :

```json
[
  {
    "sender": "ACC_0001",
    "receiver": "ACC_0002",
    "amount": 1500.50,
    "timestamp": "2024-01-15T10:30:00",
    "transaction_id": "TX_001",
    "type": "normal"
  },
  {
    "sender": "ACC_0002",
    "receiver": "ACC_0003",
    "amount": 2500.00,
    "timestamp": "2024-01-15T11:45:00",
    "transaction_id": "TX_002",
    "type": "normal"
  }
]
```

## Score de Risque

Le système calcule un score de risque entre 0 et 1 pour chaque alerte détectée :

| Score | Niveau | Interprétation |
|-------|--------|----------------|
| 0.0 - 0.3 | LOW | Faible risque |
| 0.3 - 0.5 | MEDIUM | Risque modéré |
| 0.5 - 0.7 | HIGH | Risque élevé |
| 0.7 - 1.0 | CRITICAL | Risque très élevé (requiert une investigation) |

Le score est calculé en combinant plusieurs facteurs :
- **Montant** : Plus le montant est élevé, plus le score augmente
- **Durée** : Plus la durée est courte, plus le score augmente
- **Répétition** : Plus le nombre de transactions est élevé, plus le score augmente
- **Centralité** : Pour les anomalies de réseau, les métriques de centralité sont prises en compte

## Typage Python

Le projet utilise le typage Python pour améliorer la maintenabilité et la lisibilité du code :

```python
from typing import List, Dict, Any, Optional

def detect(self, graph: nx.DiGraph) -> List[Dict[str, Any]]:
    """
    Détecte les fraudes dans le graphe.
    
    Args:
        graph: Le graphe de transactions à analyser.
    
    Returns:
        Liste des alertes de fraude détectées.
    """
    ...
```

## Documentation

Pour plus de détails sur les approches algorithmiques et les fondements théoriques, consultez le rapport technique :

```bash
cat docs/technical_report.md
```

## Dépannage

### Erreur : "zsh: command not found: python"

**Solution :** Sur macOS, utilisez `python3` au lieu de `python`

```bash
# Utilisez python3 sur macOS
python3 -m src.fraud_detector

# Ou créez un alias (optionnel)
alias python=python3
```

### Erreur : "ModuleNotFoundError: No module named 'networkx'"

**Solution :** Installez les dépendances manquantes

```bash
pip install networkx pandas numpy matplotlib
```

### Erreur : "Le graphe n'a pas été construit"

**Solution :** Appelez [`build_graph()`](src/graph/builder.py:33) avant les méthodes de détection

```python
builder = GraphBuilder()
graph = builder.build_from_transactions(transactions)  # Nécessaire avant detect()
```

### Aucune fraude détectée

**Causes possibles :**
- Les paramètres de détection sont trop stricts
- Les données ne contiennent pas de patterns de fraude
- Le seuil de montant filtre les transactions suspectes

**Solution :** Ajustez les paramètres de détection

```python
# Pour les cycles
detector = CycleDetector(max_cycle_length=3)

# Pour le smurfing
detector = SmurfingDetector(threshold=5000.0, time_window_hours=48)

# Pour les anomalies
detector = NetworkDetector(pagerank_threshold=0.005, betweenness_threshold=0.02)
```

## Architecture et Design Patterns

### Classes de Base

Le module de détection utilise une classe abstraite [`BaseDetector`](src/detection/cycle_detector.py:16) qui définit l'interface commune pour tous les détecteurs :

```python
class BaseDetector(ABC):
    @abstractmethod
    def detect(self, graph: nx.DiGraph) -> List[Dict[str, Any]]:
        pass
    
    def _calculate_risk_score(self, amount: float, ...) -> float:
        ...
```

### Séparation des Responsabilités

Chaque module a une responsabilité unique :
- **data/** : Chargement et génération des données
- **graph/** : Construction et manipulation des graphes
- **detection/** : Détection des différents types de fraude
- **visualization/** : Visualisation des résultats

### Imports Différés

Pour éviter les dépendances circulaires, le pipeline utilise des imports différés :

```python
def detect_cycles(self, builder: Any, ...) -> List[Dict[str, Any]]:
    from detection.cycle_detector import CycleDetector
    detector = CycleDetector(max_cycle_length=max_cycle_length)
    ...
```

## Licence

Ce projet est réalisé dans le cadre académique de l'ECE.

## Contact

- **Malak El Idrissi**
- **Joe Boueri**
- **Groupe 42**
- **ECE - Ingénierie Financière et IA**
