# 📦 Fiche de Livraison - Projet de Détection de Fraude Financière par Graphes

**Groupe 42**  
**Membres :** Malak El Idrissi et Joe Boueri  
**Projet :** Détection de fraude financière par graphes  
**Date de livraison :** Janvier 2026

---

## 1. Résumé du Projet

Ce projet académique s'inscrit dans le cadre du cours d'Intelligence Artificielle appliquée aux finances de l'ECE (ING4). L'objectif principal est de développer un système de détection de fraude financière basé sur l'analyse de graphes avec une architecture modulaire.

### Objectifs Spécifiques

- **Analyse de réseaux financiers** : Modéliser les transactions financières sous forme de graphes pour identifier des patterns suspects
- **Détection d'anomalies** : Implémenter des algorithmes de détection de fraude utilisant les propriétés structurelles des graphes
- **Architecture modulaire** : Concevoir une architecture modulaire et extensible facilitant la maintenance et l'évolution du système
- **Typage Python** : Utiliser le typage Python pour améliorer la robustesse et la lisibilité du code
- **Documentation complète** : Fournir une documentation technique et une présentation claire des résultats

### Approche Méthodologique

Le projet combine des techniques d'analyse de graphes (Graph Neural Networks, Community Detection, Centrality Measures) avec des méthodes d'apprentissage automatique pour identifier les comportements frauduleux dans les transactions financières. L'architecture modulaire permet une séparation claire des responsabilités entre les différents composants du système.

---

## 2. Structure du Projet

```
groupe-42-fraude-graphes/
│
├── README.md                    # Guide d'installation et d'utilisation
│   - Description du projet
│   - Architecture modulaire
│   - Instructions d'installation des dépendances
│   - Guide de démarrage rapide
│   - Exemples d'utilisation (CLI et programmatique)
│   - Format des données
│   - Score de risque
│
├── DELIVERY.md                  # Ce fichier - Fiche de livraison
│
├── src/                         # Code source du projet
│   ├── __init__.py
│   ├── fraud_detector.py        # Point d'entrée CLI et pipeline principal
│   │   - Classe FraudDetectionPipeline
│   │   - Interface en ligne de commande
│   │   - Orchestration du flux de détection
│   │
│   ├── data/                   # Module de gestion des données
│   │   ├── __init__.py
│   │   ├── loader.py          # Chargement CSV/JSON
│   │   │   - Classe TransactionLoader
│   │   │   - Validation des données
│   │   │   - Parsing flexible des timestamps
│   │   │
│   │   └── generator.py       # Générateur de données synthétiques
│   │       - Classe TransactionGenerator
│   │       - Injection de fraudes (cycles, smurfing, anomalies)
│   │       - Reproductibilité via graine aléatoire
│   │
│   ├── graph/                  # Module de construction de graphes
│   │   ├── __init__.py
│   │   └── builder.py         # Transformation en nx.DiGraph
│   │       - Classe GraphBuilder
│   │       - Construction de graphes dirigés
│   │       - Agrégation des transactions
│   │       - Calcul de statistiques
│   │       - Export GEXF/GraphML
│   │
│   ├── detection/              # Module de détection de fraude
│   │   ├── __init__.py
│   │   ├── cycle_detector.py  # Détection de boucles
│   │   │   - Classe BaseDetector (abstraite)
│   │   │   - Classe CycleDetector
│   │   │   - Algorithme de Johnson
│   │   │
│   │   ├── smurfing_detector.py # Analyse des dépôts fractionnés
│   │   │   - Classe SmurfingDetector
│   │   │   - Fenêtre temporelle glissante
│   │   │
│   │   └── network_detector.py # Outliers de centralité
│   │       - Classe NetworkDetector
│   │       - Métriques de centralité (PageRank, Betweenness)
│   │
│   ├── visualization/          # Module de visualisation
│   │   ├── __init__.py
│   │   └── plotter.py         # Visualisation Matplotlib
│   │       - Classe GraphPlotter
│   │       - Graphes complets et sous-graphes
│   │       - Heatmaps de centralité
│   │
│   └── utils.py                # Fonctions utilitaires
│
├── docs/                        # Documentation technique
│   └── technical_report.md      # Rapport technique détaillé
│       - Architecture modulaire
│       - Revue de la littérature
│       - Méthodologie
│       - Score de risque (0-1)
│       - Typage Python
│       - Résultats expérimentaux
│       - Discussion et analyse
│
└── slides/                      # Supports de présentation
    ├── .gitkeep
    └── presentation.md         # Slides de présentation orale
        - Introduction et contexte
        - Architecture modulaire
        - Méthodologie
        - Résultats
        - Conclusion et perspectives
```

### Description des Fichiers Principaux

| Fichier | Description |
|---------|-------------|
| [`README.md`](README.md) | Documentation utilisateur pour l'installation et l'exécution du projet |
| [`src/fraud_detector.py`](src/fraud_detector.py) | Point d'entrée CLI et pipeline principal [`FraudDetectionPipeline`](src/fraud_detector.py:30) |
| [`src/data/loader.py`](src/data/loader.py) | Module [`TransactionLoader`](src/data/loader.py:17) pour le chargement CSV/JSON |
| [`src/data/generator.py`](src/data/generator.py) | Module [`TransactionGenerator`](src/data/generator.py:13) pour la génération de données synthétiques |
| [`src/graph/builder.py`](src/graph/builder.py) | Module [`GraphBuilder`](src/graph/builder.py:13) pour la construction de graphes |
| [`src/detection/cycle_detector.py`](src/detection/cycle_detector.py) | Module [`CycleDetector`](src/detection/cycle_detector.py:87) pour la détection de cycles |
| [`src/detection/smurfing_detector.py`](src/detection/smurfing_detector.py) | Module [`SmurfingDetector`](src/detection/smurfing_detector.py:19) pour la détection de smurfing |
| [`src/detection/network_detector.py`](src/detection/network_detector.py) | Module [`NetworkDetector`](src/detection/network_detector.py:17) pour la détection d'anomalies |
| [`src/visualization/plotter.py`](src/visualization/plotter.py) | Module [`GraphPlotter`](src/visualization/plotter.py:17) pour la visualisation |
| [`docs/technical_report.md`](docs/technical_report.md) | Rapport technique avec l'analyse théorique et les résultats |
| [`slides/presentation.md`](slides/presentation.md) | Support de présentation pour la soutenance orale |

---

## 3. Instructions de Livraison

### Étapes de Préparation

1. **Vérifier la complétude du code**
   - S'assurer que tous les fichiers source sont présents dans `src/`
   - Vérifier que tous les modules sont correctement importés
   - Confirmer que le code s'exécute sans erreur
   - Vérifier que les dépendances sont correctement listées

2. **Finaliser la documentation**
   - Compléter le [`README.md`](README.md) avec les dernières instructions
   - Vérifier que le rapport technique est à jour avec la nouvelle architecture
   - S'assurer que les slides de présentation sont complètes
   - Vérifier que les liens entre les fichiers de documentation sont corrects

3. **Tester le pipeline complet**
   - Exécuter le pipeline avec les paramètres par défaut
   - Tester l'interface CLI avec différentes options
   - Vérifier la génération des fichiers de sortie (CSV, GEXF, PNG)
   - Valider les résultats de détection

4. **Préparer l'archive de livraison**
   - Créer une archive ZIP ou TAR du dossier complet
   - Nommer l'archive : `groupe-42-fraude-graphes-livraison.zip`
   - Inclure tous les fichiers et dossiers
   - Exclure les fichiers temporaires et les caches

5. **Soumission au professeur**
   - Envoyer l'archive par la plateforme spécifiée (Moodle, email, etc.)
   - Respecter la date limite de soumission
   - Conserver une copie de l'archive

### Format de Livraison Attendu

- **Archive compressée** : Format `.zip` ou `.tar.gz`
- **Nom du fichier** : `groupe-42-fraude-graphes-livraison.[extension]`
- **Contenu** : L'intégralité du dossier `groupe-42-fraude-graphes/`

---

## 4. Checklist de Livraison

Avant de soumettre le projet, vérifier que tous les éléments suivants sont en place :

### Code Source
- [ ] Tous les fichiers Python sont présents dans `src/`
- [ ] Structure modulaire respectée (data/, graph/, detection/, visualization/)
- [ ] Le code s'exécute sans erreur
- [ ] Les dépendances sont listées dans le [`README.md`](README.md)
- [ ] Le code est commenté et lisible
- [ ] Le typage Python est utilisé correctement
- [ ] Les classes abstraites sont implémentées correctement
- [ ] Les imports différés sont utilisés pour éviter les dépendances circulaires

### Module `data/`
- [ ] [`loader.py`](src/data/loader.py) implémente [`TransactionLoader`](src/data/loader.py:17)
- [ ] Chargement CSV fonctionnel
- [ ] Chargement JSON fonctionnel
- [ ] Validation des données implémentée
- [ ] [`generator.py`](src/data/generator.py) implémente [`TransactionGenerator`](src/data/generator.py:13)
- [ ] Génération de transactions normales fonctionnelle
- [ ] Injection de cycles de blanchiment fonctionnelle
- [ ] Injection de smurfing fonctionnelle
- [ ] Injection d'anomalies de réseau fonctionnelle

### Module `graph/`
- [ ] [`builder.py`](src/graph/builder.py) implémente [`GraphBuilder`](src/graph/builder.py:13)
- [ ] Construction de graphe depuis les transactions fonctionnelle
- [ ] Agrégation des transactions fonctionnelle
- [ ] Calcul de statistiques fonctionnel
- [ ] Export GEXF fonctionnel
- [ ] Export GraphML fonctionnel

### Module `detection/`
- [ ] [`cycle_detector.py`](src/detection/cycle_detector.py) implémente [`CycleDetector`](src/detection/cycle_detector.py:87)
- [ ] Détection de cycles fonctionnelle
- [ ] Filtrage des cycles fonctionnel
- [ ] Calcul du score de risque fonctionnel
- [ ] [`smurfing_detector.py`](src/detection/smurfing_detector.py) implémente [`SmurfingDetector`](src/detection/smurfing_detector.py:19)
- [ ] Détection de smurfing fonctionnelle
- [ ] Groupement par fenêtre temporelle fonctionnel
- [ ] [`network_detector.py`](src/detection/network_detector.py) implémente [`NetworkDetector`](src/detection/network_detector.py:17)
- [ ] Calcul des métriques de centralité fonctionnel
- [ ] Détection d'anomalies fonctionnelle

### Module `visualization/`
- [ ] [`plotter.py`](src/visualization/plotter.py) implémente [`GraphPlotter`](src/visualization/plotter.py:17)
- [ ] Visualisation de graphe complet fonctionnelle
- [ ] Visualisation de sous-graphe fonctionnelle
- [ ] Visualisation d'alerte fonctionnelle
- [ ] Heatmap de centralité fonctionnelle

### Point d'Entrée
- [ ] [`fraud_detector.py`](src/fraud_detector.py) implémente [`FraudDetectionPipeline`](src/fraud_detector.py:30)
- [ ] Interface CLI fonctionnelle
- [ ] Pipeline complet fonctionnel
- [ ] Génération de rapports fonctionnelle
- [ ] Logging configuré correctement

### Documentation
- [ ] [`README.md`](README.md) est complet et à jour
- [ ] Architecture modulaire documentée
- [ ] Instructions d'installation claires
- [ ] Exemples d'utilisation fonctionnels (CLI et programmatique)
- [ ] Format des données documenté
- [ ] Score de risque documenté
- [ ] [`docs/technical_report.md`](docs/technical_report.md) contient toutes les sections requises
- [ ] Architecture modulaire décrite dans le rapport technique
- [ ] Typage Python documenté
- [ ] Références bibliographiques complètes

### Présentation
- [ ] [`slides/presentation.md`](slides/presentation.md) est complet
- [ ] Architecture modulaire présentée
- [ ] Structure de la présentation est logique
- [ ] Les résultats sont clairement présentés
- [ ] Les visuels (si présents) sont lisibles

### Livraison
- [ ] L'archive de livraison est créée
- [ ] Le nom de l'archive respecte le format demandé
- [ ] Tous les fichiers sont inclus dans l'archive
- [ ] L'archive peut être décompressée sans erreur
- [ ] Le projet fonctionne après extraction de l'archive

### Divers
- [ ] Les noms des membres du groupe sont correctement indiqués
- [ ] Le numéro de groupe (42) est mentionné
- [ ] Aucun fichier temporaire ou inutile n'est inclus
- [ ] Les fichiers `.gitkeep` sont présents si nécessaire
- [ ] Les fichiers `__init__.py` sont présents dans tous les modules

---

## 5. Notes pour la Présentation Orale

### Structure Recommandée de la Présentation

1. **Introduction (2-3 minutes)**
   - Présentation du groupe et du sujet
   - Contexte et problématique de la fraude financière
   - Objectifs du projet
   - Importance de l'architecture modulaire

2. **Architecture Modulaire (3-4 minutes)**
   - Présentation de la structure des modules
   - Séparation des responsabilités
   - Avantages de l'approche modulaire
   - Typage Python et classes abstraites

3. **État de l'Art (3-4 minutes)**
   - Revue des méthodes existantes
   - Pourquoi l'approche par graphes ?
   - Avantages et limites des différentes approches

4. **Méthodologie (5-6 minutes)**
   - Architecture du système proposé
   - Description des modules (data, graph, detection, visualization)
   - Algorithmes utilisés
   - Données et prétraitement
   - Score de risque (0-1)

5. **Résultats (4-5 minutes)**
   - Métriques d'évaluation
   - Comparaison des approches
   - Analyse des résultats
   - Exemples de détection

6. **Discussion et Perspectives (2-3 minutes)**
   - Forces et faiblesses de la solution
   - Améliorations possibles
   - Conclusion

### Conseils de Présentation

- **Préparation** : Répéter la présentation plusieurs fois pour maîtriser le timing
- **Visuels** : Utiliser des graphiques et schémas pour illustrer les concepts
- **Clarté** : Expliquer les termes techniques simplement
- **Interaction** : Prévoir des questions/réponses à la fin
- **Confiance** : Bien connaître le sujet pour répondre aux questions du jury

### Points Forts à Mettre en Avant

- Architecture modulaire et extensible
- Typage Python pour la robustesse
- Code bien documenté et maintenable
- Interface CLI complète
- Résultats expérimentaux solides
- Score de risque normalisé (0-1)
- Perspectives d'amélioration réalistes

### Questions Anticipées

- Pourquoi avoir choisi cette approche par graphes ?
- Quels sont les avantages de l'architecture modulaire ?
- Comment le typage Python améliore-t-il le projet ?
- Quelles sont les limites de votre solution ?
- Comment votre solution se compare-t-elle aux méthodes traditionnelles ?
- Quelles améliorations envisagez-vous pour l'avenir ?
- Comment votre solution pourrait-elle être déployée en production ?
- Comment le score de risque est-il calculé ?

---

## 6. Tests de Validation

### Test 1 : Pipeline Complet

```bash
# Exécuter le pipeline avec paramètres par défaut
python -m src.fraud_detector --seed 42 --verbose
```

**Attendu :**
- Génération de 1000 transactions normales
- Injection de 5 cycles, 3 cas de smurfing, 3 anomalies
- Construction du graphe
- Détection des fraudes
- Génération de la visualisation
- Rapport de résumé dans les logs

### Test 2 : Chargement de Données

```python
from src.data.loader import TransactionLoader

loader = TransactionLoader()
transactions = loader.load_from_csv("transactions.csv")
print(f"Transactions chargées : {len(transactions)}")
```

**Attendu :**
- Chargement réussi des transactions
- Validation des données
- Statistiques correctes

### Test 3 : Construction de Graphe

```python
from src.graph.builder import GraphBuilder

builder = GraphBuilder()
graph = builder.build_from_transactions(transactions)
stats = builder.get_graph_statistics()
print(f"Nœuds : {stats['num_nodes']}, Arêtes : {stats['num_edges']}")
```

**Attendu :**
- Graphe construit correctement
- Statistiques cohérentes

### Test 4 : Détection de Fraudes

```python
from src.detection.cycle_detector import CycleDetector

detector = CycleDetector(max_cycle_length=5)
cycles = detector.detect(graph)
print(f"Cycles détectés : {len(cycles)}")
```

**Attendu :**
- Détection des cycles de blanchiment
- Scores de risque calculés
- Niveaux de risque assignés

### Test 5 : Visualisation

```python
from src.visualization.plotter import GraphPlotter

plotter = GraphPlotter()
plotter.plot_graph(graph, output_file="test_graph.png")
```

**Attendu :**
- Génération de l'image
- Nœuds frauduleux en rouge
- Nœuds normaux en bleu

---

## Informations de Contact

**Groupe 42**
- Malak El Idrissi
- Joe Boueri

**Projet** : Détection de fraude financière par graphes  
**Cours** : Intelligence Artificielle appliquée aux finances  
**Année** : 2025-2026 - ING4 - ECE

---

## Annexe : Dépendances

### Dépendances Python

```
networkx>=3.0
pandas>=2.0
numpy>=1.24
matplotlib>=3.5
```

### Installation

```bash
pip install networkx pandas numpy matplotlib
```

### Version Python

- Python 3.10 ou supérieur

---

*Document généré le 28 janvier 2026*
