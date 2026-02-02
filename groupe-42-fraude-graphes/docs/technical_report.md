# Rapport Technique : Détection de Fraude Financière par Graphes

**Projet académique ECE - Groupe 42**

**Auteurs:** Malak El Idrissi et Joe Boueri

---

## Table des Matières

1. [Introduction](#introduction)
2. [Architecture Modulaire](#architecture-modulaire)
3. [Les Trois Types de Fraude](#les-trois-types-de-fraude)
4. [Approches Algorithmiques](#approches-algorithmiques)
5. [Technologies Employées](#technologies-employées)
6. [Implémentation Technique](#implémentation-technique)
7. [Score de Risque](#score-de-risque)
8. [Typage Python et Modularité](#typage-python-et-modularité)
9. [Références Bibliographiques](#références-bibliographiques)

---

## Introduction

La détection de fraude financière est un domaine critique pour les institutions bancaires et les organismes de régulation. Les méthodes traditionnelles basées sur des règles simples sont de plus en plus contournées par des fraudeurs sophistiqués. L'analyse de graphes offre une approche puissante pour identifier des patterns complexes et des structures suspectes dans les réseaux de transactions financières.

### Pourquoi l'Analyse de Graphes ?

Les transactions financières forment naturellement un réseau où :
- Les **nœuds** représentent les comptes bancaires ou les entités
- Les **arêtes** représentent les transferts de fonds
- Les **attributs** incluent les montants, les horodatages, etc.

Cette représentation permet d'appliquer des algorithmes de théorie des graphes pour détecter des patterns qui seraient invisibles avec des méthodes traditionnelles.

### Avantages de l'Approche par Graphes

1. **Détection de patterns structurels** : Identification de cycles, de hubs et de communautés
2. **Analyse contextuelle** : Compréhension des relations entre les entités
3. **Scalabilité** : Algorithmes efficaces pour de grands volumes de données
4. **Interprétabilité** : Résultats explicables et traçables

---

## Architecture Modulaire

Le projet adopte une architecture modulaire avec séparation claire des responsabilités, facilitant la maintenance, l'extension et les tests.

### Structure des Modules

```
src/
├── fraud_detector.py        # Point d'entrée CLI et pipeline principal
├── data/                   # Module de gestion des données
│   ├── loader.py          # Chargement CSV/JSON
│   └── generator.py       # Générateur de données synthétiques
├── graph/                  # Module de construction de graphes
│   └── builder.py         # Transformation en nx.DiGraph
├── detection/              # Module de détection de fraude
│   ├── cycle_detector.py  # Détection de boucles
│   ├── smurfing_detector.py # Analyse des dépôts fractionnés
│   └── network_detector.py # Outliers de centralité
└── visualization/          # Module de visualisation
    └── plotter.py         # Visualisation Matplotlib
```

### Module `data/`

#### TransactionLoader

La classe [`TransactionLoader`](../src/data/loader.py:17) est responsable du chargement des transactions depuis différents formats de fichiers.

**Fonctionnalités principales :**
- Chargement depuis CSV avec support de différents délimiteurs
- Chargement depuis JSON
- Validation des données (types, valeurs cohérentes)
- Parsing flexible des timestamps (ISO 8601, Unix, formats personnalisés)
- Statistiques sur les données chargées

**Méthodes clés :**
- [`load_from_csv()`](../src/data/loader.py:41) : Charge les transactions depuis un fichier CSV
- [`load_from_json()`](../src/data/loader.py:102) : Charge les transactions depuis un fichier JSON
- [`validate()`](../src/data/loader.py:292) : Valide les transactions chargées
- [`get_statistics()`](../src/data/loader.py:353) : Retourne des statistiques sur les données

#### TransactionGenerator

La classe [`TransactionGenerator`](../src/data/generator.py:13) permet de générer des données synthétiques pour tester les algorithmes de détection.

**Fonctionnalités principales :**
- Génération de transactions normales aléatoires
- Injection de cycles de blanchiment
- Injection de patterns de smurfing
- Injection d'anomalies de réseau
- Reproductibilité via graine aléatoire

**Méthodes clés :**
- [`generate_normal_transactions()`](../src/data/generator.py:44) : Génère des transactions normales
- [`inject_money_laundering_cycles()`](../src/data/generator.py:99) : Injecte des cycles de blanchiment
- [`inject_smurfing()`](../src/data/generator.py:156) : Injecte des cas de smurfing
- [`inject_network_anomalies()`](../src/data/generator.py:220) : Injecte des anomalies de réseau
- [`generate_complete_dataset()`](../src/data/generator.py:340) : Génère un jeu de données complet

### Module `graph/`

#### GraphBuilder

La classe [`GraphBuilder`](../src/graph/builder.py:13) transforme les transactions en un graphe dirigé NetworkX.

**Fonctionnalités principales :**
- Construction de graphes dirigés à partir de transactions
- Agrégation des transactions entre mêmes nœuds
- Calcul des statistiques de graphe
- Export aux formats GEXF et GraphML
- Identification des nœuds frauduleux

**Méthodes clés :**
- [`build_from_transactions()`](../src/graph/builder.py:33) : Construit le graphe depuis les transactions
- [`get_graph_statistics()`](../src/graph/builder.py:353) : Retourne les statistiques du graphe
- [`get_fraudulent_nodes()`](../src/graph/builder.py:257) : Retourne les nœuds frauduleux
- [`export_to_gexf()`](../src/graph/builder.py:380) : Exporte le graphe au format GEXF

**Attributs des nœuds :**
- `account_id` : Identifiant du compte
- `total_sent` : Montant total envoyé
- `total_received` : Montant total reçu
- `transaction_count` : Nombre de transactions
- `is_fraudulent` : Indicateur de fraude

**Attributs des arêtes :**
- `total_amount` : Montant total des transactions
- `transaction_count` : Nombre de transactions
- `transactions` : Liste des transactions
- `first_timestamp` : Timestamp de la première transaction
- `last_timestamp` : Timestamp de la dernière transaction

### Module `detection/`

#### BaseDetector

La classe abstraite [`BaseDetector`](../src/detection/cycle_detector.py:16) définit l'interface commune pour tous les détecteurs de fraude.

**Méthodes abstraites :**
- [`detect()`](../src/detection/cycle_detector.py:25) : Détecte les fraudes dans le graphe

**Méthodes communes :**
- [`_calculate_risk_score()`](../src/detection/cycle_detector.py:37) : Calcule un score de risque entre 0 et 1

#### CycleDetector

La classe [`CycleDetector`](../src/detection/cycle_detector.py:87) détecte les cycles de blanchiment dans le graphe.

**Algorithme utilisé :**
- Algorithme de Johnson pour trouver tous les cycles élémentaires
- Filtrage préalable des nœuds ne pouvant pas former de cycles
- Limitation de la longueur des cycles pour éviter les blocages

**Paramètres :**
- `max_cycle_length` : Longueur maximale des cycles (défaut: 5)
- `min_cycle_length` : Longueur minimale des cycles (défaut: 3)
- `max_cycles` : Nombre maximum de cycles à détecter (défaut: 50)

**Méthodes clés :**
- [`detect()`](../src/detection/cycle_detector.py:125) : Détecte les cycles de blanchiment
- [`get_high_risk_cycles()`](../src/detection/cycle_detector.py:287) : Filtre les cycles à haut risque
- [`get_cycles_by_node()`](../src/detection/cycle_detector.py:307) : Retourne les cycles impliquant un nœud

#### SmurfingDetector

La classe [`SmurfingDetector`](../src/detection/smurfing_detector.py:19) détecte les patterns de smurfing (dépôts fractionnés).

**Approche :**
- Analyse des flux entrants vers chaque nœud
- Groupement des transactions par fenêtre temporelle
- Identification des comptes pivots recevant de nombreuses petites transactions

**Paramètres :**
- `threshold` : Seuil de montant pour les dépôts fractionnés (défaut: 1000.0)
- `time_window_hours` : Fenêtre temporelle en heures (défaut: 24)
- `min_deposits` : Nombre minimum de dépôts (défaut: 5)

**Méthodes clés :**
- [`detect()`](../src/detection/smurfing_detector.py:59) : Détecte les cas de smurfing
- [`get_high_risk_cases()`](../src/detection/smurfing_detector.py:278) : Filtre les cas à haut risque
- [`get_sender_network()`](../src/detection/smurfing_detector.py:318) : Analyse le réseau d'émetteurs

#### NetworkDetector

La classe [`NetworkDetector`](../src/detection/network_detector.py:17) détecte les anomalies de réseau basées sur les métriques de centralité.

**Métriques calculées :**
- Centralité de degré (in, out, total)
- Centralité d'intermédiarité (betweenness)
- PageRank

**Paramètres :**
- `pagerank_threshold` : Seuil de PageRank (défaut: auto)
- `betweenness_threshold` : Seuil de betweenness (défaut: auto)
- `percentile_threshold` : Percentile pour les seuils automatiques (défaut: 95)

**Méthodes clés :**
- [`detect()`](../src/detection/network_detector.py:60) : Détecte les anomalies de réseau
- [`get_top_anomalies()`](../src/detection/network_detector.py:323) : Retourne les top N anomalies
- [`get_anomaly_summary()`](../src/detection/network_detector.py:347) : Retourne un résumé des anomalies

### Module `visualization/`

#### GraphPlotter

La classe [`GraphPlotter`](../src/visualization/plotter.py:17) permet de visualiser les graphes avec Matplotlib.

**Fonctionnalités principales :**
- Visualisation de graphes complets
- Mise en évidence des nœuds frauduleux
- Différents algorithmes de layout (spring, circular, kamada_kawai, etc.)
- Visualisation de sous-graphes
- Visualisation d'alertes spécifiques
- Heatmaps de centralité

**Méthodes clés :**
- [`plot_graph()`](../src/visualization/plotter.py:57) : Visualise le graphe complet
- [`plot_subgraph()`](../src/visualization/plotter.py:298) : Visualise un sous-graphe
- [`plot_alert()`](../src/visualization/plotter.py:336) : Visualise une alerte spécifique
- [`plot_centrality_heatmap()`](../src/visualization/plotter.py:426) : Visualise avec heatmap de centralité

### Point d'Entrée

#### FraudDetectionPipeline

La classe [`FraudDetectionPipeline`](../src/fraud_detector.py:30) orchestre l'ensemble du flux de détection.

**Flux de traitement :**
1. Génération/Chargement des données
2. Construction du graphe
3. Détection des cycles
4. Détection du smurfing
5. Détection des anomalies de réseau
6. Visualisation
7. Génération de rapports

**Méthodes clés :**
- [`generate_data()`](../src/fraud_detector.py:54) : Génère des données synthétiques
- [`build_graph()`](../src/fraud_detector.py:101) : Construit le graphe
- [`detect_cycles()`](../src/fraud_detector.py:130) : Détecte les cycles
- [`detect_smurfing()`](../src/fraud_detector.py:157) : Détecte le smurfing
- [`detect_network_anomalies()`](../src/fraud_detector.py:186) : Détecte les anomalies
- [`visualize()`](../src/fraud_detector.py:218) : Visualise le graphe
- [`run_full_pipeline()`](../src/fraud_detector.py:246) : Exécute le pipeline complet

---

## Les Trois Types de Fraude

### 1. Cycles de Blanchiment

#### Définition

Le blanchiment d'argent par cycles consiste à faire circuler des fonds à travers plusieurs comptes intermédiaires avant de les ramener à leur point d'origine. Cette technique vise à masquer l'origine illicite des fonds en créant une chaîne de transactions complexe.

#### Structure Typique

```
Compte A → Compte B → Compte C → Compte D → Compte A
   $50k      $50k      $50k      $50k      $50k
```

#### Caractéristiques

- **Boucle fermée** : Les fonds reviennent à l'expéditeur initial
- **Montants similaires** : Les montants transférés sont souvent proches
- **Délai court** : Les transactions s'effectuent sur une période limitée
- **Intermédiaires multiples** : Utilisation de comptes "mules" pour brouiller les pistes

#### Indicateurs de Risque

1. **Longueur du cycle** : Plus le cycle est long, plus il est suspect
2. **Variation des montants** : Une faible variation indique un pattern artificiel
3. **Délai temporel** : Un cycle rapide suggère une coordination
4. **Répétition** : Cycles multiples impliquant les mêmes comptes

#### Exemple Réel

L'affaire **Danske Bank (2018)** a révélé un schéma de blanchiment massif où des fonds non déclarés circulaient à travers des comptes offshore en Estonie, en Russie et dans d'autres juridictions, formant des cycles complexes pour masquer l'origine des fonds.

---

### 2. Smurfing (Schtroumpfage)

#### Définition

Le smurfing, ou schtroumpfage, est une technique de fractionnement de gros montants en plusieurs petites transactions pour éviter les seuils de déclaration obligatoire. Ces petites transactions sont ensuite acheminées vers un compte pivot qui reconstitue le montant initial.

#### Structure Typique

```
Compte Mule 1 ──┐
Compte Mule 2 ──┤
Compte Mule 3 ──┼──→ Compte Pivot (reçoit ~100k€)
Compte Mule 4 ──┤      via 12 transactions de ~8k€
Compte Mule 5 ──┘
```

#### Caractéristiques

- **Fractionnement** : Montants inférieurs au seuil de déclaration (ex: 10 000€ en Europe)
- **Convergence** : Plusieurs émetteurs vers un même destinataire
- **Similitude des montants** : Les transactions ont des montants proches
- **Fenêtre temporelle** : Transactions rapprochées dans le temps

#### Indicateurs de Risque

1. **Nombre de transactions** : Volume élevé de petites transactions
2. **Proximité du seuil** : Montants juste en dessous du seuil réglementaire
3. **Coefficient de variation** : Faible variation entre les montants
4. **Concentration temporelle** : Transactions groupées sur une courte période

#### Exemple Réel

L'affaire **Bank of New York (1999)** impliquait un schéma de smurfing où des fonds russes étaient fractionnés en milliers de petites transactions pour éviter les déclarations, totalisant plus de 7 milliards de dollars sur plusieurs années.

---

### 3. Anomalies de Réseau

#### Définition

Les anomalies de réseau regroupent divers comportements atypiques dans la structure des transactions financières. Elles incluent les hubs, les rafales de transactions et les communautés isolées.

#### Types d'Anomalies

##### 3.1 Hubs (Comptes Centraux)

Un hub est un compte avec une centralité anormalement élevée, effectuant un nombre disproportionné de transactions avec de nombreux partenaires différents.

**Caractéristiques :**
- Degré de centralité élevé
- Forte intermédiarité
- PageRank élevé

**Indicateurs de risque :**
- Centralité > moyenne + 2 écarts-types
- Nombre de partenaires anormalement élevé
- Volume de transactions disproportionné

##### 3.2 Bursts (Rafales)

Un burst est une activité transactionnelle intense sur une très courte période, suggérant une coordination ou une automatisation.

**Caractéristiques :**
- Grand nombre de transactions en peu de temps
- Densité temporelle élevée
- Souvent vers des destinataires variés

**Indicateurs de risque :**
- > 20 transactions en 2 heures
- Densité > 10 transactions/heure
- Pattern inhabituel pour le compte

##### 3.3 Communautés Isolées

Une communauté isolée est un groupe de comptes qui effectuent principalement des transactions entre eux, formant un sous-graphe dense avec peu de connexions externes.

**Caractéristiques :**
- Forte densité interne
- Faible connectivité externe
- Structure en clique ou quasi-clique

**Indicateurs de risque :**
- Ratio de transactions internes > 70%
- Taille de communauté significative (> 3 comptes)
- Activité cyclique interne

#### Exemple Réel

L'affaire **1MDB (2015)** a révélé un réseau complexe de sociétés écrans et de comptes offshore formant des communautés isolées pour détourner des fonds publics malaisiens, avec des hubs centraux coordonnant les transferts.

---

## Approches Algorithmiques

### Détection de Cycles

#### Algorithme de Johnson

L'algorithme de Johnson est utilisé pour trouver tous les cycles élémentaires dans un graphe orienté. Il est particulièrement adapté aux graphes de taille moyenne.

**Complexité :** O((n + e)(c + 1)) où n est le nombre de nœuds, e le nombre d'arêtes et c le nombre de cycles.

**Avantages :**
- Trouve tous les cycles élémentaires
- Efficace pour les graphes modérément denses
- Implémentation disponible dans NetworkX

**Limitations :**
- Peut être lent sur les très grands graphes
- Nombre exponentiel de cycles possibles

#### Filtrage Temporel

Pour réduire les faux positifs, nous appliquons un filtrage temporel :

```python
# Seuls les cycles complétés dans une fenêtre temporelle sont considérés
if (max_timestamp - min_timestamp) <= time_window_hours:
    # Cycle suspect
```

#### Scoring des Cycles

Un score de risque composite est calculé :

```
Score = 0.4 × (montant / 100k) +
        0.3 × (1 - durée / 72h) +
        0.3 × (répétition / 10)
```

---

### Détection de Smurfing

#### Approche par Fenêtre Glissante

Nous utilisons une fenêtre temporelle glissante pour identifier les concentrations de transactions :

1. Pour chaque compte pivot, collecter les transactions entrantes
2. Trier par timestamp
3. Appliquer une fenêtre glissante de taille T
4. Identifier les fenêtres avec N+ transactions sous le seuil S

#### Détection de Similarité des Montants

Le coefficient de variation (CV) mesure la similarité des montants :

```
CV = écart-type / moyenne
```

Un CV faible (< 0.2) indique des montants similaires, caractéristique du smurfing.

#### Scoring du Smurfing

```
Score = 0.4 × (montant_total / 100k) +
        0.3 × (nb_transactions / 20) +
        0.3 × (1 - durée / 48h)
```

---

### Détection d'anomalies de Réseau

#### Centralité de Degré

La centralité de degré mesure l'importance d'un nœud basée sur le nombre de connexions :

```
C_D(v) = deg(v) / (n - 1)
```

Un nœud avec une centralité > moyenne + 2σ est considéré comme un hub suspect.

#### Centralité d'Intermédiarité

La centralité d'intermédiarité mesure la fréquence à laquelle un nœud se trouve sur les plus courts chemins :

```
C_B(v) = Σ (σ_st(v) / σ_st)
```

Où σ_st est le nombre de plus courts chemins entre s et t, et σ_st(v) le nombre passant par v.

#### PageRank

PageRank mesure l'importance d'un nœud basée sur la qualité de ses connexions :

```
PR(v) = (1 - d) / n + d × Σ (PR(u) / L(u))
```

Où d est le facteur d'amortissement (typiquement 0.85) et L(u) le nombre de liens sortants de u.

#### Détection de Communautés (Louvain)

L'algorithme de Louvain détecte les communautés en maximisant la modularité :

```
Q = (1 / 2m) × Σ [A_ij - (k_i × k_j) / 2m] × δ(c_i, c_j)
```

Où A_ij est la matrice d'adjacence, k_i le degré du nœud i, et δ la fonction de Kronecker.

---

## Technologies Employées

### NetworkX

**Description :** Bibliothèque Python pour la création, la manipulation et l'étude de graphes.

**Fonctionnalités utilisées :**
- Création de graphes orientés (`DiGraph`)
- Algorithmes de détection de cycles (`simple_cycles`)
- Métriques de centralité (`degree_centrality`, `betweenness_centrality`, `pagerank`)
- Détection de communautés (via `python-louvain`)
- Export de graphes (`write_gexf`)

**Avantages :**
- API intuitive et bien documentée
- Large gamme d'algorithmes de graphes
- Intégration facile avec l'écosystème Python

### Pandas

**Description :** Bibliothèque Python pour la manipulation et l'analyse de données.

**Fonctionnalités utilisées :**
- Chargement de données CSV/JSON
- Manipulation de DataFrames
- Agrégations et statistiques

**Avantages :**
- Performance optimisée pour les grands datasets
- Syntaxe expressive pour les opérations sur les données

### NumPy

**Description :** Bibliothèque Python pour le calcul scientifique.

**Fonctionnalités utilisées :**
- Calculs statistiques (moyenne, écart-type)
- Opérations vectorielles
- Gestion des tableaux multidimensionnels

**Avantages :**
- Performance native via C
- Large éventail de fonctions mathématiques

### Matplotlib

**Description :** Bibliothèque Python pour la visualisation de données.

**Fonctionnalités utilisées :**
- Visualisation de graphes
- Heatmaps de centralité
- Export d'images haute résolution

**Avantages :**
- Flexibilité de personnalisation
- Support de nombreux formats de sortie

---

## Implémentation Technique

### Flux de Traitement

```
1. Chargement des données
   ↓
2. Validation des données
   ↓
3. Construction du graphe
   ↓
4. Détection de cycles
   ↓
5. Détection de smurfing
   ↓
6. Détection d'anomalies
   ↓
7. Génération du rapport
   ↓
8. Export des résultats
```

### Génération de Données Synthétiques

Pour les tests, le système peut générer des données synthétiques avec des fraudes intégrées :

```python
generate_complete_dataset(
    num_normal=1000,
    num_cycles=5,
    num_smurfing=3,
    num_anomalies=3
)
```

Les fraudes générées incluent :
- **Cycles** : Boucles de 4-6 comptes avec des montants similaires
- **Smurfing** : 5-10 transactions de < 10k€ vers un compte pivot
- **Anomalies** : Hubs, bursts ou communautés isolées

### Paramétrage

Les paramètres de détection sont configurables :

```python
# Cycles
detector = CycleDetector(max_cycle_length=5, min_cycle_length=3)

# Smurfing
detector = SmurfingDetector(threshold=1000.0, time_window_hours=24)

# Anomalies
detector = NetworkDetector(pagerank_threshold=0.01, betweenness_threshold=0.05)
```

### Performance

**Complexité temporelle :**
- Construction du graphe : O(n) où n est le nombre de transactions
- Détection de cycles : O((n + e)(c + 1))
- Détection de smurfing : O(n × m) où m est le nombre de comptes
- Détection d'anomalies : O(n × log n) pour les métriques de centralité

**Optimisations :**
- Filtrage préalable des transactions
- Utilisation d'index pour les recherches
- Parallélisation possible pour les grands datasets

---

## Score de Risque

Le système calcule un score de risque entre 0 et 1 pour chaque alerte détectée. Ce score est calculé par la méthode [`_calculate_risk_score()`](../src/detection/cycle_detector.py:37) de la classe [`BaseDetector`](../src/detection/cycle_detector.py:16).

### Formule de Calcul

```
Score = 0.4 × Score_Montant +
        0.3 × Score_Durée +
        0.3 × Score_Répétition
```

#### Score de Montant

Basé sur une échelle logarithmique pour éviter que les très gros montants dominent :

```
Score_Montant = min(1.0, montant / 100000.0)
```

#### Score de Durée

Plus la durée est courte, plus le score est élevé :

```
Score_Durée = 1.0    si durée ≤ 1h
Score_Durée = 0.8    si durée ≤ 24h
Score_Durée = 0.5    si durée ≤ 168h (1 semaine)
Score_Durée = 0.2    sinon
```

#### Score de Répétition

Basé sur le nombre de transactions ou la longueur du cycle :

```
Score_Répétition = min(1.0, répétition / 10.0)
```

### Niveaux de Risque

| Score | Niveau | Interprétation |
|-------|--------|----------------|
| 0.0 - 0.3 | LOW | Faible risque - Surveillance normale |
| 0.3 - 0.5 | MEDIUM | Risque modéré - Vérification recommandée |
| 0.5 - 0.7 | HIGH | Risque élevé - Investigation requise |
| 0.7 - 1.0 | CRITICAL | Risque très élevé - Action immédiate |

### Ajustement pour les Anomalies de Réseau

Pour les anomalies de réseau, le score est ajusté en fonction des métriques de centralité :

```
Score_Final = 0.7 × Score_Base + 0.3 × Score_Centralité

Score_Centralité = 0.5 × (PageRank / Seuil_PageRank) +
                   0.5 × (Betweenness / Seuil_Betweenness)
```

---

## Typage Python et Modularité

Le projet utilise le typage Python (type hints) pour améliorer la maintenabilité, la lisibilité et la robustesse du code.

### Avantages du Typage

1. **Documentation intégrée** : Les types servent de documentation
2. **Détection précoce des erreurs** : Les outils comme mypy peuvent détecter les erreurs de type
3. **Meilleure autocomplétion** : Les IDE peuvent fournir des suggestions plus précises
4. **Refactoring plus sûr** : Les modifications sont plus faciles à valider

### Exemples de Typage

#### Définition de Classe

```python
class TransactionLoader:
    def __init__(self) -> None:
        self.transactions: List[Dict[str, Any]] = []
        self.errors: List[str] = []
```

#### Méthodes avec Types

```python
def load_from_csv(
    self,
    filepath: str,
    encoding: str = "utf-8",
    delimiter: str = ","
) -> List[Dict[str, Any]]:
    ...
```

#### Types Complexes

```python
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

### Imports Différés

Pour éviter les dépendances circulaires, le pipeline utilise des imports différés :

```python
def detect_cycles(self, builder: Any, ...) -> List[Dict[str, Any]]:
    from detection.cycle_detector import CycleDetector
    detector = CycleDetector(max_cycle_length=max_cycle_length)
    ...
```

### Classes Abstraites

L'utilisation de classes abstraites garantit que tous les détecteurs implémentent la même interface :

```python
class BaseDetector(ABC):
    @abstractmethod
    def detect(self, graph: nx.DiGraph) -> List[Dict[str, Any]]:
        pass
```

---

## Références Bibliographiques

### Ouvrages

1. **Newman, M. E. J.** (2018). *Networks* (2nd ed.). Oxford University Press.
   - Référence fondamentale sur la théorie des graphes et l'analyse de réseaux.

2. **Easley, D., & Kleinberg, J.** (2010). *Networks, Crowds, and Markets: Reasoning About a Highly Connected World*. Cambridge University Press.
   - Analyse des réseaux sociaux et économiques.

3. **Chandola, V., Banerjee, A., & Kumar, V.** (2009). "Anomaly Detection: A Survey". *ACM Computing Surveys*, 41(3), 1-58.
   - État de l'art sur la détection d'anomalies.

### Articles Scientifiques

4. **Liu, X., et al.** (2018). "Detecting Money Laundering in Financial Networks". *IEEE International Conference on Data Mining*.
   - Approches de détection de blanchiment par analyse de graphes.

5. **Savage, D., et al.** (2014). "Detecting Financial Fraud Using Graph Mining". *International Journal of Data Mining and Knowledge Management*.
   - Techniques de mining de graphes pour la fraude financière.

6. **Van Vlasselaer, V., et al.** (2015). "APATE: A Novel Approach for Automated Credit Card Transaction Fraud Detection Using Network-Based Extensions". *Decision Support Systems*.
   - Détection de fraude par analyse de réseaux de transactions.

7. **Blondel, V. D., et al.** (2008). "Fast Unfolding of Communities in Large Networks". *Journal of Statistical Mechanics: Theory and Experiment*.
   - Algorithme de Louvain pour la détection de communautés.

8. **Johnson, D. B.** (1975). "Finding All the Elementary Circuits of a Directed Graph". *SIAM Journal on Computing*, 4(1), 77-84.
   - Algorithme de Johnson pour la détection de cycles.

### Rapports et Études de Cas

9. **Financial Action Task Force (FATF)** (2023). *Money Laundering Using Trade-Based Cryptocurrencies*.
   - Étude sur les nouvelles techniques de blanchiment.

10. **United Nations Office on Drugs and Crime (UNODC)** (2022). *The Role of Cryptocurrencies in Money Laundering*.
    - Analyse des crypto-monnaies dans le blanchiment d'argent.

11. **European Banking Authority (EBA)** (2021). *Guidelines on AML/CFT*.
    - Lignes directrices européennes sur la lutte contre le blanchiment.

12. **US Financial Crimes Enforcement Network (FinCEN)** (2020). *Advisory on Illicit Activity Involving Convertible Virtual Currency*.
    - Avis sur les activités illicites impliquant les crypto-monnaies.

### Études de Cas Réelles

13. **Danske Bank Money Laundering Scandal** (2018).
    - Blanchiment de 200 milliards d'euros via des comptes estoniens.

14. **1MDB Scandal** (2015).
    - Détournement de 4,5 milliards de dollars via un réseau complexe.

15. **Bank of New York Scandal** (1999).
    - Smurfing de 7 milliards de dollars de fonds russes.

16. **Panama Papers** (2016).
    - Révélation de 214 000 sociétés offshore dans des paradis fiscaux.

### Ressources Techniques

17. **NetworkX Documentation**. https://networkx.org/documentation/stable/
    - Documentation officielle de NetworkX.

18. **Pandas Documentation**. https://pandas.pydata.org/docs/
    - Documentation officielle de Pandas.

19. **Python-Louvain**. https://python-louvain.readthedocs.io/
    - Documentation de l'algorithme de Louvain.

20. **Gephi**. https://gephi.org/
    - Logiciel de visualisation et d'analyse de graphes.

---

## Conclusion

Ce projet a permis de développer un système complet de détection de fraude financière basé sur l'analyse de graphes. L'approche combinant des algorithmes de théorie des graphes avec des techniques de scoring permet d'identifier efficacement trois types majeurs de fraudes : les cycles de blanchiment, le smurfing et les anomalies de réseau.

L'architecture modulaire adoptée facilite la maintenance et l'extension du système, tandis que l'utilisation du typage Python améliore la robustesse et la lisibilité du code.

Les perspectives d'amélioration incluent :
- Intégration de l'apprentissage automatique pour le scoring
- Analyse temporelle avancée avec des séries temporelles
- Détection de patterns plus complexes (ex: structures en étoile)
- Intégration de données externes (KYC, sanctions, etc.)

---

**Document rédigé par :** Malak El Idrissi et Joe Boueri  
**Groupe :** 42  
**Date :** Janvier 2026  
**Institution :** ECE - Ingénierie Financière et IA
