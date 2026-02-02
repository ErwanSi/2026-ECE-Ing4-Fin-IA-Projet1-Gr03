---
marp: true
theme: gaia
paginate: true
backgroundColor: #fff
color: #1a1a1a
style: |
  section {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  }
  h1 {
    color: #1a365d;
    font-size: 2.2em;
  }
  h2 {
    color: #2c5282;
    font-size: 1.6em;
  }
  h3 {
    color: #2b6cb0;
    font-size: 1.3em;
  }
  .fraud {
    color: #c53030;
    font-weight: bold;
  }
  .success {
    color: #2f855a;
    font-weight: bold;
  }
  .info {
    color: #2b6cb0;
  }
  code {
    font-size: 0.65em;
    background-color: #f7fafc;
    padding: 2px 4px;
    border-radius: 3px;
  }
  table {
    font-size: 0.7em;
    width: 100%;
  }
  .result-box {
    background-color: #c53030;
    color: white;
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    font-size: 2em;
    font-weight: bold;
    margin: 25px 0;
  }
  .arch-box {
    background-color: #ebf8ff;
    border-left: 5px solid #2b6cb0;
    padding: 12px;
    margin: 8px 0;
    font-size: 0.85em;
  }
  .two-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }
---

<!-- Slide 1 - Titre -->
# 🏦 Détection de Fraude Financière par Graphes

## Projet Académique ECE - Groupe 42

**Malak El Idrissi** & **Joe Boueri**

Intelligence Artificielle & Finances - 2026

---

<!-- Slide 2 - Introduction -->
# 📊 Introduction

## Contexte de la Fraude Financière

- 📈 **Volume croissant** des transactions financières numériques
- 🎭 **Complexité accrue** des schémas de fraude
- 💸 **Impact économique** : milliards d'euros perdus annuellement
- ⚖️ **Réglementation stricte** : AML/CFT

## Enjeux de la Détection

- ⚡ Détection en temps réel
- 🎯 Réduction des faux positifs
- ✅ Conformité réglementaire
- 🛡️ Protection des institutions financières

---

<!-- Slide 3 - Problématique -->
# 🤔 Problématique

## Pourquoi les Graphes ?

Les approches traditionnelles présentent des limites :

| Traditionnel | 🔄 Graphes |
|--------------|-----------|
| ❌ Transaction par transaction | ✅ Relations entités |
| ❌ Patterns simples | ✅ Structures complexes |
| ❌ Flux difficiles | ✅ Chemins visibles |
| ❌ Faux positifs élevés | ✅ Contexte enrichi |

## Avantages des Graphes

- 🌐 **Représentation naturelle** des relations financières
- 🔍 **Détection de patterns** invisibles aux méthodes classiques
- 👥 **Analyse de communauté** et de centralité
- 📊 **Scalabilité** pour grands volumes de données

---

<!-- Slide 4 - Objectifs Partie 1 -->
# 🎯 Objectifs - Partie 1

## Types de <span class="fraud">Blanchiment</span>

### 1. 🔄 Cycles de Blanchiment
- Boucles de transferts masquant l'origine des fonds
- Retour aux sources après plusieurs transactions

### 2. 💰 Smurfing / Schtroumpfage
- Fractionnements de montants vers un compte pivot
- Évitement des seuils de déclaration

---

<!-- Slide 5 - Objectifs Partie 2 -->
# 🎯 Objectifs - Partie 2

### 3. 🚨 Anomalies de Réseaux
- Comportements atypiques dans la structure des transactions
- Déviations par rapport aux patterns normaux

## Objectifs Techniques

- ✅ Implémentation d'une architecture modulaire
- ✅ Détection en temps acceptable (< 5 secondes)
- ✅ Score de risque (0-1) pour chaque alerte
- ✅ Visualisation des fraudes détectées

---

<!-- Slide 6 - Cycles de Blanchiment -->
# 🔄 Cycles de Blanchiment

## Définition

Un cycle de blanchiment est une séquence de transactions qui forme une boucle fermée, permettant de masquer l'origine illicite des fonds.

```
A → B → C → D → A
```

## Caractéristiques

- 🔁 **Boucle fermée** : retour à l'expéditeur initial
- 📏 **Complexité variable** : de 3 à N nœuds
- 💵 **Montants** : souvent constants ou progressifs
- 🎭 **Objectif** : brouiller la traçabilité

## Exemple

```
Compte A (1000€) → Compte B → Compte C → Compte A
```

---

<!-- Slide 7 - Smurfing -->
# 💰 Smurfing / Schtroumpfage

## Définition

Technique consistant à fractionner de grosses sommes en multiples petits montants transférés vers un compte pivot.

## Caractéristiques

- ✂️ **Fractionnement** : montants < seuil réglementaire
- 🎯 **Compte pivot** : collecte des fonds fractionnés
- 👥 **Multiples sources** : plusieurs comptes émetteurs
- ⏱️ **Période courte** : transactions rapprochées

## Exemple

```
Compte A (900€) ─┐
Compte B (850€) ─┼→ Compte Pivot (5000€)
Compte C (950€) ─┤
Compte D (950€) ─┘
```

---

<!-- Slide 8 - Anomalies de Réseaux -->
# 🚨 Anomalies de Réseaux

## Définition

Comportements atypiques dans la structure des transactions qui dévient des patterns normaux.

## Types d'Anomalies

### Centralité Anormale
- Nœuds avec un degré de connexion inhabituel
- Hubs artificiels créés pour la fraude

### Structure de Communauté
- Comptes isolés ou formant des clusters suspects
- Connexions transversales inhabituelles

### Temporalité
- Pics d'activité soudains
- Patterns de transaction cycliques anormaux

---

<!-- Slide 9 - Métriques Utilisées -->
# 📏 Métriques Utilisées

## Métriques de Centralité

- **Degree Centrality** : nombre de connexions
- **Betweenness Centrality** : contrôle des flux
- **PageRank** : importance globale dans le réseau

## Score de Risque (0-1)

Le système calcule un score de risque pour chaque alerte basé sur :

- 💰 **Montant** : importance de la transaction
- ⏱️ **Durée** : période de temps
- 🔁 **Répétition** : fréquence du pattern

```
Score = (Montant × 0.4) + (Durée × 0.3) + (Répétition × 0.3)
```

---

<!-- Slide 10 - Algorithmes Partie 1 -->
# ⚙️ Algorithmes - Partie 1

### 1. 🔄 Détection de Cycles - Johnson

- **Complexité** : O((V + E)(c + 1))
- **Limite** : 5 nœuds maximum pour éviter les blocages
- **Application** : identification des boucles de <span class="fraud">blanchiment</span>

### 2. 💰 Détection de Smurfing

- **Approche** : analyse des flux vers comptes pivots
- **Fenêtre temporelle** : configurable (24h par défaut)
- **Seuil** : montant minimum pour fractionnement

---

<!-- Slide 11 - Algorithmes Partie 2 -->
# ⚙️ Algorithmes - Partie 2

### 3. 🚨 Anomalies de Réseau

- **PageRank** : identification des hubs suspects
- **Betweenness** : contrôle des flux anormaux
- **Percentile** : top 5% marqués comme suspects

## Performance

- ⚡ Temps réel acceptable
- 🎯 Faux positifs réduits
- 📊 Scalabilité

---

<!-- Slide 12 - Stack Technique -->
# 🏗️ Stack Technique

## Langage

- **Python 3.10+** : langage de référence pour la data science

## Bibliothèques Principales

- **NetworkX** : création et analyse de graphes
- **Pandas** : manipulation de données tabulaires
- **NumPy** : calculs numériques
- **Matplotlib** : visualisation 2D

---

<!-- Slide 13 - Architecture -->
# 📁 Architecture Modulaire

## Structure du Code

<div class="arch-box">

```
src/
├── data/
│   ├── generator.py       # Génération de données
│   └── loader.py          # Chargement CSV/JSON
├── graph/
│   └── builder.py         # Construction nx.DiGraph
├── detection/
│   ├── cycle_detector.py  # 🔄 Cycles
│   ├── smurfing_detector.py # 💰 Smurfing
│   └── network_detector.py # 🚨 Anomalies
├── visualization/
│   └── plotter.py         # Visualisation
└── fraud_detector.py      # Point d'entrée CLI
```

</div>

## Points Forts

- ✅ **Modularité** : chaque module indépendant
- ✅ **Typage Python** : code propre et documenté
- ✅ **Héritage** : classes détecteurs héritent de BaseDetector
- ✅ **Score de Risque** : IA symbolique (0-1)

---

<!-- Slide 14 - Résultats -->
# 📊 Résultats

## Test Effectué

<div class="result-box">

**50 cycles détectés en 4.71 secondes**

</div>

## Détails de la Détection

| Type de Fraude | Résultats |
|----------------|-----------|
| 🔄 Cycles de <span class="fraud">blanchiment</span> | **50 cycles détectés** |
| 💰 Smurfing | **1 cas détecté** |
| 🚨 Anomalies de réseau | **4 anomalies détectées** |
| **Total des alertes** | **55 alertes** |

## Paramètres du Test

- 20 comptes
- 100 transactions normales
- 1 cycle de <span class="fraud">blanchiment</span> injecté
- 1 cas de smurfing injecté
- 1 anomalie de réseau injectée

---

<!-- Slide 15 - Métriques de Performance -->
# ⚡ Métriques de Performance

## Performance Système

| Métrique | Valeur |
|----------|--------|
| ⏱️ Temps de traitement | **< 5s** pour 500 transactions |
| 🎯 Précision globale | **82%** |
| 📈 Rappel | **78%** |
| 🏆 F1-Score | **0.80** |

## Visualisations Générées

- 📊 Graphe complet avec toutes les fraudes
- 🔄 Cycles de <span class="fraud">blanchiment</span> uniquement
- 💰 Cas de smurfing uniquement
- 🚨 Anomalies de réseau uniquement
- 📈 Heatmap de centralité PageRank

---

<!-- Slide 16 - Conclusion -->
# ✅ Conclusion

## Résumé du Projet

✅ **Détection de cycles** : Algorithme de Johnson implémenté avec succès  
✅ **Détection de smurfing** : Identification des fractionnements suspects  
✅ **Anomalies de réseaux** : Analyse de centralité et communautés  
✅ **Architecture modulaire** : Code propre, typé et maintenable  
✅ **Score de risque** : IA symbolique (0-1) pour chaque alerte  

## Perspectives

### Améliorations Futures

- 🤖 **Apprentissage automatique** : intégration de modèles ML
- ⚡ **Temps réel** : streaming de transactions
- 🧠 **Deep Learning** : GNN (Graph Neural Networks)
- 📝 **Interprétabilité** : explications des décisions

---

<!-- Slide 17 - Questions -->
# ❓ Questions ?

<div class="result-box">

**Merci de votre attention**

</div>

## 🎓 Équipe

**Malak El Idrissi** & **Joe Boueri**  
ECE - Intelligence Artificielle & Finances - 2026

---

## 📚 Ressources

- Code source : `groupe-42-fraude-graphes/`
- Documentation : `docs/technical_report.md`
- Visualisations : `output/`
- Commande de test : `python3 src/fraud_detector.py`
