#!/usr/bin/env python3
"""
Script pour générer plusieurs visualisations pour les différents types de fraude.

Ce script crée des visualisations séparées pour :
1. Le graphe complet avec toutes les fraudes
2. Les cycles de blanchiment uniquement
3. Les cas de smurfing uniquement
4. Les anomalies de réseau uniquement

Projet académique ECE - Groupe 42 : Malak El Idrissi et Joe Boueri.
"""

import os
import sys
from typing import List, Dict, Any

# Ajouter le répertoire src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.generator import TransactionGenerator
from graph.builder import GraphBuilder
from detection.cycle_detector import CycleDetector
from detection.smurfing_detector import SmurfingDetector
from detection.network_detector import NetworkDetector
from visualization.plotter import GraphPlotter


def generate_all_visualizations(
    num_accounts: int = 30,
    num_normal: int = 200,
    num_cycles: int = 2,
    num_smurfing: int = 2,
    num_anomalies: int = 2,
    output_dir: str = "output"
) -> None:
    """
    Génère toutes les visualisations pour les différents types de fraude.
    
    Args:
        num_accounts: Nombre de comptes bancaires.
        num_normal: Nombre de transactions normales.
        num_cycles: Nombre de cycles de blanchiment.
        num_smurfing: Nombre de cas de smurfing.
        num_anomalies: Nombre d'anomalies de réseau.
        output_dir: Répertoire de sortie pour les visualisations.
    """
    print("=" * 60)
    print("GÉNÉRATION DES VISUALISATIONS")
    print("=" * 60)
    
    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Générer les données
    print("\n1. Génération des données...")
    generator = TransactionGenerator(num_accounts=num_accounts)
    transactions = generator.generate_complete_dataset(
        num_normal=num_normal,
        num_cycles=num_cycles,
        num_smurfing=num_smurfing,
        num_anomalies=num_anomalies
    )
    print(f"   Transactions générées : {len(transactions)}")
    
    # 2. Construire le graphe
    print("\n2. Construction du graphe...")
    builder = GraphBuilder()
    graph = builder.build_from_transactions(transactions)
    stats = builder.get_graph_statistics()
    print(f"   Nœuds : {stats['num_nodes']}")
    print(f"   Arêtes : {stats['num_edges']}")
    print(f"   Nœuds frauduleux : {stats['num_fraudulent_nodes']}")
    
    # 3. Détecter les fraudes
    print("\n3. Détection des fraudes...")
    
    # Cycles de blanchiment
    cycle_detector = CycleDetector(max_cycle_length=5)
    cycles = cycle_detector.detect(graph)
    print(f"   Cycles détectés : {len(cycles)}")
    
    # Smurfing
    smurfing_detector = SmurfingDetector(threshold=1000.0, time_window_hours=24)
    smurfing_cases = smurfing_detector.detect(graph)
    print(f"   Cas de smurfing détectés : {len(smurfing_cases)}")
    
    # Anomalies de réseau
    network_detector = NetworkDetector(pagerank_threshold=0.01, betweenness_threshold=0.05)
    anomalies = network_detector.detect(graph)
    print(f"   Anomalies de réseau détectées : {len(anomalies)}")
    
    # 4. Générer les visualisations
    print("\n4. Génération des visualisations...")
    plotter = GraphPlotter()
    
    # 4.1 Graphe complet
    print("   - Graphe complet...")
    plotter.plot_graph(
        graph,
        output_file=os.path.join(output_dir, "01_complete_graph.png"),
        highlight_fraud=True,
        title="Graphe complet - Toutes les fraudes"
    )
    
    # 4.2 Cycles de blanchiment
    print("   - Cycles de blanchiment...")
    if cycles:
        # Extraire les nœuds impliqués dans les cycles
        cycle_nodes = set()
        for cycle in cycles:
            cycle_nodes.update(cycle['nodes'])
        
        # Créer un sous-graphe avec les cycles
        cycle_subgraph = graph.subgraph(cycle_nodes)
        plotter.plot_graph(
            cycle_subgraph,
            output_file=os.path.join(output_dir, "02_cycles_graph.png"),
            highlight_fraud=True,
            title=f"Cycles de blanchiment ({len(cycles)} cycles)"
        )
    
    # 4.3 Smurfing
    print("   - Smurfing...")
    if smurfing_cases:
        # Extraire les nœuds impliqués dans le smurfing
        smurfing_nodes = set()
        for case in smurfing_cases:
            smurfing_nodes.add(case['pivot_account'])
            smurfing_nodes.update(case['senders'])
        
        # Créer un sous-graphe avec le smurfing
        smurfing_subgraph = graph.subgraph(smurfing_nodes)
        plotter.plot_graph(
            smurfing_subgraph,
            output_file=os.path.join(output_dir, "03_smurfing_graph.png"),
            highlight_fraud=True,
            title=f"Smurfing ({len(smurfing_cases)} cas)"
        )
    
    # 4.4 Anomalies de réseau
    print("   - Anomalies de réseau...")
    if anomalies:
        # Extraire les nœuds anormaux
        anomaly_nodes = set()
        for anomaly in anomalies:
            anomaly_nodes.add(anomaly['node'])
        
        # Créer un sous-graphe avec les anomalies
        anomaly_subgraph = graph.subgraph(anomaly_nodes)
        plotter.plot_graph(
            anomaly_subgraph,
            output_file=os.path.join(output_dir, "04_anomalies_graph.png"),
            highlight_fraud=True,
            title=f"Anomalies de réseau ({len(anomalies)} anomalies)"
        )
    
    # 4.5 Heatmap de centralité
    print("   - Heatmap de centralité...")
    plotter.plot_centrality_heatmap(
        graph,
        output_file=os.path.join(output_dir, "05_centrality_heatmap.png"),
        title="Heatmap de centralité PageRank"
    )
    
    print("\n" + "=" * 60)
    print("VISUALISATIONS GÉNÉRÉES AVEC SUCCÈS")
    print("=" * 60)
    print(f"\nFichiers générés dans le répertoire '{output_dir}':")
    print("  - 01_complete_graph.png : Graphe complet avec toutes les fraudes")
    print("  - 02_cycles_graph.png : Cycles de blanchiment")
    print("  - 03_smurfing_graph.png : Cas de smurfing")
    print("  - 04_anomalies_graph.png : Anomalies de réseau")
    print("  - 05_centrality_heatmap.png : Heatmap de centralité")
    print(f"\nTotal des alertes : {len(cycles) + len(smurfing_cases) + len(anomalies)}")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_visualizations(
        num_accounts=30,
        num_normal=200,
        num_cycles=2,
        num_smurfing=2,
        num_anomalies=2,
        output_dir="output"
    )
