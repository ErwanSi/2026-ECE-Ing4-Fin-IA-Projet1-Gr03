#!/usr/bin/env python3
"""
Point d'entrée principal pour la détection de fraude financière par graphes.

Ce module fournit une interface en ligne de commande pour orchestrer
l'ensemble du flux de détection de fraude : génération de données,
construction de graphe, détection et visualisation.

Projet académique ECE - Groupe 42 : Malak El Idrissi et Joe Boueri.
"""

import argparse
import logging
import sys
from typing import Optional, List, Dict, Any
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("fraud_detection.log")
    ]
)
logger = logging.getLogger(__name__)


class FraudDetectionPipeline:
    """
    Pipeline principal de détection de fraude financière.
    
    Cette classe orchestre l'ensemble du flux de détection :
    1. Génération/Chargement des données
    2. Construction du graphe
    3. Détection des fraudes
    4. Génération de rapports
    """
    
    def __init__(self, verbose: bool = False) -> None:
        """
        Initialise le pipeline de détection.
        
        Args:
            verbose: Active le mode verbeux pour plus de logs.
        """
        self.verbose = verbose
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        logger.info("Initialisation du pipeline de détection de fraude")
    
    def load_data(
        self,
        input_file: str
    ) -> List[Dict[str, Any]]:
        """
        Charge des données depuis un fichier CSV existant.
        
        Args:
            input_file: Fichier CSV à charger.
        
        Returns:
            Liste des transactions chargées.
        """
        logger.info(f"Chargement des données depuis {input_file}...")
        
        from src.data.loader import TransactionLoader
        
        loader = TransactionLoader()
        transactions = loader.load_from_csv(input_file)
        
        logger.info(f"Données chargées : {len(transactions)} transactions")
        
        return transactions
    
    def generate_data(
        self,
        num_accounts: int = 100,
        num_normal: int = 1000,
        num_cycles: int = 5,
        num_smurfing: int = 3,
        num_anomalies: int = 3,
        seed: Optional[int] = None,
        output_file: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Génère des données synthétiques pour la détection.
        
        Args:
            num_accounts: Nombre de comptes bancaires.
            num_normal: Nombre de transactions normales.
            num_cycles: Nombre de cycles de blanchiment.
            num_smurfing: Nombre de cas de smurfing.
            num_anomalies: Nombre d'anomalies de réseau.
            seed: Graine aléatoire pour la reproductibilité.
            output_file: Fichier de sortie pour les données générées.
        
        Returns:
            Liste des transactions générées.
        """
        logger.info("Génération des données synthétiques...")
        
        from src.data.generator import TransactionGenerator
        
        generator = TransactionGenerator(num_accounts=num_accounts, seed=seed)
        transactions = generator.generate_complete_dataset(
            num_normal=num_normal,
            num_cycles=num_cycles,
            num_smurfing=num_smurfing,
            num_anomalies=num_anomalies
        )
        
        logger.info(f"Données générées : {len(transactions)} transactions")
        logger.info(f"  - Transactions normales : {len(generator.get_normal_transactions())}")
        logger.info(f"  - Transactions frauduleuses : {len(generator.get_fraudulent_transactions())}")
        
        if output_file:
            generator.save_to_csv(output_file)
            logger.info(f"Données sauvegardées dans {output_file}")
        
        return transactions
    
    def build_graph(
        self,
        transactions: List[Dict[str, Any]]
    ) -> Any:
        """
        Construit le graphe à partir des transactions.
        
        Args:
            transactions: Liste des transactions.
        
        Returns:
            Le graphe construit.
        """
        logger.info("Construction du graphe de transactions...")
        
        from src.graph.builder import GraphBuilder
        
        builder = GraphBuilder()
        graph = builder.build_from_transactions(transactions)
        
        stats = builder.get_graph_statistics()
        logger.info(f"Graphe construit :")
        logger.info(f"  - Nœuds (comptes) : {stats['num_nodes']}")
        logger.info(f"  - Arêtes (transactions) : {stats['num_edges']}")
        logger.info(f"  - Nœuds frauduleux : {stats['num_fraudulent_nodes']}")
        logger.info(f"  - Densité : {stats['density']:.4f}")
        
        return builder
    
    def detect_cycles(
        self,
        builder: Any,
        max_cycle_length: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Détecte les cycles de blanchiment dans le graphe.
        
        Args:
            builder: Le constructeur de graphe.
            max_cycle_length: Longueur maximale des cycles à détecter.
        
        Returns:
            Liste des cycles détectés avec leurs scores de risque.
        """
        logger.info(f"Détection des cycles de blanchiment (max_length={max_cycle_length})...")
        
        # Import différé pour éviter les dépendances circulaires
        from src.detection.cycle_detector import CycleDetector
        
        detector = CycleDetector(max_cycle_length=max_cycle_length)
        cycles = detector.detect(builder.get_graph())
        
        logger.info(f"Cycles détectés : {len(cycles)}")
        
        return cycles
    
    def detect_smurfing(
        self,
        builder: Any,
        threshold: float = 1000.0,
        time_window_hours: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Détecte les cas de smurfing dans le graphe.
        
        Args:
            builder: Le constructeur de graphe.
            threshold: Seuil de montant pour les dépôts fractionnés.
            time_window_hours: Fenêtre temporelle en heures.
        
        Returns:
            Liste des cas de smurfing détectés avec leurs scores de risque.
        """
        logger.info(f"Détection du smurfing (threshold={threshold}, window={time_window_hours}h)...")
        
        # Import différé pour éviter les dépendances circulaires
        from src.detection.smurfing_detector import SmurfingDetector
        
        detector = SmurfingDetector(threshold=threshold, time_window_hours=time_window_hours)
        smurfing_cases = detector.detect(builder.get_graph())
        
        logger.info(f"Cas de smurfing détectés : {len(smurfing_cases)}")
        
        return smurfing_cases
    
    def detect_network_anomalies(
        self,
        builder: Any,
        pagerank_threshold: float = 0.01,
        betweenness_threshold: float = 0.05
    ) -> List[Dict[str, Any]]:
        """
        Détecte les anomalies de réseau dans le graphe.
        
        Args:
            builder: Le constructeur de graphe.
            pagerank_threshold: Seuil de PageRank.
            betweenness_threshold: Seuil de betweenness centrality.
        
        Returns:
            Liste des anomalies détectées avec leurs scores de risque.
        """
        logger.info(f"Détection des anomalies de réseau (pagerank={pagerank_threshold}, betweenness={betweenness_threshold})...")
        
        # Import différé pour éviter les dépendances circulaires
        from src.detection.network_detector import NetworkDetector
        
        detector = NetworkDetector(
            pagerank_threshold=pagerank_threshold,
            betweenness_threshold=betweenness_threshold
        )
        anomalies = detector.detect(builder.get_graph())
        
        logger.info(f"Anomalies de réseau détectées : {len(anomalies)}")
        
        return anomalies
    
    def visualize(
        self,
        builder: Any,
        output_file: str = "fraud_graph.png",
        highlight_fraud: bool = True
    ) -> None:
        """
        Visualise le graphe avec les fraudes mises en évidence.
        
        Args:
            builder: Le constructeur de graphe.
            output_file: Fichier de sortie pour la visualisation.
            highlight_fraud: Met en évidence les nœuds frauduleux.
        """
        logger.info(f"Génération de la visualisation dans {output_file}...")
        
        # Import différé pour éviter les dépendances circulaires
        from src.visualization.plotter import GraphPlotter
        
        plotter = GraphPlotter()
        plotter.plot_graph(
            builder.get_graph(),
            output_file=output_file,
            highlight_fraud=highlight_fraud
        )
        
        logger.info(f"Visualisation sauvegardée dans {output_file}")
    
    def run_full_pipeline(
        self,
        input_file: Optional[str] = None,
        num_accounts: int = 100,
        num_normal: int = 1000,
        num_cycles: int = 5,
        num_smurfing: int = 3,
        num_anomalies: int = 3,
        seed: Optional[int] = None,
        data_output: Optional[str] = None,
        graph_output: Optional[str] = None,
        visualization_output: str = "fraud_graph.png",
        max_cycle_length: int = 5,
        smurfing_threshold: float = 1000.0,
        pagerank_threshold: float = 0.01,
        betweenness_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Exécute le pipeline complet de détection de fraude.
        
        Args:
            num_accounts: Nombre de comptes bancaires.
            num_normal: Nombre de transactions normales.
            num_cycles: Nombre de cycles de blanchiment.
            num_smurfing: Nombre de cas de smurfing.
            num_anomalies: Nombre d'anomalies de réseau.
            seed: Graine aléatoire pour la reproductibilité.
            data_output: Fichier de sortie pour les données générées.
            graph_output: Fichier de sortie pour le graphe.
            visualization_output: Fichier de sortie pour la visualisation.
            max_cycle_length: Longueur maximale des cycles à détecter.
            smurfing_threshold: Seuil de montant pour le smurfing.
            pagerank_threshold: Seuil de PageRank.
            betweenness_threshold: Seuil de betweenness centrality.
        
        Returns:
            Dictionnaire contenant les résultats de la détection.
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("DÉBUT DU PIPELINE DE DÉTECTION DE FRAUDE")
        logger.info("=" * 60)
        
        results = {
            "start_time": start_time.isoformat(),
            "transactions": None,
            "graph_stats": None,
            "cycles": [],
            "smurfing": [],
            "anomalies": []
        }
        
        # Étape 1 : Chargement ou Génération des données
        try:
            if input_file:
                # Charger des données existantes
                transactions = self.load_data(input_file=input_file)
            else:
                # Générer des données synthétiques
                transactions = self.generate_data(
                    num_accounts=num_accounts,
                    num_normal=num_normal,
                    num_cycles=num_cycles,
                    num_smurfing=num_smurfing,
                    num_anomalies=num_anomalies,
                    seed=seed,
                    output_file=data_output
                )
            results["transactions"] = transactions
        except Exception as e:
            logger.error(f"Erreur lors du chargement/génération des données : {e}")
            return results
        
        # Étape 2 : Construction du graphe
        try:
            builder = self.build_graph(transactions)
            results["graph_stats"] = builder.get_graph_statistics()
            
            if graph_output:
                builder.export_to_gexf(graph_output)
                logger.info(f"Graphe exporté dans {graph_output}")
        except Exception as e:
            logger.error(f"Erreur lors de la construction du graphe : {e}")
            return results
        
        # Étape 3 : Détection des cycles
        try:
            cycles = self.detect_cycles(builder, max_cycle_length=max_cycle_length)
            results["cycles"] = cycles
        except Exception as e:
            logger.error(f"Erreur lors de la détection des cycles : {e}")
        
        # Étape 4 : Détection du smurfing
        try:
            smurfing = self.detect_smurfing(builder, threshold=smurfing_threshold)
            results["smurfing"] = smurfing
        except Exception as e:
            logger.error(f"Erreur lors de la détection du smurfing : {e}")
        
        # Étape 5 : Détection des anomalies de réseau
        try:
            anomalies = self.detect_network_anomalies(
                builder,
                pagerank_threshold=pagerank_threshold,
                betweenness_threshold=betweenness_threshold
            )
            results["anomalies"] = anomalies
        except Exception as e:
            logger.error(f"Erreur lors de la détection des anomalies : {e}")
        
        # Étape 6 : Visualisation
        try:
            self.visualize(builder, output_file=visualization_output)
        except Exception as e:
            logger.error(f"Erreur lors de la visualisation : {e}")
        
        # Résumé
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        results["end_time"] = end_time.isoformat()
        results["duration_seconds"] = duration
        
        logger.info("=" * 60)
        logger.info("RÉSUMÉ DE LA DÉTECTION")
        logger.info("=" * 60)
        logger.info(f"Durée totale : {duration:.2f} secondes")
        logger.info(f"Cycles de blanchiment détectés : {len(results['cycles'])}")
        logger.info(f"Cas de smurfing détectés : {len(results['smurfing'])}")
        logger.info(f"Anomalies de réseau détectées : {len(results['anomalies'])}")
        logger.info(f"Total des alertes : {len(results['cycles']) + len(results['smurfing']) + len(results['anomalies'])}")
        logger.info("=" * 60)
        
        return results


def main() -> int:
    """
    Point d'entrée principal du programme.
    
    Returns:
        Code de sortie (0 pour succès, 1 pour erreur).
    """
    parser = argparse.ArgumentParser(
        description="Moteur de détection de fraude financière par graphes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation :
  # Exécution avec paramètres par défaut
  python -m src.fraud_detector
  
  # Génération de données personnalisées
  python -m src.fraud_detector --accounts 200 --normal 2000 --cycles 10
  
  # Utilisation d'une graine pour la reproductibilité
  python -m src.fraud_detector --seed 42
  
  # Export des données et du graphe
  python -m src.fraud_detector --data-output transactions.csv --graph-output graph.gexf
  
  # Mode verbeux
  python -m src.fraud_detector --verbose
        """
    )
    
    # Arguments de données
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Fichier CSV à charger (désactive la génération de données)"
    )
    parser.add_argument(
        "--accounts",
        type=int,
        default=100,
        help="Nombre de comptes bancaires (défaut: 100)"
    )
    parser.add_argument(
        "--normal",
        type=int,
        default=1000,
        help="Nombre de transactions normales (défaut: 1000)"
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=5,
        help="Nombre de cycles de blanchiment à injecter (défaut: 5)"
    )
    parser.add_argument(
        "--smurfing",
        type=int,
        default=3,
        help="Nombre de cas de smurfing à injecter (défaut: 3)"
    )
    parser.add_argument(
        "--anomalies",
        type=int,
        default=3,
        help="Nombre d'anomalies de réseau à injecter (défaut: 3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Graine aléatoire pour la reproductibilité"
    )
    
    # Arguments de sortie
    parser.add_argument(
        "--data-output",
        type=str,
        default=None,
        help="Fichier de sortie pour les données générées (CSV)"
    )
    parser.add_argument(
        "--graph-output",
        type=str,
        default=None,
        help="Fichier de sortie pour le graphe (GEXF)"
    )
    parser.add_argument(
        "--viz-output",
        type=str,
        default="fraud_graph.png",
        help="Fichier de sortie pour la visualisation (défaut: fraud_graph.png)"
    )
    
    # Arguments de détection
    parser.add_argument(
        "--max-cycle-length",
        type=int,
        default=5,
        help="Longueur maximale des cycles à détecter (défaut: 5)"
    )
    parser.add_argument(
        "--smurfing-threshold",
        type=float,
        default=1000.0,
        help="Seuil de montant pour le smurfing (défaut: 1000.0)"
    )
    parser.add_argument(
        "--pagerank-threshold",
        type=float,
        default=0.01,
        help="Seuil de PageRank (défaut: 0.01)"
    )
    parser.add_argument(
        "--betweenness-threshold",
        type=float,
        default=0.05,
        help="Seuil de betweenness centrality (défaut: 0.05)"
    )
    
    # Arguments généraux
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Active le mode verbeux"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0"
    )
    
    args = parser.parse_args()
    
    # Création et exécution du pipeline
    try:
        pipeline = FraudDetectionPipeline(verbose=args.verbose)
        results = pipeline.run_full_pipeline(
            input_file=args.input,
            num_accounts=args.accounts,
            num_normal=args.normal,
            num_cycles=args.cycles,
            num_smurfing=args.smurfing,
            num_anomalies=args.anomalies,
            seed=args.seed,
            data_output=args.data_output,
            graph_output=args.graph_output,
            visualization_output=args.viz_output,
            max_cycle_length=args.max_cycle_length,
            smurfing_threshold=args.smurfing_threshold,
            pagerank_threshold=args.pagerank_threshold,
            betweenness_threshold=args.betweenness_threshold
        )
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Interruption par l'utilisateur")
        return 130
    except Exception as e:
        logger.error(f"Erreur fatale : {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
