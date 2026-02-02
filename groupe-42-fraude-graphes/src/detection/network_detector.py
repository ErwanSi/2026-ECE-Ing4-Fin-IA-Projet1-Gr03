"""
Détecteur d'anomalies de réseau.

Ce module permet de détecter les anomalies basées sur les métriques
de centralité dans le graphe de transactions.

Projet académique ECE - Groupe 42 : Malak El Idrissi et Joe Boueri.
"""

import networkx as nx
from typing import List, Dict, Any, Optional
from collections import defaultdict

from .cycle_detector import BaseDetector


class NetworkDetector(BaseDetector):
    """
    Détecteur d'anomalies de réseau basé sur les métriques de centralité.
    
    Cette classe détecte les anomalies dans le graphe de transactions
    en analysant les métriques de centralité (PageRank, Betweenness)
    pour identifier les nœuds avec un comportement atypique.
    
    Attributes:
        pagerank_threshold (float): Seuil de PageRank pour détecter
            les nœuds à haute centralité.
        betweenness_threshold (float): Seuil de betweenness centrality
            pour détecter les nœuds à haute intermédiarité.
        percentile_threshold (float): Percentile pour détecter les
            outliers (défaut: 95ème percentile).
    
    Example:
        >>> detector = NetworkDetector(
        ...     pagerank_threshold=0.01,
        ...     betweenness_threshold=0.05
        ... )
        >>> anomalies = detector.detect(graph)
        >>> print(f"Détecté {len(anomalies)} anomalies")
    """
    
    def __init__(
        self,
        pagerank_threshold: Optional[float] = None,
        betweenness_threshold: Optional[float] = None,
        percentile_threshold: float = 95.0
    ) -> None:
        """
        Initialise le détecteur d'anomalies de réseau.
        
        Args:
            pagerank_threshold: Seuil de PageRank (None = auto).
            betweenness_threshold: Seuil de betweenness (None = auto).
            percentile_threshold: Percentile pour les seuils automatiques.
        """
        self.pagerank_threshold = pagerank_threshold
        self.betweenness_threshold = betweenness_threshold
        self.percentile_threshold = percentile_threshold
    
    def detect(self, graph: nx.DiGraph) -> List[Dict[str, Any]]:
        """
        Détecte les anomalies de réseau dans le graphe.
        
        La méthode calcule les métriques de centralité pour chaque nœud
        et identifie les outliers basés sur les seuils spécifiés ou
        calculés automatiquement.
        
        Args:
            graph: Le graphe de transactions à analyser.
        
        Returns:
            Liste des anomalies détectées avec leurs scores de risque.
        """
        if graph.number_of_nodes() == 0:
            return []
        
        # Calculer les métriques de centralité
        metrics = self._compute_centrality_metrics(graph)
        
        # Calculer les seuils automatiques si nécessaire
        pagerank_thresh = self.pagerank_threshold
        betweenness_thresh = self.betweenness_threshold
        
        if pagerank_thresh is None:
            pagerank_thresh = self._compute_percentile_threshold(
                [m["pagerank"] for m in metrics.values()],
                self.percentile_threshold
            )
        
        if betweenness_thresh is None:
            betweenness_thresh = self._compute_percentile_threshold(
                [m["betweenness"] for m in metrics.values()],
                self.percentile_threshold
            )
        
        # Détecter les anomalies
        anomalies = []
        
        for node_id, node_metrics in metrics.items():
            anomaly = self._detect_node_anomaly(
                node_id,
                node_metrics,
                graph,
                pagerank_thresh,
                betweenness_thresh
            )
            if anomaly:
                anomalies.append(anomaly)
        
        # Trier par score de risque décroissant
        anomalies.sort(key=lambda x: x.get("risk_score", 0), reverse=True)
        
        return anomalies
    
    def _compute_centrality_metrics(
        self,
        graph: nx.DiGraph
    ) -> Dict[str, Dict[str, float]]:
        """
        Calcule les métriques de centralité pour chaque nœud.
        
        Args:
            graph: Le graphe de transactions.
        
        Returns:
            Dictionnaire mappant chaque nœud à ses métriques.
        """
        metrics = {}
        
        # Centralité de degré
        degree_centrality = nx.degree_centrality(graph)
        in_degree_centrality = nx.in_degree_centrality(graph)
        out_degree_centrality = nx.out_degree_centrality(graph)
        
        # Centralité d'intermédiarité
        try:
            betweenness_centrality = nx.betweenness_centrality(graph)
        except Exception:
            betweenness_centrality = {node: 0.0 for node in graph.nodes()}
        
        # PageRank
        try:
            pagerank = nx.pagerank(graph)
        except Exception:
            pagerank = {node: 0.0 for node in graph.nodes()}
        
        # Combiner les métriques
        for node in graph.nodes():
            metrics[node] = {
                "degree_centrality": degree_centrality.get(node, 0.0),
                "in_degree_centrality": in_degree_centrality.get(node, 0.0),
                "out_degree_centrality": out_degree_centrality.get(node, 0.0),
                "betweenness": betweenness_centrality.get(node, 0.0),
                "pagerank": pagerank.get(node, 0.0)
            }
        
        return metrics
    
    def _compute_percentile_threshold(
        self,
        values: List[float],
        percentile: float
    ) -> float:
        """
        Calcule le seuil pour un percentile donné.
        
        Args:
            values: Liste des valeurs.
            percentile: Percentile à calculer (0-100).
        
        Returns:
            Seuil correspondant au percentile.
        """
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100.0)
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]
    
    def _detect_node_anomaly(
        self,
        node_id: str,
        metrics: Dict[str, float],
        graph: nx.DiGraph,
        pagerank_threshold: float,
        betweenness_threshold: float
    ) -> Optional[Dict[str, Any]]:
        """
        Détecte si un nœud est une anomalie.
        
        Args:
            node_id: Identifiant du nœud.
            metrics: Métriques de centralité du nœud.
            graph: Le graphe de transactions.
            pagerank_threshold: Seuil de PageRank.
            betweenness_threshold: Seuil de betweenness.
        
        Returns:
            Dictionnaire contenant les informations de l'anomalie,
            ou None si ce n'est pas une anomalie.
        """
        pagerank = metrics.get("pagerank", 0.0)
        betweenness = metrics.get("betweenness", 0.0)
        
        # Vérifier si le nœud dépasse les seuils
        is_pagerank_anomaly = pagerank >= pagerank_threshold
        is_betweenness_anomaly = betweenness >= betweenness_threshold
        
        if not (is_pagerank_anomaly or is_betweenness_anomaly):
            return None
        
        # Récupérer les attributs du nœud
        node_data = graph.nodes.get(node_id, {})
        total_sent = node_data.get("total_sent", 0.0)
        total_received = node_data.get("total_received", 0.0)
        transaction_count = node_data.get("transaction_count", 0)
        
        # Calculer le score de risque basé sur les métriques
        risk_score = self._calculate_risk_score(
            amount=max(total_sent, total_received),
            repetition=transaction_count
        )
        
        # Ajuster le score en fonction des métriques de centralité
        centrality_score = (
            0.5 * (pagerank / max(pagerank_threshold, 0.001)) +
            0.5 * (betweenness / max(betweenness_threshold, 0.001))
        )
        risk_score = min(1.0, risk_score * 0.7 + centrality_score * 0.3)
        
        # Déterminer le type d'anomalie
        anomaly_types = []
        if is_pagerank_anomaly:
            anomaly_types.append("high_pagerank")
        if is_betweenness_anomaly:
            anomaly_types.append("high_betweenness")
        
        return {
            "alert_type": "network_anomaly",
            "node_id": node_id,
            "anomaly_types": anomaly_types,
            "metrics": {
                "pagerank": round(pagerank, 6),
                "betweenness": round(betweenness, 6),
                "degree_centrality": round(metrics.get("degree_centrality", 0.0), 6),
                "in_degree_centrality": round(metrics.get("in_degree_centrality", 0.0), 6),
                "out_degree_centrality": round(metrics.get("out_degree_centrality", 0.0), 6)
            },
            "transaction_stats": {
                "total_sent": round(total_sent, 2),
                "total_received": round(total_received, 2),
                "transaction_count": transaction_count
            },
            "thresholds": {
                "pagerank": round(pagerank_threshold, 6),
                "betweenness": round(betweenness_threshold, 6)
            },
            "risk_score": round(risk_score, 3),
            "risk_level": self._get_risk_level(risk_score)
        }
    
    def _get_risk_level(self, risk_score: float) -> str:
        """
        Retourne le niveau de risque basé sur le score.
        
        Args:
            risk_score: Score de risque entre 0 et 1.
        
        Returns:
            Niveau de risque (LOW, MEDIUM, HIGH, CRITICAL).
        """
        if risk_score >= 0.8:
            return "CRITICAL"
        elif risk_score >= 0.6:
            return "HIGH"
        elif risk_score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def get_high_risk_anomalies(
        self,
        anomalies: List[Dict[str, Any]],
        min_risk_score: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Filtre les anomalies pour ne garder que celles à haut risque.
        
        Args:
            anomalies: Liste des anomalies détectées.
            min_risk_score: Score de risque minimum.
        
        Returns:
            Liste des anomalies à haut risque.
        """
        return [
            anomaly for anomaly in anomalies
            if anomaly.get("risk_score", 0) >= min_risk_score
        ]
    
    def get_anomalies_by_type(
        self,
        anomalies: List[Dict[str, Any]],
        anomaly_type: str
    ) -> List[Dict[str, Any]]:
        """
        Filtre les anomalies par type.
        
        Args:
            anomalies: Liste des anomalies détectées.
            anomaly_type: Type d'anomalie (high_pagerank, high_betweenness).
        
        Returns:
            Liste des anomalies du type spécifié.
        """
        return [
            anomaly for anomaly in anomalies
            if anomaly_type in anomaly.get("anomaly_types", [])
        ]
    
    def get_top_anomalies(
        self,
        anomalies: List[Dict[str, Any]],
        metric: str = "pagerank",
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retourne les top N anomalies selon une métrique spécifique.
        
        Args:
            anomalies: Liste des anomalies détectées.
            metric: Métrique de tri (pagerank, betweenness, risk_score).
            top_n: Nombre d'anomalies à retourner.
        
        Returns:
            Liste des top N anomalies.
        """
        if metric == "risk_score":
            key = lambda x: x.get("risk_score", 0)
        else:
            key = lambda x: x.get("metrics", {}).get(metric, 0)
        
        return sorted(anomalies, key=key, reverse=True)[:top_n]
    
    def get_anomaly_summary(
        self,
        anomalies: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Retourne un résumé des anomalies détectées.
        
        Args:
            anomalies: Liste des anomalies détectées.
        
        Returns:
            Dictionnaire contenant le résumé.
        """
        if not anomalies:
            return {
                "total_anomalies": 0,
                "by_type": {},
                "by_risk_level": {},
                "avg_risk_score": 0.0
            }
        
        # Compter par type
        by_type = defaultdict(int)
        for anomaly in anomalies:
            for anomaly_type in anomaly.get("anomaly_types", []):
                by_type[anomaly_type] += 1
        
        # Compter par niveau de risque
        by_risk_level = defaultdict(int)
        for anomaly in anomalies:
            risk_level = anomaly.get("risk_level", "UNKNOWN")
            by_risk_level[risk_level] += 1
        
        # Score moyen
        avg_risk_score = sum(
            a.get("risk_score", 0) for a in anomalies
        ) / len(anomalies)
        
        return {
            "total_anomalies": len(anomalies),
            "by_type": dict(by_type),
            "by_risk_level": dict(by_risk_level),
            "avg_risk_score": round(avg_risk_score, 3)
        }
