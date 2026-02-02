"""
Détecteur de cycles de blanchiment d'argent.

Ce module permet de détecter les cycles dans le graphe de transactions,
ce qui est un indicateur typique de blanchiment d'argent.

Projet académique ECE - Groupe 42 : Malak El Idrissi et Joe Boueri.
"""

import networkx as nx
from typing import List, Dict, Any, Optional
from datetime import datetime
from abc import ABC, abstractmethod
import signal
import logging
import time

logger = logging.getLogger(__name__)


class BaseDetector(ABC):
    """
    Classe de base pour tous les détecteurs de fraude.
    
    Cette classe abstraite définit l'interface commune que tous les
    détecteurs doivent implémenter.
    """
    
    @abstractmethod
    def detect(self, graph: nx.DiGraph) -> List[Dict[str, Any]]:
        """
        Détecte les fraudes dans le graphe.
        
        Args:
            graph: Le graphe de transactions à analyser.
        
        Returns:
            Liste des alertes de fraude détectées.
        """
        pass
    
    def _calculate_risk_score(
        self,
        amount: float,
        duration_hours: Optional[float] = None,
        repetition: int = 1
    ) -> float:
        """
        Calcule un score de risque entre 0 et 1.
        
        Le score est basé sur des règles logiques:
        - Plus le montant est élevé, plus le score augmente
        - Plus la durée est courte, plus le score augmente
        - Plus la répétition est élevée, plus le score augmente
        
        Args:
            amount: Montant total des transactions.
            duration_hours: Durée en heures (optionnel).
            repetition: Nombre de répétitions.
        
        Returns:
            Score de risque entre 0 et 1.
        """
        # Score basé sur le montant (échelle logarithmique)
        amount_score = min(1.0, amount / 100000.0)
        
        # Score basé sur la durée (plus court = plus risqué)
        duration_score = 0.5
        if duration_hours is not None:
            if duration_hours <= 1:
                duration_score = 1.0
            elif duration_hours <= 24:
                duration_score = 0.8
            elif duration_hours <= 168:  # 1 semaine
                duration_score = 0.5
            else:
                duration_score = 0.2
        
        # Score basé sur la répétition
        repetition_score = min(1.0, repetition / 10.0)
        
        # Combinaison pondérée des scores
        risk_score = (
            0.4 * amount_score +
            0.3 * duration_score +
            0.3 * repetition_score
        )
        
        return round(risk_score, 3)


class CycleDetector(BaseDetector):
    """
    Détecteur de cycles de blanchiment d'argent.
    
    Cette classe détecte les cycles élémentaires dans le graphe de
    transactions, ce qui est un indicateur typique de blanchiment
    d'argent où les fonds circulent entre plusieurs comptes avant
    de revenir au point de départ.
    
    Attributes:
        max_cycle_length (int): Longueur maximale des cycles à détecter.
        min_cycle_length (int): Longueur minimale des cycles à détecter.
        max_cycles (int): Nombre maximum de cycles à détecter.
        timeout_seconds (int): Timeout en secondes pour la détection.
    
    Example:
        >>> detector = CycleDetector(max_cycle_length=5)
        >>> cycles = detector.detect(graph)
        >>> print(f"Détecté {len(cycles)} cycles")
    """
    
    def __init__(
        self,
        max_cycle_length: int = 5,
        min_cycle_length: int = 3,
        max_cycles: int = 100,
        timeout_seconds: int = 15
    ) -> None:
        """
        Initialise le détecteur de cycles.
        
        Args:
            max_cycle_length: Longueur maximale des cycles à détecter.
            min_cycle_length: Longueur minimale des cycles à détecter.
            max_cycles: Nombre maximum de cycles à détecter.
            timeout_seconds: Timeout en secondes pour la détection.
        """
        self.max_cycle_length = max_cycle_length
        self.min_cycle_length = min_cycle_length
        self.max_cycles = max_cycles
        self.timeout_seconds = timeout_seconds
    
    def detect(self, graph: nx.DiGraph) -> List[Dict[str, Any]]:
        """
        Détecte les cycles de blanchiment dans le graphe.
        
        La méthode filtre d'abord le graphe pour supprimer les nœuds
        qui ne peuvent pas faire partie d'un cycle (degré < 2),
        puis recherche les cycles élémentaires avec une longueur
        limitée pour éviter les blocages de calcul.
        
        Un chronomètre est utilisé pour arrêter la détection après
        timeout_seconds et retourner les cycles déjà trouvés.
        
        Args:
            graph: Le graphe de transactions à analyser.
        
        Returns:
            Liste des cycles détectés avec leurs scores de risque.
        """
        cycles = []
        
        # Étape 1: Filtrer le graphe - supprimer les nœuds qui ne peuvent pas
        # faire partie d'un cycle (doivent avoir au moins une arête entrante
        # et une arête sortante)
        filtered_graph = self._filter_graph(graph)
        
        if filtered_graph.number_of_nodes() < self.min_cycle_length:
            logger.info(f"Graphe filtré trop petit ({filtered_graph.number_of_nodes()} nœuds)")
            return cycles
        
        logger.info(f"Graphe filtré : {filtered_graph.number_of_nodes()} nœuds, {filtered_graph.number_of_edges()} arêtes")
        
        # Étape 2: Recherche des cycles avec limites et timeout
        start_time = time.time()
        timeout_reached = False
        
        try:
            cycle_count = 0
            for cycle in nx.simple_cycles(filtered_graph):
                # Vérifier le timeout
                elapsed_time = time.time() - start_time
                if elapsed_time > self.timeout_seconds:
                    logger.warning(f"  Timeout atteint après {elapsed_time:.2f}s (limite: {self.timeout_seconds}s)")
                    timeout_reached = True
                    break
                
                # Vérifier la longueur minimale et maximale
                if len(cycle) >= self.min_cycle_length and len(cycle) <= self.max_cycle_length:
                    cycle_alert = self._create_cycle_alert(cycle, graph)
                    cycles.append(cycle_alert)
                    cycle_count += 1
                    
                    # Log de progression tous les 10 cycles
                    if cycle_count % 10 == 0:
                        elapsed = time.time() - start_time
                        logger.info(f"  {cycle_count} cycles détectés... ({elapsed:.2f}s)")
                    
                    # Arrêter si on a atteint la limite
                    if cycle_count >= self.max_cycles:
                        logger.info(f"  Limite de {self.max_cycles} cycles atteinte")
                        break
        except nx.NetworkXError as e:
            logger.warning(f"Erreur NetworkX lors de la détection de cycles : {e}")
        except Exception as e:
            logger.warning(f"Erreur lors de la détection de cycles : {e}")
        
        total_time = time.time() - start_time
        if timeout_reached:
            logger.warning(f"  Détection interrompue par timeout - {len(cycles)} cycles trouvés en {total_time:.2f}s")
        else:
            logger.info(f"  Total : {len(cycles)} cycles détectés en {total_time:.2f}s")
        
        return cycles
    
    def _filter_graph(self, graph: nx.DiGraph) -> nx.DiGraph:
        """
        Filtre le graphe pour ne garder que les nœuds pouvant former des cycles.
        
        Un nœud doit avoir au moins une arête entrante et une arête sortante
        pour pouvoir faire partie d'un cycle.
        
        Args:
            graph: Le graphe original.
        
        Returns:
            Le graphe filtré.
        """
        filtered_graph = graph.copy()
        nodes_to_remove = []
        
        for node in graph.nodes():
            in_degree = graph.in_degree(node)
            out_degree = graph.out_degree(node)
            if in_degree < 1 or out_degree < 1:
                nodes_to_remove.append(node)
        
        if nodes_to_remove:
            filtered_graph.remove_nodes_from(nodes_to_remove)
            logger.info(f"  {len(nodes_to_remove)} nœuds élagués (degré < 2)")
        
        return filtered_graph
    
    def _create_cycle_alert(
        self,
        cycle: List[str],
        graph: nx.DiGraph
    ) -> Dict[str, Any]:
        """
        Crée une alerte de fraude pour un cycle détecté.
        
        Args:
            cycle: Liste des nœuds formant le cycle.
            graph: Le graphe de transactions.
        
        Returns:
            Dictionnaire contenant les informations de l'alerte.
        """
        # Calculer le montant total du cycle
        total_amount = 0.0
        transactions = []
        timestamps = []
        
        for i in range(len(cycle)):
            sender = cycle[i]
            receiver = cycle[(i + 1) % len(cycle)]
            
            if graph.has_edge(sender, receiver):
                edge_data = graph[sender][receiver]
                total_amount += edge_data.get("total_amount", 0.0)
                
                # Récupérer les transactions de l'arête
                edge_transactions = edge_data.get("transactions", [])
                transactions.extend(edge_transactions)
                
                # Récupérer les timestamps
                first_ts = edge_data.get("first_timestamp")
                last_ts = edge_data.get("last_timestamp")
                if first_ts:
                    timestamps.append(first_ts)
                if last_ts:
                    timestamps.append(last_ts)
        
        # Calculer la durée du cycle
        duration_hours = None
        if timestamps:
            try:
                parsed_timestamps = [
                    datetime.fromisoformat(ts) if isinstance(ts, str) else ts
                    for ts in timestamps
                ]
                min_time = min(parsed_timestamps)
                max_time = max(parsed_timestamps)
                duration_hours = (max_time - min_time).total_seconds() / 3600.0
            except (ValueError, TypeError):
                pass
        
        # Calculer le score de risque
        risk_score = self._calculate_risk_score(
            amount=total_amount,
            duration_hours=duration_hours,
            repetition=len(cycle)
        )
        
        return {
            "alert_type": "money_laundering_cycle",
            "cycle_nodes": cycle,
            "cycle_length": len(cycle),
            "total_amount": round(total_amount, 2),
            "transaction_count": len(transactions),
            "duration_hours": round(duration_hours, 2) if duration_hours else None,
            "risk_score": risk_score,
            "risk_level": self._get_risk_level(risk_score),
            "transactions": transactions
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
    
    def get_high_risk_cycles(
        self,
        cycles: List[Dict[str, Any]],
        min_risk_score: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Filtre les cycles pour ne garder que ceux à haut risque.
        
        Args:
            cycles: Liste des cycles détectés.
            min_risk_score: Score de risque minimum.
        
        Returns:
            Liste des cycles à haut risque.
        """
        return [
            cycle for cycle in cycles
            if cycle.get("risk_score", 0) >= min_risk_score
        ]
    
    def get_cycles_by_node(
        self,
        cycles: List[Dict[str, Any]],
        node_id: str
    ) -> List[Dict[str, Any]]:
        """
        Retourne tous les cycles impliquant un nœud spécifique.
        
        Args:
            cycles: Liste des cycles détectés.
            node_id: Identifiant du nœud.
        
        Returns:
            Liste des cycles impliquant le nœud.
        """
        return [
            cycle for cycle in cycles
            if node_id in cycle.get("cycle_nodes", [])
        ]
