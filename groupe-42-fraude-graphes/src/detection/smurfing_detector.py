"""
Détecteur de smurfing (dépôts fractionnés).

Ce module permet de détecter les patterns de smurfing, où une grande
somme est divisée en plusieurs petites transactions pour éviter les
seuils de déclaration.

Projet académique ECE - Groupe 42 : Malak El Idrissi et Joe Boueri.
"""

import networkx as nx
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict

from .cycle_detector import BaseDetector


class SmurfingDetector(BaseDetector):
    """
    Détecteur de smurfing (dépôts fractionnés).
    
    Cette classe détecte les patterns de smurfing dans le graphe de
    transactions, caractérisés par de nombreuses petites transactions
    vers un compte pivot sur une courte période.
    
    Attributes:
        threshold (float): Seuil de montant pour considérer une transaction
            comme un dépôt fractionné.
        time_window_hours (int): Fenêtre temporelle en heures pour
            regrouper les transactions.
        min_deposits (int): Nombre minimum de dépôts pour considérer
            un cas comme du smurfing.
    
    Example:
        >>> detector = SmurfingDetector(threshold=1000.0, time_window_hours=24)
        >>> smurfing_cases = detector.detect(graph)
        >>> print(f"Détecté {len(smurfing_cases)} cas de smurfing")
    """
    
    def __init__(
        self,
        threshold: float = 1000.0,
        time_window_hours: int = 24,
        min_deposits: int = 5
    ) -> None:
        """
        Initialise le détecteur de smurfing.
        
        Args:
            threshold: Seuil de montant pour les dépôts fractionnés.
            time_window_hours: Fenêtre temporelle en heures.
            min_deposits: Nombre minimum de dépôts pour un cas de smurfing.
        """
        self.threshold = threshold
        self.time_window_hours = time_window_hours
        self.min_deposits = min_deposits
    
    def detect(self, graph: nx.DiGraph) -> List[Dict[str, Any]]:
        """
        Détecte les cas de smurfing dans le graphe.
        
        La méthode analyse les flux entrants vers chaque nœud et identifie
        les comptes pivots qui reçoivent de nombreuses petites transactions
        sur une courte période.
        
        Args:
            graph: Le graphe de transactions à analyser.
        
        Returns:
            Liste des cas de smurfing détectés avec leurs scores de risque.
        """
        smurfing_cases = []
        
        # Analyser chaque nœud comme compte pivot potentiel
        for node in graph.nodes():
            case = self._analyze_pivot_account(node, graph)
            if case:
                smurfing_cases.append(case)
        
        # Trier par score de risque décroissant
        smurfing_cases.sort(key=lambda x: x.get("risk_score", 0), reverse=True)
        
        return smurfing_cases
    
    def _analyze_pivot_account(
        self,
        pivot_node: str,
        graph: nx.DiGraph
    ) -> Optional[Dict[str, Any]]:
        """
        Analyse un compte pivot pour détecter du smurfing.
        
        Args:
            pivot_node: Le nœud à analyser comme compte pivot.
            graph: Le graphe de transactions.
        
        Returns:
            Dictionnaire contenant les informations du cas de smurfing,
            ou None si aucun cas n'est détecté.
        """
        # Récupérer toutes les transactions entrantes vers le pivot
        incoming_edges = list(graph.in_edges(pivot_node, data=True))
        
        if len(incoming_edges) < self.min_deposits:
            return None
        
        # Récupérer toutes les transactions entrantes
        incoming_transactions = []
        for sender, receiver, edge_data in incoming_edges:
            transactions = edge_data.get("transactions", [])
            incoming_transactions.extend(transactions)
        
        # Filtrer les transactions sous le seuil
        small_transactions = [
            tx for tx in incoming_transactions
            if tx.get("amount", 0) <= self.threshold
        ]
        
        if len(small_transactions) < self.min_deposits:
            return None
        
        # Grouper les transactions par fenêtre temporelle
        time_groups = self._group_by_time_window(small_transactions)
        
        # Trouver le groupe avec le plus de transactions
        max_group = max(time_groups, key=lambda g: len(g), default=[])
        
        if len(max_group) < self.min_deposits:
            return None
        
        # Créer l'alerte de smurfing
        return self._create_smurfing_alert(pivot_node, max_group, graph)
    
    def _group_by_time_window(
        self,
        transactions: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Groupe les transactions par fenêtre temporelle.
        
        Args:
            transactions: Liste des transactions à grouper.
        
        Returns:
            Liste de groupes de transactions.
        """
        if not transactions:
            return []
        
        # Parser les timestamps
        parsed_transactions = []
        for tx in transactions:
            try:
                timestamp_str = tx.get("timestamp")
                if isinstance(timestamp_str, str):
                    timestamp = datetime.fromisoformat(timestamp_str)
                else:
                    timestamp = timestamp_str
                parsed_transactions.append((timestamp, tx))
            except (ValueError, TypeError):
                continue
        
        # Trier par timestamp
        parsed_transactions.sort(key=lambda x: x[0])
        
        # Grouper par fenêtre temporelle
        groups = []
        current_group = []
        window_start = None
        
        for timestamp, tx in parsed_transactions:
            if window_start is None:
                window_start = timestamp
                current_group = [tx]
            else:
                time_diff = (timestamp - window_start).total_seconds() / 3600.0
                if time_diff <= self.time_window_hours:
                    current_group.append(tx)
                else:
                    if current_group:
                        groups.append(current_group)
                    window_start = timestamp
                    current_group = [tx]
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _create_smurfing_alert(
        self,
        pivot_node: str,
        transactions: List[Dict[str, Any]],
        graph: nx.DiGraph
    ) -> Dict[str, Any]:
        """
        Crée une alerte de fraude pour un cas de smurfing détecté.
        
        Args:
            pivot_node: Le compte pivot.
            transactions: Liste des transactions de smurfing.
            graph: Le graphe de transactions.
        
        Returns:
            Dictionnaire contenant les informations de l'alerte.
        """
        # Calculer le montant total
        total_amount = sum(tx.get("amount", 0) for tx in transactions)
        
        # Récupérer les comptes émetteurs uniques
        senders = set(tx.get("sender") for tx in transactions if tx.get("sender"))
        
        # Calculer la durée
        timestamps = []
        for tx in transactions:
            try:
                timestamp_str = tx.get("timestamp")
                if isinstance(timestamp_str, str):
                    timestamp = datetime.fromisoformat(timestamp_str)
                else:
                    timestamp = timestamp_str
                timestamps.append(timestamp)
            except (ValueError, TypeError):
                continue
        
        duration_hours = None
        if timestamps:
            min_time = min(timestamps)
            max_time = max(timestamps)
            duration_hours = (max_time - min_time).total_seconds() / 3600.0
        
        # Calculer le montant moyen
        avg_amount = total_amount / len(transactions) if transactions else 0.0
        
        # Calculer le score de risque
        risk_score = self._calculate_risk_score(
            amount=total_amount,
            duration_hours=duration_hours,
            repetition=len(transactions)
        )
        
        return {
            "alert_type": "smurfing",
            "pivot_account": pivot_node,
            "sender_accounts": list(senders),
            "transaction_count": len(transactions),
            "unique_senders": len(senders),
            "total_amount": round(total_amount, 2),
            "average_amount": round(avg_amount, 2),
            "duration_hours": round(duration_hours, 2) if duration_hours else None,
            "time_window_hours": self.time_window_hours,
            "threshold": self.threshold,
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
    
    def get_high_risk_cases(
        self,
        cases: List[Dict[str, Any]],
        min_risk_score: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Filtre les cas pour ne garder que ceux à haut risque.
        
        Args:
            cases: Liste des cas de smurfing détectés.
            min_risk_score: Score de risque minimum.
        
        Returns:
            Liste des cas à haut risque.
        """
        return [
            case for case in cases
            if case.get("risk_score", 0) >= min_risk_score
        ]
    
    def get_cases_by_pivot(
        self,
        cases: List[Dict[str, Any]],
        pivot_account: str
    ) -> List[Dict[str, Any]]:
        """
        Retourne tous les cas impliquant un compte pivot spécifique.
        
        Args:
            cases: Liste des cas de smurfing détectés.
            pivot_account: Identifiant du compte pivot.
        
        Returns:
            Liste des cas impliquant le compte pivot.
        """
        return [
            case for case in cases
            if case.get("pivot_account") == pivot_account
        ]
    
    def get_sender_network(
        self,
        cases: List[Dict[str, Any]]
    ) -> Dict[str, Set[str]]:
        """
        Analyse le réseau de comptes émetteurs impliqués dans le smurfing.
        
        Args:
            cases: Liste des cas de smurfing détectés.
        
        Returns:
            Dictionnaire mappant chaque compte pivot à ses émetteurs.
        """
        network = {}
        for case in cases:
            pivot = case.get("pivot_account")
            senders = set(case.get("sender_accounts", []))
            if pivot:
                network[pivot] = senders
        return network
    
    def get_overlapping_senders(
        self,
        cases: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """
        Identifie les émetteurs qui participent à plusieurs cas de smurfing.
        
        Args:
            cases: Liste des cas de smurfing détectés.
        
        Returns:
            Dictionnaire mappant chaque émetteur à la liste des pivots
            vers lesquels il envoie des fonds.
        """
        sender_to_pivots = defaultdict(list)
        
        for case in cases:
            pivot = case.get("pivot_account")
            senders = case.get("sender_accounts", [])
            for sender in senders:
                if pivot:
                    sender_to_pivots[sender].append(pivot)
        
        # Filtrer pour ne garder que les émetteurs avec plusieurs pivots
        return {
            sender: pivots
            for sender, pivots in sender_to_pivots.items()
            if len(pivots) > 1
        }
