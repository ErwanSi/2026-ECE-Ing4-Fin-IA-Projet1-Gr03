"""
Constructeur de graphes pour la détection de fraude financière.

Ce module permet de transformer des transactions bancaires en graphes dirigés
pour l'analyse de fraude à l'aide de NetworkX.
"""

import networkx as nx
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime


class GraphBuilder:
    """
    Constructeur de graphes dirigés pour l'analyse de transactions.
    
    Cette classe transforme des transactions bancaires en un graphe dirigé
    où les nœuds représentent les comptes et les arêtes représentent les
    transactions entre comptes.
    
    Attributes:
        graph (nx.DiGraph): Le graphe dirigé construit.
        transactions (List[Dict[str, Any]]): Liste des transactions utilisées.
    """
    
    def __init__(self) -> None:
        """
        Initialise le constructeur de graphe.
        """
        self.graph: nx.DiGraph = nx.DiGraph()
        self.transactions: List[Dict[str, Any]] = []
    
    def build_from_transactions(
        self,
        transactions: List[Dict[str, Any]]
    ) -> nx.DiGraph:
        """
        Construit un graphe dirigé à partir d'une liste de transactions.
        
        Args:
            transactions: Liste des transactions à transformer.
        
        Returns:
            Le graphe dirigé construit.
        """
        self.transactions = transactions
        self.graph = nx.DiGraph()
        
        for tx in transactions:
            self._add_transaction_to_graph(tx)
        
        return self.graph
    
    def _add_transaction_to_graph(self, transaction: Dict[str, Any]) -> None:
        """
        Ajoute une transaction au graphe.
        
        Args:
            transaction: La transaction à ajouter.
        """
        sender = transaction.get("sender")
        receiver = transaction.get("receiver")
        amount = transaction.get("amount", 0.0)
        timestamp = transaction.get("timestamp")
        tx_type = transaction.get("type", "normal")
        tx_id = transaction.get("transaction_id")
        
        # Ajouter les nœuds s'ils n'existent pas
        if sender and not self.graph.has_node(sender):
            self.graph.add_node(
                sender,
                account_id=sender,
                total_sent=0.0,
                total_received=0.0,
                transaction_count=0,
                is_fraudulent=False
            )
        
        if receiver and not self.graph.has_node(receiver):
            self.graph.add_node(
                receiver,
                account_id=receiver,
                total_sent=0.0,
                total_received=0.0,
                transaction_count=0,
                is_fraudulent=False
            )
        
        # Ajouter l'arête avec les attributs de transaction
        if sender and receiver:
            edge_key = (sender, receiver)
            
            if self.graph.has_edge(sender, receiver):
                # Mettre à jour l'arête existante
                edge_data = self.graph[sender][receiver]
                edge_data["total_amount"] += amount
                edge_data["transaction_count"] += 1
                edge_data["transactions"].append(transaction)
            else:
                # Créer une nouvelle arête
                self.graph.add_edge(
                    sender,
                    receiver,
                    total_amount=amount,
                    transaction_count=1,
                    transactions=[transaction],
                    first_timestamp=timestamp,
                    last_timestamp=timestamp
                )
            
            # Mettre à jour les attributs des nœuds
            self.graph.nodes[sender]["total_sent"] += amount
            self.graph.nodes[sender]["transaction_count"] += 1
            self.graph.nodes[receiver]["total_received"] += amount
            self.graph.nodes[receiver]["transaction_count"] += 1
            
            # Marquer les nœuds impliqués dans une fraude
            if tx_type != "normal":
                self.graph.nodes[sender]["is_fraudulent"] = True
                self.graph.nodes[receiver]["is_fraudulent"] = True
    
    def add_node(
        self,
        node_id: str,
        **attributes: Any
    ) -> None:
        """
        Ajoute un nœud au graphe avec des attributs personnalisés.
        
        Args:
            node_id: Identifiant du nœud.
            **attributes: Attributs supplémentaires du nœud.
        """
        if not self.graph.has_node(node_id):
            default_attrs = {
                "account_id": node_id,
                "total_sent": 0.0,
                "total_received": 0.0,
                "transaction_count": 0,
                "is_fraudulent": False
            }
            default_attrs.update(attributes)
            self.graph.add_node(node_id, **default_attrs)
        else:
            # Mettre à jour les attributs existants
            for key, value in attributes.items():
                self.graph.nodes[node_id][key] = value
    
    def add_edge(
        self,
        source: str,
        target: str,
        amount: float = 0.0,
        **attributes: Any
    ) -> None:
        """
        Ajoute une arête au graphe avec des attributs personnalisés.
        
        Args:
            source: Nœud source de l'arête.
            target: Nœud cible de l'arête.
            amount: Montant de la transaction.
            **attributes: Attributs supplémentaires de l'arête.
        """
        # S'assurer que les nœuds existent
        if not self.graph.has_node(source):
            self.add_node(source)
        if not self.graph.has_node(target):
            self.add_node(target)
        
        edge_attrs = {
            "total_amount": amount,
            "transaction_count": 1,
            "transactions": [],
            "first_timestamp": None,
            "last_timestamp": None
        }
        edge_attrs.update(attributes)
        
        if self.graph.has_edge(source, target):
            # Mettre à jour l'arête existante
            edge_data = self.graph[source][target]
            edge_data["total_amount"] += amount
            edge_data["transaction_count"] += 1
            for key, value in attributes.items():
                if key not in edge_data:
                    edge_data[key] = value
        else:
            self.graph.add_edge(source, target, **edge_attrs)
        
        # Mettre à jour les attributs des nœuds
        self.graph.nodes[source]["total_sent"] += amount
        self.graph.nodes[source]["transaction_count"] += 1
        self.graph.nodes[target]["total_received"] += amount
        self.graph.nodes[target]["transaction_count"] += 1
    
    def get_graph(self) -> nx.DiGraph:
        """
        Retourne le graphe construit.
        
        Returns:
            Le graphe dirigé.
        """
        return self.graph
    
    def get_nodes(self) -> List[str]:
        """
        Retourne la liste des nœuds du graphe.
        
        Returns:
            Liste des identifiants des nœuds.
        """
        return list(self.graph.nodes())
    
    def get_edges(self) -> List[Tuple[str, str]]:
        """
        Retourne la liste des arêtes du graphe.
        
        Returns:
            Liste des tuples (source, cible) des arêtes.
        """
        return list(self.graph.edges())
    
    def get_node_attributes(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Retourne les attributs d'un nœud.
        
        Args:
            node_id: Identifiant du nœud.
        
        Returns:
            Dictionnaire des attributs du nœud, ou None si le nœud n'existe pas.
        """
        if self.graph.has_node(node_id):
            return dict(self.graph.nodes[node_id])
        return None
    
    def get_edge_attributes(
        self,
        source: str,
        target: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retourne les attributs d'une arête.
        
        Args:
            source: Nœud source de l'arête.
            target: Nœud cible de l'arête.
        
        Returns:
            Dictionnaire des attributs de l'arête, ou None si l'arête n'existe pas.
        """
        if self.graph.has_edge(source, target):
            return dict(self.graph[source][target])
        return None
    
    def get_fraudulent_nodes(self) -> List[str]:
        """
        Retourne la liste des nœuds marqués comme frauduleux.
        
        Returns:
            Liste des identifiants des nœuds frauduleux.
        """
        return [
            node for node, data in self.graph.nodes(data=True)
            if data.get("is_fraudulent", False)
        ]
    
    def get_normal_nodes(self) -> List[str]:
        """
        Retourne la liste des nœuds normaux (non frauduleux).
        
        Returns:
            Liste des identifiants des nœuds normaux.
        """
        return [
            node for node, data in self.graph.nodes(data=True)
            if not data.get("is_fraudulent", False)
        ]
    
    def get_node_degree(self, node_id: str) -> Dict[str, int]:
        """
        Retourne le degré d'un nœud (entrant, sortant, total).
        
        Args:
            node_id: Identifiant du nœud.
        
        Returns:
            Dictionnaire avec les degrés in, out et total.
        """
        if not self.graph.has_node(node_id):
            return {"in": 0, "out": 0, "total": 0}
        
        return {
            "in": self.graph.in_degree(node_id),
            "out": self.graph.out_degree(node_id),
            "total": self.graph.degree(node_id)
        }
    
    def get_high_volume_nodes(
        self,
        threshold: float = 10000.0,
        metric: str = "total_sent"
    ) -> List[str]:
        """
        Retourne les nœuds avec un volume de transactions élevé.
        
        Args:
            threshold: Seuil de volume.
            metric: Métrique à utiliser ("total_sent", "total_received", "transaction_count").
        
        Returns:
            Liste des identifiants des nœuds à haut volume.
        """
        return [
            node for node, data in self.graph.nodes(data=True)
            if data.get(metric, 0) >= threshold
        ]
    
    def get_subgraph(
        self,
        nodes: List[str]
    ) -> nx.DiGraph:
        """
        Retourne un sous-graphe contenant uniquement les nœuds spécifiés.
        
        Args:
            nodes: Liste des nœuds à inclure dans le sous-graphe.
        
        Returns:
            Le sous-graphe dirigé.
        """
        return self.graph.subgraph(nodes).copy()
    
    def get_weakly_connected_components(self) -> List[Set[str]]:
        """
        Retourne les composantes faiblement connexes du graphe.
        
        Returns:
            Liste des ensembles de nœuds pour chaque composante.
        """
        return list(nx.weakly_connected_components(self.graph))
    
    def get_strongly_connected_components(self) -> List[Set[str]]:
        """
        Retourne les composantes fortement connexes du graphe.
        
        Returns:
            Liste des ensembles de nœuds pour chaque composante.
        """
        return list(nx.strongly_connected_components(self.graph))
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Retourne des statistiques sur le graphe.
        
        Returns:
            Dictionnaire contenant diverses statistiques.
        """
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_fraudulent_nodes": len(self.get_fraudulent_nodes()),
            "num_normal_nodes": len(self.get_normal_nodes()),
            "density": nx.density(self.graph),
            "is_weakly_connected": nx.is_weakly_connected(self.graph),
            "is_strongly_connected": nx.is_strongly_connected(self.graph),
            "num_weakly_connected_components": nx.number_weakly_connected_components(self.graph),
            "num_strongly_connected_components": nx.number_strongly_connected_components(self.graph),
            "average_degree": sum(dict(self.graph.degree()).values()) / max(self.graph.number_of_nodes(), 1)
        }
    
    def clear(self) -> None:
        """
        Efface le graphe et les transactions.
        """
        self.graph = nx.DiGraph()
        self.transactions = []
    
    def export_to_gexf(self, filepath: str) -> None:
        """
        Exporte le graphe au format GEXF avec gestion d'erreurs.
        
        Args:
            filepath: Chemin du fichier de sortie.
        """
        import os
        
        try:
            # Créer une copie du graphe pour l'export
            export_graph = self.graph.copy()
            
            # Convertir tous les attributs de nœuds en types compatibles GEXF
            for node, data in export_graph.nodes(data=True):
                for key, value in data.items():
                    if key == "transactions":
                        # Ignorer la liste de transactions (trop complexe pour GEXF)
                        export_graph.nodes[node][key] = str(len(value))
                    elif isinstance(value, bool):
                        export_graph.nodes[node][key] = str(value)
                    elif isinstance(value, (int, float)):
                        export_graph.nodes[node][key] = float(value)
                    elif isinstance(value, datetime):
                        export_graph.nodes[node][key] = value.isoformat()
                    elif value is None:
                        export_graph.nodes[node][key] = ""
                    else:
                        export_graph.nodes[node][key] = str(value)
            
            # Convertir tous les attributs d'arêtes en types compatibles GEXF
            for source, target, data in export_graph.edges(data=True):
                for key, value in data.items():
                    if key == "transactions":
                        # Ignorer la liste de transactions (trop complexe pour GEXF)
                        export_graph[source][target][key] = str(len(value))
                    elif isinstance(value, bool):
                        export_graph[source][target][key] = str(value)
                    elif isinstance(value, (int, float)):
                        export_graph[source][target][key] = float(value)
                    elif isinstance(value, datetime):
                        export_graph[source][target][key] = value.isoformat()
                    elif value is None:
                        export_graph[source][target][key] = ""
                    else:
                        export_graph[source][target][key] = str(value)
            
            # Écrire le fichier GEXF
            nx.write_gexf(export_graph, filepath)
            
        except Exception as e:
            # En cas d'erreur, supprimer le fichier corrompu s'il existe
            if os.path.exists(filepath):
                os.remove(filepath)
            raise RuntimeError(f"Erreur lors de l'export GEXF : {e}")
    
    def export_to_graphml(self, filepath: str) -> None:
        """
        Exporte le graphe au format GraphML.
        
        Args:
            filepath: Chemin du fichier de sortie.
        """
        nx.write_graphml(self.graph, filepath)
