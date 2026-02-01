"""
Visualiseur de graphes pour la détection de fraude financière.

Ce module permet de visualiser les graphes de transactions avec
Matplotlib, en mettant en évidence les nœuds frauduleux.

Projet académique ECE - Groupe 42 : Malak El Idrissi et Joe Boueri.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path


class GraphPlotter:
    """
    Visualiseur de graphes pour l'analyse de fraude financière.
    
    Cette classe permet de visualiser les graphes de transactions
    avec différentes options de mise en page et de mise en évidence
    des nœuds frauduleux.
    
    Attributes:
        figure_size (Tuple[int, int]): Taille de la figure en pouces.
        node_size (int): Taille des nœuds.
        edge_width (float): Largeur des arêtes.
        font_size (int): Taille de la police pour les labels.
    
    Example:
        >>> plotter = GraphPlotter()
        >>> plotter.plot_graph(graph, output_file="fraud_graph.png")
    """
    
    def __init__(
        self,
        figure_size: Tuple[int, int] = (16, 12),
        node_size: int = 500,
        edge_width: float = 1.0,
        font_size: int = 8
    ) -> None:
        """
        Initialise le visualiseur de graphes.
        
        Args:
            figure_size: Taille de la figure (largeur, hauteur).
            node_size: Taille des nœuds.
            edge_width: Largeur des arêtes.
            font_size: Taille de la police pour les labels.
        """
        self.figure_size = figure_size
        self.node_size = node_size
        self.edge_width = edge_width
        self.font_size = font_size
    
    def plot_graph(
        self,
        graph: nx.DiGraph,
        output_file: str = "fraud_graph.png",
        highlight_fraud: bool = True,
        show_labels: bool = True,
        layout: str = "spring",
        title: Optional[str] = None,
        dpi: int = 300
    ) -> None:
        """
        Visualise le graphe de transactions.
        
        Args:
            graph: Le graphe de transactions à visualiser.
            output_file: Chemin du fichier de sortie.
            highlight_fraud: Met en évidence les nœuds frauduleux en rouge.
            show_labels: Afficher les labels des nœuds.
            layout: Algorithme de layout (spring, circular, random, kamada_kawai).
            title: Titre du graphique.
            dpi: Résolution de l'image en points par pouce.
        """
        if graph.number_of_nodes() == 0:
            raise ValueError("Le graphe ne contient aucun nœud")
        
        # Créer la figure
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Calculer le layout
        pos = self._get_layout(graph, layout)
        
        # Séparer les nœuds frauduleux et normaux
        fraudulent_nodes = self._get_fraudulent_nodes(graph)
        normal_nodes = [n for n in graph.nodes() if n not in fraudulent_nodes]
        
        # Dessiner les arêtes
        self._draw_edges(ax, graph, pos)
        
        # Dessiner les nœuds normaux
        if normal_nodes:
            self._draw_nodes(
                ax,
                graph,
                normal_nodes,
                pos,
                color="lightblue",
                label="Comptes normaux"
            )
        
        # Dessiner les nœuds frauduleux
        if highlight_fraud and fraudulent_nodes:
            self._draw_nodes(
                ax,
                graph,
                fraudulent_nodes,
                pos,
                color="red",
                label="Comptes frauduleux",
                node_size=self.node_size * 1.5
            )
        
        # Afficher les labels si demandé
        if show_labels:
            self._draw_labels(ax, graph, pos)
        
        # Configurer le graphique
        ax.set_title(
            title or f"Graphe de transactions ({graph.number_of_nodes()} nœuds, {graph.number_of_edges()} arêtes)",
            fontsize=14,
            fontweight="bold"
        )
        ax.axis("off")
        
        # Ajouter la légende
        if highlight_fraud and fraudulent_nodes:
            self._add_legend(ax)
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        # Sauvegarder le graphique
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()
    
    def _get_layout(
        self,
        graph: nx.DiGraph,
        layout: str
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calcule le layout du graphe.
        
        Args:
            graph: Le graphe de transactions.
            layout: Algorithme de layout.
        
        Returns:
            Dictionnaire mappant chaque nœud à sa position.
        """
        layout = layout.lower()
        
        if layout == "spring":
            return nx.spring_layout(graph, k=1, iterations=50)
        elif layout == "circular":
            return nx.circular_layout(graph)
        elif layout == "random":
            return nx.random_layout(graph)
        elif layout == "kamada_kawai":
            return nx.kamada_kawai_layout(graph)
        elif layout == "shell":
            return nx.shell_layout(graph)
        else:
            return nx.spring_layout(graph, k=1, iterations=50)
    
    def _get_fraudulent_nodes(self, graph: nx.DiGraph) -> Set[str]:
        """
        Retourne l'ensemble des nœuds frauduleux.
        
        Args:
            graph: Le graphe de transactions.
        
        Returns:
            Ensemble des nœuds marqués comme frauduleux.
        """
        fraudulent = set()
        for node, data in graph.nodes(data=True):
            if data.get("is_fraudulent", False):
                fraudulent.add(node)
        return fraudulent
    
    def _draw_edges(
        self,
        ax: plt.Axes,
        graph: nx.DiGraph,
        pos: Dict[str, Tuple[float, float]]
    ) -> None:
        """
        Dessine les arêtes du graphe.
        
        Args:
            ax: L'axe matplotlib.
            graph: Le graphe de transactions.
            pos: Positions des nœuds.
        """
        # Dessiner les arêtes avec des flèches
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            edge_color="gray",
            width=self.edge_width,
            alpha=0.5,
            arrows=True,
            arrowsize=20,
            arrowstyle="->"
        )
    
    def _draw_nodes(
        self,
        ax: plt.Axes,
        graph: nx.DiGraph,
        nodes: List[str],
        pos: Dict[str, Tuple[float, float]],
        color: str,
        label: str,
        node_size: Optional[int] = None
    ) -> None:
        """
        Dessine les nœuds du graphe.
        
        Args:
            ax: L'axe matplotlib.
            graph: Le graphe de transactions.
            nodes: Liste des nœuds à dessiner.
            pos: Positions des nœuds.
            color: Couleur des nœuds.
            label: Label pour la légende.
            node_size: Taille des nœuds (optionnel).
        """
        if node_size is None:
            node_size = self.node_size
        
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=nodes,
            ax=ax,
            node_color=color,
            node_size=node_size,
            alpha=0.8,
            edgecolors="black",
            linewidths=1.0,
            label=label
        )
    
    def _draw_labels(
        self,
        ax: plt.Axes,
        graph: nx.DiGraph,
        pos: Dict[str, Tuple[float, float]]
    ) -> None:
        """
        Dessine les labels des nœuds.
        
        Args:
            ax: L'axe matplotlib.
            graph: Le graphe de transactions.
            pos: Positions des nœuds.
        """
        # Pour les grands graphes, n'afficher que les nœuds importants
        if graph.number_of_nodes() > 50:
            # Afficher uniquement les nœuds avec un degré élevé
            degrees = dict(graph.degree())
            threshold = sorted(degrees.values(), reverse=True)[min(20, len(degrees) - 1)]
            important_nodes = [n for n, d in degrees.items() if d >= threshold]
            labels = {n: n for n in important_nodes}
        else:
            labels = {n: n for n in graph.nodes()}
        
        nx.draw_networkx_labels(
            graph,
            pos,
            labels=labels,
            ax=ax,
            font_size=self.font_size,
            font_weight="bold"
        )
    
    def _add_legend(self, ax: plt.Axes) -> None:
        """
        Ajoute une légende au graphique.
        
        Args:
            ax: L'axe matplotlib.
        """
        normal_patch = mpatches.Patch(color="lightblue", label="Comptes normaux")
        fraud_patch = mpatches.Patch(color="red", label="Comptes frauduleux")
        ax.legend(handles=[normal_patch, fraud_patch], loc="upper right")
    
    def plot_subgraph(
        self,
        graph: nx.DiGraph,
        nodes: List[str],
        output_file: str = "subgraph.png",
        highlight_fraud: bool = True,
        show_labels: bool = True,
        layout: str = "spring",
        title: Optional[str] = None,
        dpi: int = 300
    ) -> None:
        """
        Visualise un sous-graphe contenant uniquement les nœuds spécifiés.
        
        Args:
            graph: Le graphe de transactions original.
            nodes: Liste des nœuds à inclure dans le sous-graphe.
            output_file: Chemin du fichier de sortie.
            highlight_fraud: Met en évidence les nœuds frauduleux en rouge.
            show_labels: Afficher les labels des nœuds.
            layout: Algorithme de layout.
            title: Titre du graphique.
            dpi: Résolution de l'image en points par pouce.
        """
        # Créer le sous-graphe
        subgraph = graph.subgraph(nodes).copy()
        
        # Utiliser la même méthode de visualisation
        self.plot_graph(
            subgraph,
            output_file=output_file,
            highlight_fraud=highlight_fraud,
            show_labels=show_labels,
            layout=layout,
            title=title or f"Sous-graphe ({len(nodes)} nœuds)",
            dpi=dpi
        )
    
    def plot_alert(
        self,
        graph: nx.DiGraph,
        alert: Dict[str, Any],
        output_file: str = "alert.png",
        show_labels: bool = True,
        layout: str = "spring",
        dpi: int = 300
    ) -> None:
        """
        Visualise une alerte de fraude spécifique.
        
        Args:
            graph: Le graphe de transactions.
            alert: Dictionnaire contenant les informations de l'alerte.
            output_file: Chemin du fichier de sortie.
            show_labels: Afficher les labels des nœuds.
            layout: Algorithme de layout.
            dpi: Résolution de l'image en points par pouce.
        """
        alert_type = alert.get("alert_type", "unknown")
        nodes_to_include = []
        
        # Extraire les nœuds pertinents selon le type d'alerte
        if alert_type == "money_laundering_cycle":
            nodes_to_include = alert.get("cycle_nodes", [])
        elif alert_type == "smurfing":
            nodes_to_include = [alert.get("pivot_account")] + alert.get("sender_accounts", [])
        elif alert_type == "network_anomaly":
            nodes_to_include = [alert.get("node_id")]
        
        if not nodes_to_include:
            raise ValueError(f"Impossible d'extraire les nœuds de l'alerte de type {alert_type}")
        
        # Créer le titre
        risk_level = alert.get("risk_level", "UNKNOWN")
        risk_score = alert.get("risk_score", 0.0)
        title = f"Alerte: {alert_type} - Niveau: {risk_level} (Score: {risk_score:.2f})"
        
        # Visualiser le sous-graphe
        self.plot_subgraph(
            graph,
            nodes_to_include,
            output_file=output_file,
            highlight_fraud=True,
            show_labels=show_labels,
            layout=layout,
            title=title,
            dpi=dpi
        )
    
    def plot_multiple_alerts(
        self,
        graph: nx.DiGraph,
        alerts: List[Dict[str, Any]],
        output_dir: str = "alerts",
        show_labels: bool = True,
        layout: str = "spring",
        dpi: int = 300
    ) -> None:
        """
        Visualise plusieurs alertes de fraude.
        
        Args:
            graph: Le graphe de transactions.
            alerts: Liste des alertes à visualiser.
            output_dir: Répertoire de sortie pour les images.
            show_labels: Afficher les labels des nœuds.
            layout: Algorithme de layout.
            dpi: Résolution de l'image en points par pouce.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for idx, alert in enumerate(alerts):
            alert_type = alert.get("alert_type", "unknown")
            output_file = output_path / f"alert_{idx:03d}_{alert_type}.png"
            
            try:
                self.plot_alert(
                    graph,
                    alert,
                    str(output_file),
                    show_labels=show_labels,
                    layout=layout,
                    dpi=dpi
                )
            except Exception as e:
                print(f"Erreur lors de la visualisation de l'alerte {idx}: {e}")
    
    def plot_centrality_heatmap(
        self,
        graph: nx.DiGraph,
        metric: str = "pagerank",
        output_file: str = "centrality_heatmap.png",
        layout: str = "spring",
        title: Optional[str] = None,
        dpi: int = 300
    ) -> None:
        """
        Visualise le graphe avec une heatmap de centralité.
        
        Args:
            graph: Le graphe de transactions.
            metric: Métrique de centralité (pagerank, betweenness, degree).
            output_file: Chemin du fichier de sortie.
            layout: Algorithme de layout.
            title: Titre du graphique.
            dpi: Résolution de l'image en points par pouce.
        """
        if graph.number_of_nodes() == 0:
            raise ValueError("Le graphe ne contient aucun nœud")
        
        # Calculer la métrique de centralité
        if metric == "pagerank":
            centrality = nx.pagerank(graph)
        elif metric == "betweenness":
            centrality = nx.betweenness_centrality(graph)
        elif metric == "degree":
            centrality = nx.degree_centrality(graph)
        else:
            raise ValueError(f"Métrique inconnue: {metric}")
        
        # Créer la figure
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Calculer le layout
        pos = self._get_layout(graph, layout)
        
        # Dessiner les arêtes
        self._draw_edges(ax, graph, pos)
        
        # Dessiner les nœuds avec la couleur basée sur la centralité
        nodes = list(graph.nodes())
        colors = [centrality.get(n, 0.0) for n in nodes]
        
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=nodes,
            ax=ax,
            node_color=colors,
            node_size=self.node_size,
            alpha=0.8,
            edgecolors="black",
            linewidths=1.0,
            cmap="YlOrRd"
        )
        
        # Ajouter une barre de couleur
        sm = plt.cm.ScalarMappable(
            cmap="YlOrRd",
            norm=plt.Normalize(vmin=min(colors), vmax=max(colors))
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(f"Centralité ({metric})", rotation=270, labelpad=20)
        
        # Configurer le graphique
        ax.set_title(
            title or f"Centralité {metric} ({graph.number_of_nodes()} nœuds)",
            fontsize=14,
            fontweight="bold"
        )
        ax.axis("off")
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        # Sauvegarder le graphique
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()
