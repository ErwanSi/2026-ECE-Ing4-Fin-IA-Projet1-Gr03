"""
Fonctions utilitaires pour le projet de détection de fraude financière par graphes.

Ce module contient des fonctions pour:
- Charger des données transactionnelles (CSV, JSON)
- Générer des données synthétiques avec des fraudes intégrées
- Valider les données
- Manipuler des graphes

Auteurs: Malak El Idrissi et Joe Boueri
Groupe: 42
Projet: Détection de fraude financière par graphes
"""

import csv
import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
import networkx as nx


# ============================================================================
# CHARGEMENT DE DONNÉES
# ============================================================================

def load_transactions_from_csv(filepath: str) -> List[Dict[str, Any]]:
    """
    Charge les transactions depuis un fichier CSV.

    Le fichier CSV doit contenir les colonnes suivantes:
    - sender_id: Identifiant de l'émetteur
    - receiver_id: Identifiant du destinataire
    - amount: Montant de la transaction
    - timestamp: Horodatage de la transaction (format ISO ou timestamp Unix)
    - transaction_id: Identifiant unique de la transaction (optionnel)

    Args:
        filepath: Chemin vers le fichier CSV

    Returns:
        Liste de dictionnaires représentant les transactions

    Raises:
        FileNotFoundError: Si le fichier n'existe pas
        ValueError: Si le format des données est invalide

    Example:
        >>> transactions = load_transactions_from_csv("data/transactions.csv")
        >>> print(len(transactions))
        1000
    """
    transactions = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            # Vérifier les colonnes requises
            required_columns = {'sender_id', 'receiver_id', 'amount', 'timestamp'}
            if not required_columns.issubset(set(reader.fieldnames or [])):
                missing = required_columns - set(reader.fieldnames or [])
                raise ValueError(f"Colonnes manquantes dans le CSV: {missing}")
            
            for row in reader:
                # Convertir le montant en float
                try:
                    amount = float(row['amount'])
                except (ValueError, TypeError):
                    raise ValueError(f"Montant invalide: {row['amount']}")
                
                # Normaliser le timestamp
                timestamp = parse_timestamp(row['timestamp'])
                
                transaction = {
                    'sender_id': row['sender_id'],
                    'receiver_id': row['receiver_id'],
                    'amount': amount,
                    'timestamp': timestamp,
                    'transaction_id': row.get('transaction_id', f"tx_{len(transactions)}")
                }
                transactions.append(transaction)
                
    except FileNotFoundError:
        raise FileNotFoundError(f"Fichier non trouvé: {filepath}")
    
    return transactions


def load_transactions_from_json(filepath: str) -> List[Dict[str, Any]]:
    """
    Charge les transactions depuis un fichier JSON.

    Le fichier JSON doit contenir un tableau d'objets avec les propriétés:
    - sender_id: Identifiant de l'émetteur
    - receiver_id: Identifiant du destinataire
    - amount: Montant de la transaction
    - timestamp: Horodatage de la transaction

    Args:
        filepath: Chemin vers le fichier JSON

    Returns:
        Liste de dictionnaires représentant les transactions

    Raises:
        FileNotFoundError: Si le fichier n'existe pas
        ValueError: Si le format des données est invalide

    Example:
        >>> transactions = load_transactions_from_json("data/transactions.json")
        >>> print(len(transactions))
        500
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        if not isinstance(data, list):
            raise ValueError("Le fichier JSON doit contenir un tableau de transactions")
        
        transactions = []
        for idx, row in enumerate(data):
            # Vérifier les champs requis
            required_fields = {'sender_id', 'receiver_id', 'amount', 'timestamp'}
            if not required_fields.issubset(set(row.keys())):
                missing = required_fields - set(row.keys())
                raise ValueError(f"Transaction {idx}: champs manquants: {missing}")
            
            # Convertir le montant
            try:
                amount = float(row['amount'])
            except (ValueError, TypeError):
                raise ValueError(f"Transaction {idx}: montant invalide: {row['amount']}")
            
            # Normaliser le timestamp
            timestamp = parse_timestamp(row['timestamp'])
            
            transaction = {
                'sender_id': row['sender_id'],
                'receiver_id': row['receiver_id'],
                'amount': amount,
                'timestamp': timestamp,
                'transaction_id': row.get('transaction_id', f"tx_{idx}")
            }
            transactions.append(transaction)
            
    except FileNotFoundError:
        raise FileNotFoundError(f"Fichier non trouvé: {filepath}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Erreur de parsing JSON: {e}")
    
    return transactions


def parse_timestamp(timestamp_str: str) -> datetime:
    """
    Parse une chaîne de caractères en objet datetime.

    Supporte plusieurs formats:
    - ISO 8601: 2024-01-15T10:30:00
    - Date simple: 2024-01-15
    - Timestamp Unix: 1705318200

    Args:
        timestamp_str: Chaîne représentant un timestamp

    Returns:
        Objet datetime

    Raises:
        ValueError: Si le format n'est pas reconnu
    """
    # Essayer timestamp Unix
    try:
        return datetime.fromtimestamp(float(timestamp_str))
    except (ValueError, TypeError):
        pass
    
    # Formats ISO et date simple
    formats = [
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M:%S.%f',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d',
        '%d/%m/%Y %H:%M:%S',
        '%d/%m/%Y'
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(timestamp_str, fmt)
        except ValueError:
            continue
    
    raise ValueError(f"Format de timestamp non reconnu: {timestamp_str}")


# ============================================================================
# GÉNÉRATION DE DONNÉES SYNTHÉTIQUES
# ============================================================================

def generate_synthetic_transactions(
    num_accounts: int = 100,
    num_transactions: int = 1000,
    num_money_laundering_cycles: int = 3,
    num_smurfing_patterns: int = 2,
    num_network_anomalies: int = 2,
    date_start: Optional[datetime] = None,
    date_end: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """
    Génère des transactions synthétiques avec des fraudes intégrées.

    Cette fonction crée un ensemble de transactions normales et injecte
    délibérément des patterns de fraude pour tester les algorithmes de détection.

    Args:
        num_accounts: Nombre de comptes uniques à générer
        num_transactions: Nombre total de transactions à générer
        num_money_laundering_cycles: Nombre de cycles de blanchiment à insérer
        num_smurfing_patterns: Nombre de patterns de smurfing à insérer
        num_network_anomalies: Nombre d'anomalies de réseau à insérer
        date_start: Date de début des transactions (défaut: 30 jours avant aujourd'hui)
        date_end: Date de fin des transactions (défaut: aujourd'hui)

    Returns:
        Liste de dictionnaires représentant les transactions

    Example:
        >>> transactions = generate_synthetic_transactions(
        ...     num_accounts=50,
        ...     num_transactions=500,
        ...     num_money_laundering_cycles=2
        ... )
        >>> print(len(transactions))
        500
    """
    # Configuration des dates
    if date_end is None:
        date_end = datetime.now()
    if date_start is None:
        date_start = date_end - timedelta(days=30)
    
    # Générer les identifiants de comptes
    accounts = [f"ACC_{i:04d}" for i in range(num_accounts)]
    
    transactions = []
    tx_counter = 0
    
    # 1. Générer des transactions normales
    normal_tx_count = num_transactions - (
        num_money_laundering_cycles * 5 +  # ~5 tx par cycle
        num_smurfing_patterns * 10 +       # ~10 tx par smurfing
        num_network_anomalies * 15         # ~15 tx par anomalie
    )
    
    for _ in range(normal_tx_count):
        sender = random.choice(accounts)
        receiver = random.choice([a for a in accounts if a != sender])
        amount = round(random.uniform(100, 10000), 2)
        timestamp = random_timestamp(date_start, date_end)
        
        transactions.append({
            'transaction_id': f"tx_{tx_counter}",
            'sender_id': sender,
            'receiver_id': receiver,
            'amount': amount,
            'timestamp': timestamp,
            'type': 'normal'
        })
        tx_counter += 1
    
    # 2. Générer des cycles de blanchiment
    for i in range(num_money_laundering_cycles):
        cycle_transactions = generate_money_laundering_cycle(
            accounts, date_start, date_end, cycle_id=i
        )
        for tx in cycle_transactions:
            tx['transaction_id'] = f"tx_{tx_counter}"
            tx['type'] = 'money_laundering_cycle'
            transactions.append(tx)
            tx_counter += 1
    
    # 3. Générer des patterns de smurfing
    for i in range(num_smurfing_patterns):
        smurfing_transactions = generate_smurfing_pattern(
            accounts, date_start, date_end, pattern_id=i
        )
        for tx in smurfing_transactions:
            tx['transaction_id'] = f"tx_{tx_counter}"
            tx['type'] = 'smurfing'
            transactions.append(tx)
            tx_counter += 1
    
    # 4. Générer des anomalies de réseau
    for i in range(num_network_anomalies):
        anomaly_transactions = generate_network_anomaly(
            accounts, date_start, date_end, anomaly_id=i
        )
        for tx in anomaly_transactions:
            tx['transaction_id'] = f"tx_{tx_counter}"
            tx['type'] = 'network_anomaly'
            transactions.append(tx)
            tx_counter += 1
    
    # Trier par timestamp
    transactions.sort(key=lambda x: x['timestamp'])
    
    return transactions


def generate_money_laundering_cycle(
    accounts: List[str],
    date_start: datetime,
    date_end: datetime,
    cycle_id: int
) -> List[Dict[str, Any]]:
    """
    Génère un cycle de blanchiment d'argent.

    Un cycle typique: A -> B -> C -> D -> A
    Le but est de masquer l'origine des fonds en les faisant circuler
    à travers plusieurs comptes intermédiaires.

    Args:
        accounts: Liste des comptes disponibles
        date_start: Date de début
        date_end: Date de fin
        cycle_id: Identifiant du cycle

    Returns:
        Liste de transactions formant un cycle
    """
    # Sélectionner 4-6 comptes pour le cycle
    cycle_length = random.randint(4, 6)
    cycle_accounts = random.sample(accounts, cycle_length)
    
    transactions = []
    base_timestamp = random_timestamp(date_start, date_end - timedelta(hours=cycle_length))
    
    # Créer le cycle: A -> B -> C -> ... -> A
    amount = round(random.uniform(5000, 50000), 2)
    
    for i in range(cycle_length):
        sender = cycle_accounts[i]
        receiver = cycle_accounts[(i + 1) % cycle_length]
        
        # Le montant peut varier légèrement pour paraître naturel
        variation = random.uniform(0.95, 1.05)
        tx_amount = round(amount * variation, 2)
        
        timestamp = base_timestamp + timedelta(hours=i)
        
        transactions.append({
            'sender_id': sender,
            'receiver_id': receiver,
            'amount': tx_amount,
            'timestamp': timestamp,
            'cycle_id': cycle_id
        })
    
    return transactions


def generate_smurfing_pattern(
    accounts: List[str],
    date_start: datetime,
    date_end: datetime,
    pattern_id: int
) -> List[Dict[str, Any]]:
    """
    Génère un pattern de smurfing (schtroumpfage).

    Le smurfing consiste à fractionner un montant important en plusieurs
    petites transactions vers un compte pivot pour éviter les seuils de déclaration.

    Args:
        accounts: Liste des comptes disponibles
        date_start: Date de début
        date_end: Date de fin
        pattern_id: Identifiant du pattern

    Returns:
        Liste de transactions de smurfing
    """
    # Sélectionner un compte pivot et plusieurs comptes émetteurs
    pivot_account = random.choice(accounts)
    num_mules = random.randint(5, 10)
    mule_accounts = random.sample([a for a in accounts if a != pivot_account], num_mules)
    
    transactions = []
    base_timestamp = random_timestamp(date_start, date_end - timedelta(days=1))
    
    # Montant total à fractionner
    total_amount = round(random.uniform(50000, 200000), 2)
    # Fractionner en montants inférieurs au seuil typique (ex: 10000€)
    threshold = 10000
    num_transactions = int(total_amount / (threshold * 0.8)) + 1
    amount_per_tx = round(total_amount / num_transactions, 2)
    
    for i in range(num_transactions):
        sender = random.choice(mule_accounts)
        
        # Variation légère du montant
        variation = random.uniform(0.9, 1.0)
        tx_amount = round(amount_per_tx * variation, 2)
        
        # Transactions rapprochées dans le temps
        timestamp = base_timestamp + timedelta(
            hours=random.randint(0, 24),
            minutes=random.randint(0, 59)
        )
        
        transactions.append({
            'sender_id': sender,
            'receiver_id': pivot_account,
            'amount': tx_amount,
            'timestamp': timestamp,
            'pattern_id': pattern_id,
            'pivot_account': pivot_account
        })
    
    return transactions


def generate_network_anomaly(
    accounts: List[str],
    date_start: datetime,
    date_end: datetime,
    anomaly_id: int
) -> List[Dict[str, Any]]:
    """
    Génère une anomalie de réseau.

    Une anomalie de réseau peut être:
    - Un compte avec une centralité anormalement élevée (hub)
    - Une communauté isolée avec des transactions internes denses
    - Un compte avec un comportement transactionnel atypique

    Args:
        accounts: Liste des comptes disponibles
        date_start: Date de début
        date_end: Date de fin
        anomaly_id: Identifiant de l'anomalie

    Returns:
        Liste de transactions formant une anomalie
    """
    anomaly_type = random.choice(['hub', 'isolated_community', 'burst'])
    
    if anomaly_type == 'hub':
        return generate_hub_anomaly(accounts, date_start, date_end, anomaly_id)
    elif anomaly_type == 'isolated_community':
        return generate_isolated_community(accounts, date_start, date_end, anomaly_id)
    else:
        return generate_burst_anomaly(accounts, date_start, date_end, anomaly_id)


def generate_hub_anomaly(
    accounts: List[str],
    date_start: datetime,
    date_end: datetime,
    anomaly_id: int
) -> List[Dict[str, Any]]:
    """
    Génère une anomalie de type hub (compte central).

    Un compte effectue un nombre anormalement élevé de transactions
    avec de nombreux partenaires différents.
    """
    hub_account = random.choice(accounts)
    num_partners = random.randint(15, 25)
    partners = random.sample([a for a in accounts if a != hub_account], num_partners)
    
    transactions = []
    base_timestamp = random_timestamp(date_start, date_end - timedelta(days=7))
    
    for partner in partners:
        # Transaction aller-retour
        amount = round(random.uniform(1000, 15000), 2)
        
        # Transaction hub -> partner
        timestamp1 = base_timestamp + timedelta(
            days=random.randint(0, 7),
            hours=random.randint(0, 23)
        )
        transactions.append({
            'sender_id': hub_account,
            'receiver_id': partner,
            'amount': amount,
            'timestamp': timestamp1,
            'anomaly_id': anomaly_id,
            'anomaly_type': 'hub'
        })
        
        # Transaction partner -> hub (retour)
        timestamp2 = timestamp1 + timedelta(hours=random.randint(1, 48))
        return_amount = round(amount * random.uniform(0.9, 1.1), 2)
        transactions.append({
            'sender_id': partner,
            'receiver_id': hub_account,
            'amount': return_amount,
            'timestamp': timestamp2,
            'anomaly_id': anomaly_id,
            'anomaly_type': 'hub'
        })
    
    return transactions


def generate_isolated_community(
    accounts: List[str],
    date_start: datetime,
    date_end: datetime,
    anomaly_id: int
) -> List[Dict[str, Any]]:
    """
    Génère une anomalie de type communauté isolée.

    Un groupe de comptes effectue principalement des transactions
    entre eux, formant une communauté fermée.
    """
    community_size = random.randint(5, 8)
    community = random.sample(accounts, community_size)
    
    transactions = []
    base_timestamp = random_timestamp(date_start, date_end - timedelta(days=5))
    
    # Créer un sous-graphe dense
    num_internal_tx = community_size * (community_size - 1)
    
    for _ in range(num_internal_tx):
        sender, receiver = random.sample(community, 2)
        amount = round(random.uniform(500, 5000), 2)
        timestamp = base_timestamp + timedelta(
            days=random.randint(0, 5),
            hours=random.randint(0, 23)
        )
        
        transactions.append({
            'sender_id': sender,
            'receiver_id': receiver,
            'amount': amount,
            'timestamp': timestamp,
            'anomaly_id': anomaly_id,
            'anomaly_type': 'isolated_community'
        })
    
    return transactions


def generate_burst_anomaly(
    accounts: List[str],
    date_start: datetime,
    date_end: datetime,
    anomaly_id: int
) -> List[Dict[str, Any]]:
    """
    Génère une anomalie de type burst (rafale).

    Un compte effectue un grand nombre de transactions
    sur une très courte période.
    """
    burst_account = random.choice(accounts)
    num_burst_tx = random.randint(20, 30)
    
    # Sélectionner quelques destinataires
    num_receivers = random.randint(3, 5)
    receivers = random.sample([a for a in accounts if a != burst_account], num_receivers)
    
    transactions = []
    # Toutes les transactions dans une fenêtre de 2 heures
    burst_start = random_timestamp(date_start, date_end - timedelta(hours=2))
    
    for i in range(num_burst_tx):
        receiver = random.choice(receivers)
        amount = round(random.uniform(100, 2000), 2)
        # Transactions espacées de quelques minutes
        timestamp = burst_start + timedelta(minutes=random.randint(0, 120))
        
        transactions.append({
            'sender_id': burst_account,
            'receiver_id': receiver,
            'amount': amount,
            'timestamp': timestamp,
            'anomaly_id': anomaly_id,
            'anomaly_type': 'burst'
        })
    
    return transactions


def random_timestamp(date_start: datetime, date_end: datetime) -> datetime:
    """
    Génère un timestamp aléatoire entre deux dates.

    Args:
        date_start: Date de début
        date_end: Date de fin

    Returns:
        Timestamp aléatoire
    """
    delta = date_end - date_start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return date_start + timedelta(seconds=random_seconds)


# ============================================================================
# VALIDATION DE DONNÉES
# ============================================================================

def validate_transactions(transactions: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """
    Valide la structure et le contenu des transactions.

    Vérifie:
    - Présence des champs requis
    - Types de données corrects
    - Valeurs cohérentes (montants positifs, etc.)

    Args:
        transactions: Liste de transactions à valider

    Returns:
        Tuple (is_valid, errors) où is_valid indique si toutes les
        transactions sont valides et errors contient la liste des erreurs

    Example:
        >>> is_valid, errors = validate_transactions(transactions)
        >>> if not is_valid:
        ...     print("Erreurs:", errors)
    """
    errors = []
    required_fields = {'sender_id', 'receiver_id', 'amount', 'timestamp'}
    
    for idx, tx in enumerate(transactions):
        # Vérifier les champs requis
        missing_fields = required_fields - set(tx.keys())
        if missing_fields:
            errors.append(f"Transaction {idx}: champs manquants: {missing_fields}")
            continue
        
        # Vérifier le montant
        try:
            amount = float(tx['amount'])
            if amount <= 0:
                errors.append(f"Transaction {idx}: montant doit être positif: {amount}")
        except (ValueError, TypeError):
            errors.append(f"Transaction {idx}: montant invalide: {tx['amount']}")
        
        # Vérifier le timestamp
        if not isinstance(tx['timestamp'], datetime):
            errors.append(f"Transaction {idx}: timestamp doit être un objet datetime")
        
        # Vérifier que sender et receiver sont différents
        if tx['sender_id'] == tx['receiver_id']:
            errors.append(f"Transaction {idx}: sender et receiver identiques")
    
    return len(errors) == 0, errors


def validate_graph(graph: nx.DiGraph) -> Tuple[bool, List[str]]:
    """
    Valide la structure d'un graphe transactionnel.

    Vérifie:
    - Le graphe est bien un DiGraph
    - Les nœuds ont les attributs requis
    - Les arêtes ont les attributs requis

    Args:
        graph: Graphe NetworkX à valider

    Returns:
        Tuple (is_valid, errors)
    """
    errors = []
    
    if not isinstance(graph, nx.DiGraph):
        errors.append("Le graphe doit être un DiGraph (graphe orienté)")
        return False, errors
    
    # Vérifier les attributs des arêtes
    for u, v, data in graph.edges(data=True):
        required_edge_attrs = {'amount', 'timestamp', 'transaction_id'}
        missing_attrs = required_edge_attrs - set(data.keys())
        if missing_attrs:
            errors.append(f"Arête {u}->{v}: attributs manquants: {missing_attrs}")
    
    return len(errors) == 0, errors


# ============================================================================
# UTILITAIRES POUR GRAPHES
# ============================================================================

def build_transaction_graph(
    transactions: List[Dict[str, Any]],
    min_amount: Optional[float] = None,
    max_amount: Optional[float] = None,
    date_start: Optional[datetime] = None,
    date_end: Optional[datetime] = None
) -> nx.DiGraph:
    """
    Construit un graphe orienté à partir des transactions.

    Chaque compte est un nœud, chaque transaction est une arête orientée
    de l'émetteur vers le destinataire.

    Args:
        transactions: Liste des transactions
        min_amount: Montant minimum pour inclure une transaction
        max_amount: Montant maximum pour inclure une transaction
        date_start: Date de début pour filtrer les transactions
        date_end: Date de fin pour filtrer les transactions

    Returns:
        Graphe NetworkX orienté

    Example:
        >>> graph = build_transaction_graph(transactions, min_amount=1000)
        >>> print(graph.number_of_nodes(), graph.number_of_edges())
        50 200
    """
    graph = nx.DiGraph()
    
    for tx in transactions:
        # Appliquer les filtres
        if min_amount is not None and tx['amount'] < min_amount:
            continue
        if max_amount is not None and tx['amount'] > max_amount:
            continue
        if date_start is not None and tx['timestamp'] < date_start:
            continue
        if date_end is not None and tx['timestamp'] > date_end:
            continue
        
        sender = tx['sender_id']
        receiver = tx['receiver_id']
        
        # Ajouter les nœuds s'ils n'existent pas
        if not graph.has_node(sender):
            graph.add_node(sender)
        if not graph.has_node(receiver):
            graph.add_node(receiver)
        
        # Ajouter l'arête avec les attributs
        graph.add_edge(
            sender,
            receiver,
            amount=tx['amount'],
            timestamp=tx['timestamp'],
            transaction_id=tx.get('transaction_id', '')
        )
    
    return graph


def find_cycles_in_graph(
    graph: nx.DiGraph,
    min_length: int = 3,
    max_length: int = 5,
    max_cycles: int = 50
) -> List[List[str]]:
    """
    Trouve les cycles dans un graphe orienté avec optimisations de performance.

    Cette fonction utilise plusieurs optimisations:
    - Filtre les nœuds avec degré < 2 (ne peuvent pas faire partie d'un cycle)
    - Limite la longueur maximale des cycles
    - Arrête la recherche après avoir trouvé un nombre maximum de cycles
    - Affiche des logs de progression

    Args:
        graph: Graphe NetworkX orienté
        min_length: Longueur minimale des cycles à détecter
        max_length: Longueur maximale des cycles à détecter
        max_cycles: Nombre maximum de cycles à trouver avant d'arrêter

    Returns:
        Liste de cycles (chaque cycle est une liste de nœuds)

    Example:
        >>> cycles = find_cycles_in_graph(graph, min_length=3, max_length=5)
        >>> print(f"Trouvé {len(cycles)} cycles")
    """
    cycles = []
    
    # Étape 1: Filtrer le graphe - supprimer les nœuds qui ne peuvent pas faire partie d'un cycle
    # Un nœud doit avoir au moins une arête entrante et une arête sortante
    print("  → Filtrage du graphe (suppression des nœuds avec degré < 2)...")
    filtered_graph = graph.copy()
    nodes_to_remove = []
    
    for node in graph.nodes():
        in_degree = graph.in_degree(node)
        out_degree = graph.out_degree(node)
        if in_degree < 1 or out_degree < 1:
            nodes_to_remove.append(node)
    
    if nodes_to_remove:
        filtered_graph.remove_nodes_from(nodes_to_remove)
        print(f"  → {len(nodes_to_remove)} nœuds supprimés (degré < 2)")
        print(f"  → Graphe filtré: {filtered_graph.number_of_nodes()} nœuds, {filtered_graph.number_of_edges()} arêtes")
    else:
        print(f"  → Aucun nœud à supprimer")
        print(f"  → Graphe: {filtered_graph.number_of_nodes()} nœuds, {filtered_graph.number_of_edges()} arêtes")
    
    # Si le graphe filtré est vide ou trop petit, retourner
    if filtered_graph.number_of_nodes() < min_length:
        print("  → Graphe trop petit pour contenir des cycles")
        return cycles
    
    # Étape 2: Recherche des cycles avec limites
    print(f"  → Recherche des cycles (max_length={max_length}, max_cycles={max_cycles})...")
    
    try:
        cycle_count = 0
        for cycle in nx.simple_cycles(filtered_graph):
            # Vérifier la longueur minimale et maximale
            if len(cycle) >= min_length and len(cycle) <= max_length:
                cycles.append(cycle)
                cycle_count += 1
                
                # Log de progression tous les 10 cycles
                if cycle_count % 10 == 0:
                    print(f"  → {cycle_count} cycles trouvés...")
                
                # Arrêter si on a atteint la limite
                if cycle_count >= max_cycles:
                    print(f"  → Limite de {max_cycles} cycles atteinte")
                    break
    except nx.NetworkXError as e:
        print(f"  → Erreur lors de la recherche de cycles: {e}")
        pass
    
    print(f"  → Recherche terminée: {len(cycles)} cycles trouvés")
    
    return cycles


def compute_centrality_metrics(graph: nx.DiGraph) -> Dict[str, Dict[str, float]]:
    """
    Calcule plusieurs métriques de centralité pour chaque nœud.

    Args:
        graph: Graphe NetworkX orienté

    Returns:
        Dictionnaire avec les métriques pour chaque nœud:
        {
            'node_id': {
                'degree_centrality': float,
                'in_degree_centrality': float,
                'out_degree_centrality': float,
                'betweenness_centrality': float,
                'pagerank': float
            }
        }
    """
    metrics = {}
    
    # Centralité de degré
    degree_centrality = nx.degree_centrality(graph)
    in_degree_centrality = nx.in_degree_centrality(graph)
    out_degree_centrality = nx.out_degree_centrality(graph)
    
    # Centralité d'intermédiarité (peut être lent sur les grands graphes)
    try:
        betweenness_centrality = nx.betweenness_centrality(graph)
    except:
        betweenness_centrality = {node: 0.0 for node in graph.nodes()}
    
    # PageRank
    pagerank = nx.pagerank(graph)
    
    # Combiner les métriques
    for node in graph.nodes():
        metrics[node] = {
            'degree_centrality': degree_centrality.get(node, 0.0),
            'in_degree_centrality': in_degree_centrality.get(node, 0.0),
            'out_degree_centrality': out_degree_centrality.get(node, 0.0),
            'betweenness_centrality': betweenness_centrality.get(node, 0.0),
            'pagerank': pagerank.get(node, 0.0)
        }
    
    return metrics


def detect_communities(graph: nx.Graph) -> List[set]:
    """
    Détecte les communautés dans un graphe non orienté.

    Utilise l'algorithme de Louvain pour la détection de communautés.

    Args:
        graph: Graphe NetworkX (sera converti en non orienté)

    Returns:
        Liste de communautés (chaque communauté est un set de nœuds)
    """
    # Convertir en graphe non orienté
    undirected_graph = graph.to_undirected()
    
    try:
        # Utiliser l'algorithme de Louvain via community_louvain
        import community as community_louvain
        partition = community_louvain.best_partition(undirected_graph)
        
        # Grouper les nœuds par communauté
        communities = {}
        for node, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = set()
            communities[comm_id].add(node)
        
        return list(communities.values())
    except ImportError:
        # Fallback: utiliser connected components
        return list(nx.connected_components(undirected_graph))


def get_account_statistics(
    transactions: List[Dict[str, Any]],
    account_id: str
) -> Dict[str, Any]:
    """
    Calcule des statistiques pour un compte spécifique.

    Args:
        transactions: Liste des transactions
        account_id: Identifiant du compte

    Returns:
        Dictionnaire de statistiques:
        {
            'total_sent': float,
            'total_received': float,
            'num_sent': int,
            'num_received': int,
            'unique_partners': int,
            'avg_sent_amount': float,
            'avg_received_amount': float
        }
    """
    sent = [tx for tx in transactions if tx['sender_id'] == account_id]
    received = [tx for tx in transactions if tx['receiver_id'] == account_id]
    
    total_sent = sum(tx['amount'] for tx in sent)
    total_received = sum(tx['amount'] for tx in received)
    
    partners = set()
    for tx in sent:
        partners.add(tx['receiver_id'])
    for tx in received:
        partners.add(tx['sender_id'])
    
    return {
        'total_sent': total_sent,
        'total_received': total_received,
        'num_sent': len(sent),
        'num_received': len(received),
        'unique_partners': len(partners),
        'avg_sent_amount': total_sent / len(sent) if sent else 0,
        'avg_received_amount': total_received / len(received) if received else 0
    }


def export_transactions_to_csv(
    transactions: List[Dict[str, Any]],
    filepath: str
) -> None:
    """
    Exporte les transactions vers un fichier CSV.

    Args:
        transactions: Liste des transactions
        filepath: Chemin du fichier de sortie
    """
    if not transactions:
        raise ValueError("Aucune transaction à exporter")
    
    # Déterminer les colonnes
    fieldnames = set()
    for tx in transactions:
        fieldnames.update(tx.keys())
    fieldnames = sorted(fieldnames)
    
    with open(filepath, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        for tx in transactions:
            # Convertir les datetime en string
            row = tx.copy()
            for key, value in row.items():
                if isinstance(value, datetime):
                    row[key] = value.isoformat()
            writer.writerow(row)


def export_graph_to_gexf(graph: nx.DiGraph, filepath: str) -> None:
    """
    Exporte un graphe vers un fichier GEXF.

    Args:
        graph: Graphe NetworkX
        filepath: Chemin du fichier de sortie
    """
    nx.write_gexf(graph, filepath)
