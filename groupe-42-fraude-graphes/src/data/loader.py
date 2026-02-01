"""
Chargeur de données pour la détection de fraude financière.

Ce module permet de charger des transactions bancaires depuis des fichiers
CSV et JSON avec validation des données.

Projet académique ECE - Groupe 42 : Malak El Idrissi et Joe Boueri.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


class TransactionLoader:
    """
    Chargeur de transactions bancaires depuis des fichiers CSV et JSON.
    
    Cette classe permet de charger des transactions depuis différents formats
    de fichiers, de valider les données et de fournir des statistiques.
    
    Attributes:
        transactions (List[Dict[str, Any]]): Liste des transactions chargées.
        errors (List[str]): Liste des erreurs rencontrées lors du chargement.
    
    Example:
        >>> loader = TransactionLoader()
        >>> transactions = loader.load_from_csv("data/transactions.csv")
        >>> print(f"Chargé {len(transactions)} transactions")
    """
    
    def __init__(self) -> None:
        """
        Initialise le chargeur de transactions.
        """
        self.transactions: List[Dict[str, Any]] = []
        self.errors: List[str] = []
    
    def load_from_csv(
        self,
        filepath: str,
        encoding: str = "utf-8",
        delimiter: str = ","
    ) -> List[Dict[str, Any]]:
        """
        Charge les transactions depuis un fichier CSV.
        
        Le fichier CSV doit contenir les colonnes suivantes:
        - sender: Identifiant de l'émetteur (ou sender_id)
        - receiver: Identifiant du destinataire (ou receiver_id)
        - amount: Montant de la transaction
        - timestamp: Horodatage de la transaction
        - transaction_id: Identifiant unique de la transaction (optionnel)
        - type: Type de transaction (optionnel)
        
        Args:
            filepath: Chemin vers le fichier CSV.
            encoding: Encodage du fichier (défaut: utf-8).
            delimiter: Délimiteur du CSV (défaut: virgule).
        
        Returns:
            Liste des transactions chargées.
        
        Raises:
            FileNotFoundError: Si le fichier n'existe pas.
            ValueError: Si le format des données est invalide.
        """
        self.transactions = []
        self.errors = []
        
        file_path = Path(filepath)
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier non trouvé: {filepath}")
        
        try:
            with open(file_path, "r", encoding=encoding) as csvfile:
                reader = csv.DictReader(csvfile, delimiter=delimiter)
                
                # Vérifier les colonnes requises
                required_columns = self._get_required_columns(reader.fieldnames or [])
                if not required_columns:
                    raise ValueError(
                        "Colonnes requises manquantes. "
                        "Nécessite: sender/sender_id, receiver/receiver_id, amount, timestamp"
                    )
                
                for row_idx, row in enumerate(reader):
                    try:
                        transaction = self._parse_transaction_row(row, row_idx)
                        if transaction:
                            self.transactions.append(transaction)
                    except ValueError as e:
                        self.errors.append(f"Ligne {row_idx + 2}: {e}")
        
        except csv.Error as e:
            raise ValueError(f"Erreur de lecture CSV: {e}")
        
        return self.transactions
    
    def load_from_json(
        self,
        filepath: str,
        encoding: str = "utf-8"
    ) -> List[Dict[str, Any]]:
        """
        Charge les transactions depuis un fichier JSON.
        
        Le fichier JSON doit contenir un tableau d'objets avec les propriétés:
        - sender: Identifiant de l'émetteur (ou sender_id)
        - receiver: Identifiant du destinataire (ou receiver_id)
        - amount: Montant de la transaction
        - timestamp: Horodatage de la transaction
        - transaction_id: Identifiant unique de la transaction (optionnel)
        - type: Type de transaction (optionnel)
        
        Args:
            filepath: Chemin vers le fichier JSON.
            encoding: Encodage du fichier (défaut: utf-8).
        
        Returns:
            Liste des transactions chargées.
        
        Raises:
            FileNotFoundError: Si le fichier n'existe pas.
            ValueError: Si le format des données est invalide.
        """
        self.transactions = []
        self.errors = []
        
        file_path = Path(filepath)
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier non trouvé: {filepath}")
        
        try:
            with open(file_path, "r", encoding=encoding) as jsonfile:
                data = json.load(jsonfile)
            
            if not isinstance(data, list):
                raise ValueError("Le fichier JSON doit contenir un tableau de transactions")
            
            for idx, row in enumerate(data):
                try:
                    transaction = self._parse_transaction_row(row, idx)
                    if transaction:
                        self.transactions.append(transaction)
                except ValueError as e:
                    self.errors.append(f"Transaction {idx}: {e}")
        
        except json.JSONDecodeError as e:
            raise ValueError(f"Erreur de parsing JSON: {e}")
        
        return self.transactions
    
    def _get_required_columns(self, fieldnames: List[str]) -> List[str]:
        """
        Vérifie que les colonnes requises sont présentes.
        
        Args:
            fieldnames: Liste des noms de colonnes.
        
        Returns:
            Liste des colonnes requises trouvées.
        """
        field_set = set(fieldnames)
        
        # Vérifier les variantes de noms de colonnes
        sender_found = "sender" in field_set or "sender_id" in field_set
        receiver_found = "receiver" in field_set or "receiver_id" in field_set
        amount_found = "amount" in field_set
        timestamp_found = "timestamp" in field_set
        
        if sender_found and receiver_found and amount_found and timestamp_found:
            return ["sender", "receiver", "amount", "timestamp"]
        return []
    
    def _parse_transaction_row(
        self,
        row: Dict[str, Any],
        row_idx: int
    ) -> Optional[Dict[str, Any]]:
        """
        Parse et valide une ligne de transaction.
        
        Args:
            row: Dictionnaire contenant les données de la transaction.
            row_idx: Index de la ligne pour les messages d'erreur.
        
        Returns:
            Transaction parsée ou None si invalide.
        
        Raises:
            ValueError: Si les données sont invalides.
        """
        # Normaliser les noms de champs
        sender = row.get("sender") or row.get("sender_id")
        receiver = row.get("receiver") or row.get("receiver_id")
        amount = row.get("amount")
        timestamp = row.get("timestamp")
        
        # Vérifier les champs requis
        if not sender:
            raise ValueError("Champ 'sender' ou 'sender_id' manquant")
        if not receiver:
            raise ValueError("Champ 'receiver' ou 'receiver_id' manquant")
        if amount is None:
            raise ValueError("Champ 'amount' manquant")
        if timestamp is None:
            raise ValueError("Champ 'timestamp' manquant")
        
        # Vérifier que sender et receiver sont différents
        if sender == receiver:
            raise ValueError(f"Sender et receiver identiques: {sender}")
        
        # Convertir le montant
        try:
            amount_float = float(amount)
        except (ValueError, TypeError):
            raise ValueError(f"Montant invalide: {amount}")
        
        if amount_float <= 0:
            raise ValueError(f"Montant doit être positif: {amount_float}")
        
        # Parser le timestamp
        timestamp_dt = self._parse_timestamp(timestamp)
        
        # Construire la transaction
        transaction = {
            "transaction_id": row.get("transaction_id") or f"TXN_{row_idx:06d}",
            "sender": sender,
            "receiver": receiver,
            "amount": round(amount_float, 2),
            "timestamp": timestamp_dt.isoformat(),
            "type": row.get("type", "normal")
        }
        
        # Ajouter les champs supplémentaires
        for key, value in row.items():
            if key not in ["sender", "sender_id", "receiver", "receiver_id", "amount", "timestamp", "transaction_id", "type"]:
                transaction[key] = value
        
        return transaction
    
    def _parse_timestamp(self, timestamp: Any) -> datetime:
        """
        Parse un timestamp dans différents formats.
        
        Supporte:
        - ISO 8601: 2024-01-15T10:30:00
        - Date simple: 2024-01-15
        - Timestamp Unix: 1705318200
        
        Args:
            timestamp: Timestamp à parser.
        
        Returns:
            Objet datetime.
        
        Raises:
            ValueError: Si le format n'est pas reconnu.
        """
        # Si c'est déjà un datetime
        if isinstance(timestamp, datetime):
            return timestamp
        
        # Essayer timestamp Unix
        try:
            return datetime.fromtimestamp(float(timestamp))
        except (ValueError, TypeError):
            pass
        
        # Formats ISO et date simple
        if isinstance(timestamp, str):
            formats = [
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
                "%d/%m/%Y %H:%M:%S",
                "%d/%m/%Y"
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(timestamp, fmt)
                except ValueError:
                    continue
        
        raise ValueError(f"Format de timestamp non reconnu: {timestamp}")
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Valide les transactions chargées.
        
        Vérifie:
        - Présence des champs requis
        - Types de données corrects
        - Valeurs cohérentes
        
        Returns:
            Tuple (is_valid, errors) où is_valid indique si toutes les
            transactions sont valides et errors contient la liste des erreurs.
        """
        errors = []
        required_fields = {"sender", "receiver", "amount", "timestamp"}
        
        for idx, tx in enumerate(self.transactions):
            # Vérifier les champs requis
            missing_fields = required_fields - set(tx.keys())
            if missing_fields:
                errors.append(f"Transaction {idx}: champs manquants: {missing_fields}")
            
            # Vérifier le montant
            try:
                amount = float(tx["amount"])
                if amount <= 0:
                    errors.append(f"Transaction {idx}: montant doit être positif: {amount}")
            except (ValueError, TypeError):
                errors.append(f"Transaction {idx}: montant invalide: {tx['amount']}")
            
            # Vérifier que sender et receiver sont différents
            if tx.get("sender") == tx.get("receiver"):
                errors.append(f"Transaction {idx}: sender et receiver identiques")
        
        return len(errors) == 0, errors
    
    def get_transactions(self) -> List[Dict[str, Any]]:
        """
        Retourne les transactions chargées.
        
        Returns:
            Liste des transactions.
        """
        return self.transactions
    
    def get_errors(self) -> List[str]:
        """
        Retourne les erreurs rencontrées lors du chargement.
        
        Returns:
            Liste des messages d'erreur.
        """
        return self.errors
    
    def clear(self) -> None:
        """
        Efface toutes les transactions et erreurs.
        """
        self.transactions = []
        self.errors = []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retourne des statistiques sur les transactions chargées.
        
        Returns:
            Dictionnaire contenant les statistiques.
        """
        if not self.transactions:
            return {
                "total_transactions": 0,
                "total_amount": 0.0,
                "unique_senders": 0,
                "unique_receivers": 0,
                "by_type": {}
            }
        
        total_amount = sum(tx["amount"] for tx in self.transactions)
        unique_senders = len(set(tx["sender"] for tx in self.transactions))
        unique_receivers = len(set(tx["receiver"] for tx in self.transactions))
        
        by_type = {}
        for tx in self.transactions:
            tx_type = tx.get("type", "normal")
            by_type[tx_type] = by_type.get(tx_type, 0) + 1
        
        return {
            "total_transactions": len(self.transactions),
            "total_amount": round(total_amount, 2),
            "unique_senders": unique_senders,
            "unique_receivers": unique_receivers,
            "by_type": by_type
        }
