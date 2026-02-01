"""
Générateur de données synthétiques pour la détection de fraude financière.

Ce module permet de générer des transactions bancaires normales et d'injecter
différents types de fraude pour tester les algorithmes de détection.
"""

import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional


class TransactionGenerator:
    """
    Générateur de transactions bancaires synthétiques.
    
    Cette classe permet de créer des ensembles de données de transactions
    normales et d'injecter des schémas de fraude spécifiques pour tester
    les algorithmes de détection.
    
    Attributes:
        num_accounts (int): Nombre de comptes bancaires à générer.
        seed (Optional[int]): Graine aléatoire pour la reproductibilité.
        accounts (List[str]): Liste des identifiants de comptes générés.
        transactions (List[Dict[str, Any]]): Liste des transactions générées.
    """
    
    def __init__(self, num_accounts: int = 100, seed: Optional[int] = None) -> None:
        """
        Initialise le générateur de transactions.
        
        Args:
            num_accounts: Nombre de comptes bancaires à générer.
            seed: Graine aléatoire pour la reproductibilité.
        """
        self.num_accounts = num_accounts
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        
        self.accounts: List[str] = [f"ACC_{i:06d}" for i in range(num_accounts)]
        self.transactions: List[Dict[str, Any]] = []
    
    def generate_normal_transactions(
        self,
        num_transactions: int = 1000,
        min_amount: float = 100.0,
        max_amount: float = 10000.0,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Génère des transactions bancaires normales.
        
        Args:
            num_transactions: Nombre de transactions à générer.
            min_amount: Montant minimum des transactions.
            max_amount: Montant maximum des transactions.
            start_date: Date de début des transactions.
            end_date: Date de fin des transactions.
        
        Returns:
            Liste des transactions générées.
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        transactions = []
        
        for _ in range(num_transactions):
            sender = random.choice(self.accounts)
            receiver = random.choice(self.accounts)
            
            # Éviter les transactions vers soi-même
            while receiver == sender:
                receiver = random.choice(self.accounts)
            
            amount = round(random.uniform(min_amount, max_amount), 2)
            timestamp = start_date + timedelta(
                seconds=random.randint(0, int((end_date - start_date).total_seconds()))
            )
            
            transaction = {
                "transaction_id": f"TXN_{len(self.transactions) + len(transactions):06d}",
                "sender": sender,
                "receiver": receiver,
                "amount": amount,
                "timestamp": timestamp.isoformat(),
                "type": "normal"
            }
            
            transactions.append(transaction)
        
        self.transactions.extend(transactions)
        return transactions
    
    def inject_money_laundering_cycles(
        self,
        num_cycles: int = 5,
        cycle_length: int = 3,
        base_amount: float = 5000.0,
        start_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Injecte des cycles de blanchiment d'argent dans les transactions.
        
        Un cycle de blanchiment est une série de transactions qui reviennent
        au compte d'origine, créant une boucle dans le graphe.
        
        Args:
            num_cycles: Nombre de cycles à injecter.
            cycle_length: Longueur de chaque cycle (nombre de comptes impliqués).
            base_amount: Montant de base des transactions du cycle.
            start_date: Date de début des transactions de fraude.
        
        Returns:
            Liste des transactions de fraude injectées.
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=15)
        
        fraud_transactions = []
        
        for cycle_idx in range(num_cycles):
            # Sélectionner des comptes aléatoires pour le cycle
            cycle_accounts = random.sample(self.accounts, cycle_length)
            
            # Créer le cycle de transactions
            for i in range(cycle_length):
                sender = cycle_accounts[i]
                receiver = cycle_accounts[(i + 1) % cycle_length]
                
                # Le montant diminue légèrement à chaque étape (frais de blanchiment)
                amount = round(base_amount * (0.95 ** i), 2)
                timestamp = start_date + timedelta(
                    hours=cycle_idx * 24 + i
                )
                
                transaction = {
                    "transaction_id": f"TXN_{len(self.transactions) + len(fraud_transactions):06d}",
                    "sender": sender,
                    "receiver": receiver,
                    "amount": amount,
                    "timestamp": timestamp.isoformat(),
                    "type": "money_laundering",
                    "cycle_id": f"CYCLE_{cycle_idx:03d}"
                }
                
                fraud_transactions.append(transaction)
        
        self.transactions.extend(fraud_transactions)
        return fraud_transactions
    
    def inject_smurfing(
        self,
        num_smurfing_cases: int = 3,
        num_deposits_per_case: int = 10,
        deposit_amount: float = 900.0,
        pivot_account: Optional[str] = None,
        start_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Injecte des cas de smurfing (dépôts fractionnés) dans les transactions.
        
        Le smurfing consiste à diviser une grande somme en plusieurs petits
        dépôts pour éviter les déclarations de seuil.
        
        Args:
            num_smurfing_cases: Nombre de cas de smurfing à injecter.
            num_deposits_per_case: Nombre de dépôts fractionnés par cas.
            deposit_amount: Montant de chaque dépôt fractionné.
            pivot_account: Compte pivot qui reçoit les dépôts.
            start_date: Date de début des transactions de fraude.
        
        Returns:
            Liste des transactions de fraude injectées.
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=10)
        
        fraud_transactions = []
        
        for case_idx in range(num_smurfing_cases):
            # Sélectionner un compte pivot
            if pivot_account is None:
                pivot = random.choice(self.accounts)
            else:
                pivot = pivot_account
            
            # Sélectionner des comptes mules pour les dépôts
            mule_accounts = random.sample(
                [acc for acc in self.accounts if acc != pivot],
                num_deposits_per_case
            )
            
            # Créer les dépôts fractionnés
            for deposit_idx, mule in enumerate(mule_accounts):
                amount = round(deposit_amount + random.uniform(-50, 50), 2)
                timestamp = start_date + timedelta(
                    hours=case_idx * 24 + deposit_idx
                )
                
                transaction = {
                    "transaction_id": f"TXN_{len(self.transactions) + len(fraud_transactions):06d}",
                    "sender": mule,
                    "receiver": pivot,
                    "amount": amount,
                    "timestamp": timestamp.isoformat(),
                    "type": "smurfing",
                    "smurfing_case_id": f"SMURF_{case_idx:03d}"
                }
                
                fraud_transactions.append(transaction)
        
        self.transactions.extend(fraud_transactions)
        return fraud_transactions
    
    def inject_network_anomalies(
        self,
        num_anomalies: int = 3,
        high_volume_amount: float = 50000.0,
        start_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Injecte des anomalies de réseau dans les transactions.
        
        Les anomalies de réseau sont caractérisées par des comptes avec
        une centralité anormalement élevée (beaucoup de transactions).
        
        Args:
            num_anomalies: Nombre d'anomalies à injecter.
            high_volume_amount: Montant élevé des transactions anormales.
            start_date: Date de début des transactions de fraude.
        
        Returns:
            Liste des transactions de fraude injectées.
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=5)
        
        fraud_transactions = []
        
        for anomaly_idx in range(num_anomalies):
            # Sélectionner un compte hub avec centralité élevée
            hub_account = random.choice(self.accounts)
            
            # Créer de nombreuses transactions vers et depuis ce hub
            num_transactions = random.randint(20, 50)
            
            for tx_idx in range(num_transactions):
                # Alterner entre entrant et sortant
                if tx_idx % 2 == 0:
                    sender = random.choice([acc for acc in self.accounts if acc != hub_account])
                    receiver = hub_account
                else:
                    sender = hub_account
                    receiver = random.choice([acc for acc in self.accounts if acc != hub_account])
                
                amount = round(high_volume_amount * random.uniform(0.5, 2.0), 2)
                timestamp = start_date + timedelta(
                    hours=anomaly_idx * 24 + tx_idx
                )
                
                transaction = {
                    "transaction_id": f"TXN_{len(self.transactions) + len(fraud_transactions):06d}",
                    "sender": sender,
                    "receiver": receiver,
                    "amount": amount,
                    "timestamp": timestamp.isoformat(),
                    "type": "network_anomaly",
                    "anomaly_id": f"ANOMALY_{anomaly_idx:03d}"
                }
                
                fraud_transactions.append(transaction)
        
        self.transactions.extend(fraud_transactions)
        return fraud_transactions
    
    def get_transactions(self) -> List[Dict[str, Any]]:
        """
        Retourne toutes les transactions générées.
        
        Returns:
            Liste de toutes les transactions générées.
        """
        return self.transactions
    
    def get_fraudulent_transactions(self) -> List[Dict[str, Any]]:
        """
        Retourne uniquement les transactions frauduleuses.
        
        Returns:
            Liste des transactions marquées comme fraude.
        """
        return [tx for tx in self.transactions if tx.get("type") != "normal"]
    
    def get_normal_transactions(self) -> List[Dict[str, Any]]:
        """
        Retourne uniquement les transactions normales.
        
        Returns:
            Liste des transactions normales.
        """
        return [tx for tx in self.transactions if tx.get("type") == "normal"]
    
    def clear_transactions(self) -> None:
        """
        Efface toutes les transactions générées.
        """
        self.transactions = []
    
    def save_to_csv(self, filepath: str) -> None:
        """
        Sauvegarde les transactions dans un fichier CSV.
        
        Args:
            filepath: Chemin du fichier CSV de sortie.
        """
        import csv
        
        if not self.transactions:
            raise ValueError("Aucune transaction à sauvegarder.")
        
        fieldnames = [
            "transaction_id", "sender", "receiver", "amount",
            "timestamp", "type", "cycle_id", "smurfing_case_id", "anomaly_id"
        ]
        
        with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for tx in self.transactions:
                # Filtrer les champs vides
                row = {k: v for k, v in tx.items() if k in fieldnames}
                writer.writerow(row)
    
    def generate_complete_dataset(
        self,
        num_normal: int = 1000,
        num_cycles: int = 5,
        num_smurfing: int = 3,
        num_anomalies: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Génère un jeu de données complet avec transactions normales et frauduleuses.
        
        Args:
            num_normal: Nombre de transactions normales.
            num_cycles: Nombre de cycles de blanchiment.
            num_smurfing: Nombre de cas de smurfing.
            num_anomalies: Nombre d'anomalies de réseau.
        
        Returns:
            Liste complète de toutes les transactions.
        """
        self.clear_transactions()
        
        # Générer les transactions normales
        self.generate_normal_transactions(num_normal)
        
        # Injecter les fraudes
        self.inject_money_laundering_cycles(num_cycles)
        self.inject_smurfing(num_smurfing)
        self.inject_network_anomalies(num_anomalies)
        
        # Trier par timestamp
        self.transactions.sort(key=lambda x: x["timestamp"])
        
        return self.transactions
