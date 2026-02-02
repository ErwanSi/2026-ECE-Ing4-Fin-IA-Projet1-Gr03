#!/usr/bin/env python3
"""
Script pour générer les fichiers CSV de datasets de test.

Ce script génère trois fichiers CSV dans data/synthetic/ :
- small_dataset.csv : 100 transactions
- medium_dataset.csv : 500 transactions
- large_dataset.csv : 2000 transactions

Projet académique ECE - Groupe 42 : Malak El Idrissi et Joe Boueri.
"""

import os
import sys

# Ajouter le répertoire src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.generator import TransactionGenerator


def generate_dataset(
    num_accounts: int,
    num_normal: int,
    num_cycles: int,
    num_smurfing: int,
    num_anomalies: int,
    output_file: str
) -> None:
    """
    Génère un dataset et le sauvegarde en CSV.
    
    Args:
        num_accounts: Nombre de comptes bancaires.
        num_normal: Nombre de transactions normales.
        num_cycles: Nombre de cycles de blanchiment.
        num_smurfing: Nombre de cas de smurfing.
        num_anomalies: Nombre d'anomalies de réseau.
        output_file: Fichier de sortie CSV.
    """
    print(f"Génération de {output_file}...")
    
    # Créer le générateur
    generator = TransactionGenerator(num_accounts=num_accounts)
    
    # Générer le dataset complet
    transactions = generator.generate_complete_dataset(
        num_normal=num_normal,
        num_cycles=num_cycles,
        num_smurfing=num_smurfing,
        num_anomalies=num_anomalies
    )
    
    # Sauvegarder en CSV
    generator.save_to_csv(output_file)
    
    print(f"  ✓ {len(transactions)} transactions générées")
    print(f"  ✓ {len(generator.get_normal_transactions())} transactions normales")
    print(f"  ✓ {len(generator.get_fraudulent_transactions())} transactions frauduleuses")
    print(f"  ✓ Sauvegardé dans {output_file}")


def main() -> None:
    """Fonction principale."""
    print("=" * 60)
    print("GÉNÉRATION DES DATASETS DE TEST")
    print("=" * 60)
    
    # Créer le répertoire de sortie
    output_dir = "data/synthetic"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nRépertoire de sortie : {output_dir}/\n")
    
    # Générer le petit dataset
    print("1. Petit dataset (small_dataset.csv)")
    generate_dataset(
        num_accounts=20,
        num_normal=80,
        num_cycles=1,
        num_smurfing=1,
        num_anomalies=1,
        output_file=os.path.join(output_dir, "small_dataset.csv")
    )
    print()
    
    # Générer le dataset moyen
    print("2. Dataset moyen (medium_dataset.csv)")
    generate_dataset(
        num_accounts=50,
        num_normal=400,
        num_cycles=3,
        num_smurfing=2,
        num_anomalies=2,
        output_file=os.path.join(output_dir, "medium_dataset.csv")
    )
    print()
    
    # Générer le grand dataset
    print("3. Grand dataset (large_dataset.csv)")
    generate_dataset(
        num_accounts=100,
        num_normal=1800,
        num_cycles=5,
        num_smurfing=3,
        num_anomalies=3,
        output_file=os.path.join(output_dir, "large_dataset.csv")
    )
    print()
    
    print("=" * 60)
    print("DATASETS GÉNÉRÉS AVEC SUCCÈS")
    print("=" * 60)
    print(f"\nFichiers créés dans {output_dir}/ :")
    print("  - small_dataset.csv (100 transactions)")
    print("  - medium_dataset.csv (500 transactions)")
    print("  - large_dataset.csv (2000 transactions)")
    print("\nUtilisation :")
    print("  python3 src/fraud_detector.py --input data/synthetic/small_dataset.csv")
    print("  python3 src/fraud_detector.py --input data/synthetic/medium_dataset.csv")
    print("  python3 src/fraud_detector.py --input data/synthetic/large_dataset.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
