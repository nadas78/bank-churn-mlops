"""
Génère des données de production avec drift pour tester la détection
"""
import pandas as pd
import numpy as np

def generate_drifted_data(original_file='data/bank_churn.csv', 
                          output_file='data/production_data.csv',
                          drift_level='medium'):
    """
    Génère des données avec différents niveaux de drift
    
    Args:
        original_file: Fichier de données d'entraînement
        output_file: Fichier de sortie pour les données driftées
        drift_level: 'low', 'medium', 'high'
    """
    # Charger les données originales
    df = pd.read_csv(original_file)
    prod_data = df.copy()
    
    # Paramètres de drift selon le niveau
    drift_params = {
        'low': {
            'age_shift': 2,
            'credit_shift': 10,
            'balance_multiplier': 1.05,
            'salary_shift': 2000
        },
        'medium': {
            'age_shift': 5,
            'credit_shift': 30,
            'balance_multiplier': 1.15,
            'salary_shift': 5000
        },
        'high': {
            'age_shift': 10,
            'credit_shift': 50,
            'balance_multiplier': 1.30,
            'salary_shift': 10000
        }
    }
    
    params = drift_params.get(drift_level, drift_params['medium'])
    
    np.random.seed(42)
    
    # Appliquer le drift sur les features continues
    print(f"\n{'='*60}")
    print(f"GÉNÉRATION DE DONNÉES AVEC DRIFT NIVEAU: {drift_level.upper()}")
    print(f"{'='*60}")
    
    # Age: Augmentation progressive (population vieillit)
    prod_data['Age'] = prod_data['Age'] + params['age_shift']
    print(f"✓ Age: +{params['age_shift']} ans (vieillissement de la population)")
    
    # CreditScore: Dégradation générale
    prod_data['CreditScore'] = prod_data['CreditScore'] - params['credit_shift']
    prod_data['CreditScore'] = prod_data['CreditScore'].clip(300, 850)
    print(f"✓ CreditScore: -{params['credit_shift']} points (dégradation)")
    
    # Balance: Augmentation (inflation)
    prod_data['Balance'] = prod_data['Balance'] * params['balance_multiplier']
    print(f"✓ Balance: x{params['balance_multiplier']} (inflation)")
    
    # EstimatedSalary: Augmentation
    prod_data['EstimatedSalary'] = prod_data['EstimatedSalary'] + params['salary_shift']
    print(f"✓ EstimatedSalary: +{params['salary_shift']}€ (augmentation)")
    
    # Changements dans les variables catégorielles
    # Plus de clients inactifs (changement de comportement)
    inactive_mask = np.random.choice([True, False], size=len(prod_data), p=[0.3, 0.7])
    prod_data.loc[inactive_mask, 'IsActiveMember'] = 0
    print(f"✓ IsActiveMember: 30% de clients deviennent inactifs")
    
    # Distribution géographique change
    if drift_level in ['medium', 'high']:
        geo_change = np.random.choice([0, 1], size=len(prod_data), p=[0.4, 0.6])
        prod_data['Geography_Germany'] = geo_change
        prod_data['Geography_Spain'] = 1 - geo_change
        print(f"✓ Geography: Changement de distribution (60% Allemagne)")
    
    # Sauvegarder
    prod_data.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"📊 STATISTIQUES COMPARATIVES")
    print(f"{'='*60}")
    
    # Comparaison des moyennes
    original = pd.read_csv(original_file)
    
    for col in ['Age', 'CreditScore', 'Balance', 'EstimatedSalary']:
        orig_mean = original[col].mean()
        prod_mean = prod_data[col].mean()
        change_pct = ((prod_mean - orig_mean) / orig_mean) * 100
        print(f"{col:20s}: {orig_mean:>12.2f} → {prod_mean:>12.2f} ({change_pct:+.1f}%)")
    
    print(f"\n✅ Données générées: {output_file}")
    print(f"📈 {len(prod_data)} lignes créées")
    print(f"{'='*60}\n")
    
    return prod_data


if __name__ == "__main__":
    import sys
    
    # Argument optionnel pour le niveau de drift
    drift_level = sys.argv[1] if len(sys.argv) > 1 else 'medium'
    
    if drift_level not in ['low', 'medium', 'high']:
        print("❌ Niveau de drift invalide. Utilisez: low, medium, ou high")
        sys.exit(1)
    
    print("\n🎯 Génération de données de production avec drift...")
    generate_drifted_data(drift_level=drift_level)
