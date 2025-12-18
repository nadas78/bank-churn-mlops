"""
Détection de Data Drift avec visualisations et alertes
"""
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def detect_drift(reference_file, production_file, threshold=0.05, 
                 output_dir='drift_reports'):
    """
    Détecte le drift entre données de référence et production
    
    Args:
        reference_file: Données d'entraînement
        production_file: Données de production
        threshold: Seuil de p-value (défaut: 0.05)
        output_dir: Dossier pour les rapports
    
    Returns:
        dict: Résultats de détection de drift
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Charger les données
    print("\n📂 Chargement des données...")
    ref_data = pd.read_csv(reference_file)
    prod_data = pd.read_csv(production_file)
    
    print(f"   Référence: {len(ref_data)} lignes")
    print(f"   Production: {len(prod_data)} lignes")
    
    drift_results = {}
    continuous_features = []
    categorical_features = []
    
    # Classifier les features
    for col in ref_data.columns:
        if col == 'Exited':
            continue
        if col in prod_data.columns:
            if ref_data[col].dtype in ['int64', 'float64'] and ref_data[col].nunique() > 10:
                continuous_features.append(col)
            else:
                categorical_features.append(col)
    
    print(f"\n🔍 Analyse de drift...")
    print(f"   Features continues: {len(continuous_features)}")
    print(f"   Features catégorielles: {len(categorical_features)}")
    
    # Test de Kolmogorov-Smirnov pour features continues
    print(f"\n{'='*70}")
    print("DÉTECTION DE DRIFT - FEATURES CONTINUES")
    print(f"{'='*70}")
    print(f"{'Feature':<20} {'P-value':<12} {'Statistic':<12} {'Status':<15}")
    print(f"{'-'*70}")
    
    for col in continuous_features:
        ref_values = ref_data[col].dropna()
        prod_values = prod_data[col].dropna()
        
        statistic, p_value = ks_2samp(ref_values, prod_values)
        drift_detected = p_value < threshold
        
        drift_results[col] = {
            'p_value': float(p_value),
            'statistic': float(statistic),
            'drift_detected': bool(drift_detected),
            'type': 'continuous',
            'ref_mean': float(ref_values.mean()),
            'prod_mean': float(prod_values.mean()),
            'ref_std': float(ref_values.std()),
            'prod_std': float(prod_values.std())
        }
        
        status = "🚨 DRIFT" if drift_detected else "✅ OK"
        print(f"{col:<20} {p_value:<12.4f} {statistic:<12.4f} {status:<15}")
    
    # Test du Chi-2 pour features catégorielles
    if categorical_features:
        print(f"\n{'='*70}")
        print("DÉTECTION DE DRIFT - FEATURES CATÉGORIELLES")
        print(f"{'='*70}")
        print(f"{'Feature':<20} {'P-value':<12} {'Chi2':<12} {'Status':<15}")
        print(f"{'-'*70}")
        
        for col in categorical_features:
            try:
                ref_counts = ref_data[col].value_counts()
                prod_counts = prod_data[col].value_counts()
                
                # Aligner les index
                all_values = set(ref_counts.index) | set(prod_counts.index)
                ref_aligned = [ref_counts.get(v, 0) for v in all_values]
                prod_aligned = [prod_counts.get(v, 0) for v in all_values]
                
                contingency_table = np.array([ref_aligned, prod_aligned])
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                
                drift_detected = p_value < threshold
                
                drift_results[col] = {
                    'p_value': float(p_value),
                    'chi2': float(chi2),
                    'drift_detected': bool(drift_detected),
                    'type': 'categorical'
                }
                
                status = "🚨 DRIFT" if drift_detected else "✅ OK"
                print(f"{col:<20} {p_value:<12.4f} {chi2:<12.4f} {status:<15}")
            except Exception as e:
                print(f"{col:<20} {'ERROR':<12} {str(e)[:10]:<12} {'⚠️  SKIP':<15}")
    
    # Résumé
    drifted_features = [f for f, r in drift_results.items() if r['drift_detected']]
    
    print(f"\n{'='*70}")
    print("📊 RÉSUMÉ DU DRIFT")
    print(f"{'='*70}")
    print(f"Seuil de significativité: {threshold}")
    print(f"Features analysées: {len(drift_results)}")
    print(f"Features avec drift: {len(drifted_features)}")
    print(f"Pourcentage de drift: {len(drifted_features)/len(drift_results)*100:.1f}%")
    
    if drifted_features:
        print(f"\n🚨 Features affectées par le drift:")
        for feature in drifted_features:
            result = drift_results[feature]
            if result['type'] == 'continuous':
                ref_mean = result['ref_mean']
                prod_mean = result['prod_mean']
                change = ((prod_mean - ref_mean) / ref_mean) * 100
                print(f"   • {feature}: {ref_mean:.2f} → {prod_mean:.2f} ({change:+.1f}%)")
            else:
                print(f"   • {feature}: distribution changée (p={result['p_value']:.4f})")
    else:
        print("\n✅ Aucun drift détecté - Les données sont stables")
    
    print(f"{'='*70}\n")
    
    # Visualisations
    print("📈 Génération des visualisations...")
    create_drift_visualizations(
        ref_data, prod_data, drift_results, 
        continuous_features, output_dir
    )
    
    # Sauvegarde du rapport
    report_file = f"{output_dir}/drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report = {
        'timestamp': datetime.now().isoformat(),
        'threshold': threshold,
        'features_analyzed': len(drift_results),
        'features_drifted': len(drifted_features),
        'drift_percentage': len(drifted_features)/len(drift_results)*100,
        'results': drift_results
    }
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Rapport sauvegardé: {report_file}")
    
    return drift_results


def create_drift_visualizations(ref_data, prod_data, drift_results, 
                                 continuous_features, output_dir):
    """Crée des visualisations pour le drift"""
    
    # Graphique 1: Distribution des features continues
    n_features = len(continuous_features)
    if n_features > 0:
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, col in enumerate(continuous_features):
            ax = axes[idx]
            
            # Histogrammes
            ax.hist(ref_data[col].dropna(), bins=30, alpha=0.5, 
                   label='Référence', color='blue', density=True)
            ax.hist(prod_data[col].dropna(), bins=30, alpha=0.5, 
                   label='Production', color='red', density=True)
            
            # Titre avec statut de drift
            drift_status = "🚨 DRIFT" if drift_results[col]['drift_detected'] else "✅ OK"
            p_val = drift_results[col]['p_value']
            ax.set_title(f"{col}\n{drift_status} (p={p_val:.4f})")
            ax.set_xlabel(col)
            ax.set_ylabel('Densité')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Cacher les axes inutilisés
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/drift_distributions.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✓ {output_dir}/drift_distributions.png")
    
    # Graphique 2: Heatmap des p-values
    features = list(drift_results.keys())
    p_values = [drift_results[f]['p_value'] for f in features]
    drift_detected = [drift_results[f]['drift_detected'] for f in features]
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(features)*0.3)))
    
    # Créer une matrice pour la heatmap
    data_matrix = np.array(p_values).reshape(-1, 1)
    
    sns.heatmap(data_matrix, 
                annot=True, 
                fmt='.4f',
                cmap='RdYlGn_r',
                yticklabels=features,
                xticklabels=['P-value'],
                cbar_kws={'label': 'P-value'},
                vmin=0, vmax=0.1,
                ax=ax)
    
    ax.set_title('Heatmap des P-values (rouge = drift)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/drift_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ {output_dir}/drift_heatmap.png")


if __name__ == "__main__":
    import sys
    
    reference_file = "data/bank_churn.csv"
    production_file = "data/production_data.csv"
    
    # Vérifier si les fichiers existent
    try:
        pd.read_csv(reference_file)
    except FileNotFoundError:
        print(f"❌ Fichier de référence manquant: {reference_file}")
        print("   Générez d'abord les données avec: python generate_data.py")
        sys.exit(1)
    
    try:
        pd.read_csv(production_file)
    except FileNotFoundError:
        print(f"⚠️  Fichier de production manquant: {production_file}")
        print("   Génération automatique avec drift moyen...")
        from generate_drift_data import generate_drifted_data
        generate_drifted_data(drift_level='medium')
    
    # Exécuter la détection
    results = detect_drift(reference_file, production_file)
    
    print("\n🎯 Détection terminée!")
    print("   Consultez le dossier 'drift_reports/' pour les visualisations")
