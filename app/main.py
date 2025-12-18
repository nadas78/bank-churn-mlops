from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from typing import List
import logging
import os
from opencensus.ext.azure.log_exporter import AzureLogHandler
from app.models import CustomerFeatures, PredictionResponse, HealthResponse
import json  
from scipy.stats import ks_2samp  
import glob  
from pathlib import Path

# Configuration du logging - AJOUT Application Insights
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AJOUT: Configuration Application Insights
APPINSIGHTS_CONN = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
if APPINSIGHTS_CONN:
    handler = AzureLogHandler(connection_string=APPINSIGHTS_CONN)
    logger.addHandler(handler)
    logger.info("Application Insights connecté")
else:
    logger.warning("Application Insights non configuré")

# Initialisation FastAPI
app = FastAPI(
    title="Bank Churn Prediction API",
    description="API de prediction de defaillance client",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS pour permettre les requetes depuis un navigateur
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement du modele au demarrage
MODEL_PATH = os.getenv("MODEL_PATH", "model/churn_model.pkl")
model = None

@app.on_event("startup")
async def load_model():
    """Charge le modele au demarrage de l'API"""
    global model
    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"Modele charge avec succes depuis {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modele : {e}")
        model = None

@app.get("/", tags=["General"])
def root():
    """Endpoint racine"""
    return {
        "message": "Bank Churn Prediction API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    """Verification de l'etat de l'API"""
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Modele non charge"
        )
    return {
        "status": "healthy",
        "model_loaded": True
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(features: CustomerFeatures):
    """
    Predit si un client va partir (churn)
    
    Retourne :
    - churn_probability : probabilite de churn (0 a 1)
    - prediction : 0 (reste) ou 1 (part)
    - risk_level : Low, Medium ou High
    """
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Modele non disponible"
        )
    
    try:
        # Preparation des features
        input_data = np.array([[
            features.CreditScore,
            features.Age,
            features.Tenure,
            features.Balance,
            features.NumOfProducts,
            features.HasCrCard,
            features.IsActiveMember,
            features.EstimatedSalary,
            features.Geography_Germany,
            features.Geography_Spain
        ]])
        
        # Prediction
        proba = model.predict_proba(input_data)[0][1]
        prediction = int(proba > 0.5)
        
        # Classification du risque
        if proba < 0.3:
            risk = "Low"
        elif proba < 0.7:
            risk = "Medium"
        else:
            risk = "High"
        
        logger.info(
            f"Prediction effectuee : proba={proba:.4f}, "
            f"prediction={prediction}, risk={risk}"
        )
        
        return {
            "churn_probability": round(float(proba), 4),
            "prediction": prediction,
            "risk_level": risk
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de la prediction : {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur de prediction : {str(e)}"
        )

@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(features_list: List[CustomerFeatures]):
    """
    Predictions en batch pour plusieurs clients
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modele non disponible")
    
    try:
        predictions = []
        
        for features in features_list:
            input_data = np.array([[
                features.CreditScore, features.Age, features.Tenure,
                features.Balance, features.NumOfProducts, features.HasCrCard,
                features.IsActiveMember, features.EstimatedSalary,
                features.Geography_Germany, features.Geography_Spain
            ]])
            
            proba = model.predict_proba(input_data)[0][1]
            prediction = int(proba > 0.5)
            
            predictions.append({
                "churn_probability": round(float(proba), 4),
                "prediction": prediction
            })
        
        logger.info(f"Batch prediction : {len(predictions)} clients traites")
        
        return {"predictions": predictions, "count": len(predictions)}
    
    except Exception as e:
        logger.error(f"Erreur batch prediction : {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# FONCTIONS UTILITAIRES POUR APPLICATION INSIGHTS
# ============================================================

def log_drift_to_insights(drift_results):
    """
    Envoie les résultats de drift vers Application Insights
    """
    if not drift_results:
        return
    
    # Calculer les métriques globales
    total_features = len(drift_results)
    drifted_features = sum(1 for r in drift_results.values() if r.get('drift_detected', False))
    drift_percentage = (drifted_features / total_features * 100) if total_features > 0 else 0
    
    # Déterminer le niveau de risque
    if drift_percentage < 20:
        risk_level = "LOW"
    elif drift_percentage < 50:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"
    
    # Logger les métriques globales
    logger.warning(
        f"🚨 DRIFT DETECTION - Risk: {risk_level} - "
        f"{drifted_features}/{total_features} features affected ({drift_percentage:.1f}%)",
        extra={
            'custom_dimensions': {
                'event_type': 'drift_detection',
                'total_features': total_features,
                'drifted_features': drifted_features,
                'drift_percentage': drift_percentage,
                'risk_level': risk_level
            }
        }
    )
    
    # Logger chaque feature avec drift
    for feature_name, details in drift_results.items():
        if details.get('drift_detected', False):
            logger.warning(
                f"⚠️  DRIFT DETECTED on feature: {feature_name}",
                extra={
                    'custom_dimensions': {
                        'event_type': 'feature_drift',
                        'feature_name': feature_name,
                        'p_value': details.get('p_value', 0),
                        'statistic': details.get('statistic', 0),
                        'type': details.get('type', 'unknown')
                    }
                }
            )


# ============================================================
# ENDPOINTS DE MONITORING DU DRIFT
# ============================================================

@app.get("/drift/status", tags=["Monitoring"])
def get_drift_status():
    """
    Récupère le dernier rapport de drift et l'envoie à Application Insights
    
    Returns:
        dict: Statut du drift avec détails des features affectées
    """
    try:
        # Chercher le dernier rapport de drift
        drift_reports = glob.glob("drift_reports/drift_report_*.json")
        
        if not drift_reports:
            logger.info("No drift report available", extra={
                'custom_dimensions': {'event_type': 'drift_check', 'status': 'no_report'}
            })
            return {
                "status": "no_report",
                "message": "Aucun rapport de drift disponible",
                "instruction": "Exécutez: python drift_detection.py"
            }
        
        # Prendre le plus récent
        latest_report = max(drift_reports, key=lambda x: Path(x).stat().st_mtime)
        
        with open(latest_report, 'r') as f:
            report = json.load(f)
        
        # Extraire les features avec drift
        drifted_features = {
            feature: details 
            for feature, details in report['results'].items() 
            if details['drift_detected']
        }
        
        # Déterminer le niveau de risque
        drift_percentage = report['drift_percentage']
        if drift_percentage < 20:
            risk_level = "LOW"
            recommendation = "Surveillance normale"
        elif drift_percentage < 50:
            risk_level = "MEDIUM"
            recommendation = "Attention requise - Envisager un réentraînement"
        else:
            risk_level = "HIGH"
            recommendation = "Action urgente - Réentraînement recommandé"
        
        # 🔥 ENVOYER À APPLICATION INSIGHTS
        log_drift_to_insights(report['results'])
        
        # Logger le résumé
        logger.info(
            f"Drift Status Check - Risk: {risk_level}",
            extra={
                'custom_dimensions': {
                    'event_type': 'drift_status_check',
                    'drift_percentage': drift_percentage,
                    'risk_level': risk_level,
                    'features_drifted': len(drifted_features),
                    'features_analyzed': report['features_analyzed'],
                    'timestamp': report['timestamp']
                }
            }
        )
        
        response = {
            "status": "ok",
            "timestamp": report['timestamp'],
            "drift_detected": len(drifted_features) > 0,
            "features_analyzed": report['features_analyzed'],
            "features_drifted": report['features_drifted'],
            "drift_percentage": round(drift_percentage, 2),
            "risk_level": risk_level,
            "recommendation": recommendation,
            "drifted_features": list(drifted_features.keys()),
            "details": drifted_features,
            "report_file": latest_report
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du statut de drift: {e}", extra={
            'custom_dimensions': {'event_type': 'drift_error', 'error': str(e)}
        })
        raise HTTPException(
            status_code=500,
            detail=f"Erreur: {str(e)}"
        )


@app.post("/drift/check", tags=["Monitoring"])
def check_drift_now(threshold: float = 0.05):
    """
    Lance une vérification de drift immédiate
    
    Args:
        threshold: Seuil de p-value (défaut: 0.05)
    
    Returns:
        dict: Résultats de la détection de drift
    """
    try:
        import subprocess
        
        # Vérifier que les fichiers existent
        if not Path("data/bank_churn.csv").exists():
            raise HTTPException(
                status_code=404,
                detail="Fichier de référence manquant: data/bank_churn.csv"
            )
        
        if not Path("data/production_data.csv").exists():
            return {
                "status": "no_production_data",
                "message": "Aucune donnée de production disponible",
                "instruction": "Générez d'abord des données: python generate_drift_data.py"
            }
        
        # Exécuter la détection de drift
        result = subprocess.run(
            ["python", "drift_detection.py"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return {
                "status": "success",
                "message": "Détection de drift exécutée avec succès",
                "output": result.stdout,
                "instruction": "Consultez /drift/status pour les résultats"
            }
        else:
            return {
                "status": "error",
                "message": "Erreur lors de l'exécution",
                "error": result.stderr
            }
            
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=504,
            detail="Timeout - La détection a pris trop de temps"
        )
    except Exception as e:
        logger.error(f"Erreur lors de la vérification de drift: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur: {str(e)}"
        )


@app.get("/drift/visualizations", tags=["Monitoring"])
def list_drift_visualizations():
    """
    Liste les visualisations de drift disponibles
    
    Returns:
        dict: Liste des fichiers de visualisation
    """
    try:
        viz_files = {
            "distributions": glob.glob("drift_reports/drift_distributions.png"),
            "heatmap": glob.glob("drift_reports/drift_heatmap.png")
        }
        
        available = {k: bool(v) for k, v in viz_files.items()}
        
        return {
            "status": "ok",
            "visualizations": available,
            "files": {k: v[0] if v else None for k, v in viz_files.items()},
            "instruction": "Les fichiers sont dans le dossier drift_reports/"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur: {str(e)}"
        )


@app.post("/drift/alert", tags=["Monitoring"])
def trigger_drift_alert(
    message: str = "Manual drift alert triggered",
    severity: str = "warning"
):
    """
    Déclenche une alerte de drift manuelle vers Application Insights
    
    Args:
        message: Message de l'alerte
        severity: Niveau (info, warning, error)
    
    Returns:
        dict: Confirmation de l'envoi
    """
    log_func = {
        'info': logger.info,
        'warning': logger.warning,
        'error': logger.error
    }.get(severity, logger.warning)
    
    log_func(
        f"🚨 MANUAL DRIFT ALERT: {message}",
        extra={
            'custom_dimensions': {
                'event_type': 'manual_drift_alert',
                'alert_message': message,
                'severity': severity,
                'triggered_by': 'api_endpoint'
            }
        }
    )
    
    return {
        "status": "alert_sent",
        "message": message,
        "severity": severity,
        "logged_to": "Application Insights"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)