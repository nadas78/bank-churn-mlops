from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd
from typing import List
import logging
import os

from app.models import CustomerFeatures, PredictionResponse, HealthResponse

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Chargement du modele ET du scaler au demarrage
MODEL_PATH = os.getenv("MODEL_PATH", "model/churn_model_optimized.pkl")  # <-- Nom corrigé
SCALER_PATH = os.getenv("SCALER_PATH", "model/scaler.pkl")
model = None
scaler = None

@app.on_event("startup")
async def load_model_and_scaler():
    """Charge le modele et le scaler au demarrage de l'API"""
    global model, scaler
    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"Modele charge avec succes depuis {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modele : {e}")
        model = None
    
    try:
        scaler = joblib.load(SCALER_PATH)
        logger.info(f"Scaler charge avec succes depuis {SCALER_PATH}")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du scaler : {e}")
        scaler = None

def prepare_features_for_model(features: CustomerFeatures):
    """
    Transforme les 10 features de base en 19 features pour le modele.
    Cette fonction doit EXACTEMENT reproduire le preprocessing d'entrainement.
    """
    # 1. Convertir en DataFrame
    input_dict = features.dict()
    df = pd.DataFrame([input_dict])
    
    # 2. Recréer les features engineered (IDENTIQUE à votre script d'entrainement)
    # Calculs de base
    df['Balance_to_Salary_Ratio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    df['Products_per_Tenure'] = df['NumOfProducts'] / (df['Tenure'] + 1)
    df['CreditScore_Age_Interaction'] = df['CreditScore'] * df['Age'] / 1000
    
    # Pour 'Is_High_Value', utilisez les médianes exactes de votre dataset d'entrainement
    # Ces valeurs doivent être les MÊMES que dans votre script d'entrainement
    BALANCE_MEDIAN = 50000  # REMPLACEZ par la vraie médiane de votre dataset
    SALARY_MEDIAN = 75000   # REMPLACEZ par la vraie médiane de votre dataset
    df['Is_High_Value'] = ((df['Balance'] > BALANCE_MEDIAN) & 
                          (df['EstimatedSalary'] > SALARY_MEDIAN)).astype(int)
    
    # 3. Créer les colonnes one-hot pour Age_Group
    # Utilisez les MÊMES bins que dans pd.cut() lors de l'entrainement
    bins = [0, 25, 35, 45, 55, 65, 100]
    labels = ['<25', '25-35', '35-45', '45-55', '55-65', '65+']
    
    age_group = pd.cut(df['Age'], bins=bins, labels=labels)
    age_dummies = pd.get_dummies(age_group, prefix='Age', drop_first=True)
    
    # 4. Assembler toutes les colonnes
    df = pd.concat([df, age_dummies], axis=1)
    
    # 5. Appliquer le StandardScaler sur les colonnes numériques
    # Assurez-vous que c'est la MÊME liste que dans l'entrainement
    numerical_cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary', 
                     'Balance_to_Salary_Ratio', 'CreditScore_Age_Interaction']
    
    if scaler is not None:
        df[numerical_cols] = scaler.transform(df[numerical_cols])
    else:
        logger.warning("Scaler non chargé, features non normalisées!")
    
    # 6. S'assurer que l'ordre des colonnes est correct
    # Le modèle s'attend à un ordre spécifique. Si vous avez sauvegardé feature_names_in_, utilisez-le
    if hasattr(model, 'feature_names_in_'):
        expected_columns = model.feature_names_in_
        # Réorganiser df pour qu'il corresponde à l'ordre attendu
        df = df.reindex(columns=expected_columns, fill_value=0)
    
    return df

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
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503, 
            detail="Modele ou scaler non charge"
        )
    return {
        "status": "healthy",
        "model_loaded": True
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(features: CustomerFeatures):
    """
    Predit si un client va partir (churn)
    """
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503, 
            detail="Modele ou scaler non disponible"
        )
    
    try:
        # Préparation des features (10 → 19)
        processed_df = prepare_features_for_model(features)
        
        # Vérification du nombre de features
        logger.info(f"Nombre de features préparées: {processed_df.shape[1]}")
        
        # Prédiction
        proba = model.predict_proba(processed_df)[0][1]
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

# Note : Le endpoint /predict/batch doit aussi être mis à jour pour utiliser prepare_features_for_model

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)