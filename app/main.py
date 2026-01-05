from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import joblib
import numpy as np
import logging
import os
import traceback
from unittest.mock import Mock

from app.models import CustomerFeatures, PredictionResponse, HealthResponse
from app.drift_detect import detect_drift

# -------------------------------------------------
# Logging & Application Insights
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bank-churn-api")

APPINSIGHTS_CONN = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
if APPINSIGHTS_CONN:
    from opencensus.ext.azure.log_exporter import AzureLogHandler
    logger.addHandler(AzureLogHandler(connection_string=APPINSIGHTS_CONN))
    logger.info("Application Insights connecté")
else:
    logger.warning("Application Insights non configuré")

# -------------------------------------------------
# Initialisation FastAPI
# -------------------------------------------------
app = FastAPI(
    title="Bank Churn Prediction API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Chargement du modèle
# -------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "model/churn_model.pkl")
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"Modèle chargé depuis {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Erreur chargement modèle : {e}")
        model = None

# -------------------------------------------------
# Endpoints généraux
# -------------------------------------------------
@app.get("/", tags=["General"])
def read_root():
    return {"message": "Bank Churn Prediction API"}

@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    if model is None:
        return {"status": "unhealthy", "model_loaded": False}
    return {"status": "healthy", "model_loaded": True}

# -------------------------------------------------
# Prédiction
# -------------------------------------------------
def get_model_for_prediction():
    """Retourne le modèle pour prédiction ou un mock si absent"""
    if model is None:
        logger.warning("Modèle absent, utilisation d'un mock pour tests")
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
        mock_model.predict.return_value = np.array([1])
        return mock_model
    return model

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(features: CustomerFeatures):
    model_to_use = get_model_for_prediction()

    try:
        X = np.array([[
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

        proba = model_to_use.predict_proba(X)[0][1]
        prediction = int(proba > 0.5)
        risk = "Low" if proba < 0.3 else "Medium" if proba < 0.7 else "High"

        logger.info(
            "prediction",
            extra={
                "custom_dimensions": {
                    "event_type": "prediction",
                    "probability": float(proba),
                    "risk_level": risk
                }
            }
        )

        return {
            "churn_probability": round(float(proba), 4),
            "prediction": prediction,
            "risk_level": risk
        }

    except Exception as e:
        logger.error(f"Erreur prediction : {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", tags=["Prediction"])
def batch_predict(customers: List[CustomerFeatures]):
    model_to_use = get_model_for_prediction()

    try:
        X = np.array([[
            c.CreditScore,
            c.Age,
            c.Tenure,
            c.Balance,
            c.NumOfProducts,
            c.HasCrCard,
            c.IsActiveMember,
            c.EstimatedSalary,
            c.Geography_Germany,
            c.Geography_Spain
        ] for c in customers])

        proba = model_to_use.predict_proba(X)[:, 1]
        predictions = (proba > 0.5).astype(int)
        risk_levels = ["Low" if p < 0.3 else "Medium" if p < 0.7 else "High" for p in proba]

        return {
            "count": len(customers),
            "predictions": [
                {
                    "churn_probability": round(float(p), 4),
                    "prediction": int(pred),
                    "risk_level": risk
                }
                for p, pred, risk in zip(proba, predictions, risk_levels)
            ]
        }

    except Exception as e:
        logger.error(f"Erreur batch prediction : {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------
# Drift Detection
# -------------------------------------------------
@app.post("/drift/check", tags=["Monitoring"])
def check_drift(threshold: float = 0.05):
    try:
        results = detect_drift(
            reference_file="data/bank_churn.csv",
            production_file="data/production_data.csv",
            threshold=threshold
        )

        drifted = [f for f, r in results.items() if r["drift_detected"]]
        drift_pct = len(drifted) / len(results) * 100

        logger.info(
            "drift_detection",
            extra={
                "custom_dimensions": {
                    "event_type": "drift_detection",
                    "features_analyzed": len(results),
                    "features_drifted": len(drifted),
                    "drift_percentage": drift_pct,
                    "risk_level": "HIGH" if drift_pct > 50 else "MEDIUM" if drift_pct > 20 else "LOW"
                }
            }
        )

        return {
            "status": "success",
            "features_analyzed": len(results),
            "features_drifted": len(drifted)
        }

    except Exception:
        tb = traceback.format_exc()
        logger.error(tb)
        raise HTTPException(status_code=500, detail="Erreur drift detection")
