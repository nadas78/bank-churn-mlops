from pydantic import BaseModel, Field

class CustomerFeatures(BaseModel):
    CreditScore: int = Field(..., ge=300, le=850)
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    Geography_Germany: int
    Geography_Spain: int

class PredictionResponse(BaseModel):
    churn_probability: float
    prediction: int
    risk_level: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

# Exemple de donnée pour tests
TEST_CUSTOMER = {
    "CreditScore": 600,
    "Age": 35,
    "Tenure": 5,
    "Balance": 10000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 50000.0,
    "Geography_Germany": 0,
    "Geography_Spain": 1
}
