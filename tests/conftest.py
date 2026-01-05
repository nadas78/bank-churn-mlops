import pytest
from app.main import app, model
from sklearn.dummy import DummyClassifier

@pytest.fixture(scope="session", autouse=True)
def mock_model():
    """
    Fixture pour injecter un modèle factice dans l'API pendant les tests.
    Cela évite l'erreur 503 "Model unavailable".
    """
    global model
    dummy = DummyClassifier(strategy="constant", constant=0)
    dummy.fit([[0]*10], [0])  # fit minimal juste pour que predict fonctionne
    model = dummy
    yield
