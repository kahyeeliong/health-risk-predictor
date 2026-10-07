from fastapi.testclient import TestClient

from app.app import app

client = TestClient(app)

HIGH = {"Pregnancies": 3, "Glucose": 175, "BloodPressure": 85, "BMI": 38.5,
        "DiabetesPedigreeFunction": 0.9, "Age": 50}
LOW = {"Pregnancies": 1, "Glucose": 95, "BloodPressure": 70, "BMI": 24.0,
       "DiabetesPedigreeFunction": 0.3, "Age": 28}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_high_risk_example():
    body = client.post("/predict", json=HIGH).json()
    assert body["risk"] == "high"
    assert 0.6 <= body["probability"] <= 1
    assert body["factors"][0]["feature"] == "Glucose"
    assert body["factors"][0]["effect"] == "raises"


def test_low_risk_example():
    body = client.post("/predict", json=LOW).json()
    assert body["risk"] == "low"
    assert body["probability"] < 0.3


def test_optional_fields_can_be_missing():
    minimal = {"Pregnancies": 0, "Glucose": 120, "BMI": 30, "Age": 40}
    response = client.post("/predict", json=minimal)
    assert response.status_code == 200
    assert len(response.json()["factors"]) == 8


def test_higher_glucose_means_higher_risk():
    low = client.post("/predict", json={**HIGH, "Glucose": 90}).json()["probability"]
    high = client.post("/predict", json={**HIGH, "Glucose": 190}).json()["probability"]
    assert high > low


def test_rejects_impossible_values():
    assert client.post("/predict", json={**HIGH, "BMI": 0}).status_code == 422
    assert client.post("/predict", json={**HIGH, "Age": -5}).status_code == 422


def test_frontend_is_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "Health Risk Predictor" in response.text


def test_model_info_has_test_metrics():
    info = client.get("/model-info").json()
    assert info["model"] == "Logistic regression"
    assert info["test"]["at_0.5"]["roc_auc"] > 0.75
