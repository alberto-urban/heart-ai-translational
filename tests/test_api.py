from fastapi.testclient import TestClient

from heart_ai.api import app

client = TestClient(app)


VALID_FEATURES = {
    "Age": 55,
    "Sex": "M",
    "ChestPainType": "ATA",
    "RestingBP": 140,
    "Cholesterol": 240,
    "FastingBS": 0,
    "RestingECG": "Normal",
    "MaxHR": 150,
    "ExerciseAngina": "N",
    "Oldpeak": 1.2,
    "ST_Slope": "Up",
}


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_available_models(trained_model):
    response = client.get("/models")

    assert response.status_code == 200
    assert response.json() == {
        "models": ["logistic_regression"],
    }


def test_prediction(trained_model):
    response = client.post(
        "/predict",
        json={
            "model": "logistic_regression",
            "features": VALID_FEATURES,
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert result["model"] == "logistic_regression"
    assert result["prediction"] in [0, 1]
    assert 0.0 <= result["probability"] <= 1.0


def test_prediction_with_unknown_model(trained_model):
    response = client.post(
        "/predict",
        json={
            "model": "unknown_model",
            "features": VALID_FEATURES,
        },
    )

    assert response.status_code == 400


def test_prediction_with_unknown_feature(trained_model):
    invalid_features = {
        **VALID_FEATURES,
        "UnknownFeature": 123,
    }

    response = client.post(
        "/predict",
        json={
            "model": "logistic_regression",
            "features": invalid_features,
        },
    )

    assert response.status_code == 400


def test_prediction_with_missing_feature(trained_model):
    incomplete_features = VALID_FEATURES.copy()
    del incomplete_features["Age"]

    response = client.post(
        "/predict",
        json={
            "model": "logistic_regression",
            "features": incomplete_features,
        },
    )

    assert response.status_code == 400