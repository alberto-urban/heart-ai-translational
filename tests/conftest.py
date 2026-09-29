import joblib
import pandas as pd
import pytest

from heart_ai.config import ALL_FEATURES
from heart_ai.pipeline import build_pipeline


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        "Age": [45, 55, 65, 50, 70, 40, 60, 35, 75, 52, 48, 58, 68, 42, 63, 37, 72, 54, 46, 61],
        "Sex": ["M", "F", "M", "M", "F", "M", "F", "M", "M", "F","F", "M", "M", "F", "M", "F", "M", "F", "M", "F"],
        "ChestPainType": ["TA", "ASY", "NAP", "ATA", "ASY", "NAP", "ATA", "TA", "ASY", "NAP","ATA", "ASY", "NAP", "TA", "ATA", "ASY", "NAP", "TA", "ASY", "ATA"],
        "RestingBP": [120, 140, 130, 125, 150, 118, 135, 110, 160, 128, 122, 138, 132, 115, 145, 112, 155, 130, 118, 142],
        "Cholesterol": [200, 240, 220, 210, 260, 190, 230, 180, 270, 215, 205, 235, 225, 195, 250, 185, 265, 218, 198, 245],
        "FastingBS": [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
        "RestingECG": ["Normal", "LVH", "Normal", "ST", "LVH", "Normal", "ST", "Normal", "LVH", "Normal", "Normal", "LVH", "ST", "Normal", "LVH", "Normal", "ST", "Normal", "LVH", "Normal"],
        "MaxHR": [160, 130, 145, 170, 120, 175, 140, 180, 115, 150, 165, 135, 142, 172, 125, 178, 118, 148, 168, 132],
        "ExerciseAngina": ["N", "Y", "N", "N", "Y", "N", "Y", "N", "Y", "N", "N", "Y", "N", "Y", "Y", "N", "Y", "N", "N", "Y"],
        "Oldpeak": [0, 2, 1, 0.2, 2.5, 0, 1.5, 0, 3, 0.5, 0.1, 1.8, 0.8, 0, 2.2, 0.3, 2.8, 0.6, 0.2, 1.7],
        "ST_Slope": ["Up", "Flat", "Up", "Up", "Down", "Up", "Flat", "Up", "Down", "Up", "Up", "Flat", "Up", "Up", "Down", "Up", "Flat", "Up", "Down", "Flat"],
        "HeartDisease": [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1],
    })

@pytest.fixture
def trained_model(tmp_path, sample_dataframe, monkeypatch):
    """Create a temporary trained model for API tests."""

    model_directory = tmp_path / "models"
    model_directory.mkdir()

    model = build_pipeline(
        model_name="logistic_regression",
        features=ALL_FEATURES,
    )

    model.fit(
        sample_dataframe[ALL_FEATURES],
        sample_dataframe["HeartDisease"],
    )

    model_path = model_directory / "logistic_regression.joblib"
    joblib.dump(model, model_path)

    monkeypatch.setattr(
        "heart_ai.api.MODELS_DIR",
        model_directory,
    )

    return model_path