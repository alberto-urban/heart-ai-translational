from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import ALL_FEATURES
from .models import MODEL_NAMES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"


app = FastAPI(
    title="Heart AI Translational API",
    description=(
        "REST API for heart disease prediction using trained "
        "machine learning pipelines."
    ),
    version="0.1.0",
)


class PredictionRequest(BaseModel):
    model: str = Field(
        default="logistic_regression",
        description="Name of the trained model to use.",
    )

    features: dict[str, Any] = Field(
        ...,
        description="Patient features used for prediction.",
    )


class PredictionResponse(BaseModel):
    model: str
    prediction: int
    probability: float


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return the API health status."""

    return {"status": "ok"}


@app.get("/models")
def available_models() -> dict[str, list[str]]:
    """Return the models supported by the API."""

    available = []

    for model_name in MODEL_NAMES:
        model_path = MODELS_DIR / f"{model_name}.joblib"

        if model_path.exists():
            available.append(model_name)

    return {"models": available}


def load_model(model_name: str):
    """Load a trained model from disk."""

    if model_name not in MODEL_NAMES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown model '{model_name}'. "
                f"Available models: {MODEL_NAMES}"
            ),
        )

    model_path = MODELS_DIR / f"{model_name}.joblib"

    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Trained model '{model_name}' not found.",
        )

    return joblib.load(model_path)


def validate_input_features(features: dict[str, Any]) -> None:
    """Validate that the request contains only known features."""

    unknown_features = set(features) - set(ALL_FEATURES)

    if unknown_features:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown feature(s): {sorted(unknown_features)}",
        )

    missing_features = set(ALL_FEATURES) - set(features)

    if missing_features:
        raise HTTPException(
            status_code=400,
            detail=f"Missing feature(s): {sorted(missing_features)}",
        )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict heart disease probability for one patient."""

    validate_input_features(request.features)

    model = load_model(request.model)

    patient = pd.DataFrame(
        [request.features],
        columns=ALL_FEATURES,
    )

    prediction = int(model.predict(patient)[0])
    probability = float(model.predict_proba(patient)[0, 1])

    return PredictionResponse(
        model=request.model,
        prediction=prediction,
        probability=probability,
    )