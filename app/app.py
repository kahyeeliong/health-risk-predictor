"""FastAPI service for the diabetes risk model.

Run from the repo root:
    uvicorn app.app:app --reload

Then open http://127.0.0.1:8000 for the web page, or /docs for Swagger.
"""

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.features import FEATURES, LABELS, ZERO_MEANS_MISSING, risk_band

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "app" / "model.joblib"
METRICS_PATH = ROOT / "app" / "metrics.json"
FRONTEND_DIR = ROOT / "frontend"

if not MODEL_PATH.exists():
    raise RuntimeError("No trained model found. Run `python -m app.train_model` first.")

model = joblib.load(MODEL_PATH)
metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}

# Pieces of the pipeline, used to explain each prediction.
imputer, scaler, classifier = model[0], model[1], model[-1]


class HealthInput(BaseModel):
    """One patient's measurements. Optional fields can be left out if unknown."""

    Pregnancies: int = Field(..., ge=0, le=20, description="Number of times pregnant")
    Glucose: float = Field(..., ge=40, le=400, description="Plasma glucose, 2h oral glucose tolerance test (mg/dL)")
    BloodPressure: Optional[float] = Field(None, ge=20, le=200, description="Diastolic blood pressure (mm Hg)")
    SkinThickness: Optional[float] = Field(None, ge=1, le=100, description="Triceps skin fold thickness (mm)")
    Insulin: Optional[float] = Field(None, ge=1, le=1000, description="2-hour serum insulin (μU/mL)")
    BMI: float = Field(..., ge=10, le=80, description="Body mass index (kg/m²)")
    DiabetesPedigreeFunction: Optional[float] = Field(
        None, ge=0, le=3, description="Family history score (higher = more diabetic relatives)"
    )
    Age: int = Field(..., ge=18, le=120, description="Age in years")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Pregnancies": 2,
                    "Glucose": 165,
                    "BloodPressure": 80,
                    "BMI": 36.5,
                    "DiabetesPedigreeFunction": 0.8,
                    "Age": 45,
                }
            ]
        }
    }


class Factor(BaseModel):
    feature: str
    label: str
    value: Optional[float]
    effect: str  # "raises" or "lowers"
    weight: float  # contribution to the log-odds vs an average patient


class Prediction(BaseModel):
    risk: str  # "low", "moderate" or "high"
    probability: float  # estimated probability of diabetes, 0 to 1
    factors: list[Factor]  # every feature, biggest influence first


app = FastAPI(
    title="Health Risk Predictor",
    description="Estimates diabetes risk from 8 health measurements and explains which ones drove the result.",
    version="2.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def to_frame(data: HealthInput) -> pd.DataFrame:
    row = pd.DataFrame([data.model_dump()], columns=FEATURES).astype(float)
    row[ZERO_MEANS_MISSING] = row[ZERO_MEANS_MISSING].replace(0, np.nan)
    return row


def explain(row: pd.DataFrame, data: HealthInput) -> list[Factor]:
    """Per-feature contribution for a logistic regression: coefficient times the
    standardised value. Zero means "same as the average training patient"."""
    z = scaler.transform(imputer.transform(row))[0]
    contributions = classifier.coef_[0] * z
    raw = data.model_dump()
    factors = [
        Factor(
            feature=f,
            label=LABELS[f],
            value=raw[f],
            effect="raises" if c > 0 else "lowers",
            weight=round(float(c), 3),
        )
        for f, c in zip(FEATURES, contributions)
    ]
    return sorted(factors, key=lambda x: abs(x.weight), reverse=True)


@app.post("/predict", response_model=Prediction)
def predict(data: HealthInput):
    row = to_frame(data)
    probability = float(model.predict_proba(row)[0, 1])
    return Prediction(
        risk=risk_band(probability),
        probability=round(probability, 3),
        factors=explain(row, data),
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    """How the model was trained and how well it scored on held-out data."""
    return metrics


# Serve the web page from the same server, so one command runs everything.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
