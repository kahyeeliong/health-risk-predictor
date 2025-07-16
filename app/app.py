from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import joblib
import numpy
import os

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(model_path)

# Define the expected input format
class HealthInput(BaseModel):
    Pregnancies: int = Field(..., ge=0)
    Glucose: float = Field(..., ge=0)
    BloodPressure: float = Field(..., ge=0)
    SkinThickness: float = Field(..., ge=0)
    Insulin: float = Field(..., ge=0)
    BMI: float = Field(..., ge=0)
    DiabetesPedigreeFunction: float = Field(..., ge=0)
    Age: int = Field(..., ge=0)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "My custom health risk API — Kah Yee"}

@app.post("/predict")
def predict_risk(data: HealthInput):
    features = numpy.array([[ 
        data.Pregnancies, data.Glucose, data.BloodPressure, data.SkinThickness,
        data.Insulin, data.BMI, data.DiabetesPedigreeFunction, data.Age
    ]])

    # Get predicted probability of class 1 (diabetes)
    proba = model.predict_proba(features)[0][1]  # confidence of positive class

    if proba >= 0.7:
        risk_level = "high"
    elif proba >= 0.4:
        risk_level = "moderate"
    else:
        risk_level = "low"

    return {
        "risk": risk_level,
        "confidence": round(proba, 2),
        "prediction_raw": int(model.predict(features)[0])
    }

@app.options("/predict")
def preflight_handler():
    return JSONResponse(
        content=None,
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        },
    )