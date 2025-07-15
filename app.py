from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy

# Load the trained model
model = joblib.load("model.pkl")

# Define the expected input format
class HealthInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Health Risk Predictor API is running."}

@app.post("/predict")
def predict_risk(data: HealthInput):
    features = numpy.array([[ 
        data.Pregnancies, data.Glucose, data.BloodPressure, data.SkinThickness,
        data.Insulin, data.BMI, data.DiabetesPedigreeFunction, data.Age
    ]])
    
    prediction = model.predict(features)[0]

    return {
        "risk": "high" if prediction == 1 else "low",
        "prediction_raw": int(prediction)
    }