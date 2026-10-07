"""Feature definitions shared by training and serving."""

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# In the Pima dataset a 0 in these columns means "not measured", not a real
# reading (nobody has a BMI of 0). They are treated as missing and filled with
# the training median.
ZERO_MEANS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

LABELS = {
    "Pregnancies": "Pregnancies",
    "Glucose": "Glucose",
    "BloodPressure": "Blood pressure",
    "SkinThickness": "Skin thickness",
    "Insulin": "Insulin",
    "BMI": "BMI",
    "DiabetesPedigreeFunction": "Family history score",
    "Age": "Age",
}

# Probability cut-offs for the risk bands returned by the API.
MODERATE_FROM = 0.3
HIGH_FROM = 0.6


def risk_band(probability: float) -> str:
    if probability >= HIGH_FROM:
        return "high"
    if probability >= MODERATE_FROM:
        return "moderate"
    return "low"
