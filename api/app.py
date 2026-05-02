import joblib
import numpy as np
import shap

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =========================
# Load model + scaler
# =========================
model = joblib.load("../models/fraud_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

# SHAP explainer
explainer = shap.TreeExplainer(model)

# =========================
# Initialize API
# =========================
app = FastAPI(title="Fraud Detection API")

# ✅ CORS FIX
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Input schema
# =========================
class Transaction(BaseModel):
    features: list[float]

# =========================
# Home
# =========================
@app.get("/")
def home():
    return {"message": "Fraud Detection API running 🚀"}

# =========================
# Predict
# =========================
@app.post("/predict")
def predict(tx: Transaction):

    data = np.array([tx.features])

    # Scale
    data_scaled = scaler.transform(data)

    # Predict
    prob = model.predict_proba(data_scaled)[0][1]
    pred = int(prob >= 0.5)

    # Decision
    if prob > 0.8:
        decision = "REVIEW"
    elif prob > 0.5:
        decision = "FLAG"
    else:
        decision = "ALLOW"

    # =========================
    # SHAP (FIXED)
    # =========================
    shap_values = explainer.shap_values(data_scaled)

    if isinstance(shap_values, list):
        contributions = shap_values[1][0]
    else:
        contributions = shap_values[0]

    contributions = np.array(contributions).flatten()

    # Top 5 features
    top_indices = np.argsort(np.abs(contributions))[-5:]

    top_features = {}
    for i in top_indices:
        value = contributions[i]
        if isinstance(value, np.ndarray):
            value = value.item()
        top_features[f"feature_{i}"] = float(value)

    return {
        "fraud_probability": float(prob),
        "prediction": pred,
        "decision": decision,
        "top_risk_factors": top_features
    }