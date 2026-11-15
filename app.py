"""FastAPI inference API for the synthetic-data demo model."""
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "fraud_detection_pipeline.joblib"
app = FastAPI(title="Dynamic Fraud Detection API", version="1.0.0",
              description="Educational API trained on synthetic data.")
bundle = None

class Transaction(BaseModel):
    amount: float = Field(gt=0, le=1_000_000)
    hour: int = Field(ge=0, le=23)
    distance_from_home_km: float = Field(ge=0, le=20_000)
    transactions_last_24h: int = Field(ge=0, le=500)
    merchant_risk_score: float = Field(ge=0, le=1)

@app.on_event("startup")
def load_model():
    global bundle
    if MODEL_PATH.exists():
        bundle = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bundle is not None,
            "note": "Demo model trained on synthetic data only."}

@app.post("/predict")
def predict(transaction: Transaction):
    if bundle is None:
        raise HTTPException(status_code=503,
            detail="Model missing. Run generate_data.py and train_model.py first.")
    row = pd.DataFrame([transaction.model_dump()], columns=bundle["features"])
    model = bundle["pipeline"]
    label = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1])
    return {
        "prediction": label,
        "label": "potential_fraud" if label else "likely_legitimate",
        "fraud_probability": round(probability, 4),
        "disclaimer": "Educational prediction from synthetic data; not financial advice."
    }
