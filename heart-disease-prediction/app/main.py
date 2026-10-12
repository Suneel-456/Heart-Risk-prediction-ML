"""FastAPI service.  Run:  uvicorn app.main:app --reload"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import config
from src.predict import load_bundle, predict_one

app = FastAPI(title="Heart Disease Risk API", version="1.0.0",
              description="Screening support only - not a medical diagnosis.")


class Patient(BaseModel):
    age: int = Field(..., ge=18, le=100)
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type")
    trestbps: int = Field(..., ge=70, le=250, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., ge=80, le=700, description="Cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: int = Field(..., ge=50, le=230, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0, le=10)
    slope: int = Field(..., ge=0, le=2)
    ca: int = Field(..., ge=0, le=4)
    thal: int = Field(..., ge=0, le=3)


class Prediction(BaseModel):
    probability: float
    prediction: int
    risk_level: str
    threshold: float
    model: str


@app.get("/health")
def health():
    ok = config.MODEL_PATH.exists()
    return {"status": "ok" if ok else "model_missing"}


@app.post("/predict", response_model=Prediction)
def predict(patient: Patient):
    if not config.MODEL_PATH.exists():
        raise HTTPException(503, "Model not trained yet. Run: python -m src.train")
    return predict_one(patient.model_dump())


@app.get("/model-info")
def model_info():
    if not config.MODEL_PATH.exists():
        raise HTTPException(503, "Model not trained yet.")
    b = load_bundle()
    return {"model": b["model_name"], "threshold": b["threshold"], "metrics": b["metrics"]}
