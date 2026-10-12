"""Load the saved model and score patient records."""
from functools import lru_cache

import joblib
import pandas as pd

from src import config


@lru_cache(maxsize=1)
def load_bundle(path=config.MODEL_PATH):
    return joblib.load(path)


def risk_band(p: float, threshold: float) -> str:
    if p >= max(0.75, threshold + 0.25):
        return "high"
    return "elevated" if p >= threshold else "low"


def predict_one(record: dict) -> dict:
    b = load_bundle()
    X = pd.DataFrame([record])[b["features"]]
    p = float(b["pipeline"].predict_proba(X)[0, 1])
    return {
        "probability": round(p, 4),
        "prediction": int(p >= b["threshold"]),
        "risk_level": risk_band(p, b["threshold"]),
        "threshold": round(b["threshold"], 4),
        "model": b["model_name"],
    }


if __name__ == "__main__":
    sample = dict(age=58, sex=1, cp=0, trestbps=140, chol=260, fbs=0, restecg=1,
                  thalach=120, exang=1, oldpeak=2.0, slope=1, ca=2, thal=3)
    print(predict_one(sample))
