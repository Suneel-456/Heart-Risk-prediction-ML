"""Load the latest registered model from the MLflow Model Registry and score a sample.

Run:  python -m src.mlflow_load
"""
import mlflow
import mlflow.sklearn
import pandas as pd

from src import config

SAMPLE = dict(age=58, sex=1, cp=0, trestbps=140, chol=260, fbs=0, restecg=1,
              thalach=120, exang=1, oldpeak=2.0, slope=1, ca=2, thal=3)

if __name__ == "__main__":
    mlflow.set_tracking_uri(config.MLFLOW_URI)
    uri = f"models:/{config.MLFLOW_MODEL_NAME}/latest"
    model = mlflow.sklearn.load_model(uri)
    p = model.predict_proba(pd.DataFrame([SAMPLE]))[0, 1]
    print(f"Loaded {uri}\nProbability of heart disease: {p:.3f}")
