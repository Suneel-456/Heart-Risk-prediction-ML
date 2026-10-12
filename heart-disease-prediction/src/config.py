"""Central configuration: paths, column groups and training settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "heart.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
EXCEL_PATH = ROOT / "data" / "heart_disease_data.xlsx"
MODEL_PATH = MODELS_DIR / "heart_model.joblib"

MLFLOW_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
MLFLOW_EXPERIMENT = "heart-disease-prediction"
MLFLOW_MODEL_NAME = "heart-disease-classifier"

TARGET = "target"  # after loading: 1 = heart disease present, 0 = absent
# In the public CSV copy of the Cleveland data used here, target is INVERTED
# (1 = no disease). Verified from the data: target=1 rows have higher max heart
# rate and lower ST depression, with no exercise angina. load_raw() flips it.
RAW_TARGET_INVERTED = True
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Minimum recall we want from the screening model. Missing a sick patient is
# costlier than a false alarm, so the decision threshold is tuned for recall.
TARGET_RECALL = 0.90

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
ENGINEERED_NUMERIC = ["hr_reserve_pct"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = NUMERIC + CATEGORICAL

FEATURE_DESCRIPTIONS = {
    "age": ("Age in years", "numeric"),
    "sex": ("Sex (1 = male, 0 = female)", "binary"),
    "cp": ("Chest pain type (0-3)", "categorical"),
    "trestbps": ("Resting blood pressure (mm Hg)", "numeric"),
    "chol": ("Serum cholesterol (mg/dl)", "numeric"),
    "fbs": ("Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)", "binary"),
    "restecg": ("Resting ECG result (0-2)", "categorical"),
    "thalach": ("Maximum heart rate achieved", "numeric"),
    "exang": ("Exercise-induced angina (1 = yes, 0 = no)", "binary"),
    "oldpeak": ("ST depression induced by exercise relative to rest", "numeric"),
    "slope": ("Slope of the peak exercise ST segment (0-2)", "categorical"),
    "ca": ("Number of major vessels coloured by fluoroscopy (0-4)", "categorical"),
    "thal": ("Thalassemia test result (0-3)", "categorical"),
    "target": ("Heart disease present (1) or absent (0). Flipped from the source CSV, where 1 meant no disease", "target"),
}
