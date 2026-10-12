"""Feature engineering and the sklearn preprocessing pipeline."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """hr_reserve_pct: achieved max heart rate as a share of the age-predicted max (220 - age)."""
    df = df.copy()
    df["hr_reserve_pct"] = df["thalach"] / (220 - df["age"])
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), config.NUMERIC + config.ENGINEERED_NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL),
        ]
    )


def build_pipeline(model) -> Pipeline:
    """Feature engineering + scaling/encoding + estimator in one object, so the
    exact same transformations run at training time and in the API."""
    return Pipeline(
        [
            ("features", FunctionTransformer(add_features)),
            ("prep", build_preprocessor()),
            ("clf", model),
        ]
    )
