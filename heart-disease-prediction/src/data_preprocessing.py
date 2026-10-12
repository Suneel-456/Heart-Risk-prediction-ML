"""Load, validate, clean and split the raw heart-disease data."""
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config

log = logging.getLogger(__name__)


def load_raw(path=config.RAW_DATA) -> pd.DataFrame:
    """Read the CSV and make target mean 1 = disease present."""
    df = pd.read_csv(path)
    if config.RAW_TARGET_INVERTED:
        df[config.TARGET] = 1 - df[config.TARGET]
    return df


def validate(df: pd.DataFrame) -> None:
    missing = set(config.FEATURES + [config.TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if not df[config.TARGET].isin([0, 1]).all():
        raise ValueError("Target must be binary (0/1)")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows and rows with missing values."""
    before = len(df)
    df = df.drop_duplicates().dropna().reset_index(drop=True)
    log.info("Cleaning removed %d rows (%d -> %d)", before - len(df), before, len(df))
    return df


def split(df: pd.DataFrame):
    """Stratified train/test split so both sets keep the class balance."""
    X = df[config.FEATURES]
    y = df[config.TARGET]
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )


def prepare():
    df = load_raw()
    validate(df)
    df = clean(df)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.PROCESSED_DIR / "heart_clean.csv", index=False)
    return split(df)
