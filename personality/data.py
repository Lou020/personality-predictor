"""Loading and one-time cleaning of the raw training data."""
import numpy as np
import pandas as pd

from personality.config import (
    AGE_MAX, AGE_MIN, DATA_PATH, FEATURES, ITEM_COLS, ITEM_MAX, ITEM_MIN, TARGET,
)


def load_raw_data() -> pd.DataFrame:
    """Read the raw dataset from data/data.csv."""
    return pd.read_csv(DATA_PATH)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning decisions from the EDA."""
    df = df.copy()

    # 1. invalid ages (birth years, garbage) -> NaN, imputed later in the pipeline
    invalid_age = ~df["age"].between(AGE_MIN, AGE_MAX)
    df.loc[invalid_age, "age"] = np.nan

    # 2. drop rows with any answer outside 1..5 (the all-zero row)
    valid_answers = df[ITEM_COLS].apply(lambda col: col.between(ITEM_MIN, ITEM_MAX)).all(axis=1)
    df = df[valid_answers]

    # 3. drop exact duplicates
    df = df.drop_duplicates()
    return df.reset_index(drop=True)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return X (features) and y (target)."""
    return df[FEATURES], df[TARGET]