"""Single source of truth for paths, column names and constants.

Notebooks and app.py import from here, so nothing is defined twice.
"""
from pathlib import Path

# Paths (relative to the repo root, so it works from any working directory)
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "data.csv"
MODEL_DIR = ROOT_DIR / "models"
BEST_MODEL_PATH = MODEL_DIR / "best_model.joblib"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

# Reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Columns
TARGET = "target"
ITEM_COLS = [
    "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9", "N10",
    "E1", "E3", "E4", "E5", "E7", "E9", "E10",
    "C4", "A4",
]
NUMERIC_FEATURES = ITEM_COLS + ["age"]
CATEGORICAL_FEATURES = ["gender", "hand"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Valid value ranges (used for cleaning and for the app inputs)
ITEM_MIN, ITEM_MAX = 1, 5
AGE_MIN, AGE_MAX = 13, 100

# Fixed class order, so every chart and confusion matrix looks the same
CLASS_ORDER = ["Moderate", "Resilient", "Overcontroller", "Undercontroller"]
