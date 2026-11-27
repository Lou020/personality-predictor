"""Single source of truth for paths, column names and constants.

Notebooks and app.py import from here, so nothing is defined twice.
"""
from pathlib import Path

# Paths (relative to the repo root, so it works from any working directory)
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "data.csv"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

# Reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
SCORING = "f1_macro"

# Experiment tracking (absolute paths, so notebooks and app use the same store)
MLFLOW_TRACKING_URI = f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACT_DIR = ROOT_DIR / "mlartifacts"
EXPERIMENT_NAME = "personality-predictor"

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

# Question texts for the app (from data/codebook.txt)
QUESTIONS = {
    "N1": "I get stressed out easily.",
    "N2": "I am relaxed most of the time.",
    "N3": "I worry about things.",
    "N4": "I seldom feel blue.",
    "N5": "I am easily disturbed.",
    "N6": "I get upset easily.",
    "N7": "I change my mood a lot.",
    "N8": "I have frequent mood swings.",
    "N9": "I get irritated easily.",
    "N10": "I often feel blue.",
    "E1": "I am the life of the party.",
    "E3": "I feel comfortable around people.",
    "E4": "I keep in the background.",
    "E5": "I start conversations.",
    "E7": "I talk to a lot of different people at parties.",
    "E9": "I don't mind being the center of attention.",
    "E10": "I am quiet around strangers.",
    "C4": "I make a mess of things.",
    "A4": "I sympathize with others' feelings.",
}
GENDER_OPTIONS = ["Female", "Male", "Other"]
HAND_OPTIONS = ["Right", "Left", "Both"]