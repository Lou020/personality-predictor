"""MLflow setup, shared by the tuning notebook and the app."""
import mlflow
import mlflow.sklearn

from personality.config import EXPERIMENT_NAME, MLFLOW_ARTIFACT_DIR, MLFLOW_TRACKING_URI

# Newer MLflow saves sklearn models with skops (safer than pickle) and needs
# to be told which extra types are safe to load.
SKOPS_TRUSTED_TYPES = ["numpy.dtype", "sklearn.tree._tree.Tree"]


def setup_mlflow() -> None:
    """Point MLflow to the project's database and select the experiment."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            EXPERIMENT_NAME, artifact_location=MLFLOW_ARTIFACT_DIR.as_uri()
        )
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_model(model) -> None:
    """Save the full fitted pipeline in the active MLflow run."""
    mlflow.sklearn.log_model(model, name="model", skops_trusted_types=SKOPS_TRUSTED_TYPES)


def load_model(run_id: str):
    """Load the full fitted pipeline (preprocessor + model) of an MLflow run."""
    setup_mlflow()
    return mlflow.sklearn.load_model(f"runs:/{run_id}/model")


def get_best_run_id(metric: str = "f1_macro") -> str:
    """Return the run id with the highest metric in the project experiment."""
    setup_mlflow()
    runs = mlflow.search_runs(order_by=[f"metrics.{metric} DESC"], max_results=1)
    if runs.empty:
        raise RuntimeError("No MLflow runs found. Run notebooks/03_tune_and_track.ipynb first.")
    return runs.loc[0, "run_id"]