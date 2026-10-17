
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "mlflow_model.joblib"
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

EXPERIMENT_NAME = "Cacao_Rating_Lab6_Registry"
REGISTERED_MODEL_NAME = "CacaoRatingRegressor"

# ============================================================
# 2. CHECK WHETHER THE SAVED MODEL EXISTS
# ============================================================

print("[INFO] Checking saved model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}\n"
        "Please run your existing training script first."
    )

print(f"[INFO] Model path: {MODEL_PATH}")

# ============================================================
# 3. LOAD THE EXISTING MODEL
# ============================================================

print("[INFO] Loading saved Cacao Rating model...")

pipeline = joblib.load(MODEL_PATH)

print("[SUCCESS] Model loaded successfully.")

# ============================================================
# 4. CONFIGURE MLFLOW
# ============================================================

tracking_uri = f"sqlite:///{MLFLOW_DB.as_posix()}"

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(EXPERIMENT_NAME)

client = MlflowClient(tracking_uri=tracking_uri)

print(f"[INFO] MLflow database: {MLFLOW_DB}")
print(f"[INFO] Experiment: {EXPERIMENT_NAME}")

# ============================================================
# 5. LOG AND REGISTER THE MODEL
# ============================================================

print("[INFO] Registering model in MLflow...")

with mlflow.start_run(
    run_name="Cacao_Rating_Model_Registration"
) as run:

    # Record model information
    mlflow.log_param(
        "source_model",
        "models/mlflow_model.joblib"
    )

    mlflow.log_param(
        "task",
        "regression"
    )

    mlflow.log_param(
        "target",
        "rating"
    )

    mlflow.log_param(
        "model_type",
        "RandomForestRegressor"
    )

    # Log the sklearn pipeline and register it
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        name="cacao_rating_model",
        registered_model_name=REGISTERED_MODEL_NAME,
        skops_trusted_types=[
            "numpy.dtype",
            "sklearn.tree._tree.Tree"
        ]
    )

    run_id = run.info.run_id

    print("[SUCCESS] Model logged successfully.")
    print(f"[INFO] MLflow run ID: {run_id}")

# ============================================================
# 6. FIND THE REGISTERED MODEL VERSION
# ============================================================

print("[INFO] Checking registered model versions...")

versions = client.search_model_versions(
    f"name='{REGISTERED_MODEL_NAME}'"
)

if not versions:
    raise RuntimeError(
        "Model registration failed: no registered versions found."
    )

latest_version = max(
    versions,
    key=lambda version: int(version.version)
)

# ============================================================
# 7. ASSIGN THE CANDIDATE ALIAS
# ============================================================

client.set_registered_model_alias(
    name=REGISTERED_MODEL_NAME,
    alias="candidate",
    version=latest_version.version
)

# ============================================================
# 8. VERIFY REGISTRATION
# ============================================================

registered_version = client.get_model_version_by_alias(
    name=REGISTERED_MODEL_NAME,
    alias="candidate"
)

print()
print("=" * 55)
print("       LAB 6: MODEL REGISTRATION RESULTS")
print("=" * 55)

print(f"Registered model : {REGISTERED_MODEL_NAME}")
print(f"Model version    : {registered_version.version}")
print("Model alias      : candidate")
print(f"Experiment       : {EXPERIMENT_NAME}")
print(f"Run ID           : {run_id}")
print(f"Tracking database: {MLFLOW_DB}")

print("=" * 55)
print("[SUCCESS] LAB 6 MODEL REGISTRATION COMPLETED.")
