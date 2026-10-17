from pathlib import Path
import warnings

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "flavors_of_cacao.csv"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = MODEL_DIR / "mlflow_model.joblib"

# MLflow SQLite database
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("[INFO] Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"[INFO] Original shape: {df.shape}")


# ============================================================
# 3. NORMALIZE COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .astype(str)
    .str.replace("\n", "_", regex=False)
    .str.replace("\r", "", regex=False)
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
)

print("[INFO] Columns after normalization:")
print(list(df.columns))


# ============================================================
# 4. RENAME DATASET COLUMNS
# ============================================================

rename_map = {
    "company\xa0_(maker-if_known)": "company",
    "company_(maker-if_known)": "company",
    "company_maker_if_known": "company",
    "company": "company",

    "specific_bean_origin_or_bar_name": "bar_name",

    "ref_": "ref",

    "review_date": "review_year",

    "cocoa_percent": "cocoa_percent",

    "company_location": "company_location",

    "bean_type": "bean_type",

    "broad_bean_origin": "bean_origin",

    "rating": "rating",
}

df = df.rename(
    columns={
        key: value
        for key, value in rename_map.items()
        if key in df.columns
    }
)


# ============================================================
# 5. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "company",
    "bar_name",
    "review_year",
    "cocoa_percent",
    "company_location",
    "bean_type",
    "bean_origin",
    "rating"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing expected columns: {missing_columns}\n"
        f"Columns found: {list(df.columns)}"
    )


# ============================================================
# 6. DATA CLEANING
# ============================================================

df = df.drop_duplicates().copy()

# Convert cocoa percentage such as 70% -> 70.0
df["cocoa_percent"] = (
    df["cocoa_percent"]
    .astype(str)
    .str.replace("%", "", regex=False)
    .str.strip()
)

df["cocoa_percent"] = pd.to_numeric(
    df["cocoa_percent"],
    errors="coerce"
)

# Convert year
df["review_year"] = pd.to_numeric(
    df["review_year"],
    errors="coerce"
)

# Convert target rating
df["rating"] = pd.to_numeric(
    df["rating"],
    errors="coerce"
)

# Remove rows without target
df = df.dropna(
    subset=["rating"]
)

print(f"[INFO] Cleaned shape: {df.shape}")


# ============================================================
# 7. FEATURE SELECTION
# ============================================================

feature_columns = [
    "company",
    "bar_name",
    "review_year",
    "cocoa_percent",
    "company_location",
    "bean_type",
    "bean_origin"
]

X = df[feature_columns].copy()
y = df["rating"].copy()


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(
    f"[INFO] Training samples: {len(X_train)}"
)

print(
    f"[INFO] Testing samples: {len(X_test)}"
)


# ============================================================
# 9. FEATURE TYPES
# ============================================================

numeric_features = [
    "review_year",
    "cocoa_percent"
]

categorical_features = [
    "company",
    "bar_name",
    "company_location",
    "bean_type",
    "bean_origin"
]


# ============================================================
# 10. PREPROCESSING PIPELINES
# ============================================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    )
])

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),
    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=3
        )
    )
])


preprocessor = ColumnTransformer([
    (
        "num",
        numeric_pipeline,
        numeric_features
    ),
    (
        "cat",
        categorical_pipeline,
        categorical_features
    )
])


# ============================================================
# 11. RANDOM FOREST MODEL
# ============================================================

n_estimators = 200
max_depth = 15
min_samples_split = 2
min_samples_leaf = 1
random_state = 42

model = RandomForestRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    random_state=random_state,
    n_jobs=-1
)


# ============================================================
# 12. COMPLETE ML PIPELINE
# ============================================================

pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        model
    )
])


# ============================================================
# 13. CONFIGURE MLFLOW
# ============================================================

# Use SQLite instead of the deprecated filesystem backend.
mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB}"
)

mlflow.set_experiment(
    "Cacao_Rating_Lab4"
)


# ============================================================
# 14. START MLFLOW RUN
# ============================================================

with mlflow.start_run():

    print("[INFO] Starting MLflow run...")

    # --------------------------------------------------------
    # Log parameters
    # --------------------------------------------------------

    mlflow.log_param(
        "model",
        "RandomForestRegressor"
    )

    mlflow.log_param(
        "n_estimators",
        n_estimators
    )

    mlflow.log_param(
        "max_depth",
        max_depth
    )

    mlflow.log_param(
        "min_samples_split",
        min_samples_split
    )

    mlflow.log_param(
        "min_samples_leaf",
        min_samples_leaf
    )

    mlflow.log_param(
        "random_state",
        random_state
    )

    mlflow.log_param(
        "test_size",
        0.20
    )

    mlflow.log_param(
        "feature_count",
        len(feature_columns)
    )


    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    pipeline.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = pipeline.predict(
        X_test
    )


    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )


    # --------------------------------------------------------
    # Log metrics to MLflow
    # --------------------------------------------------------

    mlflow.log_metric(
        "MAE",
        float(mae)
    )

    mlflow.log_metric(
        "RMSE",
        float(rmse)
    )

    mlflow.log_metric(
        "R2",
        float(r2)
    )


    # --------------------------------------------------------
    # Save model locally
    # --------------------------------------------------------

    joblib.dump(
        pipeline,
        MODEL_PATH
    )


    # --------------------------------------------------------
    # Log model to MLflow
    # --------------------------------------------------------

    mlflow.sklearn.log_model(
        pipeline,
        name="cacao_rating_model",
        skops_trusted_types=[
            "numpy.dtype",
            "sklearn.tree._tree.Tree"
        ]
    )


    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print()
    print(
        "================ MLFLOW RESULTS ================"
    )
    print()

    print(
        f"MAE  : {mae:.4f}"
    )

    print(
        f"RMSE : {rmse:.4f}"
    )

    print(
        f"R2   : {r2:.4f}"
    )

    print()

    print(
        f"[INFO] Local model saved to:"
    )

    print(
        MODEL_PATH
    )

    print()

    print(
        "[INFO] MLflow Run ID:"
    )

    print(
        mlflow.active_run().info.run_id
    )


print()
print(
    "[SUCCESS] MLflow experiment completed!"
)