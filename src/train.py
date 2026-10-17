"""
Improved training script for Cacao Rating Regression.

This version keeps the same project structure and dataset columns described
in the supplied project files, but improves model selection and tuning.

Run from the project root:
    python src/train_improved.py

It saves the selected model to:
    models/best_model.joblib
"""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# LOCATION 1: PROJECT PATHS + DATA LOADING
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "flavors_of_cacao.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "best_model.joblib"

df = pd.read_csv(DATA_PATH)

# Normalize column names so the script works with the supplied dataset format.
df.columns = (
    df.columns
    .astype(str)
    .str.replace("\n", "_", regex=False)
    .str.replace("\r", "", regex=False)
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
)

# Handle common Flavors of Cacao column spellings.
rename_map = {
    "company_maker_if_known": "company",
    "company\xa0_(maker-if_known)": "company",
    "company_(maker-if_known)": "company",
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
df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

required = [
    "company", "bar_name", "ref", "review_year",
    "cocoa_percent", "company_location", "bean_type",
    "bean_origin", "rating"
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(
        f"Missing expected columns: {missing}\n"
        f"Columns found: {list(df.columns)}"
    )

# ============================================================
# LOCATION 2: DATA CLEANING
# ============================================================

df = df.drop_duplicates().copy()

# Convert cocoa percentage such as "70%" to numeric 70.0.
df["cocoa_percent"] = (
    df["cocoa_percent"]
    .astype(str)
    .str.replace("%", "", regex=False)
    .str.strip()
)
df["cocoa_percent"] = pd.to_numeric(df["cocoa_percent"], errors="coerce")

df["review_year"] = pd.to_numeric(df["review_year"], errors="coerce")
df["ref"] = pd.to_numeric(df["ref"], errors="coerce")
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

df = df.dropna(subset=["rating"])

# ============================================================
# LOCATION 3: FEATURE SELECTION
# IMPORTANT CHANGE:
# REF is treated as an identifier rather than a quality feature.
# ============================================================

feature_columns = [
    "company",
    "bar_name",
    "review_year",
    "cocoa_percent",
    "company_location",
    "bean_type",
    "bean_origin",
]

X = df[feature_columns].copy()
y = df["rating"].copy()

# ============================================================
# LOCATION 4: TRAIN/TEST SPLIT
# Keep the original 80/20 methodology.
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

numeric_features = [
    "review_year",
    "cocoa_percent",
]

categorical_features = [
    "company",
    "bar_name",
    "company_location",
    "bean_type",
    "bean_origin",
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=3
        )
    ),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ]
)

# ============================================================
# LOCATION 5: MODEL DEFINITIONS
# IMPORTANT CHANGE:
# Add ExtraTrees and use stronger Random Forest settings.
# ============================================================

models = {
    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features=0.8,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]),

    "Extra Trees": Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            ExtraTreesRegressor(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features=1.0,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]),

    "Gradient Boosting": Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.03,
                max_depth=3,
                min_samples_leaf=3,
                loss="huber",
                random_state=42,
            ),
        ),
    ]),
}

# ============================================================
# LOCATION 6: CROSS-VALIDATION + HYPERPARAMETER SEARCH
# IMPORTANT CHANGE:
# Optimize directly for R² instead of RMSE.
# ============================================================

cv = KFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 15],
    "model__min_samples_split": [2],
    "model__min_samples_leaf": [1],
    "model__max_features": [0.8],
}

search = GridSearchCV(
    estimator=models["Random Forest"],
    param_grid=param_grid,
    scoring="r2",
    cv=cv,
    n_jobs=-1,
    verbose=1,
)

search.fit(X_train, y_train)

best_model = search.best_estimator_
best_cv_r2 = search.best_score_

# ============================================================
# LOCATION 7: COMPARE FINAL CANDIDATES
# ============================================================

results = []

# Tuned Random Forest
rf_pred = best_model.predict(X_test)
results.append({
    "Model": "Tuned Random Forest",
    "MAE": mean_absolute_error(y_test, rf_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, rf_pred)),
    "R2": r2_score(y_test, rf_pred),
})

# Extra Trees
extra = models["Extra Trees"]
extra.fit(X_train, y_train)
extra_pred = extra.predict(X_test)
results.append({
    "Model": "Extra Trees",
    "MAE": mean_absolute_error(y_test, extra_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, extra_pred)),
    "R2": r2_score(y_test, extra_pred),
})

# Gradient Boosting
gb = models["Gradient Boosting"]
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
results.append({
    "Model": "Gradient Boosting",
    "MAE": mean_absolute_error(y_test, gb_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, gb_pred)),
    "R2": r2_score(y_test, gb_pred),
})

results_df = pd.DataFrame(results).sort_values("R2", ascending=False)

print("\n================ MODEL RESULTS ================\n")
print(results_df.to_string(index=False))

print("\nBest Random Forest CV R2:", round(best_cv_r2, 4))
print("Best test R2:", round(results_df.iloc[0]["R2"], 4))

# ============================================================
# LOCATION 8: SAVE THE BEST MODEL
# IMPORTANT:
# app.py already expects models/best_model.joblib.
# ============================================================

best_name = results_df.iloc[0]["Model"]

if best_name == "Tuned Random Forest":
    final_model = best_model
elif best_name == "Extra Trees":
    final_model = extra
else:
    final_model = gb

joblib.dump(final_model, MODEL_PATH)

print(f"\nSaved best model: {MODEL_PATH}")
print("Run the Streamlit app normally after training.")
