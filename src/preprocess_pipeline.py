import os
import json
import joblib

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# ============================================================
# CACAO PREPROCESSING PIPELINE
# ============================================================

def run_preprocessing():

    print(
        "[INFO] Starting Cacao Sklearn Pipeline Preprocessing..."
    )

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    data_path = "data/flavors_of_cacao.csv"

    df = pd.read_csv(
        data_path
    )

    print(
        f"[INFO] Original shape: {df.shape}"
    )

    # ========================================================
    # 2. NORMALIZE COLUMN NAMES
    # ========================================================

    df.columns = (
        df.columns
        .astype(str)
        .str.replace("\n", "_", regex=False)
        .str.replace("\r", "", regex=False)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    # ========================================================
    # 3. RENAME COLUMNS
    # ========================================================

    rename_map = {

        "company\xa0_(maker-if_known)":
            "company",

        "company_(maker-if_known)":
            "company",

        "company_maker_if_known":
            "company",

        "specific_bean_origin_or_bar_name":
            "bar_name",

        "ref_":
            "ref",

        "review_date":
            "review_year",

        "broad_bean_origin":
            "bean_origin"
    }

    df = df.rename(
        columns={
            key: value
            for key, value in rename_map.items()
            if key in df.columns
        }
    )

    # ========================================================
    # 4. DATA CLEANING
    # ========================================================

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

    df["review_year"] = pd.to_numeric(
        df["review_year"],
        errors="coerce"
    )

    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce"
    )

    # Remove rows where target is missing
    df = df.dropna(
        subset=["rating"]
    )

    # ========================================================
    # 5. FEATURES AND TARGET
    # ========================================================

    feature_columns = [
        "company",
        "bar_name",
        "review_year",
        "cocoa_percent",
        "company_location",
        "bean_type",
        "bean_origin"
    ]

    X = df[
        feature_columns
    ].copy()

    y = df[
        "rating"
    ].copy()

    # ========================================================
    # 6. TRAIN TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # ========================================================
    # 7. COLUMN TYPES
    # ========================================================

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

    # ========================================================
    # 8. NUMERICAL PIPELINE
    # ========================================================

    numeric_pipeline = Pipeline(
        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # ========================================================
    # 9. CATEGORICAL PIPELINE
    # ========================================================

    categorical_pipeline = Pipeline(
        steps=[

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
                    min_frequency=3,
                    sparse_output=False
                )
            )
        ]
    )

    # ========================================================
    # 10. COLUMN TRANSFORMER
    # ========================================================

    preprocessor = ColumnTransformer(
        transformers=[

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
        ]
    )

    # ========================================================
    # 11. FIT TRAIN DATA
    # ========================================================

    print(
        "[INFO] Fitting preprocessing pipeline..."
    )

    X_train_final = preprocessor.fit_transform(
        X_train
    )

    # ========================================================
    # 12. TRANSFORM TEST DATA
    # ========================================================

    print(
        "[INFO] Transforming test data..."
    )

    X_test_final = preprocessor.transform(
        X_test
    )

    # ========================================================
    # 13. CREATE DIRECTORIES
    # ========================================================

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    os.makedirs(
        "models",
        exist_ok=True
    )

    # ========================================================
    # 14. SAVE PROCESSED DATA
    # ========================================================

    np.save(
        "data/processed/X_train_final.npy",
        X_train_final
    )

    np.save(
        "data/processed/X_test_final.npy",
        X_test_final
    )

    np.save(
        "data/processed/y_train.npy",
        y_train.to_numpy()
    )

    np.save(
        "data/processed/y_test.npy",
        y_test.to_numpy()
    )

    # ========================================================
    # 15. SAVE PREPROCESSOR
    # ========================================================

    joblib.dump(
        preprocessor,
        "models/preprocessor.pkl"
    )

    # ========================================================
    # 16. SAVE METADATA
    # ========================================================

    metadata = {

        "dataset_name":
            "Flavors of Cacao",

        "train_shape":
            list(X_train_final.shape),

        "test_shape":
            list(X_test_final.shape),

        "target":
            "rating",

        "numerical_features":
            numeric_features,

        "categorical_features":
            categorical_features
    }

    with open(
        "data/processed/dataset_metadata.json",
        "w"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print(
        "[SUCCESS] Cacao preprocessing completed successfully!"
    )

    print(
        "[INFO] Preprocessor saved as "
        "models/preprocessor.pkl"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_preprocessing()