from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "flavors_of_cacao.csv"

COLUMN_MAPPING = {
    "Company\u00a0\n(Maker-if known)": "company",
    "Specific Bean Origin\nor Bar Name": "bar_name",
    "REF": "ref",
    "Review\nDate": "review_year",
    "Cocoa\nPercent": "cocoa_percent",
    "Company\nLocation": "company_location",
    "Rating": "rating",
    "Bean\nType": "bean_type",
    "Broad Bean\nOrigin": "bean_origin",
}


def load_and_clean_data(path=DEFAULT_DATA_PATH):
    """Load the raw dataset and perform deterministic cleaning."""
    df = pd.read_csv(path)
    df = df.rename(columns=COLUMN_MAPPING)

    expected = set(COLUMN_MAPPING.values())
    missing_columns = expected.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")

    df = df.drop_duplicates().copy()
    df["cocoa_percent"] = (
        df["cocoa_percent"].astype(str).str.replace("%", "", regex=False)
    )
    df["cocoa_percent"] = pd.to_numeric(df["cocoa_percent"], errors="coerce")

    text_columns = ["company", "bar_name", "company_location", "bean_type", "bean_origin"]
    for column in text_columns:
        df[column] = df[column].astype("string").str.strip()
        df[column] = df[column].replace({"": pd.NA, "\u00a0": pd.NA})
        # scikit-learn imputers expect np.nan rather than pandas' scalar pd.NA.
        df[column] = df[column].astype(object).where(df[column].notna(), np.nan)

    for column in ["ref", "review_year", "rating"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["rating"])
    return df


def split_features_target(df):
    """Return model inputs and prediction target."""
    feature_columns = [
        "company",
        "bar_name",
        "ref",
        "review_year",
        "cocoa_percent",
        "company_location",
        "bean_type",
        "bean_origin",
    ]
    return df[feature_columns], df["rating"]
