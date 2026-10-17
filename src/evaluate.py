from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "flavors_of_cacao.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_PATH = OUTPUT_DIR / "evaluation_results.txt"


def evaluate_model():

    print("[INFO] Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    # Normalize column names
    df.columns = (
        df.columns
        .astype(str)
        .str.replace("\n", "_", regex=False)
        .str.replace("\r", "", regex=False)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

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

    df = df.rename(
        columns={k: v for k, v in rename_map.items() if k in df.columns}
    )

    # Clean numeric columns
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

    df = df.dropna(subset=["rating"])

    feature_columns = [
        "company",
        "bar_name",
        "review_year",
        "cocoa_percent",
        "company_location",
        "bean_type",
        "bean_origin",
    ]

    X = df[feature_columns]
    y = df["rating"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("[INFO] Loading trained model...")

    model = joblib.load(MODEL_PATH)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\n================ EVALUATION RESULTS ================\n")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

    with open(RESULT_PATH, "w") as file:
        file.write("Cacao Rating Model Evaluation\n")
        file.write("=============================\n")
        file.write(f"MAE  : {mae:.4f}\n")
        file.write(f"RMSE : {rmse:.4f}\n")
        file.write(f"R2   : {r2:.4f}\n")

    print(f"\n[INFO] Results saved to: {RESULT_PATH}")


if __name__ == "__main__":
    evaluate_model()