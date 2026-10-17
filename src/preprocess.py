from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "flavors_of_cacao.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "processed_data.csv"


def preprocess_data():
    print("[INFO] Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"[INFO] Original shape: {df.shape}")

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

    # Rename columns
    rename_map = {
        "company_maker_if_known": "company",
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

    # Remove duplicates
    df = df.drop_duplicates()

    # Convert numeric columns
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

    df["ref"] = pd.to_numeric(
        df["ref"],
        errors="coerce"
    )

    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce"
    )

    # Remove rows where target is missing
    df = df.dropna(subset=["rating"])

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"[INFO] Processed shape: {df.shape}")
    print(f"[INFO] Processed data saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    preprocess_data()