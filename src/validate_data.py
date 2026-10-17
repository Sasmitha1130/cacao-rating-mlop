import os
import pandas as pd


DATA_PATH = "data/flavors_of_cacao.csv"


def load_data():

    print("[INFO] Loading Cacao dataset...")

    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Dataset not found: {DATA_PATH}")
        return None

    df = pd.read_csv(DATA_PATH)

    print(f"[INFO] Original shape: {df.shape}")

    # Normalize column names
    df.columns = (
        df.columns
        .astype(str)
        .str.replace("\n", "_", regex=False)
        .str.replace("\r", "", regex=False)
        .str.replace("\xa0", " ", regex=False)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    # Rename columns to the names used by the project
    rename_map = {
        "company__(maker-if_known)": "company",
    "company_(maker-if_known)": "company",
    "company_maker-if_known": "company",
    "specific_bean_origin_or_bar_name": "bar_name",
    "ref_": "ref",
    "review_date": "review_year",
    "broad_bean_origin": "bean_origin"
}

    df.rename(
        columns=rename_map,
        inplace=True
    )

    print("[INFO] Final columns:")
    print(list(df.columns))

    return df


def validate_data(df):

    print()
    print("[INFO] Starting data validation...")

    errors = []

    # ---------------------------------------------------------
    # 1. Check required columns
    # ---------------------------------------------------------

    required_columns = [
        "company",
        "bar_name",
        "ref",
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

        errors.append(
            f"Missing columns: {missing_columns}"
        )

    else:

        print("[PASS] All required columns are present.")

    # ---------------------------------------------------------
    # 2. Check number of records
    # ---------------------------------------------------------

    if len(df) == 0:

        errors.append(
            "Dataset contains zero records."
        )

    else:

        print(
            f"[PASS] Dataset contains {len(df)} records."
        )

    # ---------------------------------------------------------
    # 3. Check rating
    # ---------------------------------------------------------

    if "rating" in df.columns:

        rating = pd.to_numeric(
            df["rating"],
            errors="coerce"
        )

        invalid_rating = (
            rating.dropna()
            .loc[
                (rating.dropna() < 0) |
                (rating.dropna() > 5)
            ]
        )

        if len(invalid_rating) > 0:

            errors.append(
                f"Invalid rating values found: "
                f"{len(invalid_rating)}"
            )

        else:

            print(
                "[PASS] Rating values are within 0-5."
            )

    # ---------------------------------------------------------
    # 4. Check cocoa percentage
    # ---------------------------------------------------------

    if "cocoa_percent" in df.columns:

        cocoa = (
            df["cocoa_percent"]
            .astype(str)
            .str.replace("%", "", regex=False)
        )

        cocoa = pd.to_numeric(
            cocoa,
            errors="coerce"
        )

        invalid_cocoa = (
            cocoa.dropna()
            .loc[
                (cocoa.dropna() < 0) |
                (cocoa.dropna() > 100)
            ]
        )

        if len(invalid_cocoa) > 0:

            errors.append(
                f"Invalid cocoa percentage values found: "
                f"{len(invalid_cocoa)}"
            )

        else:

            print(
                "[PASS] Cocoa percentage values are valid."
            )

    # ---------------------------------------------------------
    # 5. Check duplicate rows
    # ---------------------------------------------------------

    duplicates = df.duplicated().sum()

    print(
        f"[INFO] Duplicate rows found: {duplicates}"
    )

    # ---------------------------------------------------------
    # 6. Check missing values
    # ---------------------------------------------------------

    print("[INFO] Missing values:")

    missing_values = df.isnull().sum()

    print(
        missing_values.to_string()
    )

    # ---------------------------------------------------------
    # FINAL RESULT
    # ---------------------------------------------------------

    if errors:

        print()
        print("[ERROR] Data Validation FAILED.")

        for error in errors:
            print(f"[ERROR] {error}")

        os.makedirs(
            "artifacts",
            exist_ok=True
        )

        report = pd.DataFrame({
            "validation_errors": errors
        })

        report_path = (
            "artifacts/cacao_validation_report.csv"
        )

        report.to_csv(
            report_path,
            index=False
        )

        print(
            f"[INFO] Validation report saved to: "
            f"{report_path}"
        )

        return False

    print()
    print("[SUCCESS] Data Validation PASSED.")
    print("[SUCCESS] Dataset is ready for preprocessing.")

    return True


if __name__ == "__main__":

    df = load_data()

    if df is None:
        raise SystemExit(1)

    if not validate_data(df):
        raise SystemExit(1)