import os
import json
import numpy as np


def validate_preprocessing_outputs():

    print("[INFO] Validating Cacao Preprocessing Outputs...")

    try:
        X_train = np.load("data/processed/X_train_final.npy")
        X_test = np.load("data/processed/X_test_final.npy")
        y_train = np.load("data/processed/y_train.npy")
        y_test = np.load("data/processed/y_test.npy")

    except FileNotFoundError as error:
        print(f"[ERROR] Processed file not found: {error}")
        return False

    errors = []

    # Check for missing values
    if np.isnan(X_train).sum() > 0:
        errors.append("NaN values found in X_train.")

    if np.isnan(X_test).sum() > 0:
        errors.append("NaN values found in X_test.")

    # Check feature dimensions
    if X_train.shape[1] != X_test.shape[1]:
        errors.append(
            "X_train and X_test have different numbers of features."
        )

    # Check training rows
    if X_train.shape[0] != y_train.shape[0]:
        errors.append(
            "X_train and y_train have different numbers of rows."
        )

    # Check testing rows
    if X_test.shape[0] != y_test.shape[0]:
        errors.append(
            "X_test and y_test have different numbers of rows."
        )

    report = {
        "validation_status": "PASSED" if not errors else "FAILED",

        "shapes": {
            "X_train": list(X_train.shape),
            "X_test": list(X_test.shape),
            "y_train": list(y_train.shape),
            "y_test": list(y_test.shape)
        },

        "missing_values": {
            "X_train": int(np.isnan(X_train).sum()),
            "X_test": int(np.isnan(X_test).sum())
        },

        "errors": errors
    }

    os.makedirs("artifacts", exist_ok=True)

    report_path = (
        "artifacts/preprocessing_summary_report.json"
    )

    with open(report_path, "w") as file:
        json.dump(report, file, indent=4)

    if errors:

        print("[ERROR] Output Validation FAILED!")

        for error in errors:
            print(f" - {error}")

        return False

    print("[SUCCESS] Output Validation PASSED.")
    print("[SUCCESS] Features are clean and dimensionally consistent.")
    print(f"[INFO] Report saved to: {report_path}")

    return True


if __name__ == "__main__":

    success = validate_preprocessing_outputs()

    if not success:
        raise SystemExit(1)