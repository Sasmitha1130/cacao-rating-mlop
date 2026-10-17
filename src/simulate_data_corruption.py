
"""Simulate data corruption and report detected data-quality failures."""

from pathlib import Path
import json

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "flavors_of_cacao.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def validate_dataset(df: pd.DataFrame) -> list[str]:
    """Return detected data-quality errors."""
    errors = []

    if df.empty:
        errors.append("Dataset is empty.")

    if "Rating" not in df.columns:
        errors.append("Required column 'Rating' is missing.")
    else:
        ratings = pd.to_numeric(df["Rating"], errors="coerce")
        if ratings.isna().any():
            errors.append("Rating contains missing or non-numeric values.")

    cocoa_column = "Cocoa\nPercent"
    if cocoa_column in df.columns:
        cocoa = (
            df[cocoa_column]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        cocoa = pd.to_numeric(cocoa, errors="coerce")

        if cocoa.isna().any():
            errors.append("Cocoa percentage contains invalid values.")

        if ((cocoa < 0) | (cocoa > 100)).any():
            errors.append("Cocoa percentage is outside the 0–100 range.")

    return errors



def simulate_corruption(df: pd.DataFrame) -> pd.DataFrame:
    """Create a corrupted copy without modifying the original dataset."""
    corrupted = df.copy()

    if corrupted.empty:
        raise ValueError("Cannot corrupt an empty dataset.")

    # Convert Rating to object so it can contain deliberately invalid text.
    if "Rating" in corrupted.columns:
        corrupted["Rating"] = corrupted["Rating"].astype(object)
        corrupted.loc[corrupted.index[0], "Rating"] = "INVALID_RATING"

    return corrupted



def run_simulation(input_path: Path = DEFAULT_INPUT) -> dict:
    """Run the corruption test and save its reports."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    original = pd.read_csv(input_path)
    corrupted = simulate_corruption(original)
    errors = validate_dataset(corrupted)

    corrupted_path = ARTIFACTS_DIR / "corrupted_chocolate_data.csv"
    errors_path = ARTIFACTS_DIR / "corrupted_simulation_errors.csv"
    report_path = ARTIFACTS_DIR / "failure_simulation_report.json"

    corrupted.to_csv(corrupted_path, index=False)

    pd.DataFrame(
        [{"error": error} for error in errors],
        columns=["error"],
    ).to_csv(errors_path, index=False)

    report = {
        "input_file": str(input_path.relative_to(PROJECT_ROOT))
        if input_path.is_relative_to(PROJECT_ROOT)
        else input_path.name,
        "original_rows": len(original),
        "corrupted_rows": len(corrupted),
        "corruption_injected": True,
        "errors_detected": len(errors),
        "errors": errors,
        "failure_detected": len(errors) > 0,
        "status": "PASSED" if errors else "FAILED",
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run_simulation()
    print(json.dumps(result, indent=2))

    if not result["failure_detected"]:
        raise SystemExit("Failure simulation did not detect the injected corruption.")

    print("\nFailure simulation completed successfully.")
