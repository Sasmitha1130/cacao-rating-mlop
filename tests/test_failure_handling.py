
import pandas as pd

from src.simulate_data_corruption import (
    simulate_corruption,
    validate_dataset,
)


def test_valid_dataset_has_no_errors():
    df = pd.DataFrame({
        "Rating": [3.5, 4.0],
        "Cocoa\nPercent": ["70%", "80%"],
    })

    assert validate_dataset(df) == []


def test_corrupted_rating_is_detected():
    df = pd.DataFrame({
        "Rating": [3.5, 4.0],
        "Cocoa\nPercent": ["70%", "80%"],
    })

    corrupted = simulate_corruption(df)
    errors = validate_dataset(corrupted)

    assert any("Rating" in error for error in errors)


def test_missing_rating_column_is_detected():
    df = pd.DataFrame({
        "Cocoa\nPercent": ["70%", "80%"],
    })

    errors = validate_dataset(df)

    assert any("Rating" in error for error in errors)


def test_invalid_cocoa_percentage_is_detected():
    df = pd.DataFrame({
        "Rating": [3.5],
        "Cocoa\nPercent": ["not-a-number"],
    })

    errors = validate_dataset(df)

    assert any("Cocoa percentage" in error for error in errors)
