import argparse

import joblib
import pandas as pd

from data_utils import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description="Predict a chocolate bar rating.")
    parser.add_argument("--company", default="A. Morin")
    parser.add_argument("--bar-name", default="New Chocolate Bar")
    parser.add_argument("--ref", type=int, default=2000)
    parser.add_argument("--review-year", type=int, default=2017)
    parser.add_argument("--cocoa-percent", type=float, default=70.0)
    parser.add_argument("--company-location", default="France")
    parser.add_argument("--bean-type", default="Trinitario")
    parser.add_argument("--bean-origin", default="Venezuela")
    args = parser.parse_args()

    model_path = PROJECT_ROOT / "models" / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Run `python src/train.py` before prediction.")
    model = joblib.load(model_path)
    sample = pd.DataFrame([{
        "company": args.company,
        "bar_name": args.bar_name,
        "ref": args.ref,
        "review_year": args.review_year,
        "cocoa_percent": args.cocoa_percent,
        "company_location": args.company_location,
        "bean_type": args.bean_type,
        "bean_origin": args.bean_origin,
    }])
    prediction = float(model.predict(sample)[0])
    prediction = max(1.0, min(5.0, prediction))
    print(f"Predicted chocolate rating: {prediction:.2f} / 5.00")


if __name__ == "__main__":
    main()

