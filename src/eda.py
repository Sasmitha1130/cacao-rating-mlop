from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from data_utils import DEFAULT_DATA_PATH, PROJECT_ROOT, load_and_clean_data


def main():
    output_dir = PROJECT_ROOT / "outputs" / "eda"
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_and_clean_data(DEFAULT_DATA_PATH)

    print(f"Rows: {len(df)}, Columns: {df.shape[1]}")
    print("\nMissing values:\n", df.isna().sum())
    print("\nDuplicate rows:", df.duplicated().sum())
    print("\nNumeric summary:\n", df.describe())

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.histplot(df["rating"], bins=16, kde=True, color="#6f4e37")
    plt.title("Distribution of Chocolate Ratings")
    plt.tight_layout()
    plt.savefig(output_dir / "rating_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="cocoa_percent", y="rating", alpha=0.45, color="#8b5a2b")
    plt.title("Cocoa Percentage vs Rating")
    plt.tight_layout()
    plt.savefig(output_dir / "cocoa_vs_rating.png", dpi=160)
    plt.close()

    yearly = df.groupby("review_year", as_index=False)["rating"].mean()
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=yearly, x="review_year", y="rating", marker="o", color="#4b2e20")
    plt.title("Average Rating by Review Year")
    plt.tight_layout()
    plt.savefig(output_dir / "yearly_average_rating.png", dpi=160)
    plt.close()

    top_locations = df["company_location"].value_counts().head(10).index
    location_rating = (
        df[df["company_location"].isin(top_locations)]
        .groupby("company_location", as_index=False)["rating"].mean()
        .sort_values("rating")
    )
    plt.figure(figsize=(9, 6))
    sns.barplot(data=location_rating, x="rating", y="company_location", color="#a66a3f")
    plt.title("Average Rating: Top 10 Company Locations by Count")
    plt.xlim(0, 5)
    plt.tight_layout()
    plt.savefig(output_dir / "location_average_rating.png", dpi=160)
    plt.close()

    print(f"\nEDA charts saved in: {output_dir}")


if __name__ == "__main__":
    main()

