import pandas as pd
import os

def generate_spatial_folds(metadata_path, out_path):
    """
    Create 4 spatial CV folds by splitting the city into 4 quadrants.
    Quadrants are defined by the median x and median y of patch centers.
    
    Output: folds.csv with a new column 'fold' ∈ {1,2,3,4}
    
    Fold assignment:
        Fold 1 → Q1 (upper-left)
        Fold 2 → Q2 (upper-right)
        Fold 3 → Q3 (lower-left)
        Fold 4 → Q4 (lower-right)
    """

    print("▶ Loading metadata...")
    df = pd.read_csv(metadata_path)

    # Check columns exist
    if not {"x", "y"}.issubset(df.columns):
        raise ValueError("metadata.csv must contain x and y columns.")

    # Compute medians → splitting lines
    x_med = df["x"].median()
    y_med = df["y"].median()

    print(f"   Median X: {x_med}")
    print(f"   Median Y: {y_med}")

    def assign_fold(row):
        x, y = row["x"], row["y"]

        if x <= x_med and y >= y_med:
            return 1  # Q1
        elif x > x_med and y >= y_med:
            return 2  # Q2
        elif x <= x_med and y < y_med:
            return 3  # Q3
        else:
            return 4  # Q4

    print("▶ Assigning folds to patches...")
    df["fold"] = df.apply(assign_fold, axis=1)

    print("▶ Saving folds file...")
    df.to_csv(out_path, index=False)

    # Diagnostics
    print("\n========================")
    print(" Spatial CV Statistics ")
    print("========================")
    for k in range(1, 5):
        n = (df["fold"] == k).sum()
        frac = n / len(df) * 100
        print(f"Fold {k}: {n} patches ({frac:.2f}%)")

    print("\nFile saved to:", out_path)
    print("Done ✓")


if __name__ == "__main__":
    metadata = "dataset/metadata.csv"
    out_csv  = "dataset/folds.csv"

    generate_spatial_folds(metadata, out_csv)